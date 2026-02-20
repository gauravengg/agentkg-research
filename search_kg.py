"""
Knowledge Graph Search Module
Pre-defined search functions (Safe & Fast)
"""
from config import Config, Neo4jConnection

class KnowledgeGraphSearch:
    def __init__(self):
        self.config = Config()
        self.db = None
    
    def connect(self):
        self.db = Neo4jConnection(self.config)
    
    def close(self):
        if self.db:
            self.db.close()
    
    # ========================================
    # BASIC SEARCH FUNCTIONS
    # ========================================
    
    def search_papers_by_keyword(self, keyword):
        """Search papers by keyword in title or abstract"""
        query = """
        MATCH (p:Paper)
        WHERE toLower(p.title) CONTAINS toLower($keyword)
           OR toLower(p.abstract) CONTAINS toLower($keyword)
        OPTIONAL MATCH (a:Author)-[:WROTE]->(p)
        OPTIONAL MATCH (p)-[:ABOUT]->(t:Topic)
        OPTIONAL MATCH (a)-[:AFFILIATED_WITH]->(i:Institution)
        RETURN p.title as title, 
               p.year as year, 
               p.citation_count as citations,
               p.abstract as abstract,
               collect(DISTINCT a.name) as authors,
               collect(DISTINCT t.name) as topics,
               collect(DISTINCT i.name) as institutions
        ORDER BY p.citation_count DESC
        """
        return self.db.execute_query(query, {'keyword': keyword})
    
    def search_by_author(self, author_name):
        """Search papers by author name"""
        query = """
        MATCH (a:Author)-[:WROTE]->(p:Paper)
        WHERE toLower(a.name) CONTAINS toLower($author_name)
        OPTIONAL MATCH (a)-[:AFFILIATED_WITH]->(i:Institution)
        OPTIONAL MATCH (p)-[:ABOUT]->(t:Topic)
        RETURN a.name as author,
               a.h_index as h_index,
               i.name as institution,
               p.title as title,
               p.year as year,
               p.citation_count as citations,
               collect(DISTINCT t.name) as topics
        ORDER BY p.citation_count DESC
        """
        return self.db.execute_query(query, {'author_name': author_name})
    
    def search_by_institution(self, institution_name):
        """Search papers from specific institution"""
        query = """
        MATCH (i:Institution)<-[:AFFILIATED_WITH]-(a:Author)-[:WROTE]->(p:Paper)
        WHERE toLower(i.name) CONTAINS toLower($institution_name)
        OPTIONAL MATCH (p)-[:ABOUT]->(t:Topic)
        RETURN i.name as institution,
               a.name as author,
               p.title as title,
               p.year as year,
               p.citation_count as citations,
               collect(DISTINCT t.name) as topics
        ORDER BY p.citation_count DESC
        """
        return self.db.execute_query(query, {'institution_name': institution_name})
    
    def search_by_topic(self, topic):
        """Search papers about a specific topic"""
        query = """
        MATCH (p:Paper)-[:ABOUT]->(t:Topic)
        WHERE toLower(t.name) CONTAINS toLower($topic)
        OPTIONAL MATCH (a:Author)-[:WROTE]->(p)
        OPTIONAL MATCH (a)-[:AFFILIATED_WITH]->(i:Institution)
        RETURN t.name as topic,
               p.title as title,
               p.year as year,
               p.citation_count as citations,
               collect(DISTINCT a.name) as authors,
               collect(DISTINCT i.name) as institutions
        ORDER BY p.citation_count DESC
        """
        return self.db.execute_query(query, {'topic': topic})
    
    def get_most_cited_papers(self, limit=5):
        """Get most cited papers"""
        query = """
        MATCH (p:Paper)
        WHERE p.citation_count IS NOT NULL
        OPTIONAL MATCH (a:Author)-[:WROTE]->(p)
        OPTIONAL MATCH (p)-[:ABOUT]->(t:Topic)
        OPTIONAL MATCH (a)-[:AFFILIATED_WITH]->(i:Institution)
        RETURN p.title as title,
               p.year as year,
               p.citation_count as citations,
               collect(DISTINCT a.name) as authors,
               collect(DISTINCT t.name) as topics,
               collect(DISTINCT i.name) as institutions
        ORDER BY p.citation_count DESC
        LIMIT $limit
        """
        return self.db.execute_query(query, {'limit': limit})
    
    def find_collaborations(self, author_name):
        """Find who collaborates with an author"""
        query = """
        MATCH (a1:Author)-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
        WHERE toLower(a1.name) CONTAINS toLower($author_name)
          AND a1 <> a2
        OPTIONAL MATCH (a2)-[:AFFILIATED_WITH]->(i:Institution)
        WITH a2, i, count(DISTINCT p) as collaborations
        RETURN a2.name as collaborator,
               i.name as institution,
               collaborations
        ORDER BY collaborations DESC
        """
        return self.db.execute_query(query, {'author_name': author_name})
    
    def get_statistics(self):
        """Get overall graph statistics"""
        query = """
        MATCH (p:Paper)
        WITH count(p) as total_papers, 
             sum(p.citation_count) as total_citations,
             avg(p.citation_count) as avg_citations
        MATCH (a:Author)
        WITH total_papers, total_citations, avg_citations,
             count(a) as total_authors
        MATCH (t:Topic)
        WITH total_papers, total_citations, avg_citations, total_authors,
             count(t) as total_topics
        OPTIONAL MATCH (i:Institution)
        RETURN total_papers, total_citations, avg_citations, 
               total_authors, total_topics, count(i) as total_institutions
        """
        results = self.db.execute_query(query)
        return results[0] if results else {}
