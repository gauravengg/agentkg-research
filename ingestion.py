"""Data Ingestion into Neo4j"""
from typing import List
from loguru import logger
from config import Neo4jConnection
from models import Paper, Topic

class KnowledgeGraphIngester:
    def __init__(self, db: Neo4jConnection):
        self.db = db
    
    def ingest_paper(self, paper: Paper):
        query = """
        MERGE (p:Paper {paper_id: $paper_id})
        SET p.title = $title,
            p.abstract = $abstract,
            p.year = $year,
            p.citation_count = $citation_count,
            p.url = $url
        """
        params = {
            'paper_id': paper.paper_id,
            'title': paper.title,
            'abstract': paper.abstract,
            'year': paper.year,
            'citation_count': paper.citation_count,
            'url': paper.url
        }
        self.db.execute_write(query, params)
        logger.debug(f"✓ Ingested: {paper.title[:50]}...")
    
    def link_author_to_paper(self, author_id: str, paper_id: str):
        query = """
        MATCH (p:Paper {paper_id: $paper_id})
        MERGE (a:Author {author_id: $author_id})
        MERGE (a)-[:WROTE]->(p)
        """
        self.db.execute_write(query, {'author_id': author_id, 'paper_id': paper_id})
    
    def link_paper_to_topic(self, paper_id: str, topic_name: str):
        query = """
        MATCH (p:Paper {paper_id: $paper_id})
        MERGE (t:Topic {name: $topic_name})
        MERGE (p)-[:ABOUT]->(t)
        """
        self.db.execute_write(query, {'paper_id': paper_id, 'topic_name': topic_name})
    
    def ingest_paper_full(self, paper: Paper):
        self.ingest_paper(paper)
        for author_id in paper.author_ids:
            self.link_author_to_paper(author_id, paper.paper_id)
        for topic in paper.topics:
            self.link_paper_to_topic(paper.paper_id, topic)
        logger.info(f"✓ Fully ingested: {paper.title[:50]}...")
    
    def ingest_papers_batch(self, papers: List[Paper]):
        count = 0
        for paper in papers:
            self.ingest_paper_full(paper)
            count += 1
        logger.info(f"✓ Ingested {count}/{len(papers)} papers")
        return count
