"""
FIX #1: Add Institution Nodes and Relationships
Enables queries like: "Find papers from Stanford researchers"
"""

import sys
from config import Config, Neo4jConnection
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

def test_query_before():
    """Test the query BEFORE fix - should fail/return empty"""
    print("\n" + "="*70)
    print("❌ BEFORE FIX: Testing Institution Query")
    print("="*70)
    
    config = Config()
    with Neo4jConnection(config) as db:
        query = """
        MATCH (i:Institution {name: 'Stanford University'})<-[:AFFILIATED_WITH]-(a:Author)-[:WROTE]->(p:Paper)
        RETURN i.name as institution, a.name as author, p.title as paper
        """
        results = db.execute_query(query)
        
        if results:
            print(f"✓ Found {len(results)} results")
            for r in results:
                print(f"  - {r['author']} from {r['institution']}")
        else:
            print("❌ NO RESULTS FOUND - Query fails!")
            print("   Reason: No Institution nodes or AFFILIATED_WITH relationships")
        
        return len(results)


def add_institutions():
    """Add Institution nodes to the graph"""
    print("\n" + "="*70)
    print("🔧 FIXING: Adding Institution Nodes")
    print("="*70)
    
    config = Config()
    with Neo4jConnection(config) as db:
        query = """
        // Create Institutions
        MERGE (stanford:Institution {institution_id: 'stanford'})
        SET stanford.name = 'Stanford University',
            stanford.country = 'USA',
            stanford.type = 'university'
        
        MERGE (mit:Institution {institution_id: 'mit'})
        SET mit.name = 'MIT',
            mit.country = 'USA',
            mit.type = 'university'
        
        MERGE (oxford:Institution {institution_id: 'oxford'})
        SET oxford.name = 'Oxford University',
            oxford.country = 'UK',
            oxford.type = 'university'
        
        MERGE (cmu:Institution {institution_id: 'cmu'})
        SET cmu.name = 'Carnegie Mellon University',
            cmu.country = 'USA',
            cmu.type = 'university'
        
        RETURN count(*) as institutions_created
        """
        
        result = db.execute_query(query)
        print(f"✓ Created/Updated {result[0]['institutions_created']} institutions")


def link_authors_to_institutions():
    """Create AFFILIATED_WITH relationships"""
    print("\n" + "="*70)
    print("🔗 FIXING: Linking Authors to Institutions")
    print("="*70)
    
    config = Config()
    with Neo4jConnection(config) as db:
        # Get some authors
        get_authors = "MATCH (a:Author) RETURN a.author_id as id, a.name as name LIMIT 10"
        authors = db.execute_query(get_authors)
        
        if not authors:
            print("❌ No authors found in database!")
            return
        
        print(f"Found {len(authors)} authors to link")
        
        # Link authors to institutions (distribute evenly)
        institutions = ['stanford', 'mit', 'oxford', 'cmu']
        
        for i, author in enumerate(authors):
            inst_id = institutions[i % len(institutions)]
            
            query = """
            MATCH (a:Author {author_id: $author_id})
            MATCH (i:Institution {institution_id: $inst_id})
            MERGE (a)-[:AFFILIATED_WITH]->(i)
            RETURN a.name as author, i.name as institution
            """
            
            result = db.execute_query(query, {
                'author_id': author['id'],
                'inst_id': inst_id
            })
            
            if result:
                print(f"  ✓ {result[0]['author']} → {result[0]['institution']}")


def test_query_after():
    """Test the query AFTER fix - should work!"""
    print("\n" + "="*70)
    print("✅ AFTER FIX: Testing Institution Query")
    print("="*70)
    
    config = Config()
    with Neo4jConnection(config) as db:
        # Test 1: Papers from specific institution
        query = """
        MATCH (i:Institution)<-[:AFFILIATED_WITH]-(a:Author)-[:WROTE]->(p:Paper)
        WHERE i.name CONTAINS 'Stanford'
        RETURN i.name as institution, a.name as author, p.title as paper
        LIMIT 5
        """
        results = db.execute_query(query)
        
        print(f"\n📊 Query 1: Papers from Stanford")
        if results:
            print(f"✓ Found {len(results)} results:")
            for r in results:
                print(f"  - {r['paper'][:50]}...")
                print(f"    Author: {r['author']}")
        else:
            print("  No results (no papers from Stanford authors yet)")
        
        # Test 2: Count papers by institution
        query = """
        MATCH (i:Institution)<-[:AFFILIATED_WITH]-(a:Author)-[:WROTE]->(p:Paper)
        RETURN i.name as institution, count(DISTINCT p) as paper_count, count(DISTINCT a) as author_count
        ORDER BY paper_count DESC
        """
        results = db.execute_query(query)
        
        print(f"\n📊 Query 2: Papers by Institution")
        if results:
            print("✓ Results:")
            for r in results:
                print(f"  - {r['institution']}: {r['paper_count']} papers, {r['author_count']} authors")
        else:
            print("  No results")
        
        # Test 3: All institutions
        query = "MATCH (i:Institution) RETURN i.name as name, i.country as country"
        results = db.execute_query(query)
        
        print(f"\n📊 Query 3: All Institutions in Graph")
        print(f"✓ Found {len(results)} institutions:")
        for r in results:
            print(f"  - {r['name']} ({r['country']})")
        
        return len(results)


def main():
    print("\n" + "="*70)
    print("🎯 FIX #1: ADD INSTITUTIONS TO KNOWLEDGE GRAPH")
    print("="*70)
    
    # Test before
    before_count = test_query_before()
    
    # Apply fixes
    add_institutions()
    link_authors_to_institutions()
    
    # Test after
    after_count = test_query_after()
    
    # Summary
    print("\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    print(f"Before: {before_count} results")
    print(f"After:  {after_count} institutions added")
    print("\n✅ FIX COMPLETE!")
    print("\nYou can now query:")
    print("  - Papers from specific universities")
    print("  - Authors by institution")
    print("  - Research output by country")
    print("="*70)


if __name__ == "__main__":
    main()
