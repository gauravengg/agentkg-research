"""
Test all pre-defined search functions
"""
from search_kg import KnowledgeGraphSearch

def test_all_functions():
    searcher = KnowledgeGraphSearch()
    searcher.connect()
    
    print("="*70)
    print("TESTING ALL SEARCH FUNCTIONS")
    print("="*70)
    
    # Test 1: Statistics
    print("\n1️⃣  Testing get_statistics()...")
    stats = searcher.get_statistics()
    print(f"✓ Stats: {stats}")
    
    # Test 2: Most cited papers
    print("\n2️⃣  Testing get_most_cited_papers()...")
    papers = searcher.get_most_cited_papers(3)
    print(f"✓ Found {len(papers)} papers")
    for p in papers[:2]:
        print(f"   - {p['title'][:50]}... ({p['citations']} citations)")
    
    # Test 3: Search by keyword
    print("\n3️⃣  Testing search_papers_by_keyword('graph')...")
    papers = searcher.search_papers_by_keyword('graph')
    print(f"✓ Found {len(papers)} papers")
    for p in papers[:2]:
        print(f"   - {p['title'][:50]}...")
    
    # Test 4: Search by author
    print("\n4️⃣  Testing search_by_author('Sarah')...")
    papers = searcher.search_by_author('Sarah')
    print(f"✓ Found {len(papers)} papers")
    if papers:
        print(f"   - Author: {papers[0]['author']}")
        print(f"   - Institution: {papers[0]['institution']}")
    
    # Test 5: Search by institution
    print("\n5️⃣  Testing search_by_institution('IIT')...")
    papers = searcher.search_by_institution('IIT')
    print(f"✓ Found {len(papers)} papers")
    for p in papers[:2]:
        print(f"   - {p['title'][:50]}...")
        print(f"     By: {p['author']}")
    
    # Test 6: Search by topic
    print("\n6️⃣  Testing search_by_topic('Graph Neural')...")
    papers = searcher.search_by_topic('Graph Neural')
    print(f"✓ Found {len(papers)} papers")
    for p in papers[:2]:
        print(f"   - {p['title'][:50]}...")
    
    # Test 7: Find collaborations
    print("\n7️⃣  Testing find_collaborations('Sarah')...")
    collabs = searcher.find_collaborations('Sarah')
    print(f"✓ Found {len(collabs)} collaborators")
    for c in collabs:
        print(f"   - {c['collaborator']} ({c['collaborations']} papers)")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE!")
    print("="*70)
    
    searcher.close()

if __name__ == "__main__":
    test_all_functions()
