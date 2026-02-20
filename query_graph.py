"""
Interactive Knowledge Graph Query Tool (Terminal)
Supports Institution / University wise queries
"""

from config import Config, Neo4jConnection
from loguru import logger
import sys

# --------------------
# Logger setup
# --------------------
logger.remove()
logger.add(sys.stderr, level="INFO")


def print_menu():
    print("\n" + "=" * 60)
    print("📊 Knowledge Graph Query Menu")
    print("=" * 60)
    print("1. Show all papers")
    print("2. Show all authors")
    print("3. Show citations")
    print("4. Show collaborations")
    print("5. Papers by topic")
    print("6. Run custom Cypher query")
    print("7. University-wise authors & papers")
    print("0. Exit")
    print("=" * 60)


def run_queries():
    config = Config()

    with Neo4jConnection(config) as db:
        while True:
            print_menu()
            choice = input("Enter your choice: ").strip()

            # --------------------
            # Exit
            # --------------------
            if choice == "0":
                print("👋 Exiting Knowledge Graph Tool")
                break

            # --------------------
            # 1. Show all papers
            # --------------------
            elif choice == "1":
                query = """
                MATCH (p:Paper)
                RETURN p.title AS title,
                       p.year AS year,
                       p.citation_count AS citations
                ORDER BY citations DESC
                """
                results = db.execute_query(query)
                for r in results:
                    print(f"• {r['title']} ({r['year']}) - {r['citations']} citations")

            # --------------------
            # 2. Show all authors
            # --------------------
            elif choice == "2":
                query = """
                MATCH (a:Author)
                RETURN a.name AS name,
                       a.h_index AS h_index
                ORDER BY h_index DESC
                """
                results = db.execute_query(query)
                for r in results:
                    print(f"• {r['name']} (h-index: {r['h_index']})")

            # --------------------
            # 3. Show citations
            # --------------------
            elif choice == "3":
                query = """
                MATCH (p1:Paper)-[:CITES]->(p2:Paper)
                RETURN p1.title AS citing,
                       p2.title AS cited
                """
                results = db.execute_query(query)
                if results:
                    for r in results:
                        print(f"• {r['citing'][:40]} → {r['cited'][:40]}")
                else:
                    print("❌ No citations found")

            # --------------------
            # 4. Show collaborations
            # --------------------
            elif choice == "4":
                query = """
                MATCH (a1:Author)-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
                WHERE a1.author_id < a2.author_id
                RETURN a1.name AS author1,
                       a2.name AS author2,
                       p.title AS title
                """
                results = db.execute_query(query)
                for r in results:
                    print(f"• {r['author1']} & {r['author2']}")
                    print(f"  Paper: {r['title'][:60]}")

            # --------------------
            # 5. Papers by topic
            # --------------------
            elif choice == "5":
                query = """
                MATCH (p:Paper)-[:ABOUT]->(t:Topic)
                RETURN t.name AS topic,
                       count(p) AS paper_count
                ORDER BY paper_count DESC
                """
                results = db.execute_query(query)
                for r in results:
                    print(f"• {r['topic']}: {r['paper_count']} papers")

            # --------------------
            # 6. Custom Cypher query
            # --------------------
            elif choice == "6":
                print("\nEnter your Cypher query:")
                custom_query = input(">>> ")
                try:
                    results = db.execute_query(custom_query)
                    for r in results:
                        print(r)
                except Exception as e:
                    print("❌ Error executing query:", e)

            # --------------------
            # 7. University-wise authors & papers (NEW)
            # --------------------
            elif choice == "7":
                inst_name = input("Enter university/institution name: ").strip()

                query = """
                MATCH (i:Institution {name: $inst_name})
                      <-[:AFFILIATED_WITH]-(a:Author)
                      -[:WROTE]->(p:Paper)
                RETURN a.name AS author,
                       p.title AS paper,
                       p.year AS year,
                       p.citation_count AS citations
                ORDER BY citations DESC
                """

                results = db.execute_query(query, {"inst_name": inst_name})

                if results:
                    print(f"\n🏫 Institution: {inst_name}")
                    print("-" * 50)
                    for r in results:
                        print(f"• Author: {r['author']}")
                        print(f"  Paper : {r['paper'][:70]}")
                        print(f"  Year  : {r['year']} | Citations: {r['citations']}")
                        print()
                else:
                    print("❌ No data found for this institution")

            # --------------------
            # Invalid input
            # --------------------
            else:
                print("❌ Invalid choice. Try again.")


if __name__ == "__main__":
    run_queries()
