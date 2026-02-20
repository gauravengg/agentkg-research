"""
Add Sample Research Papers with Institution Support
Idempotent version using MERGE (safe to re-run)
"""

from config import Config, Neo4jConnection, initialize_schema
from loguru import logger
import sys

# --------------------
# Logger setup
# --------------------
logger.remove()
logger.add(sys.stderr, level="INFO")


def add_sample_data():
    config = Config()

    with Neo4jConnection(config) as db:
        initialize_schema(db)

        logger.info("Adding sample research papers (safe mode)...")

        query = """
        // ============================================================
        // Papers
        // ============================================================
        MERGE (p1:Paper {paper_id: 'sample1'})
        SET p1.title = 'Graph Neural Networks: A Review of Methods and Applications',
            p1.year = 2021,
            p1.citation_count = 1245,
            p1.abstract = 'This paper surveys graph neural networks and their applications.'

        MERGE (p2:Paper {paper_id: 'sample2'})
        SET p2.title = 'Deep Learning on Graphs: A Survey',
            p2.year = 2020,
            p2.citation_count = 892,
            p2.abstract = 'We provide a comprehensive survey of deep learning methods for graphs.'

        MERGE (p3:Paper {paper_id: 'sample3'})
        SET p3.title = 'Attention Mechanisms in Graph Neural Networks',
            p3.year = 2022,
            p3.citation_count = 456,
            p3.abstract = 'This work introduces attention mechanisms for graph neural networks.'

        // ============================================================
        // Institution
        // ============================================================
        MERGE (i1:Institution {inst_id: 'iit_tpt'})
        SET i1.name = 'IIT Tirupati',
            i1.country = 'India'

        // ============================================================
        // Authors
        // ============================================================
        MERGE (a1:Author {author_id: 'auth1'})
        SET a1.name = 'Dr. Sarah Johnson',
            a1.h_index = 45

        MERGE (a2:Author {author_id: 'auth2'})
        SET a2.name = 'Dr. Michael Chen',
            a2.h_index = 52

        MERGE (a3:Author {author_id: 'auth3'})
        SET a3.name = 'Dr. Emma Williams',
            a3.h_index = 38

        // ============================================================
        // Topics
        // ============================================================
        MERGE (t1:Topic {name: 'Graph Neural Networks'})
        MERGE (t2:Topic {name: 'Deep Learning'})
        MERGE (t3:Topic {name: 'Machine Learning'})

        // ============================================================
        // Author -> Paper
        // ============================================================
        MERGE (a1)-[:WROTE]->(p1)
        MERGE (a2)-[:WROTE]->(p1)
        MERGE (a2)-[:WROTE]->(p2)
        MERGE (a3)-[:WROTE]->(p3)

        // ============================================================
        // Paper -> Topic
        // ============================================================
        MERGE (p1)-[:ABOUT]->(t1)
        MERGE (p1)-[:ABOUT]->(t2)
        MERGE (p2)-[:ABOUT]->(t1)
        MERGE (p2)-[:ABOUT]->(t3)
        MERGE (p3)-[:ABOUT]->(t1)

        // ============================================================
        // Citations
        // ============================================================
        MERGE (p3)-[:CITES]->(p1)
        MERGE (p3)-[:CITES]->(p2)

        // ============================================================
        // NEW: Author -> Institution
        // ============================================================
        MERGE (a1)-[:AFFILIATED_WITH]->(i1)
        MERGE (a2)-[:AFFILIATED_WITH]->(i1)
        MERGE (a3)-[:AFFILIATED_WITH]->(i1)
        """

        db.execute_write(query)

        logger.info("✓ Sample data added successfully")
        logger.info("✓ Script is idempotent (safe to re-run)")
        logger.info("Try this query in Neo4j Browser:")
        logger.info(
            "MATCH (a:Author)-[:AFFILIATED_WITH]->(i:Institution) "
            "RETURN a.name AS author, i.name AS institution"
        )


if __name__ == "__main__":
    add_sample_data()
