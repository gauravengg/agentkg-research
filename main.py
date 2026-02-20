"""Main Script - Research Knowledge Graph Builder"""
import sys
from loguru import logger
from config import Config, Neo4jConnection, initialize_schema
from semantic_scholar_api import SemanticScholarAPI
from ingestion import KnowledgeGraphIngester

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

def search_and_ingest():
    logger.info("=" * 60)
    logger.info("Research Knowledge Graph Builder")
    logger.info("=" * 60)
    
    config = Config()
    
    with Neo4jConnection(config) as db:
        initialize_schema(db)
        
        api = SemanticScholarAPI(api_key=config.s2_api_key, rate_limit=config.api_rate_limit)
        ingester = KnowledgeGraphIngester(db)
        
        # Get search query from user
        query = input("\nEnter research topic to search: ")
        limit = int(input("How many papers to fetch (max 20): ") or "10")
        
        logger.info(f"Searching for papers about '{query}'...")
        papers = api.search_papers(query=query, limit=min(limit, 20))
        
        logger.info(f"Found {len(papers)} papers")
        
        if papers:
            logger.info("Ingesting papers into Neo4j...")
            count = ingester.ingest_papers_batch(papers)
            logger.info(f"✓ Successfully ingested {count} papers!")
        else:
            logger.warning("No papers found")

if __name__ == "__main__":
    search_and_ingest()
