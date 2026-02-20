"""Configuration and Database Connection"""
import os
from dataclasses import dataclass
from neo4j import GraphDatabase
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

@dataclass
class Config:
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    s2_api_key: str = os.getenv("S2_API_KEY", "")
    api_rate_limit: float = 0.5

class Neo4jConnection:
    def __init__(self, config: Config):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )
        self.driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {config.neo4j_uri}")
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def execute_query(self, query: str, parameters: dict = None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def execute_write(self, query: str, parameters: dict = None):
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run(query, parameters or {}))

def initialize_schema(db: Neo4jConnection):
    logger.info("Initializing schema...")
    queries = [
        "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
        "CREATE CONSTRAINT author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.author_id IS UNIQUE",
        "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)",
    ]
    for query in queries:
        try:
            db.execute_write(query)
            logger.info(f"✓ {query[:50]}...")
        except:
            pass
    logger.info("✓ Schema initialized")
