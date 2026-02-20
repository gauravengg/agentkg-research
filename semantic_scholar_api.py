"""Semantic Scholar API Client"""
import requests
import time
from typing import Optional, List, Dict, Any
from loguru import logger
from models import Paper

class SemanticScholarAPI:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: Optional[str] = None, rate_limit: float = 0.5):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.headers = {}
        if api_key:
            self.headers['x-api-key'] = api_key
    
    def _rate_limit_delay(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None):
        self._rate_limit_delay()
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, headers=self.headers, params=params or {}, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def search_papers(self, query: str, limit: int = 10):
        endpoint = "paper/search"
        params = {
            'query': query,
            'limit': min(limit, 100),
            'fields': 'paperId,title,abstract,year,citationCount,authors,venue,url'
        }
        data = self._make_request(endpoint, params)
        papers = []
        for item in data.get('data', []):
            paper = self._parse_paper(item)
            if paper:
                papers.append(paper)
        logger.info(f"✓ Found {len(papers)} papers for '{query}'")
        return papers
    
    def _parse_paper(self, data: Dict):
        authors = data.get('authors', []) or []
        author_ids = [a.get('authorId') for a in authors if a.get('authorId')]
        
        return Paper(
            paper_id=data['paperId'],
            title=data.get('title', 'Untitled'),
            abstract=data.get('abstract'),
            year=data.get('year'),
            citation_count=data.get('citationCount', 0),
            url=data.get('url'),
            author_ids=author_ids,
            venue_id=data.get('venue')
        )
