"""Data Models for Knowledge Graph"""
from pydantic import BaseModel, Field
from typing import Optional, List

class Author(BaseModel):
    author_id: str
    name: str
    h_index: Optional[int] = None
    paper_count: Optional[int] = None
    citation_count: Optional[int] = None

class Paper(BaseModel):
    paper_id: str
    title: str
    abstract: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    citation_count: Optional[int] = 0
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    author_ids: List[str] = []
    topics: List[str] = []
    reference_ids: List[str] = []
    venue_id: Optional[str] = None

class Topic(BaseModel):
    name: str
    description: Optional[str] = None
