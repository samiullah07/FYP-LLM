# src/models.py
from typing import List, Optional, Literal
from pydantic import BaseModel

class Paper(BaseModel):
    paper_id: str
    title: str
    abstract: Optional[str] = None
    authors: List[str] = []
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    source: Literal["openalex", "semanticscholar", "manual"] = "openalex"

class Claim(BaseModel):
    text: str
    paper_ids: List[str] = []
    is_supported: Optional[bool] = None

class Citation(BaseModel):
    raw_reference: str
    matched_paper_id: Optional[str] = None
    valid: Optional[bool] = None
    error_reason: Optional[str] = None