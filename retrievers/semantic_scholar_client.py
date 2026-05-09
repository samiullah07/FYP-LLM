import os
import time
import logging
from typing import List, Dict, Any
import requests
from dotenv import load_dotenv
load_dotenv()

# Use absolute import to avoid package confusion
from src.paper_normalizer import normalize_paper
from retrievers.base_retriever import BaseRetriever

# Configure project-wide logger
logger = logging.getLogger(__name__)


class SemanticScholarRetriever(BaseRetriever):
    """Search-based Semantic Scholar paper retriever with safe parsing."""

    def __init__(self):
        self.api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
        if not self.api_key:
            raise ValueError("Missing SEMANTIC_SCHOLAR_API_KEY environment variable")

        self.headers = {
            'Content-Type': 'application/json',  # Corrected from sjson
            'X-Api-Key': self.api_key
        }
        self.cache = {}

    def fetch_papers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch papers with safe API response parsing and deduplication."""
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}"

        try:
            raw_data = self._make_request(url)
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return []

        # Validate response structure
        papers_data = raw_data.get('data', [])
        if not papers_data:
            logger.warning("No papers found in Semantic Scholar response")
            return []

        # Apply limit and deduplicate
        papers = []
        seen_hashes = set()
        for paper in papers_data[:limit]:
            # Convert Semantic Scholar author dicts to simple strings
            if isinstance(paper.get('authors'), list):
                paper['authors'] = [
                    a.get('name', '') if isinstance(a, dict) else str(a)
                    for a in paper['authors']
                ]
            # Extract DOI from externalIds if missing
            if not paper.get('doi') and isinstance(paper.get('externalIds'), dict):
                paper['doi'] = paper['externalIds'].get('DOI', '')
            # Set source provenance
            paper['source'] = 'semanticscholar'
            normalized = normalize_paper(paper)
            # Deduplicate using title+year+doi
            raw_key = f"{normalized.get('title','')}|{normalized.get('year','')}|{normalized.get('doi','')}"
            paper_hash = hash(raw_key)
            if paper_hash not in seen_hashes:
                seen_hashes.add(paper_hash)
                papers.append(normalized)

        logger.info(f"Retrieved {len(papers)} unique papers")
        return papers

    def _make_request(self, url: str) -> Dict[str, Any]:
        """Execute GET request with retry/backoff."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 429:  # Rate limited
                    sleep_time = 2 ** attempt
                    logger.warning(f"Rate limited, sleeping {sleep_time}s")
                    time.sleep(sleep_time)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)
        return {}

    def get_provenance(self) -> Dict[str, str]:
        """Return source-specific provenance metadata."""
        api_key_prefix = self.api_key[:8] + "..." if self.api_key else "unknown"
        return {
            "source": "semantic_scholar",
            "provider": "API-based",
            "identifier": f"API-{api_key_prefix}"
        }