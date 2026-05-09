import time
import logging
from typing import List, Dict, Any
import requests

from src.paper_normalizer import normalize_paper
from retrievers.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class CrossrefRetriever(BaseRetriever):
    """Crossref metadata retrieval with polite pool headers and retry/backoff."""

    def __init__(self):
        # No API key required for Crossref
        self.headers = {
            "User-Agent": "FYP-LLM/1.0 (mailto:samiullah02jan1999@gmail.com)"
        }
        self.cache = {}

    def fetch_papers(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch papers from Crossref API with retry/backoff."""
        url = f"https://api.crossref.org/works?query={query}&rows={limit}"

        try:
            raw_data = self._make_request(url)
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return []

        # Extract items from response
        items = raw_data.get("message", {}).get("items", [])
        if not items:
            logger.warning("No papers found in Crossref response")
            return []

        papers = []
        seen_hashes = set()

        for item in items[:limit]:
            # Convert Crossref-specific fields before normalization
            paper = self._preprocess_item(item)
            normalized = normalize_paper(paper)
            # Deduplicate using title+year+doi
            raw_key = f"{normalized.get('title','')}|{normalized.get('year','')}|{normalized.get('doi','')}"
            paper_hash = hash(raw_key)
            if paper_hash not in seen_hashes:
                seen_hashes.add(paper_hash)
                papers.append(normalized)

        logger.info(f"Retrieved {len(papers)} unique papers from Crossref")
        return papers

    def _preprocess_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Crossref-specific fields to our intermediate format."""
        # Title: Crossref returns a list, take first element
        title_list = item.get("title", [])
        title = title_list[0] if title_list else ""

        # Authors: Crossref uses {"given": "...", "family": "..."}
        authors = []
        for author in item.get("author", []):
            given = author.get("given", "")
            family = author.get("family", "")
            full_name = f"{given} {family}".strip()
            if full_name:
                authors.append(full_name)

        # Year: from published-print or published-online date-parts [[year, month, day]]
        year = None
        for date_key in ["published-print", "published-online"]:
            date_parts = item.get(date_key, {}).get("date-parts", [])
            if date_parts and date_parts[0]:
                year = date_parts[0][0]
                break

        # Abstract: Crossref doesn't always provide this
        abstract = item.get("abstract", "")

        # DOI
        doi = item.get("DOI", "")

        # Venue: container-title is a list, take first
        container_list = item.get("container-title", [])
        venue = container_list[0] if container_list else ""

        # URL
        url = item.get("URL", "")

        return {
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "doi": doi,
            "venue": venue,
            "url": url,
            "source": "crossref",
        }

    def _make_request(self, url: str) -> Dict[str, Any]:
        """Execute GET request with retry/backoff (same as SemanticScholarRetriever)."""
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
        return {
            "source": "crossref",
            "provider": "API-based",
            "identifier": "polite-pool"
        }
