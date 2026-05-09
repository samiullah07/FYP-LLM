# Base retriever interface for academic paper retrieval
"""
Base retriever interface defining common patterns for retrieving,
normalizing, and deduplicating academic papers from multiple sources.
"""
import hashlib
import logging
from typing import Dict, List, Any, Optional
from src.models import Paper

# Configure logging
logger = logging.getLogger(__name__)


class BaseRetriever:
    """
    Base retriever class implementing shared functionality.

    Subclasses must implement:
    - fetch_papers(query: str, limit: int) -> List[Dict]
    - _call_api(params: Dict) -> Dict
    """

    def __init__(self, source_name: str):
        """
        Initialize retriever with source name.

        Args:
            source_name: Identifier for this retriever (used in provenance tracking)
        """
        self.source_name = source_name

    def fetch_papers(self, query: str, limit: int = 10) -> List[Paper]:
        """
        Retrieve papers based on query.

        This method delegates to _call_api which must be implemented.

        Args:
            query: Search term or query string
            limit: Maximum number of papers to return

        Returns:
            List of normalized Paper objects
        """
        logger.info(f"Retrieving papers from {self.source_name}: {query}")
        raw_results = self._call_api(query, limit)

        # Normalize results and track provenance
        papers = []
        seen_hashes = set()

        for item in raw_results:
            paper = self.normalize_paper(item)
            # Skip duplicates using content hash
            paper_hash = self._hash_content(paper)
            if paper_hash not in seen_hashes:
                seen_hashes.add(paper_hash)
                papers.append(paper)

        logger.info(f"Found {len(papers)} unique papers from {self.source_name}")
        return papers

    def normalize_paper(self, raw_data: Dict[str, Any]) -> Paper:
        """
        Convert raw API response into canonical Paper schema.
        Must be overridden by subclasses.

        Args:
            raw_data: Raw API response dictionary

        Returns:
            Normalized Paper object conforming to canonical schema
        """
        raise NotImplementedError("Subclasses must implement normalize_paper()")

    def _hash_content(self, paper: Paper) -> str:
        """
        Create hash for deduplication based on key fields.

        Args:
            paper: Normalized paper object

        Returns:
            Hexadecimal hash string
        """
        # Create deterministic hash based on key bibliographic fields
        hash_str = "|".join([
            paper.get("title", ""),
            "|".join(paper.get("authors", [])),
            str(paper.get("year", "")),
            paper.get("doi", "")
        ])
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def get_provenance(self) -> Dict[str, str]:
        """
        Get source-specific provenance information.

        Returns:
            Dictionary with source metadata for audit trails
        """
        return {
            "source": self.source_name,
            "provider": "custom_retriever"
        }