# src/api_clients.py
from typing import List, Optional
import requests

from .config import settings
from .models import Paper

OPENALEX_WORKS_URL = f"{settings.openalex_base_url}/works"

# src/api_clients.py

def _paper_from_openalex_json(item: dict) -> Paper:
    """
    Convert a single OpenAlex /works JSON object into a Paper model.
    """

    authors = [
        auth.get("author", {}).get("display_name", "")
        for auth in item.get("authorships", [])
        if auth.get("author")
    ]

    # Try to get a plain abstract first
    abstract = item.get("abstract")

    # If not present, and abstract_inverted_index is available and not None,
    # try a safe reconstruction. Otherwise, keep abstract as None.
    inverted = item.get("abstract_inverted_index")
    if abstract is None and isinstance(inverted, dict) and inverted:
        try:
            max_pos = max(pos for positions in inverted.values() for pos in positions)
            words = [""] * (max_pos + 1)
            for word, positions in inverted.items():
                for pos in positions:
                    words[pos] = word
            abstract = " ".join(words)
        except Exception:
            # If anything goes wrong, just leave abstract as None
            abstract = None

    return Paper(
        paper_id=item.get("id", ""),
        title=item.get("display_name", ""),
        abstract=abstract,
        authors=authors,
        year=item.get("publication_year"),
        venue=(item.get("host_venue") or {}).get("display_name"),
        doi=item.get("doi"),
        source="openalex",
    )
def search_openalex_works(query: str, max_results: int = 10) -> List[Paper]:
    params = {"search": query, "per-page": max_results}
    resp = requests.get(OPENALEX_WORKS_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    return [_paper_from_openalex_json(item) for item in results]

def get_paper_by_doi_openalex(doi: str) -> Optional[Paper]:
    params = {"filter": f"doi:{doi}", "per-page": 1}
    resp = requests.get(OPENALEX_WORKS_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    if not results:
        return None
    return _paper_from_openalex_json(results[0])