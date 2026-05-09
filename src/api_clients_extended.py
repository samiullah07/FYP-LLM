from typing import List, Optional
import os
import time
import requests
import semanticscholar as sch

from .config import settings
from .models import Paper

def _paper_from_semanticscholar_json(item: dict) -> Paper:
    """Convert Semantic Scholar JSON object to Paper model."""
    authors = [
        auth.get('name', '')
        for auth in item.get('authors', [])
    ]

    return Paper(
        paper_id=item.get('paperId', ''),
        title=item.get('title', ''),
        abstract=item.get('abstract', ''),
        authors=authors,
        year=item.get('year'),
        venue=item.get('venue', ''),
        doi=item.get('doi', ''),
        source="semantic_scholar",
    )

def _paper_from_crossref_json(item: dict) -> Paper:
    """Convert Crossref JSON object to Paper model."""
    authors = []
    if 'author' in item:
        for auth in item['author']:
            name = auth.get('given', '') + ' ' + auth.get('family', '')
            if name.strip():
                authors.append(name.strip())

    year = None
    if 'published' in item and 'date-parts' in item['published']:
        date_parts = item['published']['date-parts'][0]
        if len(date_parts) > 0:
            year = date_parts[0]

    return Paper(
        paper_id=item.get('DOI', ''),
        title=item.get('title', [])[0] if item.get('title') else '',
        abstract='',
        authors=authors,
        year=year,
        venue=item.get('container-title', [])[0] if item.get('container-title') else '',
        doi=item.get('DOI', ''),
        source="crossref",
    )

def search_semantic_scholar(query: str, max_results: int = 5) -> List[Paper]:
    """Search Semantic Scholar API for papers."""
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

    try:
        # Initialize Semantic Scholar client
        sch_client = sch.SemanticScholar(api_key=api_key)

        # Search papers
        results = sch_client.search_paper(query, limit=max_results)

        papers = []
        for result in results[:max_results]:
            papers.append(_paper_from_semanticscholar_json(result))

        return papers

    except Exception as e:
        print(f"[Semantic Scholar] Search failed: {e}")
        return []

def search_crossref(query: str, max_results: int = 5) -> List[Paper]:
    """Search Crossref API for papers."""
    try:
        url = "https://api.crossref.org/works"
        params = {
            'query': query,
            'rows': max_results,
            'cursor': '*'
        }

        # Add API key if available
        api_key = os.environ.get("CROSS_REF_API_KEY", "")
        if api_key:
            params['mailto'] = os.environ.get("OPENALEX_EMAIL", "samiullah02jan1999@gmail.com")

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        papers = []
        for item in data.get('message', {}).get('items', [])[:max_results]:
            papers.append(_paper_from_crossref_json(item))

        return papers

    except Exception as e:
        print(f"[Crossref] Search failed: {e}")
        return []

def triangulate_citation(author: str, year: int, title: str,
                        sources: List[str] = ["openalex", "crossref", "semantic_scholar"]) -> str:
    """
    Search all sources and return consensus verdict.

    Returns:
        - "CONFIRMED" if found in 2+ sources with matching details
        - "SINGLE_SOURCE" if found in only 1 source
        - "NOT_FOUND" if found in 0 sources
    """
    from .api_clients import search_openalex_works

    confirmed_count = 0
    sources_found = []

    # Check OpenAlex
    if "openalex" in sources:
        query = f"{author} {year} {title[:100]}"
        try:
            results = search_openalex_works(query, 1)
            if results:
                sources_found.append("openalex")
                # Check if details match (simplified)
                if results[0].year == year and author.lower() in results[0].authors[0].lower():
                    confirmed_count += 1
        except Exception:
            pass

    # Check Crossref
    if "crossref" in sources:
        query = f"{author} {year} {title[:100]}"
        try:
            results = search_crossref(query, 1)
            if results:
                sources_found.append("crossref")
                # Check if details match (simplified)
                if results[0].year == year and author.lower() in ' '.join(results[0].authors).lower():
                    confirmed_count += 1
        except Exception:
            pass

    # Check Semantic Scholar
    if "semantic_scholar" in sources:
        query = f"{author} {year} {title[:100]}"
        try:
            results = search_semantic_scholar(query, 1)
            if results:
                sources_found.append("semantic_scholar")
                # Check if details match (simplified)
                if results[0].year == year and author.lower() in ' '.join(results[0].authors).lower():
                    confirmed_count += 1
        except Exception:
            pass

    # Return consensus verdict
    if confirmed_count >= 2:
        return "CONFIRMED"
    elif len(sources_found) >= 1:
        return "SINGLE_SOURCE"
    else:
        return "NOT_FOUND"