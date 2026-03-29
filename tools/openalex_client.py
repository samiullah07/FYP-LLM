import requests
from config import OPENALEX_BASE_URL, OPENALEX_EMAIL
from utils.logger import get_logger

logger = get_logger("OpenAlexClient")

def search_papers(query: str, max_results: int = 10) -> list[dict]:
    url = f"{OPENALEX_BASE_URL}/works"
    params = {
        "search": query,
        "per-page": max_results,
        "mailto": OPENALEX_EMAIL,
        "select": "id,title,authorships,publication_year,doi,primary_location"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])
        papers = []
        for r in results:
            authors = [
                a["author"]["display_name"]
                for a in r.get("authorships", [])
                if a.get("author")
            ]
            primary = r.get("primary_location") or {}
            source = primary.get("source") or {}
            papers.append({
                "title": r.get("title", ""),
                "authors": authors,
                "year": r.get("publication_year"),
                "doi": r.get("doi", ""),
                "venue": source.get("display_name", "Unknown")
            })
        return papers
    except requests.RequestException as e:
        logger.error(f"OpenAlex search failed: {e}")
        return []

def verify_citation(title: str, authors: list[str], year: int) -> dict:
    """Check if a citation matches a real paper in OpenAlex."""
    query = f"{title} {' '.join(authors[:2])}"
    papers = search_papers(query, max_results=5)
    for paper in papers:
        title_match = title.lower() in paper["title"].lower()
        year_match = paper["year"] == year
        if title_match:
            return {
                "matched": True,
                "year_correct": year_match,
                "found_year": paper["year"],
                "doi": paper["doi"],
                "status": "valid" if year_match else "partial"
            }
    return {"matched": False, "status": "hallucinated"}
