# src/paper_normalizer.py
# Canonical schema conversion for academic papers
from typing import Dict, Any


def normalize_paper(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert source-specific paper data to standardized schema.
    Handles missing fields with safe defaults.
    """
    return {
        "title": str(raw.get("title", "")).strip(),
        "paper_id": str((raw.get("externalIds", {}).get("DOI") if isinstance(raw.get("externalIds"), dict) else None) or raw.get("doi") or raw.get("paperId") or "").strip(),
        "authors": [str(a).strip() for a in (raw.get("authors") or []) if str(a).strip()],
        "year": _to_int(raw.get("year")),
        "abstract": str(raw.get("abstract", "")).strip(),
        "doi": str(raw.get("doi", "")).strip(),
        "venue": str(raw.get("venue", "")).strip(),
        "url": str(raw.get("url", "")).strip(),
        "source": str(raw.get("source", "unknown")).strip(),
        "citation_count": _to_int(raw.get("citationCount") or raw.get("citation_count") or raw.get("cited_by_count", 0)),
        "relevance_score": _to_float(raw.get("relevance_score") or raw.get("confidence", 0.5)),
    }


def _to_int(val) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _to_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.5
