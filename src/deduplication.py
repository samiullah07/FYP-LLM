# src/deduplication.py
# Minimal deduplication via content hashing
from hashlib import sha256
from typing import Dict, Any


def paper_hash(paper: Dict[str, Any]) -> str:
    """
    Create deterministic hash for deduplication based on key bibliographic fields.
    """
    # Handle authors - could be a list, string, or None
    authors = paper.get("authors")
    if isinstance(authors, list):
        authors_str = "|".join(str(a) for a in authors)
    else:
        authors_str = str(authors or "")

    hash_str = "|".join([
        str(paper.get("title") or ""),
        authors_str,
        str(paper.get("year") or ""),
        str(paper.get("doi") or ""),
    ])
    return sha256(hash_str.encode()).hexdigest()


def deduplicate(papers: list) -> list:
    """
    Remove duplicate papers from a list using content hashing.
    """
    seen = set()
    unique = []
    for p in papers:
        h = paper_hash(p)
        if h not in seen:
            seen.add(h)
            unique.append(p)
    return unique
