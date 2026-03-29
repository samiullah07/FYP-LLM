# agents/search_agent.py
"""
Search Agent for the Literature Review Pipeline.

Responsibility:
    Given a list of sub-queries from the Planner Agent, search OpenAlex
    for relevant academic papers and return a deduplicated list of papers.

Input:
    sub_queries (List[str]): list of focused academic search queries

Output:
    List[Paper]: deduplicated list of Paper objects from OpenAlex
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api_clients import search_openalex_works
from src.models import Paper
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maximum total papers to retrieve across all sub-queries
MAX_PAPERS_PER_QUERY = 5


# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------


def retrieve_papers(sub_queries: list[str]) -> list[Paper]:
    all_papers: list[Paper] = []
    seen_dois:  set[str]    = set()
    seen_titles: set[str]   = set()

    for query in sub_queries:
        print(f"[SearchAgent] Searching: '{query}'")
        try:
            papers = search_openalex_works(
                query=query,
                max_results=MAX_PAPERS_PER_QUERY,
            )
        except Exception as e:
            print(f"[SearchAgent] ERROR: {e}")
            continue

        # ADD: small delay to avoid API rate limiting
        time.sleep(0.5)

        added = 0
        for paper in papers:
            if paper.doi and paper.doi in seen_dois:
                continue
            title_key = paper.title.lower().strip()
            if title_key in seen_titles:
                continue
            if paper.doi:
                seen_dois.add(paper.doi)
            seen_titles.add(title_key)
            all_papers.append(paper)
            added += 1

        print(f"[SearchAgent] Added {added} (total: {len(all_papers)})")

    print(f"\n[SearchAgent] Final unique paper count: {len(all_papers)}")
    return all_papers
def print_papers(papers: list[Paper]) -> None:
    """
    Helper to print a summary of retrieved papers.

    Parameters
    ----------
    papers : list[Paper]
        List of Paper objects to display.
    """
    print("\n[SearchAgent] Retrieved Papers:")
    print("-" * 60)
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.authors[:2]) if p.authors else "Unknown"
        print(f"{i}. {p.title}")
        print(f"   Authors : {authors}")
        print(f"   Year    : {p.year or 'N/A'}")
        print(f"   DOI     : {p.doi or 'N/A'}")
        print(f"   Venue   : {p.venue or 'N/A'}")
        print()