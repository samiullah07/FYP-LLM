# agents/verifier_agent.py
"""
Verifier Agent for the Literature Review Pipeline.

Responsibility:
    This is the CORE contribution of the project.
    It verifies citations in a generated literature review against:
        1. Locally retrieved papers (from Search Agent)
        2. OpenAlex API (live lookup for unmatched citations)

    For each citation found in the review text, it determines:
        - "valid"       : author + year match a real retrieved paper
        - "partial"     : paper found but year is slightly off (±1)
        - "hallucinated": no matching paper found anywhere

Metrics returned:
    - total citations found
    - valid / partial / hallucinated counts
    - hallucination rate (0.0 to 1.0)
    - detailed result per citation

This directly addresses the research question:
    'Can a multi-agent system reduce LLM hallucination in
     academic literature reviews?'
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api_clients import search_openalex_works
from src.models import Paper, Citation


# ---------------------------------------------------------------------------
# Step 1: Extract citations from generated review text
# ---------------------------------------------------------------------------

def extract_citations(text: str) -> list[dict]:
    """
    Extract all (Author, Year) style citations from review text.

    Handles formats like:
        (Smith, 2023)
        (Chen et al., 2024)
        (Raiaan and Mukta, 2024)
        (Dwivedi et al., 2023)
        (Zawacki-Richter et al., 2019)
        (OpenAI, W. A. H. N., 2023)
        Smith (2023)
    """
    citations = []
    seen = set()

    # Pattern 1: (Author et al., YEAR) or (Author, YEAR)
    # Covers most Harvard inline citation styles
    patterns = [
        # (Author et al., 2023) or (Author and Author, 2023)
        r'\(([A-Z][a-zA-ZÀ-ž\s\-‐]+?(?:\s+et al\.)?(?:\s+and\s+[A-Z][a-zA-Z]+)?),\s*((?:19|20)\d{2})\)',

        # (Author et al., 2023; Author2 et al., 2024) — split these
        r'([A-Z][a-zA-ZÀ-ž\s\-‐]+?(?:\s+et al\.)?),\s*((?:19|20)\d{2})',
    ]

    # Use the broader pattern to catch all cases
    broad_pattern = r'([A-Z][A-Za-zÀ-ž\u2010\-]+(?:\s+[A-Z]?[A-Za-z\-‐]+)*(?:\s+et al\.)?(?:\s+and\s+[A-Z][a-zA-Z]+)?),?\s*((?:19|20)\d{2})'

    # Find all content inside parentheses first
    paren_contents = re.findall(r'\(([^)]+)\)', text)

    for content in paren_contents:
        # Skip if it looks like a URL or number only
        if 'http' in content or content.strip().isdigit():
            continue

        # Find all author+year pairs inside this parenthetical
        matches = re.findall(broad_pattern, content)

        for author_raw, year_str in matches:
            author = author_raw.strip().rstrip(",").strip()
            year   = int(year_str)

            # Skip very short or clearly wrong matches
            if len(author) < 3:
                continue

            # Skip if author looks like it's just numbers
            if author.replace(" ", "").isdigit():
                continue

            key = f"{author.lower()}_{year}"
            if key in seen:
                continue
            seen.add(key)

            citations.append({
                "author": author,
                "year":   year,
                "raw":    f"({author}, {year})",
            })

    return citations
# ---------------------------------------------------------------------------
# Step 2: Match a citation against locally retrieved papers
# ---------------------------------------------------------------------------

def _match_locally(
    author: str,
    year: int,
    papers: list[Paper],
) -> tuple[Paper | None, str]:
    """
    Try to match a citation against locally retrieved papers.

    Matching rules:
        - At least one significant word from author string appears
          in paper's author list
        - Year matches exactly     → "valid"
        - Year matches within ±1   → "partial"

    Parameters
    ----------
    author : str
        Author string extracted from citation.
    year   : int
        Year extracted from citation.
    papers : list[Paper]
        Papers retrieved by the Search Agent.

    Returns
    -------
    tuple[Paper | None, str]
        (matched_paper, status) where status is
        "valid", "partial", or "no_match"
    """
    # Clean author string for matching
    author_clean = (
        author
        .replace("et al.", "")
        .replace(" and ", " ")
        .replace(",", " ")
        .lower()
    )
    author_words = [w for w in author_clean.split() if len(w) > 3]

    for paper in papers:
        paper_authors_str = " ".join(paper.authors).lower()

        # Check if any significant author word matches
        author_match = any(
            word in paper_authors_str
            for word in author_words
        )

        if not author_match:
            continue

        paper_year = paper.year or 0

        if paper_year == year:
            return paper, "valid"
        elif abs(paper_year - year) <= 1:
            return paper, "partial"

    return None, "no_match"


# ---------------------------------------------------------------------------
# Step 3: Match a citation via OpenAlex live search
# ---------------------------------------------------------------------------

def _match_via_openalex(
    author: str,
    year: int,
) -> tuple[Paper | None, str]:
    """
    Search OpenAlex directly to verify a citation not found locally.

    Parameters
    ----------
    author : str
        Author string from citation.
    year   : int
        Year from citation.

    Returns
    -------
    tuple[Paper | None, str]
        (matched_paper, status)
    """
    search_query = f"{author} {year}"

    try:
        results = search_openalex_works(search_query, max_results=3)
    except Exception as e:
        print(f"[VerifierAgent] OpenAlex lookup failed for '{search_query}': {e}")
        return None, "no_match"

    author_clean = (
        author
        .replace("et al.", "")
        .replace(" and ", " ")
        .replace(",", " ")
        .lower()
    )
    author_words = [w for w in author_clean.split() if len(w) > 3]

    for paper in results:
        paper_authors_str = " ".join(paper.authors).lower()
        author_match = any(w in paper_authors_str for w in author_words)

        if not author_match:
            continue

        paper_year = paper.year or 0
        if paper_year == year:
            return paper, "valid"
        elif abs(paper_year - year) <= 1:
            return paper, "partial"

    return None, "no_match"


# ---------------------------------------------------------------------------
# Step 4: Main verification function
# ---------------------------------------------------------------------------

def verify_review(
    review_text: str,
    papers: list[Paper],
) -> dict:
    """
    Verify all citations in a generated literature review.

    Process for each citation:
        1. Try to match against locally retrieved papers.
        2. If no local match, search OpenAlex directly.
        3. Classify as valid / partial / hallucinated.

    Parameters
    ----------
    review_text : str
        The generated literature review text from Summariser Agent.
    papers : list[Paper]
        Papers retrieved by the Search Agent (used for local matching).

    Returns
    -------
    dict with keys:
        total           : int   - total citations found
        valid           : int   - exactly matched citations
        partial         : int   - year-off-by-one citations
        hallucinated    : int   - citations with no match found
        hallucination_rate : float - hallucinated / total
        citations       : list[Citation] - detailed per-citation results
    """
    print("\n[VerifierAgent] Starting citation verification...")

    # Step 1: Extract citations from review text
    raw_citations = extract_citations(review_text)
    print(f"[VerifierAgent] Found {len(raw_citations)} unique citations in review.")

    if not raw_citations:
        print("[VerifierAgent] No citations found to verify.")
        return {
            "total": 0,
            "valid": 0,
            "partial": 0,
            "hallucinated": 0,
            "hallucination_rate": 0.0,
            "citations": [],
        }

    # Step 2: Verify each citation
    results: list[Citation] = []
    valid_count = 0
    partial_count = 0
    hallucinated_count = 0

    for cit in raw_citations:
        author = cit["author"]
        year   = cit["year"]
        raw    = cit["raw"]

        print(f"\n[VerifierAgent] Checking: {raw}")

        # --- Try local match first ---
        matched_paper, status = _match_locally(author, year, papers)

        # --- If no local match, try OpenAlex ---
        if status == "no_match":
            print(f"  → No local match. Searching OpenAlex...")
            matched_paper, status = _match_via_openalex(author, year)

        # --- Classify result ---
        if status == "valid":
            valid_count += 1
            error_reason = None
            print(f"  → VALID ✓ — matched: '{matched_paper.title[:60]}...'")

        elif status == "partial":
            partial_count += 1
            error_reason = f"Year mismatch: cited {year}, paper year {matched_paper.year}"
            print(f"  → PARTIAL ⚠ — year mismatch. Paper: '{matched_paper.title[:50]}...'")

        else:
            hallucinated_count += 1
            matched_paper = None
            error_reason = "No matching paper found in local store or OpenAlex"
            print(f"  → HALLUCINATED ✗ — no match found anywhere")

        # Build Citation model
        citation_obj = Citation(
            raw_reference=raw,
            matched_paper_id=matched_paper.paper_id if matched_paper else None,
            valid=(status in ("valid", "partial")),
            error_reason=error_reason,
        )
        results.append(citation_obj)

    # Step 3: Compute hallucination rate
    total = len(raw_citations)
    hallucination_rate = round(hallucinated_count / total, 3) if total > 0 else 0.0

    # Step 4: Print summary
    print("\n" + "=" * 60)
    print("[VerifierAgent] VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Total citations   : {total}")
    print(f"  Valid             : {valid_count}")
    print(f"  Partial           : {partial_count}")
    print(f"  Hallucinated      : {hallucinated_count}")
    print(f"  Hallucination Rate: {hallucination_rate:.1%}")
    print("=" * 60)

    return {
        "total": total,
        "valid": valid_count,
        "partial": partial_count,
        "hallucinated": hallucinated_count,
        "hallucination_rate": hallucination_rate,
        "citations": results,
    }