# agents/verifier_agent.py
"""
Verifier Agent — Core Research Contribution.

This agent verifies citations in a generated literature review against:
    1. Locally retrieved papers  (fast, no API call needed)
    2. OpenAlex live lookup      (for citations not found locally)

For each citation produces:
    - status      : VALID / PARTIAL / HALLUCINATED
    - confidence  : 0.0 to 1.0 combining author + year + title signals
    - error_type  : FABRICATED_PAPER / WRONG_YEAR / WRONG_AUTHOR /
                    MISATTRIBUTION / UNKNOWN
    - matched_paper: Paper object if a match was found

Addresses IPR feedback:
    - Fuzzy author name matching
    - Title normalisation
    - Confidence scoring per citation
    - Error typology logging
    - Structured JSON logs for annotation
"""

import re
import sys
import json
import unicodedata
from collections import Counter
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api_clients import search_openalex_works
from src.models import Paper, Citation
from src.config import settings


# ---------------------------------------------------------------------------
# Configuration — tune these thresholds to improve verifier performance
# ---------------------------------------------------------------------------


AUTHOR_FUZZY_THRESHOLD = 0.85   # was 0.72 — stricter author matching
YEAR_EXACT_TOLERANCE   = 0      # exact year only for VALID
YEAR_PARTIAL_TOLERANCE = 1      # was 2 — only ±1 year for PARTIAL

# Confidence score weights (must sum to 1.0)
W_AUTHOR = 0.55   # was 0.40 — author match is most important signal
W_YEAR   = 0.35   # unchanged
W_TITLE  = 0.10   # was 0.25 — reduce title bonus impact

MAX_OPENALEX_RESULTS = 5
LOG_DIR = settings.data_dir / "eval" / "verifier_logs"


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_text(text: str) -> str:
    """
    Lowercase, remove punctuation, strip accents for robust matching.

    Examples:
        "Küchemann et al." → "kuchemann et al"
        "Zawacki‐Richter"  → "zawacki richter"
        "Smith, J."        → "smith j"
    """
    # Decompose accented characters (e.g. é → e + combining accent)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # Lowercase
    text = text.lower()
    # Replace hyphens and unicode dashes with space
    text = re.sub(r"[\-\u2010\u2011\u2012\u2013\u2014]", " ", text)
    # Remove all remaining punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fuzzy_score(a: str, b: str) -> float:
    """
    Compute fuzzy string similarity (0.0 to 1.0).
    Uses difflib SequenceMatcher ratio = 2*M / T.
    """
    return SequenceMatcher(None, a, b).ratio()


def _author_words(author_str: str) -> list[str]:
    """
    Extract significant words from a citation author string.

    Examples:
        "Dwivedi et al."   → ["dwivedi"]
        "Smith and Jones"  → ["smith", "jones"]
        "Chen, X."         → ["chen"]
        "Zawacki-Richter"  → ["zawacki", "richter"]
    """
    clean = _normalise_text(author_str)
    noise = {"et", "al", "and", "or", "the", "van", "de", "von", "der"}
    return [w for w in clean.split() if len(w) > 2 and w not in noise]


# ---------------------------------------------------------------------------
# Step 1: Extract citations from generated review text
# ---------------------------------------------------------------------------

def extract_citations(text: str) -> list[dict]:
    """
    Extract all (Author, Year) style citations from review text.

    Handles:
        (Smith, 2023)
        (Chen et al., 2024)
        (Zawacki-Richter et al., 2019)
        (Smith and Jones, 2022)
        (OpenAI, 2023)
        Multiple: (Smith, 2023; Jones, 2024)

    Parameters
    ----------
    text : str
        Generated literature review text.

    Returns
    -------
    list[dict]
        Each dict has keys: author (str), year (int), raw (str)
    """
    citations = []
    seen      = set()

    # Find all content inside parentheses
    paren_contents = re.findall(r'\(([^)]+)\)', text)

    # Broad pattern: Author string + 4-digit year
    pattern = (
        r'([A-Z][A-Za-z\u00C0-\u024F\u2010\-]+'
        r'(?:\s+[A-Z]?[A-Za-z\u00C0-\u024F\u2010\-]+)*'
        r'(?:\s+et\s+al\.)?'
        r'(?:\s+and\s+[A-Z][A-Za-z\u00C0-\u024F]+)?)'
        r',?\s*((?:19|20)\d{2})'
    )

    for content in paren_contents:
        # Skip URLs and standalone numbers
        if "http" in content or content.strip().isdigit():
            continue

        matches = re.findall(pattern, content)

        for author_raw, year_str in matches:
            author = author_raw.strip().rstrip(",").strip()
            year   = int(year_str)

            # Skip very short matches
            if len(author) < 3:
                continue

            # Skip if author is all digits
            if _normalise_text(author).replace(" ", "").isdigit():
                continue

            # Deduplicate
            key = f"{_normalise_text(author)}_{year}"
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
# Step 2: Compute confidence score for a candidate match
# ---------------------------------------------------------------------------

def _compute_confidence(
    cited_author: str,
    cited_year:   int,
    paper:        Paper,
) -> float:
    """
    Compute a confidence score (0.0 to 1.0) for a candidate paper match.

    Combines three signals:
        1. Author similarity  — fuzzy match of cited author vs paper authors
        2. Year match         — exact=1.0, within tolerance=partial, else=0.0
        3. Title bonus        — small boost if author surname in paper title

    Parameters
    ----------
    cited_author : str   — author string extracted from citation
    cited_year   : int   — year extracted from citation
    paper        : Paper — candidate paper to score against

    Returns
    -------
    float : confidence score 0.0 to 1.0
    """
    # --- Author score ---
    cited_words   = _author_words(cited_author)
    paper_authors = _normalise_text(" ".join(paper.authors or []))

    if not cited_words:
        author_score = 0.0
    else:
        matches = sum(
            1 for w in cited_words
            if w in paper_authors
            or any(
                _fuzzy_score(w, pw) >= AUTHOR_FUZZY_THRESHOLD
                for pw in paper_authors.split()
                if len(pw) > 2
            )
        )
        author_score = matches / len(cited_words)

    # --- Year score ---
    paper_year = paper.year or 0
    year_diff  = abs(paper_year - cited_year)

    if year_diff == 0:
        year_score = 1.0
    elif year_diff <= YEAR_PARTIAL_TOLERANCE:
        year_score = 1.0 - (year_diff * 0.25)
    else:
        year_score = 0.0

    # --- Title bonus ---
    paper_title = _normalise_text(paper.title or "")
    title_score = (
        0.5 if cited_words and any(w in paper_title for w in cited_words)
        else 0.0
    )

    # --- Weighted combination ---
    confidence = (
        W_AUTHOR * author_score +
        W_YEAR   * year_score   +
        W_TITLE  * title_score
    )

    return round(min(confidence, 1.0), 3)


# ---------------------------------------------------------------------------
# Step 3: Determine VALID / PARTIAL / HALLUCINATED
# ---------------------------------------------------------------------------

def _determine_status(
    cited_year: int,
    paper:      Paper,
    confidence: float,
) -> str:
    """
    UPDATED: Stricter thresholds to better detect hallucinations.

    VALID       : confidence >= 0.75 AND exact year match
    PARTIAL     : confidence >= 0.55 AND year within ±1
    HALLUCINATED: everything else
    """
    paper_year = paper.year or 0
    year_diff  = abs(paper_year - cited_year)

    if confidence >= 0.75 and year_diff == 0:
        return "VALID"
    elif confidence >= 0.55 and year_diff <= 1:
        return "PARTIAL"
    else:
        return "HALLUCINATED"
# ---------------------------------------------------------------------------
# Step 4: Classify error type
# ---------------------------------------------------------------------------

def _classify_error_type(
    cited_author:  str,
    cited_year:    int,
    matched_paper: Paper | None,
) -> str:
    """
    Classify error type for a citation that failed verification.

    Error types:
        FABRICATED_PAPER : no matching paper found anywhere
        WRONG_YEAR       : paper exists but year is wrong
        WRONG_AUTHOR     : paper exists but author does not match
        MISATTRIBUTION   : paper exists but both author and year wrong
        UNKNOWN          : cannot determine
    """
    if matched_paper is None:
        return "FABRICATED_PAPER"

    paper_year    = matched_paper.year or 0
    paper_authors = _normalise_text(" ".join(matched_paper.authors or []))
    cited_words   = _author_words(cited_author)

    year_diff    = abs(paper_year - cited_year)
    author_match = any(w in paper_authors for w in cited_words)

    if year_diff > YEAR_PARTIAL_TOLERANCE and author_match:
        return "WRONG_YEAR"
    elif not author_match and year_diff <= YEAR_PARTIAL_TOLERANCE:
        return "WRONG_AUTHOR"
    elif not author_match and year_diff > YEAR_PARTIAL_TOLERANCE:
        return "MISATTRIBUTION"
    else:
        return "UNKNOWN"


# ---------------------------------------------------------------------------
# Step 5: Local match against retrieved papers
# ---------------------------------------------------------------------------

def _match_locally(
    author: str,
    year:   int,
    papers: list[Paper],
) -> tuple[Paper | None, float]:
    """
    Find best matching paper from local corpus using confidence scoring.

    Parameters
    ----------
    author : str         — cited author string
    year   : int         — cited year
    papers : list[Paper] — locally retrieved papers

    Returns
    -------
    tuple[Paper | None, float]
        Best matching paper and confidence score.
        Returns (None, 0.0) if corpus is empty.
    """
    best_paper      = None
    best_confidence = 0.0

    for paper in papers:
        conf = _compute_confidence(author, year, paper)
        if conf > best_confidence:
            best_confidence = conf
            best_paper      = paper

    return best_paper, best_confidence


# ---------------------------------------------------------------------------
# Step 6: Live OpenAlex lookup
# ---------------------------------------------------------------------------

def _match_via_openalex(
    author: str,
    year:   int,
) -> tuple[Paper | None, float]:
    """
    Search OpenAlex for a citation not found locally.

    Parameters
    ----------
    author : str — cited author string
    year   : int — cited year

    Returns
    -------
    tuple[Paper | None, float]
        Best matching paper and confidence score.
    """
    author_words = _author_words(author)
    query = (
        f"{' '.join(author_words[:2])} {year}"
        if author_words
        else str(year)
    )

    try:
        results = search_openalex_works(query, max_results=MAX_OPENALEX_RESULTS)
    except Exception as e:
        print(f"  [Verifier] OpenAlex lookup failed for '{query}': {e}")
        return None, 0.0

    best_paper      = None
    best_confidence = 0.0

    for paper in results:
        conf = _compute_confidence(author, year, paper)
        if conf > best_confidence:
            best_confidence = conf
            best_paper      = paper

    return best_paper, best_confidence


# ---------------------------------------------------------------------------
# Step 7: Build structured log entry
# ---------------------------------------------------------------------------

def _build_log_entry(
    raw:           str,
    author:        str,
    year:          int,
    status:        str,
    confidence:    float,
    error_type:    str | None,
    matched_paper: Paper | None,
    source:        str,
) -> dict:
    """
    Build a structured log entry for one citation verification result.

    This log is used for:
        - Manual annotation in the annotation helper notebook
        - Error typology analysis in the dissertation
        - FPR results tables

    Parameters
    ----------
    raw           : raw citation string e.g. "(Smith et al., 2023)"
    author        : extracted author string
    year          : extracted year
    status        : VALID / PARTIAL / HALLUCINATED
    confidence    : float 0.0 to 1.0
    error_type    : error type label or None if VALID
    matched_paper : Paper object or None
    source        : "local" or "openalex"

    Returns
    -------
    dict : structured log entry
    """
    return {
        "raw_citation":     raw,
        "cited_author":     author,
        "cited_year":       year,
        "status":           status,
        "confidence":       confidence,
        "error_type":       error_type,
        "match_source":     source,
        "matched_paper_id": matched_paper.paper_id if matched_paper else None,
        "matched_title":    matched_paper.title[:80] if matched_paper else None,
        "matched_year":     matched_paper.year if matched_paper else None,
        "matched_authors":  matched_paper.authors[:3] if matched_paper else [],
        "matched_doi":      matched_paper.doi if matched_paper else None,
    }


# ---------------------------------------------------------------------------
# Step 8: Save verification log to JSON
# ---------------------------------------------------------------------------

def _save_verification_log(
    logs:   list[dict],
    run_id: str,
) -> Path:
    """
    Save all citation verification logs as a structured JSON file.

    File is saved to: data/eval/verifier_logs/verifier_log_{run_id}.json

    Parameters
    ----------
    logs   : list of log entry dicts (one per citation verified)
    run_id : unique run identifier e.g. "20260329_040000"

    Returns
    -------
    Path : path to saved log file
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / f"verifier_log_{run_id}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id":    run_id,
                "timestamp": datetime.now().isoformat(),
                "total":     len(logs),
                "entries":   logs,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[VerifierAgent] Log saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Step 9: Main verification function
# ---------------------------------------------------------------------------

def verify_review(
    review_text: str,
    papers:      list[Paper],
    run_id:      str | None = None,
) -> dict:
    """
    Verify all citations in a generated literature review.

    Process for each citation:
        1. Extract (Author, Year) citations from review text.
        2. Try local match against retrieved papers (fast, no API).
        3. If local confidence < 0.45, try OpenAlex live lookup.
        4. Classify as VALID / PARTIAL / HALLUCINATED.
        5. Assign confidence score and error type.
        6. Build Citation model for Assembler Agent.
        7. Build structured log entry for annotation/FPR.
        8. Save all logs to JSON file.
        9. Return full results dict.

    Parameters
    ----------
    review_text : str
        Generated literature review from Summariser Agent.
    papers : list[Paper]
        Papers retrieved by Search Agent (used for local matching).
    run_id : str | None
        Optional run ID for log file naming. Auto-generated if None.

    Returns
    -------
    dict:
        total              : int
        valid              : int
        partial            : int
        hallucinated       : int
        hallucination_rate : float   (hallucinated / total)
        citations          : list[Citation]  (for Assembler Agent)
        logs               : list[dict]      (for annotation/FPR)
    """
    # Auto-generate run ID if not provided
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n[VerifierAgent] Starting citation verification...")
    print(f"[VerifierAgent] Local corpus  : {len(papers)} papers")

    # ----------------------------------------------------------------
    # Step 1: Extract all citations from review text
    # ----------------------------------------------------------------
    raw_citations = extract_citations(review_text)
    print(f"[VerifierAgent] Found {len(raw_citations)} unique citations.")

    # Handle empty case
    if not raw_citations:
        print("[VerifierAgent] No citations found to verify.")
        return {
            "total":              0,
            "valid":              0,
            "partial":            0,
            "hallucinated":       0,
            "hallucination_rate": 0.0,
            "citations":          [],
            "logs":               [],
        }

    # ----------------------------------------------------------------
    # Step 2: Verify each citation one by one
    # ----------------------------------------------------------------
    citation_objects: list[Citation] = []
    logs:             list[dict]     = []

    valid_count        = 0
    partial_count      = 0
    hallucinated_count = 0

    for cit in raw_citations:
        author = cit["author"]
        year   = cit["year"]
        raw    = cit["raw"]

        print(f"\n[VerifierAgent] Checking: {raw}")

        # --- Try local match first (no API call) ---
        matched_paper, confidence = _match_locally(author, year, papers)
        source = "local"

        # --- If confidence too low, try OpenAlex live lookup ---
        if confidence < 0.45:
            print(
                f"  → Low local confidence ({confidence:.2f}). "
                f"Searching OpenAlex..."
            )
            oa_paper, oa_confidence = _match_via_openalex(author, year)

            # Use OpenAlex result only if it scores better
            if oa_confidence > confidence:
                matched_paper = oa_paper
                confidence    = oa_confidence
                source        = "openalex"

        # --- Determine VALID / PARTIAL / HALLUCINATED ---
        if matched_paper and confidence >= 0.45:
            status = _determine_status(year, matched_paper, confidence)
        else:
            status        = "HALLUCINATED"
            matched_paper = None

        # --- Classify error type (only for non-VALID) ---
        error_type = (
            None
            if status == "VALID"
            else _classify_error_type(author, year, matched_paper)
        )

        # --- Update counters and print result ---
        if status == "VALID":
            valid_count += 1
            print(
                f"  → VALID ✓  "
                f"(conf: {confidence:.2f}, src: {source}) "
                f"— '{(matched_paper.title or '')[:55]}...'"
            )
        elif status == "PARTIAL":
            partial_count += 1
            print(
                f"  → PARTIAL ⚠ "
                f"(conf: {confidence:.2f}, {error_type}, src: {source}) "
                f"— '{(matched_paper.title or '')[:50]}...'"
            )
        else:
            hallucinated_count += 1
            print(
                f"  → HALLUCINATED ✗ "
                f"(conf: {confidence:.2f}, {error_type})"
            )

        # --- Build Citation model (used by Assembler Agent) ---
        citation_obj = Citation(
            raw_reference    = raw,
            matched_paper_id = matched_paper.paper_id if matched_paper else None,
            valid            = (status in ("VALID", "PARTIAL")),
            error_reason     = error_type,
        )
        citation_objects.append(citation_obj)

        # --- Build structured log entry (annotation + FPR tables) ---
        log_entry = _build_log_entry(
            raw           = raw,
            author        = author,
            year          = year,
            status        = status,
            confidence    = confidence,
            error_type    = error_type,
            matched_paper = matched_paper,
            source        = source,
        )
        logs.append(log_entry)

    # ----------------------------------------------------------------
    # Step 3: Compute hallucination rate
    # ----------------------------------------------------------------
    total              = len(raw_citations)
    hallucination_rate = (
        round(hallucinated_count / total, 3) if total > 0 else 0.0
    )

    # ----------------------------------------------------------------
    # Step 4: Save structured JSON log
    # ----------------------------------------------------------------
    _save_verification_log(logs, run_id)

    # ----------------------------------------------------------------
    # Step 5: Print summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[VerifierAgent] VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Total citations     : {total}")
    print(f"  Valid               : {valid_count}")
    print(f"  Partial             : {partial_count}")
    print(f"  Hallucinated        : {hallucinated_count}")
    print(f"  Hallucination Rate  : {hallucination_rate:.1%}")
    print("=" * 60)

    # Error type breakdown
    error_types = [l["error_type"] for l in logs if l["error_type"]]
    if error_types:
        print("\n  Error type breakdown:")
        for etype, count in Counter(error_types).most_common():
            print(f"    {etype:<25} : {count}")
        print()

    # ----------------------------------------------------------------
    # Step 6: Return full results
    # ----------------------------------------------------------------
    return {
        "total":               total,
        "valid":               valid_count,
        "partial":             partial_count,
        "hallucinated":        hallucinated_count,
        "hallucination_rate":  hallucination_rate,
        "citations":           citation_objects,
        "logs":                logs,
    }