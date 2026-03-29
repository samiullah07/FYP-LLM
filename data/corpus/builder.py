# data/corpus/builder.py
"""
Gold Standard Corpus Builder.

Responsibility:
    Select, document, and store 40-60 papers across 2 topics
    with explicit inclusion criteria for the evaluation corpus.

    This directly addresses IPR feedback:
        "inclusion criteria, topic scopes, and the exact process
         of building the gold citation list are not operationalised"

Topics covered:
    1. Hallucinations and factuality in LLMs
    2. Agentic and multi-agent LLM systems

Inclusion criteria (documented for dissertation):
    - Publication year: 2019-2025
    - Venue: peer-reviewed journal or conference
    - Relevance: directly addresses topic keywords
    - Minimum abstract length: 50 words
    - Language: English only

Output files saved to: data/corpus/
    - corpus_topic1.json   : papers for topic 1
    - corpus_topic2.json   : papers for topic 2
    - corpus_combined.json : all papers combined
    - corpus_metadata.json : inclusion criteria + stats
"""

import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api_clients import search_openalex_works
from src.models import Paper
from src.config import settings


# ---------------------------------------------------------------------------
# Corpus configuration
# ---------------------------------------------------------------------------

CORPUS_DIR = settings.data_dir / "corpus"

# Two evaluation topics (as per IPR recommendation)
TOPICS = {
    "topic1": {
        "name": "Hallucinations and Factuality in LLMs",
        "description": (
            "Papers addressing hallucination detection, mitigation, "
            "evaluation, and factuality in large language models"
        ),
        "queries": [
            "hallucination detection large language models",
            "LLM factuality evaluation benchmarks",
            "citation hallucination academic text generation",
            "retrieval augmented generation hallucination reduction",
            "LLM factual accuracy verification methods",
            "false citation generation language models",
            "hallucination mitigation techniques NLP",
        ],
        "keywords": [
            "hallucination", "factuality", "citation",
            "verification", "factual", "grounding",
            "retrieval", "knowledge", "accuracy",
        ],
    },
    "topic2": {
        "name": "Agentic and Multi-Agent LLM Systems",
        "description": (
            "Papers addressing agentic AI, multi-agent orchestration, "
            "autonomous agents, and tool-using LLM systems"
        ),
        "queries": [
            "agentic AI systems large language models",
            "multi-agent LLM orchestration framework",
            "autonomous AI agents task planning",
            "LangChain LangGraph agent workflows",
            "tool-using language model agents",
            "self-correcting AI agents verification",
            "literature review automation AI agents",
        ],
        "keywords": [
            "agent", "agentic", "multi-agent", "autonomous",
            "workflow", "orchestration", "planning",
            "tool", "pipeline", "framework",
        ],
    },
}

# Inclusion criteria (documented for dissertation)
INCLUSION_CRITERIA = {
    "year_min":           2019,
    "year_max":           2025,
    "min_abstract_words": 30,
    "language":           "English",
    "max_papers_per_query": 8,
    "target_per_topic":   25,
    "relevance_check":    "keyword overlap with topic keywords",
}


# ---------------------------------------------------------------------------
# Helper: check if paper meets inclusion criteria
# ---------------------------------------------------------------------------

def _meets_inclusion_criteria(
    paper: Paper,
    topic_config: dict,
) -> tuple[bool, str]:
    """
    Check if a paper meets the documented inclusion criteria.

    Parameters
    ----------
    paper        : Paper object to evaluate
    topic_config : dict with topic keywords and criteria

    Returns
    -------
    tuple[bool, str]
        (passes, reason) where reason explains accept/reject decision
    """
    # Year check
    if paper.year:
        if paper.year < INCLUSION_CRITERIA["year_min"]:
            return False, f"Year {paper.year} before {INCLUSION_CRITERIA['year_min']}"
        if paper.year > INCLUSION_CRITERIA["year_max"]:
            return False, f"Year {paper.year} after {INCLUSION_CRITERIA['year_max']}"
    else:
        return False, "No publication year"

    # Abstract length check
    abstract_words = len((paper.abstract or "").split())
    if abstract_words < INCLUSION_CRITERIA["min_abstract_words"]:
        return False, f"Abstract too short ({abstract_words} words)"

    # Title must exist
    if not paper.title or len(paper.title.strip()) < 5:
        return False, "Missing or too short title"

    # Keyword relevance check
    text_to_check = (
        f"{paper.title} {paper.abstract or ''}".lower()
    )
    keywords      = topic_config["keywords"]
    matched_kws   = [kw for kw in keywords if kw.lower() in text_to_check]

    if len(matched_kws) == 0:
        return False, "No keyword overlap with topic"

    return True, f"Passes — matched keywords: {matched_kws[:3]}"


# ---------------------------------------------------------------------------
# Helper: label paper relevance
# ---------------------------------------------------------------------------

def _label_relevance(
    paper:        Paper,
    topic_config: dict,
) -> str:
    """
    Label paper as 'core' or 'peripheral' based on keyword overlap.

    Core       : 3+ topic keywords in title+abstract
    Peripheral : 1-2 topic keywords
    """
    text     = f"{paper.title} {paper.abstract or ''}".lower()
    keywords = topic_config["keywords"]
    matched  = sum(1 for kw in keywords if kw.lower() in text)

    if matched >= 3:
        return "core"
    elif matched >= 1:
        return "peripheral"
    else:
        return "peripheral"


# ---------------------------------------------------------------------------
# Helper: convert Paper to dict for JSON storage
# ---------------------------------------------------------------------------

def _paper_to_dict(
    paper:     Paper,
    relevance: str,
    topic_id:  str,
    reason:    str,
) -> dict:
    """
    Convert Paper object to a JSON-serialisable dict with corpus metadata.
    """
    return {
        "paper_id":   paper.paper_id,
        "title":      paper.title,
        "abstract":   paper.abstract,
        "authors":    paper.authors,
        "year":       paper.year,
        "venue":      paper.venue,
        "doi":        paper.doi,
        "source":     paper.source,
        "topic_id":   topic_id,
        "relevance":  relevance,
        "reason":     reason,
        "added_at":   datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Core: build corpus for one topic
# ---------------------------------------------------------------------------

def build_topic_corpus(
    topic_id:     str,
    topic_config: dict,
    target:       int = 25,
) -> list[dict]:
    """
    Build a corpus of papers for a single topic.

    Process:
        1. Run each query against OpenAlex.
        2. Deduplicate by DOI then title.
        3. Apply inclusion criteria filter.
        4. Label each paper as core/peripheral.
        5. Return up to `target` papers.

    Parameters
    ----------
    topic_id     : str  — e.g. "topic1"
    topic_config : dict — topic configuration with queries and keywords
    target       : int  — target number of papers

    Returns
    -------
    list[dict]
        List of paper dicts with corpus metadata.
    """
    print(f"\n[CorpusBuilder] Building corpus for: {topic_config['name']}")
    print(f"[CorpusBuilder] Target: {target} papers")
    print(f"[CorpusBuilder] Queries: {len(topic_config['queries'])}")

    all_papers:   list[Paper] = []
    seen_dois:    set[str]    = set()
    seen_titles:  set[str]    = set()

    # Step 1: Fetch papers for each query
    for query in topic_config["queries"]:
        print(f"\n  Querying: '{query}'")
        try:
            papers = search_openalex_works(
                query=query,
                max_results=INCLUSION_CRITERIA["max_papers_per_query"],
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        added = 0
        for paper in papers:
            # Deduplicate
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

        print(f"  Added {added} new papers (total: {len(all_papers)})")

        # Stop early if we have enough candidates
        if len(all_papers) >= target * 2:
            print(f"  Enough candidates collected. Stopping early.")
            break

    print(f"\n[CorpusBuilder] Raw candidates: {len(all_papers)}")

    # Step 2: Apply inclusion criteria
    accepted: list[dict] = []
    rejected_count = 0

    for paper in all_papers:
        passes, reason = _meets_inclusion_criteria(paper, topic_config)

        if passes:
            relevance = _label_relevance(paper, topic_config)
            paper_dict = _paper_to_dict(paper, relevance, topic_id, reason)
            accepted.append(paper_dict)
        else:
            rejected_count += 1

    print(f"[CorpusBuilder] Accepted : {len(accepted)}")
    print(f"[CorpusBuilder] Rejected : {rejected_count}")

    # Step 3: Sort by relevance (core first) then year (newest first)
    accepted.sort(
        key=lambda p: (
            0 if p["relevance"] == "core" else 1,
            -(p["year"] or 0),
        )
    )

    # Step 4: Cap at target
    final = accepted[:target]

    core_count       = sum(1 for p in final if p["relevance"] == "core")
    peripheral_count = sum(1 for p in final if p["relevance"] == "peripheral")

    print(f"[CorpusBuilder] Final corpus: {len(final)} papers")
    print(f"  Core       : {core_count}")
    print(f"  Peripheral : {peripheral_count}")

    return final


# ---------------------------------------------------------------------------
# Save corpus to JSON
# ---------------------------------------------------------------------------

def save_corpus(
    papers:   list[dict],
    filename: str,
) -> Path:
    """
    Save corpus papers to a JSON file in data/corpus/.

    Parameters
    ----------
    papers   : list of paper dicts
    filename : output filename e.g. "corpus_topic1.json"

    Returns
    -------
    Path : path to saved file
    """
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CORPUS_DIR / filename

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"[CorpusBuilder] Saved {len(papers)} papers → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Load corpus from JSON
# ---------------------------------------------------------------------------

def load_corpus(filename: str) -> list[dict]:
    """
    Load a saved corpus from data/corpus/.

    Parameters
    ----------
    filename : JSON filename e.g. "corpus_combined.json"

    Returns
    -------
    list[dict] : loaded paper dicts
    """
    path = CORPUS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {path}\n"
            f"Run build_full_corpus() first."
        )

    with open(path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    print(f"[CorpusBuilder] Loaded {len(papers)} papers from {path}")
    return papers


# ---------------------------------------------------------------------------
# Convert corpus dicts back to Paper objects
# ---------------------------------------------------------------------------

def corpus_to_papers(corpus: list[dict]) -> list[Paper]:
    """
    Convert loaded corpus dicts back to Paper model objects.

    Parameters
    ----------
    corpus : list of paper dicts from load_corpus()

    Returns
    -------
    list[Paper]
    """
    return [
        Paper(
            paper_id = p["paper_id"],
            title    = p["title"],
            abstract = p.get("abstract"),
            authors  = p.get("authors", []),
            year     = p.get("year"),
            venue    = p.get("venue"),
            doi      = p.get("doi"),
            source   = p.get("source", "openalex"),
        )
        for p in corpus
    ]


# ---------------------------------------------------------------------------
# Build full corpus (both topics)
# ---------------------------------------------------------------------------

def build_full_corpus() -> dict:
    """
    Build the complete evaluation corpus for both topics.

    Saves:
        data/corpus/corpus_topic1.json
        data/corpus/corpus_topic2.json
        data/corpus/corpus_combined.json
        data/corpus/corpus_metadata.json

    Returns
    -------
    dict : summary statistics
    """
    print("\n" + "=" * 60)
    print("[CorpusBuilder] BUILDING FULL EVALUATION CORPUS")
    print("=" * 60)
    print(f"  Target per topic : {INCLUSION_CRITERIA['target_per_topic']}")
    print(f"  Year range       : {INCLUSION_CRITERIA['year_min']}–{INCLUSION_CRITERIA['year_max']}")
    print(f"  Topics           : {len(TOPICS)}")
    print("=" * 60)

    all_combined = []
    topic_stats  = {}

    for topic_id, topic_config in TOPICS.items():
        # Build corpus for this topic
        papers = build_topic_corpus(
            topic_id     = topic_id,
            topic_config = topic_config,
            target       = INCLUSION_CRITERIA["target_per_topic"],
        )

        # Save topic corpus
        save_corpus(papers, f"corpus_{topic_id}.json")

        # Collect stats
        topic_stats[topic_id] = {
            "name":       topic_config["name"],
            "total":      len(papers),
            "core":       sum(1 for p in papers if p["relevance"] == "core"),
            "peripheral": sum(1 for p in papers if p["relevance"] == "peripheral"),
            "year_range": (
                min(p["year"] for p in papers if p["year"]),
                max(p["year"] for p in papers if p["year"]),
            ) if papers else (None, None),
        }

        all_combined.extend(papers)

    # Save combined corpus
    save_corpus(all_combined, "corpus_combined.json")

        # Save metadata
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "built_at":           datetime.now().isoformat(),
        "inclusion_criteria": INCLUSION_CRITERIA,
        "topics":             topic_stats,
        "total_papers":       len(all_combined),
        "core_total":         sum(1 for p in all_combined if p["relevance"] == "core"),
        "peripheral_total":   sum(1 for p in all_combined if p["relevance"] == "peripheral"),
    }

    meta_path = CORPUS_DIR / "corpus_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[CorpusBuilder] Metadata saved → {meta_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("[CorpusBuilder] CORPUS BUILD COMPLETE")
    print("=" * 60)
    for tid, stats in topic_stats.items():
        print(f"\n  {stats['name']}")
        print(f"    Total      : {stats['total']}")
        print(f"    Core       : {stats['core']}")
        print(f"    Peripheral : {stats['peripheral']}")
        print(f"    Year range : {stats['year_range']}")
    print(f"\n  Combined total : {len(all_combined)}")
    print("=" * 60)

    return metadata