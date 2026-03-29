# agents/summariser_agent.py
"""
Summariser Agent for the Literature Review Pipeline.

Responsibility:
    1. summarise_paper()        - summarise a single Paper object (2-3 sentences)
    2. write_literature_review() - write a full literature review section
                                   (300-400 words) across multiple papers,
                                   with in-text citations and a reference list.

Input:
    Paper objects from the Search Agent

Output:
    1. str         : short summary of one paper
    2. (str, list) : full review text + list of papers used
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from groq import Groq

from src.config import settings
from src.models import Paper

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_TOKENS_SUMMARY = 200    # tokens per single-paper summary
MAX_TOKENS_REVIEW  = 1500   # tokens for full literature review
MAX_PAPERS_IN_REVIEW = 15   # cap papers sent to LLM to stay within context


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

def _get_groq_client() -> Groq:
    """
    Initialise and return a Groq client using key from settings.
    """
    return Groq(api_key=settings.groq_api_key)


# ---------------------------------------------------------------------------
# Helper: format a Paper into a short string for the LLM prompt
# ---------------------------------------------------------------------------

def _format_paper_for_prompt(paper: Paper, max_authors: int = 3) -> str:
    """
    Format a Paper object into a single readable line for LLM prompts.

    Parameters
    ----------
    paper      : Paper object
    max_authors: max number of authors to include

    Returns
    -------
    str : formatted string e.g.
          "Title: ... | Authors: ... | Year: ... | Venue: ..."
    """
    authors = (
        ", ".join(paper.authors[:max_authors])
        if paper.authors
        else "Unknown"
    )
    if len(paper.authors) > max_authors:
        authors += " et al."

    venue = paper.venue or "Unknown venue"
    year  = str(paper.year) if paper.year else "Unknown year"

    return (
        f"Title: {paper.title} | "
        f"Authors: {authors} | "
        f"Year: {year} | "
        f"Venue: {venue}"
    )


# ---------------------------------------------------------------------------
# 1. Summarise a single paper
# ---------------------------------------------------------------------------

def summarise_paper(paper: Paper) -> str:
    """
    Generate a 2-3 sentence summary of a single academic paper.

    Focuses on: main finding, method, and relevance to LLM hallucination
    research (the project topic).

    Parameters
    ----------
    paper : Paper
        A single Paper object from the Search Agent.

    Returns
    -------
    str
        Short summary string.
    """
    client = _get_groq_client()

    abstract_section = (
        f"Abstract: {paper.abstract}"
        if paper.abstract
        else "(No abstract available)"
    )

    prompt = (
        "You are an academic research assistant.\n\n"
        "Summarise the following academic paper in 2-3 sentences.\n"
        "Focus on:\n"
        "  - The main research problem or question\n"
        "  - The key method or approach used\n"
        "  - The main finding or contribution\n"
        "  - Its relevance to LLM hallucination research\n\n"
        f"Title   : {paper.title}\n"
        f"Authors : {', '.join(paper.authors[:3]) if paper.authors else 'Unknown'}\n"
        f"Year    : {paper.year or 'Unknown'}\n"
        f"{abstract_section}"
    )

    response = _get_groq_client().chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS_SUMMARY,
        temperature=0.3,
    )

    summary = response.choices[0].message.content.strip()
    print(f"[SummariserAgent] Summarised: '{paper.title[:60]}...'")
    return summary


# ---------------------------------------------------------------------------
# 2. Write a full literature review across multiple papers
# ---------------------------------------------------------------------------

def write_literature_review(
    papers: list[Paper],
    topic: str,
) -> tuple[str, list[Paper]]:
    """
    Write a 300-400 word literature review section across multiple papers.

    Includes:
        - In-text citations in Harvard format: (Author, Year)
        - A reference list at the end in Harvard format
        - Thematic grouping of papers where possible

    Parameters
    ----------
    papers : list[Paper]
        Papers retrieved and deduplicated by the Search Agent.
    topic  : str
        The original research topic (used to frame the review).

    Returns
    -------
    tuple[str, list[Paper]]
        (review_text, papers_used)
        review_text  : the generated literature review string
        papers_used  : the subset of papers passed to the LLM
    """
    client = _get_groq_client()

    # Cap number of papers to avoid context overflow
    papers_used = papers[:MAX_PAPERS_IN_REVIEW]

    # Format paper list for the prompt
    paper_list_str = "\n".join(
        f"- {_format_paper_for_prompt(p)}"
        for p in papers_used
    )

    prompt = (
        "You are an expert academic writer helping a student write a "
        "literature review for their MSc Data Science dissertation.\n\n"
        f"Topic: {topic}\n\n"
        "Using ONLY the papers listed below, write a literature review "
        "section of 300-400 words.\n\n"
        "Requirements:\n"
        "  1. Use in-text citations in Harvard format: (Author, Year)\n"
        "  2. Group related papers thematically where possible\n"
        "  3. Highlight agreements, disagreements, and research gaps\n"
        "  4. End with a full reference list in Harvard format\n"
        "  5. Do NOT invent or hallucinate papers not in the list below\n"
        "  6. Only cite papers from the list provided\n\n"
        "Available papers:\n"
        f"{paper_list_str}"
    )

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS_REVIEW,
        temperature=0.4,
    )

    review_text = response.choices[0].message.content.strip()

    print(f"[SummariserAgent] Literature review written ({len(review_text)} chars)")
    print(f"[SummariserAgent] Papers used: {len(papers_used)}")

    return review_text, papers_used