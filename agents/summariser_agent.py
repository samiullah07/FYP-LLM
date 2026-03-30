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
    Write a strict literature review — ONLY cite retrieved papers.
    """
    client = _get_groq_client()
    papers_used = papers[:MAX_PAPERS_IN_REVIEW]

    # Format with full details so LLM can cite accurately
    paper_list_str = "\n".join(
        f"[{i+1}] Title: {p.title}\n"
        f"    Authors: {', '.join(p.authors[:3]) if p.authors else 'Unknown'}\n"
        f"    Year: {p.year or 'Unknown'}\n"
        f"    Venue: {p.venue or 'Unknown'}\n"
        f"    DOI: {p.doi or 'N/A'}"
        for i, p in enumerate(papers_used)
    )

    prompt = (
        f"You are an expert academic writer.\n\n"
        f"Topic: {topic}\n\n"
        f"Using ONLY the papers listed below, write a literature review "
        f"section of 400-500 words with AT LEAST 8-10 inline citations.\n\n"
        f"STRICT REQUIREMENTS:\n"
        f"1. Use in-text citations in Harvard format: (Author et al., Year)\n"
        f"2. EVERY factual claim MUST have a citation from the list below\n"
        f"3. Include AT LEAST 8 different citations throughout the review\n"
        f"4. Group papers thematically\n"
        f"5. Identify agreements, contradictions, and research gaps\n"
        f"6. End with a full reference list in Harvard format\n"
        f"7. Do NOT cite papers not in the list below\n"
        f"8. Do NOT invent authors, years, titles, or venues\n"
        f"9. Write in formal academic English\n\n"
        f"Available papers (cite ONLY from this list):\n"
        f"{paper_list_str}"
    )

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict academic writer. "
                        "You NEVER invent citations. "
                        "You ONLY cite papers explicitly provided to you. "
                        "If you are unsure, do not cite rather than guess."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.2,  # LOW temperature = strict, factual output
        )
        review_text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[SummariserAgent] ERROR: {e}")
        review_text = ""

    print(f"[SummariserAgent] Literature review written ({len(review_text)} chars)")
    print(f"[SummariserAgent] Papers used: {len(papers_used)}")

    return review_text, papers_used