# agents/planner_agent.py
"""
Planner Agent for the Literature Review Pipeline.

Responsibility:
    Given a broad research topic, use an LLM (Groq) to decompose it
    into 3-5 focused sub-queries suitable for academic database search.

Input:
    topic (str): e.g. "Agentic AI for hallucination mitigation in literature review"

Output:
    List[str]: e.g. [
        "hallucination detection in large language models",
        "multi-agent systems for academic research automation",
        "citation verification using knowledge graphs",
        ...
    ]
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so src and agents are importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from groq import Groq

from src.config import settings

# ---------------------------------------------------------------------------
# Groq client (uses GROQ_API_KEY from .env via settings)
# ---------------------------------------------------------------------------

def _get_groq_client() -> Groq:
    """
    Initialise and return a Groq client.

    Reads groq_api_key from settings (which loads it from .env).
    """
    return Groq(api_key=settings.groq_api_key)


# ---------------------------------------------------------------------------
# Core planner function
# ---------------------------------------------------------------------------

def plan_topic(topic: str) -> list[str]:
    """
    Decompose a research topic into 3-5 focused academic sub-queries.

    Parameters
    ----------
    topic : str
        High-level research topic string.

    Returns
    -------
    list[str]
        List of sub-query strings, one per line from LLM output.
    """
    client = _get_groq_client()

    prompt = (
        "You are a research assistant helping a student write an academic "
        "literature review.\n\n"
        "Break the following research topic into 3 to 5 focused sub-queries "
        "suitable for searching an academic database like OpenAlex or Semantic Scholar.\n\n"
        "Rules:\n"
        "- Return each sub-query on a new line.\n"
        "- No numbering, no bullet points, no extra commentary.\n"
        "- Each sub-query should be specific and searchable.\n"
        "- Focus on different aspects of the topic.\n\n"
        f"Topic: {topic}"
    )

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.3,  # low temperature = consistent, focused output
    )

    raw_output = response.choices[0].message.content.strip()

    # Split by newline and clean up each line
    sub_queries = [
        line.strip()
        for line in raw_output.split("\n")
        if line.strip()
    ]

    print(f"[PlannerAgent] Generated {len(sub_queries)} sub-queries for topic: '{topic}'")
    for i, q in enumerate(sub_queries, 1):
        print(f"  {i}. {q}")

    return sub_queries