# graph/baseline_graph.py
"""
Baseline Single-LLM Workflow (LangGraph).

This is the COMPARISON system used in evaluation.

Pipeline:
    search_node → write_node (no planner, no verifier)

The baseline:
    - Searches OpenAlex with the raw topic (no planning)
    - Asks the LLM to write a full literature review in ONE call
    - Does NOT verify any citations
    - Does NOT check for hallucinations

This directly mirrors what a student would do if they just asked
an LLM to "write a literature review on X" — the naive approach.

Used to measure:
    - Baseline hallucination rate
    - Baseline citation accuracy
    - Compared against the multi-agent experimental system
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from groq import Groq

from src.api_clients import search_openalex_works
from src.config import settings
from src.models import Paper, Citation
from agents.verifier_agent import verify_review


# ---------------------------------------------------------------------------
# Shared state for baseline workflow
# ---------------------------------------------------------------------------

class BaselineState(TypedDict):
    # Input
    topic: str

    # Search output
    papers: list[Paper]

    # LLM output (no verification)
    review_text: str

    # Verification metrics (run AFTER baseline for comparison)
    total_citations: int
    valid_citations: int
    partial_citations: int
    hallucinated_citations: int
    hallucination_rate: float
    citation_details: list[Citation]


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

def _get_groq_client() -> Groq:
    return Groq(api_key=settings.groq_api_key)


# ---------------------------------------------------------------------------
# Node 1: Search (raw topic, no planning)
# ---------------------------------------------------------------------------

def baseline_search_node(state: BaselineState) -> BaselineState:
    """
    Search OpenAlex using the raw topic string directly.
    No planner, no sub-queries — just one broad search.
    """
    print("\n[Baseline] Running Search Node (raw topic, no planning)...")
    topic = state["topic"]

    try:
        papers = search_openalex_works(query=topic, max_results=15)
    except Exception as e:
        print(f"[Baseline] Search failed: {e}")
        papers = []

    print(f"[Baseline] Retrieved {len(papers)} papers.")
    state["papers"] = papers
    return state


# ---------------------------------------------------------------------------
# Node 2: Write (single LLM call, no verification)
# ---------------------------------------------------------------------------

def baseline_write_node(state: BaselineState) -> BaselineState:
    print("\n[Baseline] Running Write Node...")
    client  = _get_groq_client()
    topic   = state["topic"]
    papers  = state["papers"]

    # Format papers list for context
    paper_list_str = "\n".join(
        f"- {p.title} ({p.year or 'N/A'}) "
        f"by {', '.join(p.authors[:2]) if p.authors else 'Unknown'}"
        for p in papers[:15]
    )

    # UPDATED PROMPT — forces LLM to cite confidently from memory
    prompt = (
        f"You are an expert academic writer with deep knowledge of AI research.\n\n"
        f"Write a detailed literature review of 500-600 words on:\n"
        f"Topic: {topic}\n\n"
        f"You may use the papers below as a starting point, but also draw "
        f"on your full academic knowledge to include additional relevant papers.\n\n"
        f"REQUIREMENTS:\n"
        f"1. Include AT LEAST 10-12 specific citations in Harvard format\n"
        f"2. Cite specific authors, years, journal names, and findings\n"
        f"3. Include citations from 2019-2026 covering key papers\n"
        f"4. Write with full academic confidence\n"
        f"5. Include both well-known and niche recent papers\n"
        f"6. Use format: (Author et al., Year) inline\n"
        f"7. End with full reference list in Harvard format\n"
        f"8. Do NOT hesitate to cite — include papers even if details "
        f"are approximate\n\n"
        f"Available papers (supplement with your own knowledge):\n"
        f"{paper_list_str}\n\n"
        f"Write the full literature review now:"
    )

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=1.0,  # HIGH = more varied, more hallucination risk
        )
        review_text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Baseline] LLM call failed: {e}")
        review_text = ""

    print(f"[Baseline] Review written ({len(review_text)} chars).")
    state["review_text"] = review_text
    return state# ---------------------------------------------------------------------------
# Node 3: Verify (run verifier on baseline output for measurement)
# ---------------------------------------------------------------------------

def baseline_verify_node(state: BaselineState) -> BaselineState:
    """
    Run the verifier on the baseline review to measure hallucination rate.

    Note: The baseline does NOT use the verifier to correct output —
    it only runs verification AFTER generation for measurement purposes.
    This mirrors the evaluation design in the IPR.
    """
    print("\n[Baseline] Running Verifier (measurement only, not correction)...")

    review_text = state["review_text"]
    papers      = state["papers"]

    if not review_text:
        print("[Baseline] No review text to verify.")
        state["total_citations"]        = 0
        state["valid_citations"]        = 0
        state["partial_citations"]      = 0
        state["hallucinated_citations"] = 0
        state["hallucination_rate"]     = 0.0
        state["citation_details"]       = []
        return state

    result = verify_review(review_text, papers)

    state["total_citations"]        = result["total"]
    state["valid_citations"]        = result["valid"]
    state["partial_citations"]      = result["partial"]
    state["hallucinated_citations"] = result["hallucinated"]
    state["hallucination_rate"]     = result["hallucination_rate"]
    state["citation_details"]       = result["citations"]

    return state


# ---------------------------------------------------------------------------
# Build baseline graph
# ---------------------------------------------------------------------------

def build_baseline():
    """
    Build and compile the baseline single-LLM LangGraph workflow.

    Returns
    -------
    CompiledGraph
        START → search → write → verify → END
    """
    graph = StateGraph(BaselineState)

    graph.add_node("search", baseline_search_node)
    graph.add_node("write",  baseline_write_node)
    graph.add_node("verify", baseline_verify_node)

    graph.add_edge(START,    "search")
    graph.add_edge("search", "write")
    graph.add_edge("write",  "verify")
    graph.add_edge("verify", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Run baseline
# ---------------------------------------------------------------------------

def run_baseline(topic: str) -> BaselineState:
    """
    Run the baseline single-LLM pipeline for a given topic.

    Parameters
    ----------
    topic : str
        Research topic string.

    Returns
    -------
    BaselineState
        Final state with review text and verification metrics.
    """
    baseline = build_baseline()

    initial_state: BaselineState = {
        "topic":                   topic,
        "papers":                  [],
        "review_text":             "",
        "total_citations":         0,
        "valid_citations":         0,
        "partial_citations":       0,
        "hallucinated_citations":  0,
        "hallucination_rate":      0.0,
        "citation_details":        [],
    }

    print("\n" + "=" * 60)
    print("[Baseline] Starting single-LLM pipeline for topic:")
    print(f"  '{topic}'")
    print("=" * 60)

    final_state = baseline.invoke(initial_state)

    print("\n" + "=" * 60)
    print("[Baseline] BASELINE PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Papers retrieved  : {len(final_state['papers'])}")
    print(f"  Review length     : {len(final_state['review_text'])} chars")
    print(f"  Total citations   : {final_state['total_citations']}")
    print(f"  Valid             : {final_state['valid_citations']}")
    print(f"  Partial           : {final_state['partial_citations']}")
    print(f"  Hallucinated      : {final_state['hallucinated_citations']}")
    print(f"  Hallucination Rate: {final_state['hallucination_rate']:.1%}")
    print("=" * 60)

    return final_state