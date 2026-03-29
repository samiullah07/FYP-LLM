# graph/workflow_graph.py
"""
Experimental Multi-Agent Workflow (LangGraph).

Pipeline:
    planner → search → summariser → verifier → assembler

This is the EXPERIMENTAL system compared against the baseline
single-LLM system in your evaluation.

Key improvements over baseline:
    1. Planner decomposes topic into focused sub-queries
    2. Search retrieves papers per sub-query (wider coverage)
    3. Summariser writes review with strict citation instructions
    4. Verifier checks every citation against OpenAlex metadata
    5. Assembler removes/rephrases hallucinated citations

State flows through all nodes as a shared ReviewState TypedDict.
Each node reads from state, does its job, writes results back.

Prompt version used: configs/prompts.PROMPT_VERSION
"""

import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from agents.planner_agent   import plan_topic
from agents.search_agent    import retrieve_papers
from agents.summariser_agent import write_literature_review
from agents.verifier_agent  import verify_review
from agents.assembler_agent import assemble_final_review, save_assembler_log
from src.models  import Paper, Citation
from src.config  import settings
from configs.prompts import PROMPT_VERSION


# ---------------------------------------------------------------------------
# Shared State Definition
# ALL agents read and write to this single state dict
# ---------------------------------------------------------------------------

class ReviewState(TypedDict):
    # Input
    topic: str

    # Planner output
    sub_queries: list[str]

    # Search output
    papers: list[Paper]

    # Summariser output
    draft_review: str
    papers_used: list[Paper]

    # Verifier output
    total_citations: int
    valid_citations: int
    partial_citations: int
    hallucinated_citations: int
    hallucination_rate: float
    citation_details: list[Citation]

    # Assembler output
    final_review: str
    verified_refs: list[str]
    hallucinated_refs: list[str]
    changes: dict

    # Run metadata
    run_id: str
    prompt_version: str


# ---------------------------------------------------------------------------
# Node 1: Planner
# ---------------------------------------------------------------------------

def planner_node(state: ReviewState) -> ReviewState:
    """
    Decompose the research topic into 3-5 focused sub-queries.

    Input  : state["topic"]
    Output : state["sub_queries"]
    """
    print("\n[Graph] ── Node 1: Planner ──────────────────────────────")
    topic = state["topic"]
    sub_queries = plan_topic(topic)
    state["sub_queries"] = sub_queries
    return state


# ---------------------------------------------------------------------------
# Node 2: Search
# ---------------------------------------------------------------------------

def search_node(state: ReviewState) -> ReviewState:
    """
    Retrieve papers from OpenAlex for each sub-query.

    Input  : state["sub_queries"]
    Output : state["papers"]
    """
    print("\n[Graph] ── Node 2: Search ───────────────────────────────")
    sub_queries = state["sub_queries"]
    papers = retrieve_papers(sub_queries)
    state["papers"] = papers
    return state


# ---------------------------------------------------------------------------
# Node 3: Summariser
# ---------------------------------------------------------------------------

def summariser_node(state: ReviewState) -> ReviewState:
    """
    Write a full literature review draft from retrieved papers.

    Input  : state["papers"], state["topic"]
    Output : state["draft_review"], state["papers_used"]
    """
    print("\n[Graph] ── Node 3: Summariser ───────────────────────────")
    papers = state["papers"]
    topic  = state["topic"]
    review_text, papers_used = write_literature_review(papers, topic)
    state["draft_review"] = review_text
    state["papers_used"]  = papers_used
    return state


# ---------------------------------------------------------------------------
# Node 4: Verifier
# ---------------------------------------------------------------------------

def verifier_node(state: ReviewState) -> ReviewState:
    """
    Verify all citations in the draft review against OpenAlex.

    Input  : state["draft_review"], state["papers"]
    Output : state["total_citations"], state["valid_citations"],
             state["partial_citations"], state["hallucinated_citations"],
             state["hallucination_rate"], state["citation_details"]
    """
    print("\n[Graph] ── Node 4: Verifier ─────────────────────────────")
    review_text = state["draft_review"]
    papers      = state["papers"]
    result      = verify_review(review_text, papers)

    state["total_citations"]        = result["total"]
    state["valid_citations"]        = result["valid"]
    state["partial_citations"]      = result["partial"]
    state["hallucinated_citations"] = result["hallucinated"]
    state["hallucination_rate"]     = result["hallucination_rate"]
    state["citation_details"]       = result["citations"]

    return state


# ---------------------------------------------------------------------------
# Node 5: Assembler
# ---------------------------------------------------------------------------

def assembler_node(state: ReviewState) -> ReviewState:
    """
    Produce the final clean review by removing/rephrasing
    hallucinated citations flagged by the Verifier.

    Input  : state["draft_review"], state["citation_details"],
             state["topic"]
    Output : state["final_review"], state["verified_refs"],
             state["hallucinated_refs"], state["changes"]
    """
    print("\n[Graph] ── Node 5: Assembler ────────────────────────────")
    result = assemble_final_review(
        topic        = state["topic"],
        draft_review = state["draft_review"],
        citations    = state["citation_details"],
    )

    state["final_review"]      = result["final_review"]
    state["verified_refs"]     = result["verified_refs"]
    state["hallucinated_refs"] = result["hallucinated_refs"]
    state["changes"]           = result["changes"]

    # Save structured assembler log
    run_id  = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_dir = settings.data_dir / "eval" / "logs"
    save_assembler_log(result, state["topic"], log_dir, run_id)

    return state


# ---------------------------------------------------------------------------
# Build and compile the LangGraph workflow
# ---------------------------------------------------------------------------

def build_workflow():
    """
    Build the full 5-node LangGraph pipeline.

    Graph:
        START → planner → search → summariser → verifier → assembler → END
    """
    graph = StateGraph(ReviewState)

    # Register all nodes
    graph.add_node("planner",    planner_node)
    graph.add_node("search",     search_node)
    graph.add_node("summariser", summariser_node)
    graph.add_node("verifier",   verifier_node)
    graph.add_node("assembler",  assembler_node)

    # Define sequential edges
    graph.add_edge(START,        "planner")
    graph.add_edge("planner",    "search")
    graph.add_edge("search",     "summariser")
    graph.add_edge("summariser", "verifier")
    graph.add_edge("verifier",   "assembler")
    graph.add_edge("assembler",  END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Run workflow
# ---------------------------------------------------------------------------

def run_workflow(topic: str) -> ReviewState:
    """
    Run the full 5-agent pipeline for a given research topic.

    Parameters
    ----------
    topic : str
        Research topic string.

    Returns
    -------
    ReviewState
        Final state with all agent outputs populated.
    """
    workflow = build_workflow()
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initial state — only topic and metadata pre-filled
    initial_state: ReviewState = {
        "topic":                   topic,
        "sub_queries":             [],
        "papers":                  [],
        "draft_review":            "",
        "papers_used":             [],
        "total_citations":         0,
        "valid_citations":         0,
        "partial_citations":       0,
        "hallucinated_citations":  0,
        "hallucination_rate":      0.0,
        "citation_details":        [],
        "final_review":            "",
        "verified_refs":           [],
        "hallucinated_refs":       [],
        "changes":                 {},
        "run_id":                  run_id,
        "prompt_version":          PROMPT_VERSION,
    }

    print("\n" + "=" * 60)
    print(f"[Workflow] Run ID      : {run_id}")
    print(f"[Workflow] Prompt ver  : {PROMPT_VERSION}")
    print(f"[Workflow] Topic       : {topic[:55]}")
    print("=" * 60)

    final_state = workflow.invoke(initial_state)

    # Print final summary
    print("\n" + "=" * 60)
    print("[Workflow] PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Sub-queries         : {len(final_state['sub_queries'])}")
    print(f"  Papers retrieved    : {len(final_state['papers'])}")
    print(f"  Draft length        : {len(final_state['draft_review'])} chars")
    print(f"  Total citations     : {final_state['total_citations']}")
    print(f"  Valid               : {final_state['valid_citations']}")
    print(f"  Partial             : {final_state['partial_citations']}")
    print(f"  Hallucinated        : {final_state['hallucinated_citations']}")
    print(f"  Hallucination Rate  : {final_state['hallucination_rate']:.1%}")
    print(f"  Final review length : {len(final_state['final_review'])} chars")
    print(f"  Words removed       : {final_state['changes'].get('words_removed', 0)}")
    print(f"  Prompt version      : {final_state['prompt_version']}")
    print("=" * 60)

    return final_state