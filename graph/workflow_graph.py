# graph/workflow_graph.py
"""
Experimental Multi-Agent Workflow (LangGraph).

Pipeline:
    planner_node → search_node → summariser_node → verifier_node

This is the EXPERIMENTAL system compared against the baseline
single-LLM system in your evaluation.

State flows through all nodes as a shared dictionary (ReviewState).
Each node reads from state, does its job, and writes results back.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from agents.planner_agent import plan_topic
from agents.search_agent import retrieve_papers
from agents.summariser_agent import write_literature_review
from agents.verifier_agent import verify_review
from src.models import Paper, Citation


# ---------------------------------------------------------------------------
# Shared state definition
# All agents read and write to this state dict as it flows through the graph
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


# ---------------------------------------------------------------------------
# Node 1: Planner
# ---------------------------------------------------------------------------

def planner_node(state: ReviewState) -> ReviewState:
    """
    Decompose the topic into sub-queries using the Planner Agent.
    """
    print("\n[Graph] Running Planner Node...")
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
    """
    print("\n[Graph] Running Search Node...")
    sub_queries = state["sub_queries"]
    papers = retrieve_papers(sub_queries)
    state["papers"] = papers
    return state


# ---------------------------------------------------------------------------
# Node 3: Summariser
# ---------------------------------------------------------------------------

def summariser_node(state: ReviewState) -> ReviewState:
    """
    Write a full literature review using retrieved papers.
    """
    print("\n[Graph] Running Summariser Node...")
    papers = state["papers"]
    topic  = state["topic"]
    review_text, papers_used = write_literature_review(papers, topic)
    state["draft_review"]  = review_text
    state["papers_used"]   = papers_used
    return state


# ---------------------------------------------------------------------------
# Node 4: Verifier
# ---------------------------------------------------------------------------

def verifier_node(state: ReviewState) -> ReviewState:
    """
    Verify citations in the generated review using the Verifier Agent.
    """
    print("\n[Graph] Running Verifier Node...")
    review_text = state["draft_review"]
    papers      = state["papers"]
    result      = verify_review(review_text, papers)

    state["total_citations"]       = result["total"]
    state["valid_citations"]       = result["valid"]
    state["partial_citations"]     = result["partial"]
    state["hallucinated_citations"]= result["hallucinated"]
    state["hallucination_rate"]    = result["hallucination_rate"]
    state["citation_details"]      = result["citations"]

    return state


# ---------------------------------------------------------------------------
# Build and compile the LangGraph workflow
# ---------------------------------------------------------------------------

def build_workflow():
    """
    Build and compile the full multi-agent LangGraph workflow.

    Returns
    -------
    CompiledGraph
        A runnable LangGraph pipeline:
        START → planner → search → summariser → verifier → END
    """
    graph = StateGraph(ReviewState)

    # Register nodes
    graph.add_node("planner",    planner_node)
    graph.add_node("search",     search_node)
    graph.add_node("summariser", summariser_node)
    graph.add_node("verifier",   verifier_node)

    # Define edges (sequential pipeline)
    graph.add_edge(START,        "planner")
    graph.add_edge("planner",    "search")
    graph.add_edge("search",     "summariser")
    graph.add_edge("summariser", "verifier")
    graph.add_edge("verifier",   END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Run the workflow
# ---------------------------------------------------------------------------

def run_workflow(topic: str) -> ReviewState:
    """
    Run the full multi-agent pipeline for a given topic.

    Parameters
    ----------
    topic : str
        Research topic for the literature review.

    Returns
    -------
    ReviewState
        Final state containing all outputs from all agents.
    """
    workflow = build_workflow()

    # Initial state — only topic is needed; agents fill the rest
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
    }

    print("\n" + "=" * 60)
    print(f"[Workflow] Starting pipeline for topic:")
    print(f"  '{topic}'")
    print("=" * 60)

    final_state = workflow.invoke(initial_state)

    print("\n" + "=" * 60)
    print("[Workflow] PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Sub-queries       : {len(final_state['sub_queries'])}")
    print(f"  Papers retrieved  : {len(final_state['papers'])}")
    print(f"  Review length     : {len(final_state['draft_review'])} chars")
    print(f"  Total citations   : {final_state['total_citations']}")
    print(f"  Valid             : {final_state['valid_citations']}")
    print(f"  Partial           : {final_state['partial_citations']}")
    print(f"  Hallucinated      : {final_state['hallucinated_citations']}")
    print(f"  Hallucination Rate: {final_state['hallucination_rate']:.1%}")
    print("=" * 60)

    return final_state