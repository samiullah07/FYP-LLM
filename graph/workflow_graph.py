import sys; sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import sys
import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.graph import StateGraph, START, END

from agents.planner_agent    import plan_topic
from agents.search_agent     import retrieve_papers
from agents.summariser_agent import write_literature_review
from agents.verifier_agent   import verify_review
from agents.assembler_agent  import assemble_final_review, save_assembler_log
from src.models              import Paper, Citation
from src.config              import settings
from configs.prompts         import PROMPT_VERSION

# MAB — optional, does not break pipeline if missing
try:
    from src.mab_selector import bandit
    MAB_AVAILABLE = True
except Exception:
    MAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# NEW: Global constant for multi-pass correction
# ---------------------------------------------------------------------------
MAX_CORRECTION_PASSES = 2


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------
class ReviewState(TypedDict):
    topic:                   str
    sub_queries:             list[str]
    papers:                  list[Paper]
    draft_review:            str
    papers_used:             list[Paper]
    total_citations:         int
    valid_citations:         int
    partial_citations:       int
    hallucinated_citations:  int
    hallucination_rate:      float
    citation_details:        list[Citation]
    final_review:            str
    verified_refs:           list[str]
    hallucinated_refs:       list[str]
    changes:                 dict
    run_id:                  str
    prompt_version:          str
    retry_count:             int
    latency_seconds:         float
    token_estimate:          int
    selected_model:          str
    topic_type:              str
    mab_policy:              dict
    passes_completed:        int  # ADDED for multi-pass correction


# ---------------------------------------------------------------------------
# Node 1: Planner
# ---------------------------------------------------------------------------
def planner_node(state: ReviewState) -> ReviewState:
    print("\n[Graph] -- Node 1: Planner --------------------------")
    topic = state.get("topic", "")
    if not topic:
        print("[Graph] WARNING: topic is empty in planner_node")
        return state
    sub_queries          = plan_topic(topic)
    state["sub_queries"] = sub_queries
    return state


# ---------------------------------------------------------------------------
# Node 2: Search
# ---------------------------------------------------------------------------
def search_node(state: ReviewState) -> ReviewState:
    print("\n[Graph] -- Node 2: Search ---------------------------")
    sub_queries    = state.get("sub_queries", [])
    papers         = retrieve_papers(sub_queries)
    state["papers"] = papers
    return state


# ---------------------------------------------------------------------------
# Node 3: Summariser
# ---------------------------------------------------------------------------
def summariser_node(state: ReviewState) -> ReviewState:
    print("\n[Graph] -- Node 3: Summariser -----------------------")
    papers                = state.get("papers", [])
    topic                 = state.get("topic", "")
    review_text, papers_used = write_literature_review(papers, topic)
    state["draft_review"] = review_text
    state["papers_used"]  = papers_used
    return state


# ---------------------------------------------------------------------------
# Node 4: Verifier
# ---------------------------------------------------------------------------
def verifier_node(state: ReviewState) -> ReviewState:
    print("\n[Graph] -- Node 4: Verifier -------------------------")
    review_text = state.get("draft_review", "")
    papers      = state.get("papers", [])
    run_id      = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    result      = verify_review(review_text, papers, run_id=run_id)

    state["total_citations"]        = result["total"]
    state["valid_citations"]        = result["valid"]
    state["partial_citations"]      = result["partial"]
    state["hallucinated_citations"] = result["hallucinated"]
    state["hallucination_rate"]     = result["hallucination_rate"]
    state["citation_details"]       = result["citations"]

    # Update MAB with reward after verifier runs
    if MAB_AVAILABLE:
        try:
            model      = state.get("selected_model", settings.llm_model)
            topic_type = state.get("topic_type",     "moderate")
            bandit.update(
                model              = model,
                topic_type         = topic_type,
                hallucination_rate = result["hallucination_rate"],
            )
            state["mab_policy"] = bandit.get_policy_summary()
            print(
                f"[MAB] Updated: model={model}, "
                f"reward={1 - result['hallucination_rate']:.3f}"
            )
        except Exception as e:
            print(f"[MAB] Update failed: {e}")

    return state


# ---------------------------------------------------------------------------
# Node 5: Assembler
# ---------------------------------------------------------------------------
def assembler_node(state: ReviewState) -> ReviewState:
    print("\n[Graph] -- Node 5: Assembler ------------------------")
    result = assemble_final_review(
        topic        = state.get("topic", ""),
        draft_review = state.get("draft_review", ""),
        citations    = state.get("citation_details", []),
    )

    state["final_review"]      = result["final_review"]
    state["verified_refs"]     = result["verified_refs"]
    state["hallucinated_refs"] = result["hallucinated_refs"]
    state["changes"]           = result["changes"]

    run_id  = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_dir = settings.data_dir / "eval" / "logs"
    save_assembler_log(result, state.get("topic", ""), log_dir, run_id)

    # Gap 5 integration - log correction pass
    state["passes_completed"] = state.get("passes_completed", 0) + 1
    _log_correction_pass(state)
    print("[Assembler] Correction pass", state.get("passes_completed",0), "logged")
    return state


# ---------------------------------------------------------------------------
# NEW: Conditional routing after verifier
# ---------------------------------------------------------------------------
def route_after_verifier(state: dict) -> str:
    """Conditional edge: retry assembler or end."""
    hall_count = state.get("hallucinated_citations", 0)
    passes = state.get("passes_completed", 0)
    if hall_count > 0 and passes < MAX_CORRECTION_PASSES:
        return "assembler"
    return END


# ---------------------------------------------------------------------------
# NEW: Helper to log correction passes
# ---------------------------------------------------------------------------
def _log_correction_pass(state: dict) -> None:
    """Append per-pass stats to correction_passes_log.json."""
    entry = {
        "run_id":          state.get("run_id", "unknown"),
        "pass_number":     state.get("passes_completed", 1),
        "h_rate_before":   state.get("h_rate_before_assembly", None),
        "h_rate_after":    state.get("hallucination_rate", None),
        "claims_removed":  state.get("claims_removed", 0),
        "claims_rewritten":state.get("claims_rewritten", 0),
        "timestamp":       datetime.now().isoformat(),
    }
    os.makedirs("evaluation_results", exist_ok=True)
    log_path = os.path.join("evaluation_results", "correction_passes_log.json")
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []
    logs.append(entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)


# ---------------------------------------------------------------------------
# Build workflow — NO MAB node in graph
# MAB selection happens BEFORE graph.invoke() in run_workflow()
# ---------------------------------------------------------------------------
def build_workflow():
    graph = StateGraph(ReviewState)

    graph.add_node("planner",    planner_node)
    graph.add_node("search",     search_node)
    graph.add_node("summariser", summariser_node)
    graph.add_node("verifier",   verifier_node)
    graph.add_node("assembler",  assembler_node)

    graph.add_edge(START,        "planner")
    graph.add_edge("planner",    "search")
    graph.add_edge("search",     "summariser")
    graph.add_edge("summariser", "verifier")

    # ADDITION 6: Replace verifier->assembler with conditional edge
    graph.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"assembler": "assembler", END: END}
    )

    graph.add_edge("assembler", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Run workflow
# ---------------------------------------------------------------------------
def run_workflow(topic: str) -> ReviewState:
    """
    Run full 5-agent pipeline.
    MAB model selection happens HERE before graph starts.
    """
    workflow = build_workflow()
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    start    = time.time()

    # MAB selects model BEFORE graph invocation
    selected_model = settings.llm_model
    topic_type     = "moderate"

    if MAB_AVAILABLE:
        try:
            selected_model, topic_type = bandit.select_model(topic)
            settings.llm_model = selected_model
            print(f"\n[MAB] Selected model : {selected_model}")
            print(f"[MAB] Topic type     : {topic_type}")
        except Exception as e:
            print(f"[MAB] Selection error: {e} — using default")

    # Build initial state with ALL keys pre-populated
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
        "retry_count":             0,
        "latency_seconds":         0.0,
        "token_estimate":          0,
        "selected_model":          selected_model,
        "topic_type":              topic_type,
        "mab_policy":              {},
        "passes_completed":        0,
    }

    print("\n" + "=" * 60)
    print(f"[Workflow] Run ID      : {run_id}")
    print(f"[Workflow] Model       : {selected_model}")
    print(f"[Workflow] Topic type  : {topic_type}")
    print(f"[Workflow] Topic       : {topic[:55]}")
    print("=" * 60)

    final_state = workflow.invoke(initial_state)

    # Track latency
    latency = round(time.time() - start, 2)
    final_state["latency_seconds"] = latency
    total_text = (
        final_state.get("draft_review", "") +
        final_state.get("final_review", "")
    )
    final_state["token_estimate"] = len(total_text) // 4

    print("\n" + "=" * 60)
    print("[Workflow] PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Sub-queries         : {len(final_state.get('sub_queries', []))}")
    print(f"  Papers retrieved    : {len(final_state.get('papers', []))}")
    print(f"  Total citations     : {final_state.get('total_citations', 0)}")
    print(f"  Valid               : {final_state.get('valid_citations', 0)}")
    print(f"  Hallucinated        : {final_state.get('hallucinated_citations', 0)}")
    print(f"  Hallucination Rate  : {final_state.get('hallucination_rate', 0):.1%}")
    print(f"  Final review length : {len(final_state.get('final_review', ''))} chars")
    print(f"  Latency             : {latency}s")
    print(f"  Model used          : {final_state.get('selected_model', 'N/A')}")
    print("=" * 60)

    return final_state