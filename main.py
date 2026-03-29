# main.py
"""
Main CLI Entry Point for the Literature Review Agent System.

Usage:
    uv run main.py                          # interactive mode
    uv run main.py --mode experimental      # run experimental only
    uv run main.py --mode baseline          # run baseline only
    uv run main.py --mode evaluate          # run both + comparison
    uv run main.py --mode evaluate --topics 3  # run 3 topics
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.workflow_graph import run_workflow
from graph.baseline_graph import run_baseline
from evaluation.evaluator import run_evaluation


# ---------------------------------------------------------------------------
# Predefined topics for evaluation
# (add more topics here to run bigger experiments)
# ---------------------------------------------------------------------------

EVALUATION_TOPICS = [
    "Agentic AI for reliable academic literature review and hallucination mitigation",
    "Retrieval augmented generation for reducing hallucinations in large language models",
    "Multi-agent systems for automated fact-checking and citation verification",
    "Hallucination detection and mitigation strategies in large language models",
    "Knowledge graph integration for improving factual accuracy in LLM outputs",
]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    print("\n" + "=" * 70)
    print("  LITERATURE REVIEW AGENT SYSTEM")
    print("  MSc Data Science — University of Hertfordshire")
    print("  Agentic AI for Reliable Academic Literature Review")
    print("=" * 70)


def _print_result_summary(state: dict, system: str) -> None:
    """Print a clean summary of a single pipeline run."""
    print("\n" + "-" * 50)
    print(f"  RESULTS [{system.upper()}]")
    print("-" * 50)

    if system == "experimental":
        review   = state.get("draft_review", "")
        papers   = state.get("papers", [])
        subqs    = state.get("sub_queries", [])
        print(f"  Sub-queries used    : {len(subqs)}")
    else:
        review   = state.get("review_text", "")
        papers   = state.get("papers", [])
        print(f"  Sub-queries used    : 1 (raw topic)")

    print(f"  Papers retrieved    : {len(papers)}")
    print(f"  Review length       : {len(review)} chars")
    print(f"  Total citations     : {state.get('total_citations', 0)}")
    print(f"  Valid citations     : {state.get('valid_citations', 0)}")
    print(f"  Partial citations   : {state.get('partial_citations', 0)}")
    print(f"  Hallucinated        : {state.get('hallucinated_citations', 0)}")
    print(f"  Hallucination Rate  : {state.get('hallucination_rate', 0):.1%}")
    print("-" * 50)

    print("\n--- GENERATED REVIEW ---\n")
    print(review)
    print("\n--- END OF REVIEW ---")


# ---------------------------------------------------------------------------
# Mode: experimental
# ---------------------------------------------------------------------------

def run_experimental_mode(topic: str) -> None:
    """
    Run the full multi-agent experimental pipeline on a topic.
    """
    print(f"\n[Main] Running EXPERIMENTAL pipeline...")
    print(f"[Main] Topic: {topic}\n")

    state = run_workflow(topic)
    _print_result_summary(state, "experimental")


# ---------------------------------------------------------------------------
# Mode: baseline
# ---------------------------------------------------------------------------

def run_baseline_mode(topic: str) -> None:
    """
    Run the single-LLM baseline pipeline on a topic.
    """
    print(f"\n[Main] Running BASELINE pipeline...")
    print(f"[Main] Topic: {topic}\n")

    state = run_baseline(topic)
    _print_result_summary(state, "baseline")


# ---------------------------------------------------------------------------
# Mode: evaluate
# ---------------------------------------------------------------------------

def run_evaluate_mode(num_topics: int) -> None:
    """
    Run both systems on multiple topics and compare results.
    """
    topics = EVALUATION_TOPICS[:num_topics]

    print(f"\n[Main] Running EVALUATION mode on {len(topics)} topic(s)...")
    for i, t in enumerate(topics, 1):
        print(f"  {i}. {t}")

    results = run_evaluation(topics)

    print(f"\n[Main] Evaluation complete. {len(results)} rows saved to CSV.")
    print(f"[Main] Check data/eval/ for saved reviews and metrics.")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive_mode() -> None:
    """
    Interactive CLI mode: prompt user for topic and system choice.
    """
    print("\nChoose a mode:")
    print("  1. Run Experimental System (multi-agent pipeline)")
    print("  2. Run Baseline System     (single-LLM pipeline)")
    print("  3. Run Full Evaluation     (both systems, comparison table)")
    print("  4. Exit")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "4":
        print("Exiting. Goodbye.")
        sys.exit(0)

    if choice in ("1", "2"):
        topic = input(
            "\nEnter research topic\n"
            "(or press Enter for default): "
        ).strip()

        if not topic:
            topic = (
                "Agentic AI for reliable academic literature review "
                "and hallucination mitigation in large language models"
            )
            print(f"[Main] Using default topic: {topic}")

        if choice == "1":
            run_experimental_mode(topic)
        else:
            run_baseline_mode(topic)

    elif choice == "3":
        num = input(
            f"\nHow many topics to evaluate? "
            f"(1-{len(EVALUATION_TOPICS)}, default=2): "
        ).strip()

        try:
            num_topics = int(num) if num else 2
            num_topics = max(1, min(num_topics, len(EVALUATION_TOPICS)))
        except ValueError:
            num_topics = 2

        run_evaluate_mode(num_topics)

    else:
        print("[Main] Invalid choice. Please run again.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Literature Review Agent System CLI",
    )

    parser.add_argument(
        "--mode",
        choices=["experimental", "baseline", "evaluate", "interactive"],
        default="interactive",
        help="Pipeline mode to run (default: interactive)",
    )

    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Research topic string (used in experimental/baseline modes)",
    )

    parser.add_argument(
        "--topics",
        type=int,
        default=2,
        help="Number of topics to evaluate in evaluate mode (default: 2)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    _print_banner()
    args = parse_args()

    # Default topic if none provided
    default_topic = (
        "Agentic AI for reliable academic literature review "
        "and hallucination mitigation in large language models"
    )

    if args.mode == "interactive":
        run_interactive_mode()

    elif args.mode == "experimental":
        topic = args.topic or default_topic
        run_experimental_mode(topic)

    elif args.mode == "baseline":
        topic = args.topic or default_topic
        run_baseline_mode(topic)

    elif args.mode == "evaluate":
        run_evaluate_mode(args.topics)


if __name__ == "__main__":
    main()