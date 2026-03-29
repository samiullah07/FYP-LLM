# evaluation/evaluator.py
"""
Evaluation Runner for the Literature Review Agent System.

Responsibility:
    Run BOTH pipelines on the same topic(s) and compare:
        - Experimental : multi-agent (planner → search → summarise → verify)
        - Baseline     : single-LLM  (search → write → measure)

    Metrics compared:
        - Total citations found
        - Valid citations
        - Partial citations
        - Hallucinated citations
        - Hallucination rate (%)
        - Review length (chars)
        - Papers retrieved

    Outputs:
        - Console comparison table
        - CSV file saved to data/eval/results.csv
        - Both review texts saved to data/eval/
"""

import sys
import csv
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.workflow_graph import run_workflow
from graph.baseline_graph import run_baseline
from src.config import settings


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

EVAL_DIR = settings.data_dir / "eval"


def _ensure_eval_dir() -> None:
    """Create the eval output directory if it does not exist."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Save review text to file
# ---------------------------------------------------------------------------

def _save_review(text: str, filename: str) -> Path:
    """
    Save a review text string to a .txt file in data/eval/.

    Parameters
    ----------
    text     : str   - review text to save
    filename : str   - filename without path

    Returns
    -------
    Path : path to saved file
    """
    _ensure_eval_dir()
    out_path = EVAL_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[Evaluator] Review saved to: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Save metrics to CSV
# ---------------------------------------------------------------------------

def _save_metrics_csv(rows: list[dict], filename: str = "results.csv") -> Path:
    """
    Save evaluation metrics rows to a CSV file.

    Parameters
    ----------
    rows     : list of dicts, one per (topic, system) run
    filename : output CSV filename

    Returns
    -------
    Path : path to saved CSV
    """
    _ensure_eval_dir()
    out_path = EVAL_DIR / filename

    if not rows:
        print("[Evaluator] No rows to save.")
        return out_path

    fieldnames = list(rows[0].keys())

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Evaluator] Metrics CSV saved to: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Print comparison table to console
# ---------------------------------------------------------------------------

def _print_comparison_table(
    topic: str,
    baseline_state: dict,
    experimental_state: dict,
) -> None:
    """
    Print a side-by-side comparison table of both systems.
    """
    b = baseline_state
    e = experimental_state

    print("\n" + "=" * 70)
    print("EVALUATION COMPARISON TABLE")
    print("=" * 70)
    print(f"Topic: {topic[:65]}")
    print("-" * 70)
    print(f"{'Metric':<30} {'Baseline':>15} {'Experimental':>15}")
    print("-" * 70)

    metrics = [
        ("Papers Retrieved",
         len(b["papers"]),
         len(e["papers"])),

        ("Sub-queries Used",
         1,
         len(e["sub_queries"])),

        ("Review Length (chars)",
         len(b["review_text"]),
         len(e["draft_review"])),

        ("Total Citations",
         b["total_citations"],
         e["total_citations"]),

        ("Valid Citations",
         b["valid_citations"],
         e["valid_citations"]),

        ("Partial Citations",
         b["partial_citations"],
         e["partial_citations"]),

        ("Hallucinated Citations",
         b["hallucinated_citations"],
         e["hallucinated_citations"]),

        ("Hallucination Rate (%)",
         f"{b['hallucination_rate']:.1%}",
         f"{e['hallucination_rate']:.1%}"),
    ]

    for label, b_val, e_val in metrics:
        print(f"  {label:<28} {str(b_val):>15} {str(e_val):>15}")

    print("-" * 70)

    # Highlight winner for hallucination rate
    b_rate = b["hallucination_rate"]
    e_rate = e["hallucination_rate"]

    if e_rate < b_rate:
        diff = b_rate - e_rate
        print(f"\n  ✓ Experimental system reduced hallucination rate by {diff:.1%}")
    elif e_rate == b_rate:
        print(f"\n  = Both systems achieved the same hallucination rate")
    else:
        diff = e_rate - b_rate
        print(f"\n  ✗ Baseline had lower hallucination rate by {diff:.1%}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_evaluation(topics: list[str]) -> list[dict]:
    """
    Run both pipelines on each topic and compare results.

    Parameters
    ----------
    topics : list[str]
        List of research topic strings to evaluate.

    Returns
    -------
    list[dict]
        All metric rows (one per topic per system) for CSV export.
    """
    all_rows: list[dict] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, topic in enumerate(topics, 1):
        print("\n" + "#" * 70)
        print(f"# EXPERIMENT {i}/{len(topics)}")
        print(f"# Topic: {topic[:60]}")
        print("#" * 70)

        # --- Run Baseline ---
        print("\n>>> RUNNING BASELINE SYSTEM <<<")
        baseline_state = run_baseline(topic)

        # --- Run Experimental ---
        print("\n>>> RUNNING EXPERIMENTAL SYSTEM <<<")
        experimental_state = run_workflow(topic)

        # --- Print comparison table ---
        _print_comparison_table(topic, baseline_state, experimental_state)

        # --- Save review texts ---
        safe_topic = topic[:40].replace(" ", "_").replace("/", "-")
        _save_review(
            baseline_state["review_text"],
            f"baseline_{i}_{safe_topic}_{timestamp}.txt",
        )
        _save_review(
            experimental_state["draft_review"],
            f"experimental_{i}_{safe_topic}_{timestamp}.txt",
        )

        # --- Build CSV rows ---
        base_row = {
            "timestamp":            timestamp,
            "experiment":           i,
            "topic":                topic,
            "system":               "baseline",
            "papers_retrieved":     len(baseline_state["papers"]),
            "sub_queries_used":     1,
            "review_length_chars":  len(baseline_state["review_text"]),
            "total_citations":      baseline_state["total_citations"],
            "valid_citations":      baseline_state["valid_citations"],
            "partial_citations":    baseline_state["partial_citations"],
            "hallucinated":         baseline_state["hallucinated_citations"],
            "hallucination_rate":   baseline_state["hallucination_rate"],
        }

        exp_row = {
            "timestamp":            timestamp,
            "experiment":           i,
            "topic":                topic,
            "system":               "experimental",
            "papers_retrieved":     len(experimental_state["papers"]),
            "sub_queries_used":     len(experimental_state["sub_queries"]),
            "review_length_chars":  len(experimental_state["draft_review"]),
            "total_citations":      experimental_state["total_citations"],
            "valid_citations":      experimental_state["valid_citations"],
            "partial_citations":    experimental_state["partial_citations"],
            "hallucinated":         experimental_state["hallucinated_citations"],
            "hallucination_rate":   experimental_state["hallucination_rate"],
        }

        all_rows.append(base_row)
        all_rows.append(exp_row)

    # --- Save all metrics to CSV ---
    _save_metrics_csv(all_rows, f"results_{timestamp}.csv")

    # --- Print overall summary ---
    print("\n" + "=" * 70)
    print("OVERALL EVALUATION SUMMARY")
    print("=" * 70)

    base_rows = [r for r in all_rows if r["system"] == "baseline"]
    exp_rows  = [r for r in all_rows if r["system"] == "experimental"]

    avg_base_rate = sum(r["hallucination_rate"] for r in base_rows) / len(base_rows)
    avg_exp_rate  = sum(r["hallucination_rate"] for r in exp_rows)  / len(exp_rows)

    print(f"  Topics evaluated          : {len(topics)}")
    print(f"  Avg baseline hall. rate   : {avg_base_rate:.1%}")
    print(f"  Avg experimental hall. rate: {avg_exp_rate:.1%}")

    if avg_exp_rate < avg_base_rate:
        improvement = ((avg_base_rate - avg_exp_rate) / avg_base_rate * 100
                       if avg_base_rate > 0 else 0)
        print(f"  Improvement               : {improvement:.1f}% reduction in hallucination")
    print("=" * 70)

    return all_rows