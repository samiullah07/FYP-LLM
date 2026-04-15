"""
Evaluation Module — runs Experimental vs Baseline side-by-side.

Output:
    data/eval/eval_results_<timestamp>.csv
    data/eval/eval_results_<timestamp>.json

Run via:
    uv run test_evaluation.py
"""

import sys
import csv
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.workflow_graph import run_workflow
from graph.baseline_graph import run_baseline

EVAL_DIR = ROOT / "data" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TOPICS = [
    "Agentic AI for reliable academic literature review and hallucination mitigation",
    "Retrieval-augmented generation for reducing LLM hallucinations",
    "Multi-agent systems for automated scientific paper summarisation",
    "Transformer-based models for citation verification in academic texts",
    "Self-correcting language model agents for knowledge-intensive tasks",
]


# ─────────────────────────────────────────────────────────────────────────────
# Core runner
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_topic(topic: str) -> dict[str, Any]:
    """Run both pipelines on one topic and return merged metrics."""
    result: dict[str, Any] = {
        "topic":     topic,
        "timestamp": datetime.now().isoformat(),
    }

    # ── Experimental pipeline ─────────────────────────────────────────────
    print(f"    → Experimental pipeline ...")
    t0 = time.time()
    try:
        exp = run_workflow(topic)
        exp_lat = round(time.time() - t0, 2)
        result.update({
            "exp_papers":             len(exp.get("papers", [])),
            "exp_sub_queries":        len(exp.get("sub_queries", [])),
            "exp_citations_total":    exp.get("total_citations", 0),
            "exp_citations_valid":    exp.get("valid_citations", 0),
            "exp_citations_halluc":   exp.get("hallucinated_citations", 0),
            "exp_hallucination_rate": exp.get("hallucination_rate", 0.0),
            "exp_review_length":      len(exp.get("draft_review", "")),
            "exp_latency_s":          exp_lat,
            "exp_selected_model":     exp.get("selected_model", "N/A"),
            "exp_topic_type":         exp.get("topic_type", "N/A"),
            "exp_error":              "",
        })
    except Exception as exc:
        exp_lat = round(time.time() - t0, 2)
        print(f"    [!] Experimental ERROR: {exc}")
        result.update({
            "exp_papers": 0, "exp_sub_queries": 0,
            "exp_citations_total": 0, "exp_citations_valid": 0,
            "exp_citations_halluc": 0, "exp_hallucination_rate": 0.0,
            "exp_review_length": 0, "exp_latency_s": exp_lat,
            "exp_selected_model": "N/A", "exp_topic_type": "N/A",
            "exp_error": str(exc),
        })

    # ── Baseline pipeline ─────────────────────────────────────────────────
    print(f"    → Baseline pipeline ...")
    t0 = time.time()
    try:
        base = run_baseline(topic)
        base_lat = round(time.time() - t0, 2)
        result.update({
            "base_citations_total":    base.get("total_citations", 0),
            "base_citations_valid":    base.get("valid_citations", 0),
            "base_citations_halluc":   base.get("hallucinated_citations", 0),
            "base_hallucination_rate": base.get("hallucination_rate", 0.0),
            "base_review_length":      len(base.get("draft_review", "")),
            "base_latency_s":          base_lat,
            "base_error":              "",
        })
    except Exception as exc:
        base_lat = round(time.time() - t0, 2)
        print(f"    [!] Baseline ERROR: {exc}")
        result.update({
            "base_citations_total": 0, "base_citations_valid": 0,
            "base_citations_halluc": 0, "base_hallucination_rate": 0.0,
            "base_review_length": 0, "base_latency_s": base_lat,
            "base_error": str(exc),
        })

    # ── Derived comparison metrics ────────────────────────────────────────
    bh = result.get("base_hallucination_rate", 0.0)
    eh = result.get("exp_hallucination_rate", 0.0)
    result["hallucination_reduction_pct"] = round((bh - eh) * 100, 1) if bh > 0 else 0.0
    result["latency_overhead_s"] = round(
        result.get("exp_latency_s", 0) - result.get("base_latency_s", 0), 2
    )

    _print_topic_result(result)
    return result


def run_evaluation(
    topics: list[str] | None = None,
    save: bool = True,
) -> list[dict[str, Any]]:
    """
    Run full evaluation over a list of topics.

    Args:
        topics: List of research topics. Defaults to DEFAULT_TOPICS.
        save:   Whether to persist results to CSV + JSON.

    Returns:
        List of per-topic result dicts.
    """
    topics = topics or DEFAULT_TOPICS
    results: list[dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("EVALUATION RUNNER  —  Experimental vs Baseline")
    print(f"Topics : {len(topics)}")
    print("=" * 70)

    for i, topic in enumerate(topics, 1):
        print(f"\n[{i}/{len(topics)}] {topic[:65]}...")
        result = evaluate_topic(topic)
        results.append(result)

    _print_summary(results)

    if save and results:
        _save_results(results)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_topic_result(r: dict) -> None:
    bh        = r.get("base_hallucination_rate", 0.0)
    eh        = r.get("exp_hallucination_rate",  0.0)
    reduction = r.get("hallucination_reduction_pct", 0.0)
    print(
        f"\n    {'Metric':<30} {'Baseline':>10} {'Experimental':>14}\n"
        f"    {'─'*56}\n"
        f"    {'Hallucination Rate':<30} {bh:>9.1%} {eh:>13.1%}\n"
        f"    {'Citations (total)':<30} "
        f"{r.get('base_citations_total', 0):>10} "
        f"{r.get('exp_citations_total',  0):>14}\n"
        f"    {'Latency (s)':<30} "
        f"{r.get('base_latency_s', 0):>10.1f} "
        f"{r.get('exp_latency_s',  0):>14.1f}\n"
        f"    {'─'*56}\n"
        f"    Hallucination reduction : {reduction:+.1f} pp"
    )


def _print_summary(results: list[dict]) -> None:
    n = len(results)
    if n == 0:
        print("\n  [!] No results to summarise.")
        return

    avg_base = sum(r.get("base_hallucination_rate",      0) for r in results) / n
    avg_exp  = sum(r.get("exp_hallucination_rate",       0) for r in results) / n
    avg_red  = sum(r.get("hallucination_reduction_pct",  0) for r in results) / n
    avg_lat  = sum(r.get("exp_latency_s",                0) for r in results) / n

    print("\n" + "=" * 70)
    print("OVERALL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Topics evaluated               : {n}")
    print(f"  Avg baseline hallucination     : {avg_base:.1%}")
    print(f"  Avg experimental hallucination : {avg_exp:.1%}")
    print(f"  Avg hallucination reduction    : {avg_red:+.1f} percentage points")
    print(f"  Avg experimental latency       : {avg_lat:.1f}s")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def _save_results(results: list[dict]) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = EVAL_DIR / f"eval_results_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  [Eval] CSV  saved → {csv_path.name}")

    json_path = EVAL_DIR / f"eval_results_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  [Eval] JSON saved → {json_path.name}")