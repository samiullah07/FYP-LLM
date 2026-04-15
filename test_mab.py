# test_mab.py
"""
Test the Multi-Armed Bandit Model Selector.

This script:
    1. Runs multiple topics through the full pipeline
    2. MAB selects which LLM model to use for each topic
    3. Updates bandit after observing hallucination rate
    4. Shows learned policy after all runs
    5. Saves full results to CSV for dissertation analysis

Research contribution:
    Demonstrates adaptive model selection that LEARNS which
    LLM performs best for which type of academic topic —
    improving efficiency and reducing hallucination over time.

Run: uv run test_mab.py
"""

import sys
import csv
import json
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from src.mab_selector import bandit, UCB1BanditSelector, MODELS, TOPIC_TYPES
from graph.workflow_graph import run_workflow


# ---------------------------------------------------------------------------
# Test topics — covering all 3 difficulty levels
# ---------------------------------------------------------------------------
TEST_TOPICS = [
    # ── NICHE topics (high hallucination risk) ──────────────────────────
    {
        "topic": (
            "Neurosymbolic approaches to citation graph completion "
            "and hallucination detection in scientific literature agents"
        ),
        "difficulty":             "niche",
        "expected_hallucination": "HIGH",
    },
    {
        "topic": (
            "Legal liability frameworks for AI-generated fabricated "
            "citations in medical and legal academic publishing 2024-2026"
        ),
        "difficulty":             "niche",
        "expected_hallucination": "HIGH",
    },
    {
        "topic": (
            "Blockchain-based verification systems for detecting "
            "ghost citations in autonomous AI academic manuscript generation"
        ),
        "difficulty":             "niche",
        "expected_hallucination": "HIGH",
    },

    # ── MODERATE topics (medium hallucination risk) ──────────────────────
    {
        "topic": (
            "Multi-agent orchestration frameworks for knowledge-intensive "
            "reasoning and automated literature synthesis"
        ),
        "difficulty":             "moderate",
        "expected_hallucination": "MODERATE",
    },
    {
        "topic": (
            "Agentic AI systems using LangChain and LangGraph for "
            "automated academic knowledge retrieval and synthesis"
        ),
        "difficulty":             "moderate",
        "expected_hallucination": "MODERATE",
    },
    {
        "topic": (
            "Large language model hallucination detection benchmarks "
            "for automated academic research tasks and evaluation"
        ),
        "difficulty":             "moderate",
        "expected_hallucination": "MODERATE",
    },

    # ── WELL-COVERED topics (low hallucination risk) ─────────────────────
    {
        "topic": (
            "Retrieval augmented generation for hallucination "
            "reduction in large language model outputs"
        ),
        "difficulty":             "well_covered",
        "expected_hallucination": "LOW",
    },
    {
        "topic": (
            "Transformer architecture and attention mechanisms "
            "in large language models for natural language processing"
        ),
        "difficulty":             "well_covered",
        "expected_hallucination": "LOW",
    },
    {
        "topic": (
            "BERT GPT deep learning natural language processing "
            "text classification and sequence modelling"
        ),
        "difficulty":             "well_covered",
        "expected_hallucination": "LOW",
    },
]

# Output directory
OUTPUT_DIR = settings.data_dir / "eval" / "mab_results"


# ---------------------------------------------------------------------------
# Helper: format a result row for printing
# ---------------------------------------------------------------------------
def _print_result_row(
    run:               int,
    topic_short:       str,
    difficulty:        str,
    selected_model:    str,
    hallucination_rate: float,
    latency:           float,
    reward:            float,
) -> None:
    model_short = selected_model.split("-")[0] + "-" + selected_model.split("-")[1]
    print(
        f"  Run {run:>2} | "
        f"{difficulty:<12} | "
        f"{model_short:<20} | "
        f"Hall: {hallucination_rate:>5.1%} | "
        f"Reward: {reward:>5.3f} | "
        f"Latency: {latency:>5.1f}s | "
        f"{topic_short[:30]}"
    )


# ---------------------------------------------------------------------------
# Main MAB test function
# ---------------------------------------------------------------------------
def run_mab_test(
    topics:     list[dict] = None,
    num_rounds: int        = 1,
) -> list[dict]:
    """
    Run MAB test across all topics for a given number of rounds.

    Each round:
        1. MAB classifies topic into a context state (niche/moderate/well_covered)
        2. MAB selects best model using UCB1 algorithm
        3. Pipeline runs with selected model
        4. Verifier measures hallucination rate
        5. Bandit updates with reward = 1 - hallucination_rate
        6. Policy improves over time

    Parameters
    ----------
    topics     : list of topic dicts (default: TEST_TOPICS)
    num_rounds : how many times to cycle through all topics

    Returns
    -------
    list[dict] : all experiment results
    """
    if topics is None:
        topics = TEST_TOPICS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    run_number  = 0

    print("\n" + "=" * 70)
    print("MULTI-ARMED BANDIT MODEL SELECTION TEST")
    print("=" * 70)
    print(f"  Topics       : {len(topics)}")
    print(f"  Rounds       : {num_rounds}")
    print(f"  Total runs   : {len(topics) * num_rounds}")
    print(f"  Models (arms): {len(MODELS)}")
    print(f"  Context types: {TOPIC_TYPES}")
    print("=" * 70)

    # Print initial bandit state
    print("\n[MAB] Initial bandit state:")
    for topic_type in TOPIC_TYPES:
        print(f"  {topic_type}:")
        for model in MODELS:
            pulls = bandit.counts[topic_type][model]
            avg   = bandit.values[topic_type][model]
            print(f"    {model:<38} pulls={pulls}, avg_reward={avg:.3f}")

    print("\n" + "-" * 70)
    print(f"{'Run':>4} | {'Difficulty':<12} | {'Model':<20} | "
          f"{'Hall.':>7} | {'Reward':>7} | {'Latency':>8} | Topic")
    print("-" * 70)

    for round_num in range(1, num_rounds + 1):
        if num_rounds > 1:
            print(f"\n{'='*30} ROUND {round_num}/{num_rounds} {'='*30}")

        for topic_config in topics:
            run_number += 1
            topic      = topic_config["topic"]
            difficulty = topic_config["difficulty"]

            # Step 1: MAB selects model
            selected_model, topic_type = bandit.select_model(topic)

            # Temporarily override model
            original_model     = settings.llm_model
            settings.llm_model = selected_model

            # Step 2: Run pipeline with selected model
            start_time = time.time()
            try:
                state = run_workflow(topic)
                hallucination_rate = state.get("hallucination_rate", 0.0)
                total_citations    = state.get("total_citations",    0)
                valid_citations    = state.get("valid_citations",    0)
                hallucinated       = state.get("hallucinated_citations", 0)
                final_review       = (
                    state.get("final_review", "") or
                    state.get("draft_review", "")
                )
                success = True

            except Exception as e:
                print(f"\n[MAB] ERROR on run {run_number}: {e}")
                hallucination_rate = 1.0
                total_citations    = 0
                valid_citations    = 0
                hallucinated       = 0
                final_review       = ""
                success            = False

            finally:
                # Restore original model
                settings.llm_model = original_model

            latency = round(time.time() - start_time, 2)
            reward  = round(1.0 - hallucination_rate, 3)

            # Step 3: Update bandit with observed reward
            bandit.update(
                model              = selected_model,
                topic_type         = topic_type,
                hallucination_rate = hallucination_rate,
            )

            # Step 4: Print row result
            _print_result_row(
                run                = run_number,
                topic_short        = topic[:30],
                difficulty         = difficulty,
                selected_model     = selected_model,
                hallucination_rate = hallucination_rate,
                latency            = latency,
                reward             = reward,
            )

            # Step 5: Store result
            result = {
                "timestamp":           datetime.now().isoformat(),
                "round":               round_num,
                "run":                 run_number,
                "topic":               topic[:80],
                "topic_difficulty":    difficulty,
                "topic_type_detected": topic_type,
                "selected_model":      selected_model,
                "hallucination_rate":  hallucination_rate,
                "total_citations":     total_citations,
                "valid_citations":     valid_citations,
                "hallucinated":        hallucinated,
                "citation_precision":  round(valid_citations / total_citations, 3)
                                       if total_citations > 0 else 0,
                "reward":              reward,
                "latency_seconds":     latency,
                "success":             success,
            }
            all_results.append(result)

            # Save incrementally
            _save_results(all_results, timestamp)

    # Print final policy summary
    _print_policy_summary()

    # Print experiment summary
    _print_experiment_summary(all_results)

    return all_results


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
def _save_results(results: list[dict], timestamp: str) -> None:
    """Save results to CSV and JSON."""
    if not results:
        return

    csv_path  = OUTPUT_DIR / f"mab_results_{timestamp}.csv"
    json_path = OUTPUT_DIR / f"mab_results_{timestamp}.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "total_runs": len(results),
                "results":   results,
                "bandit_state": {
                    "counts": bandit.counts,
                    "values": bandit.values,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Print learned policy summary
# ---------------------------------------------------------------------------
def _print_policy_summary() -> None:
    """Print what the bandit has learned."""
    print("\n" + "=" * 70)
    print("LEARNED POLICY SUMMARY")
    print("=" * 70)
    print("What the MAB learned: which model works best for each topic type\n")

    policy = bandit.get_policy_summary()

    for topic_type, data in policy.items():
        print(f"  Topic Type: {topic_type.upper()}")
        print(f"  {'─'*50}")
        for model in MODELS:
            avg    = data["avg_rewards"].get(model, 0)
            pulls  = data["pull_counts"].get(model, 0)
            marker = " ← PREFERRED" if model == data["preferred_model"] else ""
            bar    = "█" * int(avg * 20)
            print(
                f"  {model:<38} "
                f"avg={avg:.3f} "
                f"|{bar:<20}| "
                f"pulls={pulls}"
                f"{marker}"
            )
        print()

    print("=" * 70)


# ---------------------------------------------------------------------------
# Print experiment summary
# ---------------------------------------------------------------------------
def _print_experiment_summary(results: list[dict]) -> None:
    """Print overall summary statistics."""
    if not results:
        return

    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    total_runs  = len(results)
    avg_hall    = sum(r["hallucination_rate"] for r in results) / total_runs
    avg_latency = sum(r["latency_seconds"]    for r in results) / total_runs
    avg_reward  = sum(r["reward"]             for r in results) / total_runs

    print(f"\n  Total runs              : {total_runs}")
    print(f"  Avg hallucination rate  : {avg_hall:.1%}")
    print(f"  Avg reward              : {avg_reward:.3f}")
    print(f"  Avg latency             : {avg_latency:.1f}s")

    # Per-model summary
    print(f"\n  {'Model':<38} {'Runs':>5} {'Avg Hall.':>10} {'Avg Reward':>11}")
    print(f"  {'─'*68}")
    for model in MODELS:
        model_results = [r for r in results if r["selected_model"] == model]
        if model_results:
            runs        = len(model_results)
            avg_h       = sum(r["hallucination_rate"] for r in model_results) / runs
            avg_r       = sum(r["reward"]             for r in model_results) / runs
            print(
                f"  {model:<38} {runs:>5} "
                f"{avg_h:>9.1%} "
                f"{avg_r:>11.3f}"
            )

    # Per-difficulty summary
    print(f"\n  {'Difficulty':<15} {'Runs':>5} {'Avg Hall.':>10} {'Best Model'}")
    print(f"  {'─'*68}")
    for difficulty in ["niche", "moderate", "well_covered"]:
        d_results = [r for r in results if r["topic_difficulty"] == difficulty]
        if d_results:
            runs    = len(d_results)
            avg_h   = sum(r["hallucination_rate"] for r in d_results) / runs
            best    = min(d_results, key=lambda x: x["hallucination_rate"])
            print(
                f"  {difficulty:<15} {runs:>5} "
                f"{avg_h:>9.1%}  "
                f"{best['selected_model']}"
            )

    print("\n" + "=" * 70)
    print("DISSERTATION INSIGHT")
    print("=" * 70)
    print(
        "\n  The MAB demonstrates that adaptive model selection can improve"  )