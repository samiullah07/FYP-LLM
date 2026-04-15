# evaluation/ablation_study.py
"""
Ablation Study: SLM vs LLM Performance on Consumer Hardware.

Measures for each model:
    - Hallucination rate
    - Latency (seconds per review)
    - Token throughput
    - Cost per review (USD)
    - Citation precision

Research Question:
    Can small language models achieve comparable hallucination
    detection accuracy to large models on consumer-grade hardware?
"""

import sys
import time
import json
import csv
import psutil
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from graph.workflow_graph import run_workflow
from graph.baseline_graph import run_baseline

# ---------------------------------------------------------------------------
# Models to ablate
# ---------------------------------------------------------------------------
ABLATION_MODELS = [
    {
        "name":           "llama-3.1-8b-instant",
        "size_b":         8,
        "context_window": 128_000,
        "tier":           "SLM",
        "description":    "Smallest — fastest, most efficient",
    },
    {
        "name":           "meta-llama/llama-4-scout-17b-16e-instruct",
        "size_b":         17,
        "context_window": 128_000,
        "tier":           "SLM",
        "description":    "Mid SLM — balanced speed/quality",
    },
    {
        "name":           "meta-llama/llama-4-maverick-17b-128e-instruct",
        "size_b":         17,
        "context_window": 128_000,
        "tier":           "SLM",
        "description":    "Mid SLM — higher quality reasoning",
    },
    {
        "name":           "llama-3.3-70b-versatile",
        "size_b":         70,
        "context_window": 128_000,
        "tier":           "LLM",
        "description":    "Large — best quality baseline",
    },
]

# ------------------------------------------------------------------------#
# Test topics — one per difficulty level
# ---------------------------------------------------------------------------
ABLATION_TOPICS = [
    {
        "topic": (
            "Retrieval augmented generation for hallucination "
            "reduction in large language model outputs"
        ),
        "difficulty": "easy",
        "expected_hallucination": "low",
    },
    {
        "topic": (
            "Agentic AI for reliable academic literature review "
            "and citation verification using multi-agent systems"
        ),
        "difficulty": "medium",
        "expected_hallucination": "moderate",
    },
    {
        "topic": (
            "Neurosymbolic approaches to citation graph completion "
            "and hallucination detection in scientific literature agents"
        ),
        "difficulty": "hard",
        "expected_hallucination": "high",
    },
]

ABLATION_OUTPUT_DIR = settings.data_dir / "eval" / "ablation"


# ---------------------------------------------------------------------------
# Memory measurement helper
# ---------------------------------------------------------------------------
def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------
def calculate_cost(
    model_config:   dict,
    review_text:    str,
    input_estimate: int = 2000,
) -> float:
    """
    Estimate USD cost for one review run.

    Parameters
    ----------
    model_config   : model dict with pricing
    review_text    : generated review text
    input_estimate : estimated input tokens

    Returns
    -------
    float : cost in USD
    """
    output_tokens = len(review_text.split()) * 1.3
    input_cost    = (input_estimate  / 1_000_000) * model_config["cost_per_1m_input"]
    output_cost   = (output_tokens   / 1_000_000) * model_config["cost_per_1m_output"]
    return round(input_cost + output_cost, 6)


# ---------------------------------------------------------------------------
# Run single ablation experiment
# ---------------------------------------------------------------------------
def run_single_experiment(
    model_config:  dict,
    topic_config:  dict,
    system:        str = "experimental",
) -> dict:
    """
    Run one experiment: one model, one topic, one system.

    Parameters
    ----------
    model_config : dict — model info from ABLATION_MODELS
    topic_config : dict — topic info from ABLATION_TOPICS
    system       : str  — "experimental" or "baseline"

    Returns
    -------
    dict : experiment results
    """
    model = model_config["name"]
    topic = topic_config["topic"]

    print(f"\n{'='*60}")
    print(f"Model  : {model} ({model_config['size_b']}B params)")
    print(f"Topic  : {topic[:55]}...")
    print(f"System : {system}")
    print(f"{'='*60}")

    # Override model in settings
    original_model     = settings.llm_model
    settings.llm_model = model

    # Measure memory before
    mem_before = get_memory_usage_mb()

    # Measure latency
    start_time = time.time()

    try:
        if system == "experimental":
            state = run_workflow(topic)
            review_text       = state.get("final_review", "") or state.get("draft_review", "")
            hallucination_rate = state.get("hallucination_rate", 0)
            total_citations   = state.get("total_citations", 0)
            valid_citations   = state.get("valid_citations", 0)
            hallucinated      = state.get("hallucinated_citations", 0)
        else:
            state = run_baseline(topic)
            review_text       = state.get("review_text", "")
            hallucination_rate = state.get("hallucination_rate", 0)
            total_citations   = state.get("total_citations", 0)
            valid_citations   = state.get("valid_citations", 0)
            hallucinated      = state.get("hallucinated_citations", 0)

        success = True

    except Exception as e:
        print(f"[Ablation] ERROR: {e}")
        review_text       = ""
        hallucination_rate = 1.0
        total_citations   = 0
        valid_citations   = 0
        hallucinated      = 0
        success           = False

    finally:
        # Restore original model
        settings.llm_model = original_model

    # Measure latency and memory
    latency_seconds = round(time.time() - start_time, 2)
    mem_after       = get_memory_usage_mb()
    mem_used_mb     = round(mem_after - mem_before, 1)

    # Estimate tokens
    word_count      = len(review_text.split())
    token_estimate  = int(word_count * 1.3)

    # Calculate cost
    cost_usd        = calculate_cost(model_config, review_text)

    # Tokens per second
    tokens_per_sec  = round(token_estimate / latency_seconds, 1) if latency_seconds > 0 else 0

    # Citation precision
    precision       = round(valid_citations / total_citations, 3) if total_citations > 0 else 0

    result = {
        # Experiment identity
        "timestamp":          datetime.now().isoformat(),
        "model":              model,
        "model_size_b":       model_config["size_b"],
        "model_type":         model_config["type"],
        "topic":              topic[:60],
        "topic_difficulty":   topic_config["difficulty"],
        "system":             system,
        "success":            success,

        # Performance metrics
        "hallucination_rate": hallucination_rate,
        "total_citations":    total_citations,
        "valid_citations":    valid_citations,
        "hallucinated":       hallucinated,
        "citation_precision": precision,

        # Efficiency metrics
        "latency_seconds":    latency_seconds,
        "token_estimate":     token_estimate,
        "tokens_per_second":  tokens_per_sec,
        "memory_used_mb":     mem_used_mb,
        "cost_usd":           cost_usd,
        "review_length_chars": len(review_text),
    }

    print(f"\n[Ablation] Results:")
    print(f"  Hallucination Rate : {hallucination_rate:.1%}")
    print(f"  Citation Precision : {precision:.1%}")
    print(f"  Latency            : {latency_seconds}s")
    print(f"  Tokens/sec         : {tokens_per_sec}")
    print(f"  Memory used        : {mem_used_mb} MB")
    print(f"  Cost               : ${cost_usd:.6f}")

    return result


# ---------------------------------------------------------------------------
# Run full ablation study
# ---------------------------------------------------------------------------
def run_ablation_study(
    systems: list[str] = ["experimental", "baseline"],
) -> list[dict]:
    """
    Run complete ablation study across all models and topics.

    Parameters
    ----------
    systems : list of systems to test

    Returns
    -------
    list[dict] : all experiment results
    """
    ABLATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")

    total = len(ABLATION_MODELS) * len(ABLATION_TOPICS) * len(systems)
    done  = 0

    print("\n" + "#" * 65)
    print("# ABLATION STUDY: SLM vs LLM Performance")
    print(f"# Models  : {len(ABLATION_MODELS)}")
    print(f"# Topics  : {len(ABLATION_TOPICS)}")
    print(f"# Systems : {len(systems)}")
    print(f"# Total experiments: {total}")
    print(f"# Estimated time   : {total * 3}-{total * 5} minutes")
    print("#" * 65)

    for system in systems:
        for model_config in ABLATION_MODELS:
            for topic_config in ABLATION_TOPICS:
                done += 1
                print(f"\n[Ablation] Experiment {done}/{total}")

                result = run_single_experiment(
                    model_config = model_config,
                    topic_config = topic_config,
                    system       = system,
                )
                all_results.append(result)

                # Save incrementally (so progress is not lost)
                _save_results(all_results, timestamp)

    # Print final summary
    _print_summary(all_results)

    return all_results


# ---------------------------------------------------------------------------
# Save results to CSV and JSON
# ---------------------------------------------------------------------------
def _save_results(results: list[dict], timestamp: str) -> None:
    """Save results to CSV and JSON."""
    csv_path  = ABLATION_OUTPUT_DIR / f"ablation_{timestamp}.csv"
    json_path = ABLATION_OUTPUT_DIR / f"ablation_{timestamp}.json"

    # CSV
    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[Ablation] Results saved → {csv_path.name}")


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
def _print_summary(results: list[dict]) -> None:
    """Print summary comparison table."""
    print("\n" + "=" * 90)
    print("ABLATION STUDY SUMMARY")
    print("=" * 90)
