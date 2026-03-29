# evaluation/baseline.py
"""
Baseline evaluation helper.

Simple wrapper used to run only the baseline system
and return its metrics for standalone testing.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.baseline_graph import run_baseline


def evaluate_baseline(topic: str) -> dict:
    """
    Run the baseline system and return metrics.

    Parameters
    ----------
    topic : str
        Research topic string.

    Returns
    -------
    dict
        Metrics from the baseline run.
    """
    state = run_baseline(topic)

    return {
        "system":               "baseline",
        "topic":                topic,
        "papers_retrieved":     len(state["papers"]),
        "total_citations":      state["total_citations"],
        "valid_citations":      state["valid_citations"],
        "partial_citations":    state["partial_citations"],
        "hallucinated":         state["hallucinated_citations"],
        "hallucination_rate":   state["hallucination_rate"],
        "review_text":          state["review_text"],
    }