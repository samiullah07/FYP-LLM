"""
Run the full side-by-side evaluation (Experimental vs Baseline).

Usage:
    uv run test_evaluation.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluator import run_evaluation

TOPICS = [
    "Agentic AI for reliable academic literature review and hallucination mitigation",
    "Retrieval-augmented generation for reducing LLM hallucinations",
    "Multi-agent systems for automated scientific paper summarisation",
]

if __name__ == "__main__":
    results = run_evaluation(topics=TOPICS, save=True)
    print(f"\nDone. {len(results)} topics evaluated.")
    print("Reload the Evaluation page in Streamlit to see results.")