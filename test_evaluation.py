# test_evaluation.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluator import run_evaluation

# 5 topics covering different difficulty levels
# Topics 1-2: designed to cause baseline hallucinations (niche)
# Topics 3-4: moderate coverage
# Topic 5:    well-covered (RAG) — should show low hallucination

topics = [
    # NICHE — LLM will invent citations
    "Autonomous self-correcting AI agents for peer review "
    "validation and academic citation fraud detection",

    # NICHE — few real papers on exact topic
    "Ethical implications of large language models in "
    "automated academic peer review and citation generation",

    # MODERATE — some hallucination expected
    "Multi-agent orchestration frameworks for knowledge-intensive "
    "reasoning and automated literature synthesis",

    # WELL-COVERED — hallucination should be lower
    "Agentic AI for reliable academic literature review "
    "and hallucination mitigation in large language models",

    # WELL-COVERED — richest body of literature
    "Retrieval augmented generation for hallucination "
    "reduction in large language model outputs",
]

if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING FULL 5-TOPIC EVALUATION")
    print("This will take approximately 15-20 minutes")
    print("=" * 65)

    results = run_evaluation(topics)

    print(f"\nTotal rows saved: {len(results)}")
    print("Check data/eval/ for saved results and reviews")