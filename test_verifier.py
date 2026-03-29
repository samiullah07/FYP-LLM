# test_verifier.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.planner_agent import plan_topic
from agents.search_agent import retrieve_papers
from agents.summariser_agent import write_literature_review
from agents.verifier_agent import verify_review


def main():
    topic = (
        "Agentic AI for reliable academic literature review "
        "and hallucination mitigation in large language models"
    )

    print("=" * 60)
    print("STEP 1: Planner Agent")
    print("=" * 60)
    sub_queries = plan_topic(topic)

    print("\n" + "=" * 60)
    print("STEP 2: Search Agent")
    print("=" * 60)
    papers = retrieve_papers(sub_queries)
    print(f"  Papers retrieved: {len(papers)}")

    print("\n" + "=" * 60)
    print("STEP 3: Summariser Agent")
    print("=" * 60)
    review_text, papers_used = write_literature_review(papers, topic)
    print(f"\n--- GENERATED REVIEW ---\n{review_text}\n")

    print("=" * 60)
    print("STEP 4: Verifier Agent")
    print("=" * 60)
    verification = verify_review(review_text, papers)

    print("\n--- FINAL METRICS ---")
    print(f"  Total      : {verification['total']}")
    print(f"  Valid      : {verification['valid']}")
    print(f"  Partial    : {verification['partial']}")
    print(f"  Hallucinated: {verification['hallucinated']}")
    print(f"  Hallucination Rate: {verification['hallucination_rate']:.1%}")


if __name__ == "__main__":
    main()