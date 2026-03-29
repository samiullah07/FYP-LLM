# test_assembler.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.workflow_graph import run_workflow


def main():
    topic = (
        "Agentic AI for reliable academic literature review "
        "and hallucination mitigation in large language models"
    )

    state = run_workflow(topic)

    print("\n" + "=" * 60)
    print("DRAFT REVIEW (before assembler):")
    print("=" * 60)
    print(state["draft_review"])

    print("\n" + "=" * 60)
    print("FINAL REVIEW (after assembler):")
    print("=" * 60)
    print(state["final_review"])

    print("\n" + "=" * 60)
    print("CHANGE LOG:")
    print("=" * 60)
    for k, v in state["changes"].items():
        print(f"  {k:<35} : {v}")

    print("\nVerified refs kept:")
    for r in state["verified_refs"]:
        print(f"  ✓ {r}")

    print("\nHallucinated refs removed:")
    for r in state["hallucinated_refs"]:
        print(f"  ✗ {r}")


if __name__ == "__main__":
    main()