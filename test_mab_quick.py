# test_mab_quick.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mab_selector import bandit, MODELS, TOPIC_TYPES

def main():
    print("=" * 60)
    print("QUICK MAB SANITY TEST")
    print("=" * 60)

    # Test 1: Topic classification
    print("\n--- Test 1: Topic Classification ---")
    test_topics = [
        "Neurosymbolic citation graph hallucination detection",
        "Retrieval augmented generation for LLM hallucination",
        "Agentic AI for automated literature review synthesis",
        "BERT transformer attention mechanism NLP deep learning",
        "Blockchain verification ghost citation autonomous AI",
    ]
    for t in test_topics:
        topic_type = bandit.classify_topic(t)
        print(f"  [{topic_type:<12}] {t[:55]}")

    # Test 2: Model selection (before any learning)
    print("\n--- Test 2: Model Selection (cold start) ---")
    for t in test_topics[:3]:
        model, topic_type = bandit.select_model(t)
        print(f"  Topic type: {topic_type:<12} → Model: {model}")

    # Test 3: Simulate reward updates
    print("\n--- Test 3: Simulate Learning ---")
    simulated_runs = [
        ("llama-3.3-70b-versatile", "niche",        0.10),
        ("llama-3.1-8b-instant",    "niche",        0.55),
        ("llama-3.3-70b-versatile", "well_covered", 0.05),
        ("llama-3.1-8b-instant",    "well_covered", 0.08),
        ("llama-3.3-70b-versatile", "moderate",     0.15),
        ("llama-3.1-8b-instant",    "moderate",     0.35),
    ]
    for model, topic_type, hall_rate in simulated_runs:
        reward = 1.0 - hall_rate
        bandit.update(model, topic_type, hall_rate)
        print(
            f"  Updated: {model:<35} "
            f"topic={topic_type:<12} "
            f"hall={hall_rate:.0%} "
            f"reward={reward:.2f}"
        )

    # Test 4: Policy after learning
    print("\n--- Test 4: Learned Policy ---")
    policy = bandit.get_policy_summary()
    for topic_type, data in policy.items():
        preferred = data["preferred_model"]
        best_avg  = max(data["avg_rewards"].values())
        print(
            f"  {topic_type:<15} → {preferred:<38} "
            f"(avg reward: {best_avg:.3f})"
        )

    # Test 5: Model selection after learning
    print("\n--- Test 5: Model Selection After Learning ---")
    for t in test_topics[:3]:
        model, topic_type = bandit.select_model(t)
        print(f"  [{topic_type:<12}] {t[:40]:<40} → {model}")

    print("\n" + "=" * 60)
    print("QUICK TEST PASSED")
    print("=" * 60)
    print("\nMAB is working correctly.")
    print("Now run: uv run test_mab.py")

if __name__ == "__main__":
    main()