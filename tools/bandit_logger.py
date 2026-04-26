import csv
import json
import math
import os
from datetime import datetime


class BanditLogger:
    """Logs and summarises UCB1 multi-armed bandit model selection pulls."""

    def __init__(self):
        self.log = []  # list of pull dicts
        os.makedirs("bandit_log", exist_ok=True)

    def log_pull(
        self,
        pull_num: int,
        topic_type: str,
        selected_model: str,
        hallucination_rate: float,
        reward: float,
    ) -> None:
        """Record one bandit pull and persist state to bandit_log/ucb1_state.json."""

        N = pull_num  # total pulls so far (1-based)

        # Per-model pull count and mean reward BEFORE appending this entry
        def pulls_for(model):
            return sum(1 for e in self.log if e["model"] == model and e["topic"] == topic_type)

        def mean_for(model):
            entries = [e for e in self.log if e["model"] == model and e["topic"] == topic_type]
            return sum(e["reward"] for e in entries) / len(entries) if entries else 0.0

        n_i = pulls_for(selected_model)  # pulls BEFORE this one

        # UCB1 score calculation with updated formula
        if n_i == 0 or N < 2:
            ucb1_score = float("inf")
        else:
            ucb1_score = mean_for(selected_model) + math.sqrt(2 * math.log(N) / n_i)

        # Update mean rewards
        seen_models = list({e["model"] for e in self.log if e["topic"] == topic_type})
        mean_rewards = {m: mean_for(m) for m in seen_models}
        mean_rewards[selected_model] = mean_for(selected_model)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "pull_num": N,
            "topic": topic_type,
            "model": selected_model,
            "hallucination_rate": hallucination_rate,
            "reward": reward,
            "n_i_before_pull": n_i,
            "N_total": N,
            "ucb1_score": round(ucb1_score, 4) if ucb1_score != float("inf") else "inf",
            "mean_rewards_snapshot": mean_rewards,
        }
        self.log.append(entry)

        # Persist full log
        state_path = os.path.join("bandit_log", "ucb1_state.json")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(self.log, f, indent=2, default=str)

    def generate_summary(self) -> None:
        """Print convergence table and save CSV with fixed UCB1 formula."""
        if not self.log:
            print("No pulls logged yet.")
            return

        topics = sorted({e["topic"] for e in self.log})
        all_models = sorted({e["model"] for e in self.log})

        rows = []
        for topic in topics:
            topic_entries = [e for e in self.log if e["topic"] == topic]
            N = len(self.log)  # total pulls across everything

            best_mean = -1.0
            best_model = None
            for model in all_models:
                model_entries = [e for e in topic_entries if e["model"] == model]
                n_i = len(model_entries)
                if n_i == 0:
                    continue
                mean_r = sum(e["reward"] for e in model_entries) / n_i
                ucb1 = mean_r + (math.sqrt(2 * math.log(N) / n_i) if N >= 2 and n_i >= 1 else float("inf"))
                # Determine best model by mean reward
                if mean_r > best_mean:
                    best_mean = mean_r
                    best_model = model

                rows.append({
                    "topic": topic,
                    "model": model,
                    "pulls": n_i,
                    "mean_reward": round(mean_r, 4),
                    "ucb1_score": ucb1 if ucb1 != float("inf") else "inf",
                    "converged_to": "YES" if model == best_model else ""
                })

        # Console table
        header = f"{'Topic':<16} {'Model':<40} {'Pulls':<7} {'Mean Reward':<14} {'UCB1 Score':<13} {'Best?'}"
        print("\n" + "=" * len(header))
        print("UCB1 BANDIT CONVERGENCE SUMMARY")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r['topic']:<16} {r['model']:<40} {r['pulls']:<7} "
                f"{r['mean_reward']:<14.4f} {r['ucb1_score']:<13} {r['converged_to']}"
            )
        print("=" * len(header))

        # CSV
        os.makedirs("bandit_log", exist_ok=True)
        csv_path = os.path.join("bandit_log", "convergence_summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary saved to {csv_path}")


if __name__ == "__main__":
    print("=== BanditLogger Test (9 pulls: 3 models x 3 topics) ===\n")
    logger = BanditLogger()
    test_data = [
        (1, "niche",        "llama-3.3-70b-versatile",              0.04, 0.96),
        (2, "niche",        "llama3-8b-8192",                       0.25, 0.75),
        (3, "niche",        "llama3-groq-8b-8192-tool-use-preview", 0.21, 0.79),
        (4, "moderate",     "llama-3.3-70b-versatile",              0.05, 0.95),
        (5, "moderate",     "llama3-8b-8192",                       0.28, 0.72),
        (6, "moderate",     "llama3-groq-8b-8192-tool-use-preview", 0.22, 0.78),
        (7, "well-covered", "llama-3.3-70b-versatile",              0.14, 0.86),
        (8, "well-covered", "llama3-8b-8192",                       0.21, 0.79),
        (9, "well-covered", "llama3-groq-8b-8192-tool-use-preview", 0.28, 0.72),
    ]
    for pull_num, topic, model, h_rate, reward in test_data:
        logger.log_pull(pull_num, topic, model, h_rate, reward)

    logger.generate_summary()