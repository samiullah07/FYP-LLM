# src/mab_selector.py
"""
Multi-Armed Bandit Model Selector.

Uses Upper Confidence Bound (UCB1) algorithm to learn
which LLM model performs best for different topic types.

State  : topic context features
Action : LLM model choice
Reward : 1 - hallucination_rate (higher is better)
"""

import json
import math
import numpy as np
from pathlib import Path
from datetime import datetime

from src.config import settings

# Available models (arms)
# ONLY currently active Groq models (April 2026)
MODELS = [
    {
        "name":               "llama-3.3-70b-versatile",
        "size_b":             70,
        "cost_per_1m_input":  0.59,
        "cost_per_1m_output": 0.79,
        "description":        "LLM — best quality, lowest hallucination",
        "tier":               "LLM",
    },
    {
        "name":               "meta-llama/llama-4-maverick-17b-128e-instruct",
        "size_b":             17,
        "cost_per_1m_input":  0.20,
        "cost_per_1m_output": 0.60,
        "description":        "SLM — high quality, fast",
        "tier":               "SLM",
    },
    {
        "name":               "meta-llama/llama-4-scout-17b-16e-instruct",
        "size_b":             17,
        "cost_per_1m_input":  0.11,
        "cost_per_1m_output": 0.34,
        "description":        "SLM — balanced speed and quality",
        "tier":               "SLM",
    },
    {
        "name":               "llama-3.1-8b-instant",
        "size_b":             8,
        "cost_per_1m_input":  0.05,
        "cost_per_1m_output": 0.08,
        "description":        "SLM — fastest, cheapest",
        "tier":               "SLM",
    },
]
# Topic context categories (states)
TOPIC_TYPES = [
    "niche",        # very specific, few papers
    "moderate",     # some coverage
    "well_covered", # rich literature
]

MAB_LOG_PATH = settings.data_dir / "eval" / "mab_log.json"


class UCB1BanditSelector:
    """
    UCB1 Multi-Armed Bandit for LLM model selection.

    UCB1 balances:
        - Exploitation: use the model that has worked best so far
        - Exploration:  try less-used models to gather more data

    Formula: UCB1 score = mean_reward + sqrt(2 * ln(N) / n_i)
        where N = total pulls, n_i = pulls for arm i
    """

    def __init__(self):
        # counts[topic_type][model] = number of times selected
        self.counts  = {
            t: {m: 0 for m in MODELS}
            for t in TOPIC_TYPES
        }
        # values[topic_type][model] = average reward
        self.values  = {
            t: {m: 0.0 for m in MODELS}
            for t in TOPIC_TYPES
        }
        # history for logging
        self.history = []

        # Load existing data if available
        self._load()

    def classify_topic(self, topic: str) -> str:
        """
        Classify a topic string into a context state.

        Simple heuristic: check for niche vs well-covered keywords.
        You can make this more sophisticated with an LLM call later.

        Parameters
        ----------
        topic : str

        Returns
        -------
        str : one of "niche", "moderate", "well_covered"
        """
        topic_lower = topic.lower()

        # Well-covered topics — rich academic literature
        well_covered_keywords = [
            "transformer", "bert", "gpt", "attention mechanism",
            "retrieval augmented", "rag", "deep learning",
            "neural network", "natural language processing",
            "hallucination detection", "large language model survey",
        ]

        # Niche topics — few real papers, high hallucination risk
        niche_keywords = [
            "neurosymbolic", "citation fraud", "ghost citation",
            "blockchain verification", "legal liability ai",
            "peer review automation ethics", "autonomous citation",
            "fabricated doi", "citation graph completion",
        ]

        well_score  = sum(1 for kw in well_covered_keywords if kw in topic_lower)
        niche_score = sum(1 for kw in niche_keywords       if kw in topic_lower)

        if niche_score >= 2 or (niche_score >= 1 and well_score == 0):
            return "niche"
        elif well_score >= 2:
            return "well_covered"
        else:
            return "moderate"

    def select_model(self, topic: str) -> tuple[str, str]:
        """
        Select the best model for a given topic using UCB1.

        Parameters
        ----------
        topic : str

        Returns
        -------
        tuple[str, str] : (selected_model, topic_type)
        """
        topic_type   = self.classify_topic(topic)
        total_counts = sum(self.counts[topic_type].values())

        # If any model has never been tried, try it first (exploration)
        for model in MODELS:
            if self.counts[topic_type][model] == 0:
                print(
                    f"[MAB] Exploring untried model: {model} "
                    f"for topic type: {topic_type}"
                )
                return model, topic_type

        # UCB1 score for each model
        ucb_scores = {}
        for model in MODELS:
            n_i          = self.counts[topic_type][model]
            mean_reward  = self.values[topic_type][model]
            exploration  = math.sqrt(2 * math.log(total_counts) / n_i)
            ucb_scores[model] = mean_reward + exploration

        selected = max(ucb_scores, key=lambda m: ucb_scores[m])

        print(f"[MAB] Topic type  : {topic_type}")
        print(f"[MAB] UCB1 scores :")
        for m, s in ucb_scores.items():
            marker = " ← selected" if m == selected else ""
            print(f"  {m:<35} : {s:.4f}{marker}")

        return selected, topic_type

    def update(
        self,
        model:             str,
        topic_type:        str,
        hallucination_rate: float,
    ) -> None:
        """
        Update bandit after observing reward.

        Reward = 1 - hallucination_rate
        (lower hallucination = higher reward)

        Parameters
        ----------
        model              : str   — model that was used
        topic_type         : str   — topic context category
        hallucination_rate : float — observed hallucination rate (0.0-1.0)
        """
        reward = 1.0 - hallucination_rate

        n = self.counts[topic_type][model]
        current_value = self.values[topic_type][model]

        # Incremental mean update: Q(n+1) = Q(n) + (reward - Q(n)) / (n+1)
        self.counts[topic_type][model] += 1
        self.values[topic_type][model]  = (
            current_value + (reward - current_value) / (n + 1)
        )

        # Log entry
        entry = {
            "timestamp":         datetime.now().isoformat(),
            "model":             model,
            "topic_type":        topic_type,
            "hallucination_rate": hallucination_rate,
            "reward":            reward,
            "new_avg_reward":    self.values[topic_type][model],
            "total_pulls":       self.counts[topic_type][model],
        }
        self.history.append(entry)

        print(
            f"[MAB] Updated {model} ({topic_type}): "
            f"reward={reward:.3f}, "
            f"avg={self.values[topic_type][model]:.3f}, "
            f"pulls={self.counts[topic_type][model]}"
        )

        self._save()

    def get_policy_summary(self) -> dict:
        """
        Return current learned policy summary.

        Shows which model is currently preferred for each topic type.
        """
        summary = {}
        for topic_type in TOPIC_TYPES:
            best_model = max(
                MODELS,
                key=lambda m: self.values[topic_type][m],
            )
            summary[topic_type] = {
                "preferred_model":  best_model,
                "avg_rewards":      dict(self.values[topic_type]),
                "pull_counts":      dict(self.counts[topic_type]),
            }
        return summary

    def _save(self) -> None:
        """Save bandit state to JSON file."""
        MAB_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "counts":  self.counts,
            "values":  self.values,
            "history": self.history[-100:],  # keep last 100 entries
        }
        with open(MAB_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def _load(self) -> None:
        """Load bandit state from JSON file if it exists."""
        if MAB_LOG_PATH.exists():
            with open(MAB_LOG_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.counts  = state.get("counts",  self.counts)
            self.values  = state.get("values",  self.values)
            self.history = state.get("history", [])
            print(
                f"[MAB] Loaded existing bandit state "
                f"({len(self.history)} history entries)"
            )


# Singleton instance
bandit = UCB1BanditSelector()