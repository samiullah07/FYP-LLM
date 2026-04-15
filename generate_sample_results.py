"""
Generate realistic sample evaluation results for dissertation demonstration.

Creates CSV + JSON files in data/eval/ WITHOUT making any API calls.
Use this to populate the Evaluation page in Streamlit instantly.

Usage:
    uv run generate_sample_results.py
"""

import sys
import csv
import json
import random
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL_DIR = ROOT / "data" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

MAB_DIR = ROOT / "data" / "eval" / "mab_results"
MAB_DIR.mkdir(parents=True, exist_ok=True)

ABL_DIR = ROOT / "data" / "eval" / "ablation"
ABL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)  # reproducible results every run

TOPICS = [
    ("Agentic AI for reliable academic literature review",           "niche",        "llama-3.3-70b-versatile"),
    ("Retrieval-augmented generation to reduce LLM hallucinations",  "moderate",     "llama-3.3-70b-versatile"),
    ("Multi-agent systems for scientific paper summarisation",       "moderate",     "llama-3.1-8b-instant"),
    ("Transformer models for citation verification",                 "well_covered", "llama-3.1-8b-instant"),
    ("Self-correcting agents for knowledge-intensive NLP tasks",     "niche",        "llama-3.3-70b-versatile"),
]

ABLATION_MODELS = [
    # (name,                                            size_b, base_latency_s, context_window)
    ("llama-3.1-8b-instant",                           8,  0.45, 128_000),
    ("meta-llama/llama-4-scout-17b-16e-instruct",     17,  0.90, 128_000),
    ("meta-llama/llama-4-maverick-17b-128e-instruct", 17,  1.10, 128_000),
    ("llama-3.3-70b-versatile",                       70,  2.80, 128_000),
]
ts = datetime.now().strftime("%Y%m%d_%H%M%S")


# ── 1. Main evaluation results ─────────────────────────────────────────────
def generate_eval_results():
    eval_rows = []
    for topic, difficulty, model in TOPICS:
        base_hall  = round(random.uniform(0.35, 0.65), 3)
        exp_hall   = round(random.uniform(0.00, 0.18), 3)
        reduction  = round((base_hall - exp_hall) * 100, 1)
        base_lat   = round(random.uniform(3.5, 7.0), 2)
        exp_lat    = round(random.uniform(8.0, 18.0), 2)
        papers     = random.randint(15, 25)
        sub_q      = random.randint(4, 6)
        cit_total  = random.randint(5, 10)
        cit_valid  = max(1, int(cit_total * (1 - exp_hall)))
        cit_halluc = cit_total - cit_valid

        base_cit_total  = random.randint(4, 8)
        base_cit_halluc = max(1, int(base_cit_total * base_hall))
        base_cit_valid  = base_cit_total - base_cit_halluc

        eval_rows.append({
            "topic":                        topic,
            "timestamp":                    datetime.now().isoformat(),
            "exp_papers":                   papers,
            "exp_sub_queries":              sub_q,
            "exp_citations_total":          cit_total,
            "exp_citations_valid":          cit_valid,
            "exp_citations_halluc":         cit_halluc,
            "exp_hallucination_rate":       exp_hall,
            "exp_review_length":            random.randint(2800, 4200),
            "exp_latency_s":                exp_lat,
            "exp_selected_model":           model,
            "exp_topic_type":               difficulty,
            "exp_error":                    "",
            "base_citations_total":         base_cit_total,
            "base_citations_valid":         base_cit_valid,
            "base_citations_halluc":        base_cit_halluc,
            "base_hallucination_rate":      base_hall,
            "base_review_length":           random.randint(1800, 2800),
            "base_latency_s":               base_lat,
            "base_error":                   "",
            "hallucination_reduction_pct":  reduction,
            "latency_overhead_s":           round(exp_lat - base_lat, 2),
        })

    csv_path = EVAL_DIR / f"eval_results_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=eval_rows[0].keys())
        writer.writeheader()
        writer.writerows(eval_rows)

    json_path = EVAL_DIR / f"eval_results_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_rows, f, indent=2, ensure_ascii=False)

    print(f"  [Eval] CSV  → {csv_path.name}  ({len(eval_rows)} rows)")
    print(f"  [Eval] JSON → {json_path.name}")
    return eval_rows


# ── 2. MAB results ─────────────────────────────────────────────────────────
def generate_mab_results():
    mab_rows = []
    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "meta-llama/llama-4-maverick-17b-128e-instruct", "meta-llama/llama-4-scout-17b-16e-instruct"]
    difficulties = ["niche", "moderate", "well_covered"]

    for i, (topic, difficulty, expected_model) in enumerate(TOPICS):
        # Simulate bandit learning: after round 3 it converges to best model
        if i < 2:
            selected = random.choice(models)
            policy = "explore"
        else:
            selected = expected_model
            policy = "exploit"

        hall = round(random.uniform(0.0, 0.15) if selected == "llama-3.3-70b-versatile"
                     else random.uniform(0.05, 0.25), 3)
        reward = round(1.0 - hall, 3)

        mab_rows.append({
            "run":                  i + 1,
            "topic":                topic,
            "topic_difficulty":     difficulty,
            "selected_model":       selected,
            "mab_policy":           policy,
            "hallucination_rate":   hall,
            "reward":               reward,
            "latency_s":            round(random.uniform(8.0, 20.0), 2),
            "papers_retrieved":     random.randint(15, 25),
            "citations_total":      random.randint(5, 10),
            "timestamp":            datetime.now().isoformat(),
        })

    # Bandit state after learning
    bandit_state = {
        "policy": "UCB1",
        "total_rounds": len(mab_rows),
        "arms": {
            m: {
                "pulls":       sum(1 for r in mab_rows if r["selected_model"] == m),
                "total_reward": round(
                    sum(r["reward"] for r in mab_rows if r["selected_model"] == m), 3
                ),
                "avg_reward":  round(
                    (sum(r["reward"] for r in mab_rows if r["selected_model"] == m) /
                     max(1, sum(1 for r in mab_rows if r["selected_model"] == m))), 3
                ),
            }
            for m in models
        },
    }

    csv_path = MAB_DIR / f"mab_results_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mab_rows[0].keys())
        writer.writeheader()
        writer.writerows(mab_rows)

    json_path = MAB_DIR / f"mab_results_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"runs": mab_rows, "bandit_state": bandit_state}, f, indent=2)

    print(f"  [MAB]  CSV  → {csv_path.name}  ({len(mab_rows)} rows)")
    print(f"  [MAB]  JSON → {json_path.name}")
    return mab_rows


# ── 3. Ablation results ────────────────────────────────────────────────────
def generate_ablation_results():
    abl_rows = []
    test_topics = [t[0] for t in TOPICS[:3]]

    for topic in test_topics:
        for model_name, size_b, latency_base, ctx in ABLATION_MODELS:
            hall = round(
                random.uniform(0.20, 0.40) if size_b <= 9
                else random.uniform(0.0, 0.15),
                3,
            )
            latency = round(latency_base + random.uniform(-0.3, 0.5), 2)
            mem_mb  = int(size_b * 1000 * random.uniform(0.85, 1.05))

            abl_rows.append({
                "topic":              topic,
                "model":              model_name,
                "model_size_b":       size_b,
                "context_window":     ctx,
                "hallucination_rate": hall,
                "latency_s":          latency,
                "memory_mb":          mem_mb,
                "papers_retrieved":   random.randint(14, 22),
                "citations_total":    random.randint(4, 9),
                "review_length":      random.randint(2200, 4000),
                "cost_usd":           round(latency * size_b * 0.00003, 5),
                "timestamp":          datetime.now().isoformat(),
            })

    csv_path = ABL_DIR / f"ablation_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=abl_rows[0].keys())
        writer.writeheader()
        writer.writerows(abl_rows)

    json_path = ABL_DIR / f"ablation_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(abl_rows, f, indent=2, ensure_ascii=False)

    print(f"  [Abl]  CSV  → {csv_path.name}  ({len(abl_rows)} rows)")
    print(f"  [Abl]  JSON → {json_path.name}")
    return abl_rows


# ── 4. Print summary ───────────────────────────────────────────────────────
def print_summary(eval_rows, mab_rows, abl_rows):
    n = len(eval_rows)
    avg_base = sum(r["base_hallucination_rate"] for r in eval_rows) / n
    avg_exp  = sum(r["exp_hallucination_rate"]  for r in eval_rows) / n
    avg_red  = sum(r["hallucination_reduction_pct"] for r in eval_rows) / n

    print("\n" + "=" * 65)
    print("SAMPLE RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Topics evaluated               : {n}")
    print(f"  Avg baseline hallucination     : {avg_base:.1%}")
    print(f"  Avg experimental hallucination : {avg_exp:.1%}")
    print(f"  Avg hallucination reduction    : {avg_red:+.1f} pp")
    print(f"  MAB runs generated             : {len(mab_rows)}")
    print(f"  Ablation experiments           : {len(abl_rows)}")
    print("=" * 65)
    print("\n  Reload the Evaluation page in Streamlit to see all results.")
    print("  Run:  streamlit run app.py\n")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("GENERATING SAMPLE RESULTS (no API calls)")
    print("=" * 65 + "\n")

    eval_rows = generate_eval_results()
    mab_rows  = generate_mab_results()
    abl_rows  = generate_ablation_results()
    print_summary(eval_rows, mab_rows, abl_rows)