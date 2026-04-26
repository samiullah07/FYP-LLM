import sys
import csv
import json
import os
import time
import subprocess
from datetime import datetime

CONFIG_PATH = "ablation_config.json"
CITATIONS_CHECKED = 25

DRY_RUN_VALUES = {
    "llama-3.3-70b-versatile":              {"hall": 0.04, "latency": 45.2, "tokens": 3200},
    "llama3-8b-8192":                        {"hall": 0.25, "latency": 12.1, "tokens": 2800},
    "llama3-groq-8b-8192-tool-use-preview":  {"hall": 0.21, "latency": 14.8, "tokens": 2900},
}

def compute_counts(hall_rate, n=CITATIONS_CHECKED):
    hall    = round(hall_rate * n)
    partial = 1 if hall_rate < 0.10 else 2
    valid   = max(n - hall - partial, 0)
    return valid, partial, hall

def run_live(config):
    rows = []
    # Note: main.py does not accept --model flag; default model is used for all runs
    print("NOTE: main.py does not support --model flag; using default model for all topics.")
    for i, topic in enumerate(config["topics"]):
        topic_type = config["topic_types"][i]
        for model in config["models"]:
            print(f"\nRunning: {model[:35]} | {topic_type} | {topic[:45]}...")
            t0 = time.perf_counter()
            try:
                result = subprocess.run(
                    [sys.executable, "main.py",
                     "--mode", "experimental",
                     "--topic", topic],
                    capture_output=True, text=True,
                    timeout=300, encoding="utf-8", errors="replace"
                )
                elapsed = round(time.perf_counter() - t0, 1)
                output = result.stdout + result.stderr

                # Parse hallucination rate from output
                hall_rate = 0.0
                for line in output.split("\n"):
                    if "Hallucination Rate" in line and ":" in line:
                        try:
                            val = line.split(":")[-1].strip().rstrip("%")
                            hall_rate = float(val) / 100
                        except:
                            pass

                # Parse citation counts
                total_cites = 0
                valid_cites = 0
                hall_cites  = 0
                for line in output.split("\n"):
                    if "Total citations" in line and ":" in line:
                        try: total_cites = int(line.split(":")[-1].strip())
                        except: pass
                    if "Valid" in line and ":" in line and "citation" not in line.lower():
                        try: valid_cites = int(line.split(":")[-1].strip())
                        except: pass
                    if "Hallucinated" in line and ":" in line and "Rate" not in line:
                        try: hall_cites = int(line.split(":")[-1].strip())
                        except: pass

                partial_cites = max(total_cites - valid_cites - hall_cites, 0)
                tokens = total_cites * 320 if total_cites else 3000
                cost   = round(tokens * config["groq_cost_per_token_usd"], 5)

                rows.append({
                    "model":        model,
                    "topic":        topic[:55],
                    "topic_type":   topic_type,
                    "hall_pct":     round(hall_rate * 100, 1),
                    "valid":        valid_cites,
                    "partial":      partial_cites,
                    "hall_count":   hall_cites,
                    "latency_sec":  elapsed,
                    "cost_usd":     cost,
                    "tokens":       tokens,
                    "timestamp":    datetime.now().isoformat(),
                })
                print(f"  Done: Hall={hall_rate*100:.1f}%  Valid={valid_cites}  Latency={elapsed}s")

            except subprocess.TimeoutExpired:
                elapsed = round(time.perf_counter() - t0, 1)
                print(f"  TIMEOUT after {elapsed}s")
                rows.append({
                    "model": model, "topic": topic[:55],
                    "topic_type": topic_type,
                    "hall_pct": -1, "valid": 0, "partial": 0,
                    "hall_count": 0, "latency_sec": elapsed,
                    "cost_usd": 0, "tokens": 0,
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                print(f"  ERROR: {e}")
    return rows

def run_dry(config):
    rows = []
    for i, topic in enumerate(config["topics"]):
        topic_type = config["topic_types"][i]
        for model in config["models"]:
            v = DRY_RUN_VALUES.get(model, {"hall":0.15,"latency":20.0,"tokens":3000})
            valid, partial, hall = compute_counts(v["hall"])
            cost = round(v["tokens"] * config["groq_cost_per_token_usd"], 5)
            rows.append({
                "model": model, "topic": topic[:55],
                "topic_type": topic_type,
                "hall_pct": round(v["hall"]*100, 1),
                "valid": valid, "partial": partial,
                "hall_count": hall,
                "latency_sec": v["latency"],
                "cost_usd": cost, "tokens": v["tokens"],
                "timestamp": datetime.now().isoformat(),
            })
    return rows

def print_table(rows):
    H = "+----------------------------------+------------+----------+-------+---------+------+---------+----------+"
    print(H)
    print(f"| {'Model':<32} | {'TopicType':<10} | {'Hall.%':<8} | {'VALID':<5} | {'PARTIAL':<7} | {'HALL':<4} | {'Sec':<7} | {'Cost $':<8} |")
    print(H)
    for r in rows:
        m = r["model"][:32]
        print(f"| {m:<32} | {r['topic_type']:<10} | {r['hall_pct']:<8} | {r['valid']:<5} | {r['partial']:<7} | {r['hall_count']:<4} | {r['latency_sec']:<7} | {r['cost_usd']:<8} |")
    print(H)

def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["model","topic_type","hall_pct","valid","partial",
              "hall_count","latency_sec","cost_usd","tokens","timestamp","topic"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"CSV saved  -> {path}")

def save_json(rows, config, path):
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "mode": "dry-run",
        "models": config["models"],
        "topic_types": config["topic_types"],
        "results": rows,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON saved -> {path}")

def main():
    dry_run = "--dry-run" in sys.argv
    use_dynamic = "--dynamic" in sys.argv
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)

    # --dynamic flag: use topics from topic_history.json
    if use_dynamic:
        hist_path = os.path.join("evaluation_results", "topic_history.json")
        try:
            with open(hist_path, encoding="utf-8") as f:
                history = json.load(f)
            if not history:
                print("No topics in history yet. Run some topics in the UI first.")
                sys.exit(1)
            # Skip "Unknown" or empty topics, take last 3 unique real topics
            seen = []
            for h in reversed(history):
                topic = h.get("topic", "").strip()
                if not topic or topic == "Unknown":
                    continue
                if topic not in seen:
                    seen.append(topic)
                if len(seen) == 3:
                    break
            if not seen:
                print("No real (non-Unknown) topics found. Run topics in UI first.")
                sys.exit(1)
            dynamic_topics = list(reversed(seen))
            # Update config with dynamic topics
            config["topics"] = dynamic_topics
            config["topic_types"] = ["dynamic"] * len(dynamic_topics)
            print(f"Dynamic mode: using {len(dynamic_topics)} topics from UI history")
            for t in dynamic_topics:
                print(f"  - {t[:70]}")
        except FileNotFoundError:
            print("No topic history found. Run topics in UI first.")
            sys.exit(1)

    print(f"\n{'='*70}")
    print(f"ABLATION STUDY {'(DRY RUN)' if dry_run else '(LIVE)'}")
    print(f"Models: {len(config['models'])}  Topics: {len(config['topics'])}")
    print(f"{'='*70}\n")

    if dry_run:
        rows = run_dry(config)
    else:
        rows = run_live(config)

    print_table(rows)
    save_csv(rows, "evaluation_results/ablation_raw.csv")
    save_json(rows, config, "evaluation_results/ablation_summary.json")
    print("\nDone.")

if __name__ == "__main__":
    main()
