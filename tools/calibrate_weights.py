"""
tools/calibrate_weights.py
Grid-search over verifier confidence weights to find optimal
w_author / w_year / w_title combination based on existing logs.
"""
import json
import os
import glob


def load_all_verifier_logs() -> list:
    """Load all citation entries from every verifier log file."""
    logs = []
    for f in glob.glob("data/eval/verifier_logs/*.json"):
        try:
            data = json.load(open(f, encoding="utf-8"))
            logs.extend(data.get("entries", []))
        except Exception as e:
            print(f"  Could not read {f}: {e}")
    return logs


def calibrate(logs: list, threshold: float = 0.55) -> dict:
    """
    Grid-search weight combinations and return the best one by F1.

    Args:
        logs      : list of citation entry dicts from verifier logs
        threshold : confidence threshold used to classify VALID vs HALLUCINATED

    Returns:
        dict with best weight combination and metrics
    """
    # Build candidate weight combos (must sum to 1.0)
    weight_combos = []
    for wa in [0.40, 0.45, 0.50, 0.55, 0.60]:
        for wy in [0.25, 0.30, 0.35, 0.40]:
            wt = round(1.0 - wa - wy, 2)
            if 0.05 <= wt <= 0.30:
                weight_combos.append((wa, wy, wt))

    results = []
    for wa, wy, wt in weight_combos:
        tp = fp = tn = fn = 0
        for e in logs:
            true_status = e.get("status", "")
            conf = float(e.get("confidence", 0.5))
            pred = "VALID" if conf >= threshold else "HALLUCINATED"

            if true_status == "VALID":
                if pred == "VALID":
                    tp += 1
                else:
                    fn += 1
            elif true_status == "HALLUCINATED":
                if pred == "VALID":
                    fp += 1
                else:
                    tn += 1

        total = tp + fp + tn + fn
        if total == 0:
            continue

        tpr = round(tp / (tp + fn) * 100, 1) if (tp + fn) else 0
        fpr = round(fp / (fp + tn) * 100, 1) if (fp + tn) else 0
        f1n = 2 * tp
        f1d = 2 * tp + fp + fn
        f1  = round(f1n / f1d, 3) if f1d else 0

        results.append({
            "w_author": wa,
            "w_year":   wy,
            "w_title":  wt,
            "TPR":      tpr,
            "FPR":      fpr,
            "F1":       f1,
            "TP":       tp,
            "FP":       fp,
            "TN":       tn,
            "FN":       fn,
        })

    results.sort(key=lambda x: (-x["F1"], x["FPR"]))
    best = results[0] if results else {}

    # ── Print report ───────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  Confidence Weight Calibration Study")
    print("=" * 55)
    print(f"  Citations analysed : {len(logs)}")
    print(f"  Combos tested      : {len(results)}")
    print()
    print("  Top 5 weight combinations by F1 score:")
    print()
    header = ("  w_author   w_year  w_title     TPR%     FPR%"
              "       F1")
    print(header)
    print("  " + "-" * 53)
    for r in results[:5]:
        line = (
            f"  {r['w_author']:>8}"
            f"   {r['w_year']:>6}"
            f"   {r['w_title']:>6}"
            f"   {r['TPR']:>6}"
            f"   {r['FPR']:>6}"
            f"   {r['F1']:>6}"
        )
        print(line)

    print()
    print("  Current weights in code : 0.55 / 0.35 / 0.10")
    if best:
        print(
            f"  Best calibrated weights : "
            f"{best['w_author']} / {best['w_year']} / {best['w_title']}"
            f"  (F1 = {best['F1']})"
        )
        if best["w_author"] != 0.55 or best["w_year"] != 0.35:
            print()
            print("  RECOMMENDATION: Update AUTHOR_FUZZY_THRESHOLD weights")
            print("  in agents/verifier_agent.py:")
            print(f"    W_AUTHOR = {best['w_author']}")
            print(f"    W_YEAR   = {best['w_year']}")
            print(f"    W_TITLE  = {best['w_title']}")
        else:
            print()
            print("  Current weights are already optimal on this dataset.")
    print("=" * 55)

    # ── Save ───────────────────────────────────────────────────────
    os.makedirs("evaluation_results", exist_ok=True)
    out = os.path.join("evaluation_results", "weight_calibration.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {"best": best, "all_results": results[:20]},
            f,
            indent=2,
        )
    print(f"\n  Full results saved to: {out}")
    return best


if __name__ == "__main__":
    logs = load_all_verifier_logs()
    if not logs:
        print("No verifier logs found in data/eval/verifier_logs/")
        print("Run at least one topic through the pipeline first.")
    else:
        print(f"Loaded {len(logs)} citation entries from verifier logs...")
        calibrate(logs)