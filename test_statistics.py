# test_statistics.py
"""
Run after test_evaluation.py to compute all statistical metrics.
"""
import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import (
    two_proportion_z_test,
    wilson_confidence_interval,
    citation_precision,
    error_typology,
)
from src.config import settings


def load_latest_results() -> list[dict]:
    eval_dir  = settings.data_dir / "eval"
    csv_files = sorted(eval_dir.glob("results_*.csv"), reverse=True)
    if not csv_files:
        print("No results CSV found. Run test_evaluation.py first.")
        return []
    latest = csv_files[0]
    print(f"Loading: {latest.name}")
    rows = []
    with open(latest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    rows = load_latest_results()
    if not rows:
        return

    # Separate baseline and experimental
    base_rows = [r for r in rows if r["system"] == "baseline"]
    exp_rows  = [r for r in rows if r["system"] == "experimental"]

    # Aggregate totals
    b_total = sum(int(r["total_citations"]) for r in base_rows)
    b_hall  = sum(int(r["hallucinated"])    for r in base_rows)
    e_total = sum(int(r["total_citations"]) for r in exp_rows)
    e_hall  = sum(int(r["hallucinated"])    for r in exp_rows)

    b_rate  = b_hall / b_total if b_total > 0 else 0
    e_rate  = e_hall / e_total if e_total > 0 else 0

    print("\n" + "=" * 65)
    print("FULL STATISTICAL ANALYSIS REPORT")
    print("=" * 65)

    # --- Aggregate summary ---
    print(f"\n{'System':<20} {'Topics':>6} {'Citations':>10} "
          f"{'Hallucinated':>13} {'Rate':>8}")
    print("-" * 62)
    print(f"{'Baseline':<20} {len(base_rows):>6} {b_total:>10} "
          f"{b_hall:>13} {b_rate:>7.1%}")
    print(f"{'Experimental':<20} {len(exp_rows):>6} {e_total:>10} "
          f"{e_hall:>13} {e_rate:>7.1%}")
    print(f"{'Improvement':<20} {'':>6} {'':>10} "
          f"{'':>13} {b_rate - e_rate:>+7.1%}")

    # --- Per topic breakdown ---
    print(f"\n--- Per-Topic Breakdown ---")
    print(f"\n{'Topic':<45} {'Base':>7} {'Exp':>7} {'Diff':>8}")
    print("-" * 72)
    topics = list({r["topic"] for r in rows})
    for topic in topics:
        br = [r for r in base_rows if r["topic"] == topic]
        er = [r for r in exp_rows  if r["topic"] == topic]
        if br and er:
            bt = int(br[0]["total_citations"])
            bh = int(br[0]["hallucinated"])
            et = int(er[0]["total_citations"])
            eh = int(er[0]["hallucinated"])
            b  = bh/bt if bt > 0 else 0
            e  = eh/et if et > 0 else 0
            print(
                f"{topic[:45]:<45} "
                f"{b:>6.1%} "
                f"{e:>6.1%} "
                f"{b-e:>+7.1%}"
            )

    # --- Two-proportion z-test ---
    print("\n--- Two-Proportion Z-Test ---")
    print("H₀: p_baseline == p_experimental")
    print("H₁: p_baseline  > p_experimental (experimental better)")

    stat = two_proportion_z_test(
        n1=b_total, h1=b_hall,
        n2=e_total, h2=e_hall,
    )
    print(f"\n  z-score      : {stat['z_score']}")
    print(f"  p-value      : {stat['p_value']}")
    print(f"  Significant  : {stat['significant']} (α = 0.05)")
    print(f"  Conclusion   : {stat['interpretation']}")

    # --- Wilson confidence intervals ---
    print("\n--- 95% Confidence Intervals (Wilson Score) ---")
    b_ci = wilson_confidence_interval(b_hall, b_total)
    e_ci = wilson_confidence_interval(e_hall, e_total)
    print(f"  Baseline     : {b_rate:.1%}  95% CI {b_ci['interval']}")
    print(f"  Experimental : {e_rate:.1%}  95% CI {e_ci['interval']}")

    # --- Citation precision ---
    print("\n--- Citation Precision ---")
    b_prec = citation_precision(
        valid        = b_total - b_hall,
        partial      = 0,
        hallucinated = b_hall,
    )
    e_prec = citation_precision(
        valid        = e_total - e_hall,
        partial      = 0,
        hallucinated = e_hall,
    )
    print(f"  Baseline     : {b_prec['strict_precision']:.1%}")
    print(f"  Experimental : {e_prec['strict_precision']:.1%}")
    print(f"  Improvement  : {e_prec['strict_precision'] - b_prec['strict_precision']:+.1%}")

    # --- Dissertation table ---
    print("\n" + "=" * 65)
    print("DISSERTATION TABLE (copy into FPR)")
    print("=" * 65)
    print(f"\n| Metric                  | Baseline   | Experimental | Improvement |")
    print(f"|-------------------------|------------|--------------|-------------|")
    print(f"| Topics Evaluated        | {len(base_rows):<10} | {len(exp_rows):<12} | —           |")
    print(f"| Total Citations         | {b_total:<10} | {e_total:<12} | —           |")
    print(f"| Valid Citations         | {b_total-b_hall:<10} | {e_total-e_hall:<12} | —           |")
    print(f"| Hallucinated Citations  | {b_hall:<10} | {e_hall:<12} | —           |")
    print(f"| Hallucination Rate      | {b_rate:<10.1%} | {e_rate:<12.1%} | {b_rate-e_rate:+.1%}        |")
    print(f"| Citation Precision      | {b_prec['strict_precision']:<10.1%} | {e_prec['strict_precision']:<12.1%} | {e_prec['strict_precision']-b_prec['strict_precision']:+.1%}        |")
    print(f"| z-score                 | —          | —            | {stat['z_score']}       |")
    print(f"| p-value                 | —          | —            | {stat['p_value']}    |")
    print(f"| H₁ Supported            | —          | —            | {stat['significant']}         |")

    # --- Final conclusion ---
    print("\n" + "=" * 65)
    print("CONCLUSION")
    print("=" * 65)
    if stat["significant"]:
        print(f"\n✅ STATISTICALLY SIGNIFICANT (p = {stat['p_value']})")
        print(f"✅ Reject H₀ — Accept H₁")
        print(
            f"✅ Experimental reduced hallucination by "
            f"{b_rate - e_rate:.1%} "
            f"({b_rate:.1%} → {e_rate:.1%})"
        )
        print(f"✅ This supports your research hypothesis at α = 0.05")
    else:
        print(f"\n⚠️  NOT significant at α = 0.05 (p = {stat['p_value']})")
        print(f"⚠️  Run more topics to increase statistical power")
        print(f"⚠️  Current trend: {b_rate:.1%} → {e_rate:.1%} favours H₁")

    # Save report
    out_path = settings.data_dir / "eval" / "statistical_report.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Statistical Report\n")
        f.write(f"Generated: {__import__('datetime').datetime.now()}\n\n")
        f.write(f"Baseline hallucination rate   : {b_rate:.1%}\n")
        f.write(f"Experimental hallucination rate: {e_rate:.1%}\n")
        f.write(f"Improvement                   : {b_rate - e_rate:+.1%}\n")
        f.write(f"z-score                       : {stat['z_score']}\n")
        f.write(f"p-value                       : {stat['p_value']}\n")
        f.write(f"Significant                   : {stat['significant']}\n")

    print(f"\n[Saved] Statistical report → {out_path}")


if __name__ == "__main__":
    main()