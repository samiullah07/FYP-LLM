# test_metrics.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import (
    citation_precision,
    citation_recall,
    verifier_performance,
    error_typology,
    two_proportion_z_test,
    wilson_confidence_interval,
    cohens_kappa,
    claim_level_accuracy,
    generate_comparison_report,
    print_metrics_report,
    save_metrics_report,
)
from datetime import datetime


def main():
    print("=" * 60)
    print("TESTING ALL METRICS")
    print("=" * 60)

    # --- Citation precision ---
    print("\n1. Citation Precision:")
    prec = citation_precision(valid=7, partial=1, hallucinated=2)
    for k, v in prec.items():
        print(f"   {k:<25} : {v}")

    # --- Citation recall ---
    print("\n2. Citation Recall:")
    rec = citation_recall(valid=7, gold_standard=25)
    print(f"   recall : {rec:.1%}")

    # --- Verifier performance ---
    print("\n3. Verifier Performance:")
    perf = verifier_performance(
        true_positives=8,
        false_positives=2,
        true_negatives=18,
        false_negatives=2,
    )
    for k, v in perf.items():
        print(f"   {k:<20} : {v}")

    # --- Error typology ---
    print("\n4. Error Typology:")
    sample_logs = [
        {"error_type": "FABRICATED_PAPER"},
        {"error_type": "WRONG_YEAR"},
        {"error_type": "FABRICATED_PAPER"},
        {"error_type": "WRONG_AUTHOR"},
        {"error_type": None},
    ]
    typo = error_typology(sample_logs)
    for k, v in typo.items():
        print(f"   {k:<20} : {v}")

    # --- Statistical test ---
    print("\n5. Two-proportion z-test:")
    stat = two_proportion_z_test(n1=30, h1=0, n2=30, h2=9)
    for k, v in stat.items():
        print(f"   {k:<20} : {v}")
        # --- Wilson confidence interval ---
    print("\n6. Wilson Confidence Interval:")
    ci = wilson_confidence_interval(successes=3, total=20)
    for k, v in ci.items():
        print(f"   {k:<20} : {v}")

    # --- Cohen's Kappa ---
    print("\n7. Cohen's Kappa:")
    labels_a = [
        "supported", "supported", "unsupported",
        "partially_supported", "supported", "unsupported",
        "supported", "supported", "unsupported", "supported",
    ]
    labels_b = [
        "supported", "supported", "unsupported",
        "supported", "supported", "unsupported",
        "supported", "partially_supported", "unsupported", "supported",
    ]
    kappa = cohens_kappa(labels_a, labels_b)
    for k, v in kappa.items():
        print(f"   {k:<20} : {v}")

    # --- Claim-level accuracy ---
    print("\n8. Claim-level Accuracy:")
    annotations = [
        {"label": "supported"},
        {"label": "supported"},
        {"label": "partially_supported"},
        {"label": "unsupported"},
        {"label": "supported"},
        {"label": "unsupported"},
        {"label": "supported"},
        {"label": "supported"},
        {"label": "partially_supported"},
        {"label": "supported"},
    ]
    acc = claim_level_accuracy(annotations)
    for k, v in acc.items():
        print(f"   {k:<30} : {v}")

    # --- Full comparison report ---
    print("\n9. Full Comparison Report:")

    # Simulate baseline and experimental states
    baseline_state = {
        "valid_citations":       8,
        "partial_citations":     0,
        "hallucinated_citations": 0,
        "total_citations":       8,
        "hallucination_rate":    0.0,
        "review_text":           "Sample baseline review text " * 50,
    }
    experimental_state = {
        "valid_citations":       7,
        "partial_citations":     0,
        "hallucinated_citations": 3,
        "total_citations":       10,
        "hallucination_rate":    0.3,
        "final_review":          "Sample experimental review text " * 50,
        "draft_review":          "Sample draft review text " * 50,
    }

    # Simulate verifier logs
    baseline_logs = [
        {"error_type": None},
        {"error_type": None},
        {"error_type": None},
    ]
    experimental_logs = [
        {"error_type": "FABRICATED_PAPER"},
        {"error_type": "WRONG_YEAR"},
        {"error_type": "FABRICATED_PAPER"},
    ]

    report = generate_comparison_report(
        baseline_state      = baseline_state,
        experimental_state  = experimental_state,
        baseline_logs       = baseline_logs,
        experimental_logs   = experimental_logs,
        gold_standard_count = 25,
    )

    print_metrics_report(report)

    # Save report
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_to = save_metrics_report(report, run_id)
    print(f"\nReport saved to: {saved_to}")


if __name__ == "__main__":
    main()