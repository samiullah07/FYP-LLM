# evaluation/metrics.py
"""
Evaluation Metrics for the Literature Review Agent System.

Implements all metrics specified in the IPR and annotator feedback:

    Citation-level metrics:
        - Citation Precision      : valid / total generated
        - Citation Recall         : valid / total in gold standard
        - Hallucination Rate      : hallucinated / total generated
        - Partial Rate            : partial / total generated

    Verifier performance metrics:
        - Sensitivity (Recall)    : hallucinations correctly flagged
        - Specificity             : valid citations correctly accepted
        - Precision               : flagged citations that are truly bad
        - F1 score

    Error typology:
        - Count per error type
        - Rate per error type

    Statistical tests:
        - Two-proportion z-test   : compare hallucination rates
        - Confidence intervals    : Wilson score interval

    Annotation metrics:
        - Cohen's Kappa           : inter/intra-annotator agreement
        - Claim-level accuracy

Addresses IPR feedback:
    "The statistical tests plan (two-proportion z-test or chi-square)
     is sensible, but relies on sample size assumptions..."
    "Consider adding verifier sensitivity and specificity"
    "Correction rate among claims initially hallucinated"
"""

import sys
import json
import math
from pathlib import Path
from datetime import datetime
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings

METRICS_DIR = settings.data_dir / "eval" / "metrics"


# ===========================================================================
# 1. Citation-level metrics
# ===========================================================================

def citation_precision(
    valid:       int,
    partial:     int,
    hallucinated: int,
    count_partial_as: str = "valid",
) -> dict:
    """
    Compute citation precision metrics.

    Strict   : only VALID counted as correct
    Lenient  : VALID + PARTIAL counted as correct

    Parameters
    ----------
    valid            : number of valid citations
    partial          : number of partial citations
    hallucinated     : number of hallucinated citations
    count_partial_as : "valid" (lenient) or "invalid" (strict)

    Returns
    -------
    dict:
        total           : int
        strict_precision: float  (valid / total)
        lenient_precision: float (valid+partial / total)
        hallucination_rate: float
        partial_rate    : float
    """
    total = valid + partial + hallucinated

    if total == 0:
        return {
            "total":              0,
            "strict_precision":   0.0,
            "lenient_precision":  0.0,
            "hallucination_rate": 0.0,
            "partial_rate":       0.0,
        }

    strict_precision  = round(valid / total, 4)
    lenient_precision = round((valid + partial) / total, 4)
    hallucination_rate = round(hallucinated / total, 4)
    partial_rate       = round(partial / total, 4)

    return {
        "total":               total,
        "valid":               valid,
        "partial":             partial,
        "hallucinated":        hallucinated,
        "strict_precision":    strict_precision,
        "lenient_precision":   lenient_precision,
        "hallucination_rate":  hallucination_rate,
        "partial_rate":        partial_rate,
    }


def citation_recall(
    valid:          int,
    gold_standard:  int,
) -> float:
    """
    Compute citation recall against a gold standard set.

    Recall = valid citations that appear in gold / total gold citations

    Parameters
    ----------
    valid         : number of generated citations matched to gold
    gold_standard : total number of papers in gold standard corpus

    Returns
    -------
    float : recall score 0.0 to 1.0
    """
    if gold_standard == 0:
        return 0.0
    return round(valid / gold_standard, 4)


def hallucination_rate_per_1000_tokens(
    hallucinated:   int,
    review_text:    str,
) -> float:
    """
    Compute normalised hallucination rate per 1000 tokens.

    Useful for cost/reliability trade-off analysis in dissertation.

    Parameters
    ----------
    hallucinated : number of hallucinated citations
    review_text  : full review text string

    Returns
    -------
    float : hallucinations per 1000 tokens
    """
    # Approximate token count (words * 1.3 is a common heuristic)
    approx_tokens = len(review_text.split()) * 1.3
    if approx_tokens == 0:
        return 0.0
    return round((hallucinated / approx_tokens) * 1000, 4)


# ===========================================================================
# 2. Verifier performance metrics
# ===========================================================================

def verifier_performance(
    true_positives:  int,
    false_positives: int,
    true_negatives:  int,
    false_negatives: int,
) -> dict:
    """
    Compute verifier sensitivity, specificity, precision, F1.

    Definitions for this task:
        Positive  = citation IS hallucinated
        Negative  = citation IS valid

        True Positive  (TP): hallucinated citation correctly flagged
        False Positive (FP): valid citation wrongly flagged as hallucinated
        True Negative  (TN): valid citation correctly accepted
        False Negative (FN): hallucinated citation missed (slipped through)

    Parameters
    ----------
    true_positives  : hallucinations correctly caught by verifier
    false_positives : valid citations wrongly flagged
    true_negatives  : valid citations correctly accepted
    false_negatives : hallucinations missed by verifier

    Returns
    -------
    dict:
        sensitivity  : TP / (TP + FN)  how many hallucinations caught
        specificity  : TN / (TN + FP)  how many valid refs kept
        precision    : TP / (TP + FP)  accuracy of flags
        f1_score     : harmonic mean of sensitivity and precision
        accuracy     : (TP + TN) / total
    """
    total = true_positives + false_positives + true_negatives + false_negatives

    sensitivity = (
        round(true_positives / (true_positives + false_negatives), 4)
        if (true_positives + false_negatives) > 0 else 0.0
    )
    specificity = (
        round(true_negatives / (true_negatives + false_positives), 4)
        if (true_negatives + false_positives) > 0 else 0.0
    )
    precision = (
        round(true_positives / (true_positives + false_positives), 4)
        if (true_positives + false_positives) > 0 else 0.0
    )
    f1_score = (
        round(2 * precision * sensitivity / (precision + sensitivity), 4)
        if (precision + sensitivity) > 0 else 0.0
    )
    accuracy = (
        round((true_positives + true_negatives) / total, 4)
        if total > 0 else 0.0
    )

    return {
        "true_positives":  true_positives,
        "false_positives": false_positives,
        "true_negatives":  true_negatives,
        "false_negatives": false_negatives,
        "sensitivity":     sensitivity,
        "specificity":     specificity,
        "precision":       precision,
        "f1_score":        f1_score,
        "accuracy":        accuracy,
    }


# ===========================================================================
# 3. Error typology metrics
# ===========================================================================

def error_typology(logs: list[dict]) -> dict:
    """
    Compute error type distribution from verifier logs.

    Parameters
    ----------
    logs : list of verifier log entry dicts
           (from verifier_agent._build_log_entry)

    Returns
    -------
    dict:
        total_errors   : int
        by_type        : dict of error_type → count
        by_type_rate   : dict of error_type → rate
        most_common    : str  most frequent error type
    """
    error_types = [
        log["error_type"]
        for log in logs
        if log.get("error_type") is not None
    ]

    if not error_types:
        return {
            "total_errors": 0,
            "by_type":      {},
            "by_type_rate": {},
            "most_common":  None,
        }

    counts     = Counter(error_types)
    total      = len(error_types)
    rates      = {k: round(v / total, 4) for k, v in counts.items()}
    most_common = counts.most_common(1)[0][0]

    return {
        "total_errors": total,
        "by_type":      dict(counts),
        "by_type_rate": rates,
        "most_common":  most_common,
    }


# ===========================================================================
# 4. Statistical tests
# ===========================================================================

def two_proportion_z_test(
    n1:   int,
    h1:   int,
    n2:   int,
    h2:   int,
) -> dict:
    """
    Two-proportion z-test comparing hallucination rates.

    Tests H0: p1 == p2 (no difference between systems)
    vs    H1: p1 != p2 (significant difference)

    Parameters
    ----------
    n1 : total citations in system 1 (baseline)
    h1 : hallucinated citations in system 1
    n2 : total citations in system 2 (experimental)
    h2 : hallucinated citations in system 2

    Returns
    -------
    dict:
        p1          : float  hallucination rate system 1
        p2          : float  hallucination rate system 2
        z_score     : float
        p_value     : float  (two-tailed, approximate)
        significant : bool   (p < 0.05)
        interpretation: str
    """
    if n1 == 0 or n2 == 0:
        return {
            "p1":             0.0,
            "p2":             0.0,
            "z_score":        0.0,
            "p_value":        1.0,
            "significant":    False,
            "interpretation": "Insufficient data for test",
        }

    p1     = h1 / n1
    p2     = h2 / n2
    p_pool = (h1 + h2) / (n1 + n2)

    denom = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    if denom == 0:
        z_score = 0.0
        p_value = 1.0
    else:
        z_score = (p1 - p2) / denom
        # Approximate two-tailed p-value using normal distribution
        # P(Z > |z|) * 2
        p_value = 2 * (1 - _normal_cdf(abs(z_score)))

    significant = p_value < 0.05

    if significant:
        direction = "higher" if p1 > p2 else "lower"
        interpretation = (
            f"Significant difference (p={p_value:.4f}). "
            f"Baseline hallucination rate ({p1:.1%}) is {direction} "
            f"than experimental ({p2:.1%})."
        )
    else:
        interpretation = (
            f"No significant difference (p={p_value:.4f}). "
            f"Cannot reject H0 at α=0.05."
        )

    return {
        "p1":              round(p1, 4),
        "p2":              round(p2, 4),
        "difference":      round(p1 - p2, 4),
        "z_score":         round(z_score, 4),
        "p_value":         round(p_value, 4),
        "significant":     significant,
        "interpretation":  interpretation,
    }


def _normal_cdf(z: float) -> float:
    """
    Approximate CDF of standard normal distribution.
    Uses math.erf for accuracy without scipy dependency.
    """
    return (1.0 + math.erf(z / math.sqrt(2.0))) / 2.0


def wilson_confidence_interval(
    successes: int,
    total:     int,
    confidence: float = 0.95,
) -> dict:
    """
    Compute Wilson score confidence interval for a proportion.

    More accurate than normal approximation for small samples.
    Used to report confidence intervals around hallucination rates.

    Parameters
    ----------
    successes  : number of hallucinated citations (or valid)
    total      : total citations
    confidence : confidence level (default 0.95 = 95%)

    Returns
    -------
    dict:
        proportion  : float
        lower       : float  lower bound
        upper       : float  upper bound
        interval    : str    formatted string
    """
    if total == 0:
        return {
            "proportion": 0.0,
            "lower":      0.0,
            "upper":      0.0,
            "interval":   "[0.000, 0.000]",
        }

    # z for confidence level (1.96 for 95%)
    alpha = 1 - confidence
    z     = _normal_cdf_inv(1 - alpha / 2)

    p      = successes / total
    n      = total
    z2     = z * z
    denom  = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom

    lower = max(0.0, round(centre - margin, 4))
    upper = min(1.0, round(centre + margin, 4))

    return {
        "proportion": round(p, 4),
        "lower":      lower,
        "upper":      upper,
        "interval":   f"[{lower:.3f}, {upper:.3f}]",
    }


def _normal_cdf_inv(p: float) -> float:
    """
    Approximate inverse normal CDF (percent-point function).
    Returns z such that P(Z < z) = p.
    Used only for Wilson interval calculation.
    """
    # Rational approximation (Abramowitz and Stegun)
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    if p < 0.5:
        return -_normal_cdf_inv(1 - p)

    t = math.sqrt(-2.0 * math.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    numerator   = c0 + c1 * t + c2 * t * t
    denominator = 1 + d1 * t + d2 * t * t + d3 * t * t * t
    return t - numerator / denominator


# ===========================================================================
# 5. Cohen's Kappa (inter/intra-annotator agreement)
# ===========================================================================

def cohens_kappa(
    annotations_a: list[str],
    annotations_b: list[str],
) -> dict:
    """
    Compute Cohen's Kappa for inter or intra-annotator agreement.

    Used to measure reliability of claim-level annotation
    (supported / partially_supported / unsupported).

    Parameters
    ----------
    annotations_a : list of labels from annotator A (or first pass)
    annotations_b : list of labels from annotator B (or second pass)

    Returns
    -------
    dict:
        kappa           : float   (-1 to 1, >0.6 = substantial agreement)
        observed_agree  : float   proportion of exact matches
        expected_agree  : float   agreement expected by chance
        interpretation  : str
        total_items     : int
    """
    if len(annotations_a) != len(annotations_b):
        raise ValueError(
            f"Annotation lists must be the same length. "
            f"Got {len(annotations_a)} and {len(annotations_b)}."
        )

    n = len(annotations_a)
    if n == 0:
        return {
            "kappa":          0.0,
            "observed_agree": 0.0,
            "expected_agree": 0.0,
            "interpretation": "No annotations",
            "total_items":    0,
        }

    # Observed agreement
    observed_agree = sum(
        1 for a, b in zip(annotations_a, annotations_b) if a == b
    ) / n

    # Expected agreement
    labels    = set(annotations_a) | set(annotations_b)
    count_a   = Counter(annotations_a)
    count_b   = Counter(annotations_b)
    expected  = sum(
        (count_a.get(label, 0) / n) * (count_b.get(label, 0) / n)
        for label in labels
    )

    if expected == 1.0:
        kappa = 1.0
    else:
        kappa = round((observed_agree - expected) / (1 - expected), 4)

    # Interpretation scale (Landis & Koch 1977)
    if kappa >= 0.81:
        interp = "Almost perfect agreement"
    elif kappa >= 0.61:
        interp = "Substantial agreement"
    elif kappa >= 0.41:
        interp = "Moderate agreement"
    elif kappa >= 0.21:
        interp = "Fair agreement"
    elif kappa >= 0.0:
        interp = "Slight agreement"
    else:
        interp = "Poor agreement (less than chance)"

    return {
        "kappa":           kappa,
        "observed_agree":  round(observed_agree, 4),
        "expected_agree":  round(expected, 4),
        "interpretation":  interp,
        "total_items":     n,
    }


# ===========================================================================
# 6. Claim-level accuracy
# ===========================================================================

def claim_level_accuracy(annotations: list[dict]) -> dict:
    """
    Compute claim-level accuracy from manual annotations.

    Parameters
    ----------
    annotations : list of dicts, each with key "label":
                  "supported" / "partially_supported" / "unsupported"

    Returns
    -------
    dict:
        total                   : int
        supported               : int
        partially_supported     : int
        unsupported             : int
        strict_accuracy         : float  supported / total
        lenient_accuracy        : float  (supported+partial) / total
        strict_hallucination    : float  unsupported / total
        lenient_hallucination   : float  (unsupported+partial) / total
    """
    total       = len(annotations)
    supported   = sum(1 for a in annotations if a.get("label") == "supported")
    partial     = sum(1 for a in annotations if a.get("label") == "partially_supported")
    unsupported = sum(1 for a in annotations if a.get("label") == "unsupported")

    if total == 0:
        return {
            "total":                 0,
            "supported":             0,
            "partially_supported":   0,
            "unsupported":           0,
            "strict_accuracy":       0.0,
            "lenient_accuracy":      0.0,
            "strict_hallucination":  0.0,
            "lenient_hallucination": 0.0,
        }

    return {
        "total":                 total,
        "supported":             supported,
        "partially_supported":   partial,
        "unsupported":           unsupported,
        "strict_accuracy":       round(supported / total, 4),
        "lenient_accuracy":      round((supported + partial) / total, 4),
        "strict_hallucination":  round(unsupported / total, 4),
        "lenient_hallucination": round((unsupported + partial) / total, 4),
    }


# ===========================================================================
# 7. Full comparison report
# ===========================================================================

def generate_comparison_report(
    baseline_state:     dict,
    experimental_state: dict,
    baseline_logs:      list[dict] = None,
    experimental_logs:  list[dict] = None,
    gold_standard_count: int = 0,
) -> dict:
    """
    Generate a full comparison report between baseline and experimental.

    Parameters
    ----------
    baseline_state       : final state dict from run_baseline()
    experimental_state   : final state dict from run_workflow()
    baseline_logs        : verifier logs from baseline run
    experimental_logs    : verifier logs from experimental run
    gold_standard_count  : number of papers in gold corpus

    Returns
    -------
    dict : full metrics report
    """
    baseline_logs     = baseline_logs or []
    experimental_logs = experimental_logs or []

    # --- Citation precision for each system ---
    base_precision = citation_precision(
        valid        = baseline_state.get("valid_citations", 0),
        partial      = baseline_state.get("partial_citations", 0),
        hallucinated = baseline_state.get("hallucinated_citations", 0),
    )
    exp_precision = citation_precision(
        valid        = experimental_state.get("valid_citations", 0),
        partial      = experimental_state.get("partial_citations", 0),
        hallucinated = experimental_state.get("hallucinated_citations", 0),
    )

    # --- Citation recall (if gold standard available) ---
    base_recall = citation_recall(
        valid         = baseline_state.get("valid_citations", 0),
        gold_standard = gold_standard_count,
    )
    exp_recall = citation_recall(
        valid         = experimental_state.get("valid_citations", 0),
        gold_standard = gold_standard_count,
    )

    # --- Statistical test ---
    stat_test = two_proportion_z_test(
        n1 = baseline_state.get("total_citations", 0),
        h1 = baseline_state.get("hallucinated_citations", 0),
        n2 = experimental_state.get("total_citations", 0),
        h2 = experimental_state.get("hallucinated_citations", 0),
    )

    # --- Confidence intervals ---
    base_ci = wilson_confidence_interval(
        successes = baseline_state.get("hallucinated_citations", 0),
        total     = baseline_state.get("total_citations", 0),
    )
    exp_ci = wilson_confidence_interval(
        successes = experimental_state.get("hallucinated_citations", 0),
        total     = experimental_state.get("total_citations", 0),
    )

    # --- Hallucination per 1000 tokens ---
    base_per_1k = hallucination_rate_per_1000_tokens(
        hallucinated = baseline_state.get("hallucinated_citations", 0),
        review_text  = baseline_state.get("review_text", ""),
    )
    exp_per_1k = hallucination_rate_per_1000_tokens(
        hallucinated = experimental_state.get("hallucinated_citations", 0),
        review_text  = experimental_state.get("final_review", ""),
    )

    # --- Error typology ---
    base_errors = error_typology(baseline_logs)
    exp_errors  = error_typology(experimental_logs)

    # --- Assemble report ---
    report = {
        "generated_at":     datetime.now().isoformat(),
        "baseline": {
            "precision":       base_precision,
            "recall":          base_recall,
            "hallucination_ci": base_ci,
            "per_1000_tokens": base_per_1k,
            "error_typology":  base_errors,
        },
        "experimental": {
            "precision":       exp_precision,
            "recall":          exp_recall,
            "hallucination_ci": exp_ci,
            "per_1000_tokens": exp_per_1k,
            "error_typology":  exp_errors,
        },
        "statistical_test":  stat_test,
    }

    return report


# ===========================================================================
# 8. Save and print report
# ===========================================================================

def save_metrics_report(report: dict, run_id: str) -> Path:
    """
    Save full metrics report as JSON.

    Parameters
    ----------
    report : dict from generate_comparison_report()
    run_id : unique run identifier

    Returns
    -------
    Path : saved report path
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / f"metrics_report_{run_id}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[Metrics] Report saved → {out_path}")
    return out_path


def print_metrics_report(report: dict) -> None:
    """
    Print a formatted metrics report to console.

    Parameters
    ----------
    report : dict from generate_comparison_report()
    """
    b = report["baseline"]
    e = report["experimental"]
    s = report["statistical_test"]

    print("\n" + "=" * 70)
    print("FULL METRICS REPORT")
    print("=" * 70)

    print(f"\n{'Metric':<35} {'Baseline':>15} {'Experimental':>15}")
    print("-" * 70)

    rows = [
        ("Total Citations",
         b["precision"]["total"],
         e["precision"]["total"]),

        ("Valid Citations",
         b["precision"]["valid"],
         e["precision"]["valid"]),

        ("Partial Citations",
         b["precision"]["partial"],
         e["precision"]["partial"]),

        ("Hallucinated Citations",
         b["precision"]["hallucinated"],
         e["precision"]["hallucinated"]),

        ("Strict Precision",
         f"{b['precision']['strict_precision']:.1%}",
         f"{e['precision']['strict_precision']:.1%}"),

        ("Lenient Precision",
         f"{b['precision']['lenient_precision']:.1%}",
         f"{e['precision']['lenient_precision']:.1%}"),

        ("Hallucination Rate",
         f"{b['precision']['hallucination_rate']:.1%}",
         f"{e['precision']['hallucination_rate']:.1%}"),

        ("Recall (vs gold)",
         f"{b['recall']:.1%}",
         f"{e['recall']:.1%}"),

        ("Halluc. per 1K tokens",
         f"{b['per_1000_tokens']:.3f}",
         f"{e['per_1000_tokens']:.3f}"),

        ("95% CI (halluc. rate)",
         b["hallucination_ci"]["interval"],
         e["hallucination_ci"]["interval"]),
    ]

    for label, b_val, e_val in rows:
        print(f"  {label:<33} {str(b_val):>15} {str(e_val):>15}")

    print("-" * 70)
    print(f"\nStatistical Test (Two-proportion z-test):")
    print(f"  z-score      : {s['z_score']}")
    print(f"  p-value      : {s['p_value']}")
    print(f"  Significant  : {s['significant']}")
    print(f"  {s['interpretation']}")

    print(f"\nError Typology — Baseline:")
    for k, v in b["error_typology"].get("by_type", {}).items():
        print(f"  {k:<25} : {v}")

    print(f"\nError Typology — Experimental:")
    for k, v in e["error_typology"].get("by_type", {}).items():
        print(f"  {k:<25} : {v}")

    print("=" * 70)