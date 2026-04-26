import json
import math
import os
from datetime import datetime


def compute_wilson_ci(n_hallucinated: int, n_total: int, confidence: float = 0.95):
    """Calculate Wilson score confidence interval for a binomial proportion.

    Args:
        n_hallucinated: Number of hallucinated citations.
        n_total: Total citations checked.
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple (lower_pct, upper_pct, warning_str).
    """
    if n_total == 0:
        return 0.0, 100.0, "No data"

    phat = n_hallucinated / n_total
    z = 1.96  # 95% confidence

    denominator = 1 + (z ** 2) / n_total
    centre = (phat + (z ** 2) / (2 * n_total)) / denominator
    margin = (z * math.sqrt(phat * (1 - phat) / n_total + (z ** 2) / (4 * n_total * n_total))) / denominator

    lower = max(centre - margin, 0.0) * 100
    upper = min(centre + margin, 1.0) * 100
    warning = "Wide interval — n < 30" if n_total < 30 else ""
    return lower, upper, warning


def log_wilson_ci(hallucinated_count: int, total_count: int) -> None:
    """Compute Wilson CI and append result to evaluation_results/wilson_ci_log.json."""
    lower, upper, warning = compute_wilson_ci(hallucinated_count, total_count)
    rate_pct = hallucinated_count / total_count * 100 if total_count else 0

    entry = {
        "timestamp": datetime.now().isoformat(),
        "n_hallucinated": hallucinated_count,
        "n_total": total_count,
        "hallucination_rate_pct": round(rate_pct, 2),
        "wilson_ci_lower_pct": round(lower, 2),
        "wilson_ci_upper_pct": round(upper, 2),
        "warning": warning,
    }

    os.makedirs("evaluation_results", exist_ok=True)
    log_path = os.path.join("evaluation_results", "wilson_ci_log.json")

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(entry)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    print(f"  Observed : {hallucinated_count}/{total_count} ({rate_pct:.2f}%)")
    print(f"  Wilson CI: [{lower:.2f}%, {upper:.2f}%]")
    if warning:
        print(f"  Warning  : {warning}")
    print(f"  Logged to: {log_path}")


if __name__ == "__main__":
    print("=== Wilson CI Test ===\n")
    tests = [
        (1, 25,  "Agentic pipeline (report value)"),
        (5, 20,  "8B baseline (report value)"),
        (0, 10,  "Zero hallucinations"),
        (2, 30,  "Small sample"),
        (10, 100,"Larger sample"),
    ]
    for h, t, label in tests:
        print(f"Test: {label}")
        log_wilson_ci(h, t)
        print()