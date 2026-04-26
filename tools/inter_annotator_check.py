import sys
import csv
import json
import os
from collections import Counter

CLASSES = ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"]

def load_annotations(filepath):
    data = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row["claim_id"]] = {
                "claim_text": row["claim_text"],
                "label":      row["label"].strip(),
            }
    return data

def compute_cohen_kappa(labels1, labels2):
    n = len(labels1)
    matches = sum(1 for a, b in zip(labels1, labels2) if a == b)
    p_obs = matches / n
    c1 = Counter(labels1)
    c2 = Counter(labels2)
    p_e = sum((c1[c] / n) * (c2[c] / n) for c in CLASSES)
    kappa = (p_obs - p_e) / (1 - p_e) if (1 - p_e) != 0 else 1.0
    return round(p_obs * 100, 2), round(kappa, 4)

def print_confusion_matrix(labels1, labels2):
    matrix = {r: {c: 0 for c in CLASSES} for r in CLASSES}
    for a, b in zip(labels1, labels2):
        matrix[a][b] += 1
    short = {"SUPPORTED": "SUP", "PARTIALLY_SUPPORTED": "PART", "UNSUPPORTED": "UNSUP"}
    sys.stdout.reconfigure(encoding='utf-8')  # Fixed import
    header = f"{'Pass1 \\ Pass2':<22}" + "".join(f"{short[c]:>8}" for c in CLASSES)
    print("\nConfusion Matrix (rows=Pass1, cols=Pass2):")
    print("-" * (22 + 8 * len(CLASSES)))
    print(header)
    print("-" * (22 + 8 * len(CLASSES)))
    for r in CLASSES:
        row_str = f"{short[r]:<22}" + "".join(f"{matrix[r][c]:>8}" for c in CLASSES)
        print(row_str)
    print("-" * (22 + 8 * len(CLASSES)))
    return matrix

def main():
    p1_path = os.path.join("data", "annotation_pass1_sample.csv")
    p2_path = os.path.join("data", "annotation_pass2_sample.csv")

    if not os.path.exists(p1_path) or not os.path.exists(p2_path):
        print("ERROR: Sample CSV files not found. Create them first.")
        return

    ann1 = load_annotations(p1_path)
    ann2 = load_annotations(p2_path)

    common_ids = sorted(set(ann1) & set(ann2))
    labels1 = [ann1[i]["label"] for i in common_ids]
    labels2 = [ann2[i]["label"] for i in common_ids]

    agreement_pct, kappa = compute_cohen_kappa(labels1, labels2)
    matrix = print_confusion_matrix(labels1, labels2)

    disagreements = []
    for cid in common_ids:
        l1, l2 = ann1[cid]["label"], ann2[cid]["label"]
        if l1 != l2:
            disagreements.append({
                "claim_id":   cid,
                "claim_text": ann1[cid]["claim_text"],
                "pass1_label": l1,
                "pass2_label": l2,
            })

    print(f"\nAgreement : {agreement_pct:.1f}%")
    print(f"Cohen Kappa: {kappa}")
    print(f"Disagreements ({len(disagreements)}):")
    for d in disagreements:
        print(f"  {d['claim_id']}: \"{d['claim_text'][:60]}\"")
        print(f"    Pass1={d['pass1_label']}  →  Pass2={d['pass2_label']}")

    os.makedirs("evaluation_results", exist_ok=True)
    out = {
        "agreement_pct": agreement_pct,
        "cohen_kappa":   kappa,
        "n_claims":      len(common_ids),
        "n_disagreements": len(disagreements),
        "confusion_matrix": matrix,
        "disagreements": disagreements,
    }
    out_path = os.path.join("evaluation_results", "annotator_consistency.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")

if __name__ == "__main__":
    main()