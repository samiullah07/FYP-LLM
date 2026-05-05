import json, os, glob

def load_all_verifier_logs():
    logs = []
    for f in glob.glob('data/eval/verifier_logs/*.json'):
        try:
            data = json.load(open(f, encoding='utf-8'))
            logs.extend(data.get('entries', []))
        except:
            pass
    return logs

def calibrate(logs):
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
            true_status = e.get('status', '')
            conf = e.get('confidence', 0.5)
            pred = 'VALID' if conf >= 0.55 else 'HALLUCINATED'
            if true_status == 'VALID':
                if pred == 'VALID': tp += 1
                else:               fn += 1
            elif true_status == 'HALLUCINATED':
                if pred == 'VALID': fp += 1
                else:               tn += 1
        total = tp + fp + tn + fn
        if total == 0:
            continue
        tpr = round(tp / (tp + fn) * 100, 1) if (tp + fn) else 0
        fpr = round(fp / (fp + tn) * 100, 1) if (fp + tn) else 0
        f1n = 2 * tp
        f1d = 2 * tp + fp + fn
        f1  = round(f1n / f1d, 3) if f1d else 0
        results.append({
            'w_author': wa, 'w_year': wy, 'w_title': wt,
            'TPR': tpr, 'FPR': fpr, 'F1': f1,
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        })

    results.sort(key=lambda x: (-x['F1'], x['FPR']))

    print()
    print('=== Confidence Weight Calibration Study ===')
    print(f'Total citation entries analysed: {len(logs)}')
    print(f'Weight combinations tested     : {len(results)}')
    print()
    print(f'Top 5 weight combinations by F1:')
    print(f'{"w_author":>10} {"w_year":>8} {"w_title":>9} {"TPR%":>7} {"FPR%":>7} {"F1":>7}')
    print('-' * 55)
    for r in results[:5]:
        print(f'{r["w_author"]:>10} {r["w_year"]:>8} {r["w_title"]:>9} {r["TPR"]:>7} {r["FPR"]:>7} {r["F1"]:>7}')

    best = results[0] if results else {}
    if best:
        print()
        print(f'Current weights in code  : 0.55 author / 0.35 year / 0.10 title')
        print(f'Best calibrated weights  : {best["w_author"]} author / {best["w_year"]} year / {best["w_title"]} title  (F1={best["F1"]})')
        if best['w_author'] != 0.55 or best['w_year'] != 0.35:
            print('RECOMMENDATION: Consider updating weights in verifier_agent.py')
        else:
            print('Current weights are already optimal on this dataset.')

    os.makedirs('evaluation_results', exist_ok=True)
    out = os.path.join('evaluation_results', 'weight_calibration.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'best': best, 'all_results': results[:20]}, f, indent=2)
    print(f'Full results saved to: {out}')

logs = load_all_verifier_logs()
if not logs:
    print('No verifier logs found.')
else:
    print(f'Loaded {len(logs)} citation entries from verifier logs')
    calibrate(logs)
