import subprocess
import json
import os
import sys

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, label, detail))
    print(f"{status}  {label}  {detail}")

print("\n" + "="*60)
print("GAP VERIFICATION TEST SUITE")
print("="*60 + "\n")

# Gap 1 - Wilson CI file exists and is importable
try:
    sys.path.insert(0, ".")
    from tools.wilson_ci import compute_wilson_ci
    lo, hi, w = compute_wilson_ci(1, 25)
    check("Gap 1 - Wilson CI function",
          0 <= lo < hi <= 100,
          f"CI=[{lo:.1f}%, {hi:.1f}%]")
except Exception as e:
    check("Gap 1 - Wilson CI function", False, str(e))

# Gap 1 - Log file created
r = subprocess.run([sys.executable, "tools/wilson_ci.py"],
                   capture_output=True, text=True)
check("Gap 1 - wilson_ci.py runs cleanly", r.returncode == 0,
      r.stderr[:80] if r.returncode != 0 else "OK")
check("Gap 1 - Log file written",
      os.path.exists("evaluation_results/wilson_ci_log.json"), "")

# Gap 2 - Bandit logger
r2 = subprocess.run([sys.executable, "tools/bandit_logger.py"],
                    capture_output=True, text=True)
check("Gap 2 - bandit_logger.py runs cleanly", r2.returncode == 0,
      r2.stderr[:80] if r2.returncode != 0 else "OK")
check("Gap 2 - UCB1 state file written",
      os.path.exists("bandit_log/ucb1_state.json"), "")
check("Gap 2 - Convergence CSV written",
      os.path.exists("bandit_log/convergence_summary.csv"), "")

# Gap 3 - Error taxonomy in verifier
try:
    from agents.verifier_agent import VerifierAgent
    v = VerifierAgent.__dict__
    check("Gap 3 - ERROR_TAXONOMY exists in VerifierAgent",
          "ERROR_TAXONOMY" in v, "")
    check("Gap 3 - classify_error method exists",
          "classify_error" in v, "")
    check("Gap 3 - log_verification method exists",
          "log_verification" in v, "")
except Exception as e:
    check("Gap 3 - VerifierAgent importable", False, str(e))

# Gap 4 - Inter annotator check
r4 = subprocess.run(
    [sys.executable, "tools/inter_annotator_check.py"],
    capture_output=True, text=True)
check("Gap 4 - inter_annotator_check.py runs cleanly",
      r4.returncode == 0,
      r4.stderr[:80] if r4.returncode != 0 else "OK")
check("Gap 4 - annotator_consistency.json written",
      os.path.exists("evaluation_results/annotator_consistency.json"), "")
if os.path.exists("evaluation_results/annotator_consistency.json"):
    with open("evaluation_results/annotator_consistency.json") as f:
        d = json.load(f)
    check("Gap 4 - Kappa value present and valid",
          "cohen_kappa" in d and isinstance(d["cohen_kappa"], float),
          f"kappa={d.get('cohen_kappa')}")

# Gap 5 - Multi-pass in workflow graph
try:
    with open("graph/workflow_graph.py", encoding="utf-8") as f:
        src = f.read()
    check("Gap 5 - MAX_CORRECTION_PASSES defined",
          "MAX_CORRECTION_PASSES" in src, "")
    check("Gap 5 - route_after_verifier defined",
          "route_after_verifier" in src, "")
    check("Gap 5 - passes_completed in state",
          "passes_completed" in src, "")
    check("Gap 5 - _log_correction_pass defined",
          "_log_correction_pass" in src, "")
except Exception as e:
    check("Gap 5 - workflow_graph.py readable", False, str(e))

# Gap 6 - Ablation runner
r6 = subprocess.run(
    [sys.executable, "tools/run_ablation.py", "--dry-run"],
    capture_output=True, text=True)
check("Gap 6 - run_ablation.py --dry-run works",
      r6.returncode == 0,
      r6.stderr[:80] if r6.returncode != 0 else "OK")
check("Gap 6 - ablation_raw.csv written",
      os.path.exists("evaluation_results/ablation_raw.csv"), "")
check("Gap 6 - ablation_summary.json written",
      os.path.exists("evaluation_results/ablation_summary.json"), "")

# Gap 7 - Search agent
try:
    with open("agents/search_agent.py", encoding="utf-8") as f:
        src7 = f.read()
    check("Gap 7 - MAG note in search_agent.py",
          "Microsoft Academic Graph" in src7 or "MAG" in src7, "")
    check("Gap 7 - verify_openalex_connection defined",
          "verify_openalex_connection" in src7, "")
    check("Gap 7 - OpenAlex URL present",
          "api.openalex.org" in src7, "")
except Exception as e:
    check("Gap 7 - search_agent.py readable", False, str(e))

# Summary
print("\n" + "="*60)
passed = sum(1 for s,_,_ in results if s == PASS)
failed = sum(1 for s,_,_ in results if s == FAIL)
print(f"RESULTS: {passed} passed  |  {failed} failed  |  {len(results)} total")
print("="*60)
if failed == 0:
    print("All gaps verified successfully.")
else:
    print("Fix the FAIL items above before submission.")