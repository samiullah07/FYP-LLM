# CLAUDE.md

## Project Context
This repository implements **Agentic AI for Reliable Academic Literature Review (Hallucination Detection)** using a multi-agent architecture built with **Groq, LangGraph, LangChain, OpenAlex, FAISS, Sentence-BERT, and Streamlit**.

The system currently includes five major agents:
- Planner Agent
- Searcher Agent
- Summariser Agent
- Verifier Agent
- Assembler Agent

This file defines the next implementation phase so Claude Code can extend the project toward a more production-ready research system without repeatedly asking for permission for common file operations.

---

## Current Objective
Implement the following four production-ready extensions shown in the latest project planning update:

1. **Fine-Tuned Verification Models**
2. **Expanded Database Integration**
3. **Real-Time Human Feedback Loop**
4. **Confidence Calibration Study**

Claude Code should treat these as approved roadmap tasks for this repository.

---

## Approved Implementation Scope

### 1. Fine-Tuned Verification Models
Goal:
Train or prepare a small verification model on annotated hallucinated citation examples so the verifier can better detect edge cases such as:
- partial author matches
- year mismatches
- weak semantic grounding
- title overlap without true evidence support

Expected implementation work:
- Create a dataset structure for annotated citation verification examples.
- Add scripts for dataset preparation, training, evaluation, and prediction.
- Add a pluggable verification backend so the project can switch between:
  - rule-based verifier
  - Sentence-BERT verifier
  - fine-tuned model verifier
- Add support for uncertainty scoring inspired by semantic entropy as an internal signal for low-confidence cases.

Suggested folders/files to create:
- `data/verification_dataset/`
- `training/prepare_verifier_data.py`
- `training/train_verifier.py`
- `training/evaluate_verifier.py`
- `models/verification/`
- `agents/finetuned_verifier.py`
- `src/uncertainty.py`

Implementation notes:
- Keep verifier backend selection configurable.
- Do not remove the existing Sentence-BERT verifier.
- Add fallback logic so the old verifier still works if the fine-tuned model is unavailable.

---

### 2. Expanded Database Integration
Goal:
Expand the academic retrieval layer beyond OpenAlex so the system can support stronger triangulation and broader domain coverage.

Approved retrieval sources to integrate:
- Semantic Scholar
- Crossref
- JSTOR
- OpenAlex

Expected implementation work:
- Re-integrate Semantic Scholar properly using API key support and rate-limit handling.
- Add Crossref retriever for metadata enrichment and DOI resolution.
- Add JSTOR connector or placeholder adapter if direct API access is constrained.
- Normalize results from multiple sources into one shared paper schema.
- Add source-level provenance tracking.
- Add de-duplication logic across OpenAlex, Semantic Scholar, and Crossref.
- Allow optional source filtering from UI/config.

Suggested folders/files to create or edit:
- `retrievers/openalex_client.py`
- `retrievers/semantic_scholar_client.py`
- `retrievers/crossref_client.py`
- `retrievers/jstor_client.py`
- `retrievers/base_retriever.py`
- `src/paper_normalizer.py`
- `src/deduplication.py`
- `configs/retrieval_sources.py`

Implementation notes:
- Use a shared schema for title, authors, year, abstract, DOI, venue, URL, source, citation_count, and relevance score.
- Add retry/backoff and rate-limit protection.
- Preserve source attribution for evaluation.

---

### 3. Real-Time Human Feedback Loop
Goal:
Allow users to accept, reject, or reclassify verifier decisions in the UI and store these corrections for future learning.

Expected implementation work:
- Add UI actions for each citation decision:
  - Accept
  - Reject
  - Reclassify
- Save user feedback as structured records.
- Feed corrected labels back into:
  - bandit selection logic
  - evaluation logs
  - future fine-tuning datasets
- Add feedback audit trail with timestamp, topic, citation, predicted label, corrected label, and reviewer note.
- Support manual override in the UI for verifier outcomes.

Suggested folders/files to create or edit:
- `feedback/feedback_store.py`
- `feedback/feedback_schema.py`
- `data/human_feedback/`
- `app.py`
- `ui/components/verification_feedback.py`
- `src/feedback_processor.py`
- `src/mabselector.py`

Implementation notes:
- Every feedback event should be persisted locally in JSON or CSV first.
- Design the schema so it can later be migrated to SQLite/Postgres.
- Human feedback should not silently overwrite raw model output; store both original and corrected decisions.

---

### 4. Confidence Calibration Study
Goal:
Replace hand-tuned verifier weighting and threshold logic with a learned calibration approach and generate stronger evaluation evidence.

Current issue to address:
The verifier currently relies on hand-tuned weights such as `0.55 / 0.35 / 0.10`, which should be replaced by a more principled learned calibration method.

Expected implementation work:
- Build a labelled calibration dataset from verifier outputs.
- Train a calibration model using regression or classification.
- Compare hand-tuned confidence scores against calibrated confidence outputs.
- Generate ROC curves and threshold analysis for:
  - VALID
  - PARTIAL
  - HALLUCINATED
- Export calibration metrics for Chapter 5 evaluation.
- Make thresholds configurable through a central config file.

Suggested folders/files to create or edit:
- `evaluation/calibration_dataset.py`
- `evaluation/calibration_study.py`
- `evaluation/roc_analysis.py`
- `evaluation/export_metrics.py`
- `configs/verifier_thresholds.py`
- `agents/verifier_agent.py`

Implementation notes:
- Preserve backward compatibility with the existing verifier logic until calibration is validated.
- Save output charts, CSVs, and threshold tables for dissertation evidence.

---

## Engineering Rules
Claude Code should follow these implementation rules:

- Keep the current project architecture intact.
- Prefer additive changes over destructive rewrites.
- Avoid breaking working baseline and experimental pipelines.
- Preserve compatibility with current Streamlit dashboard views.
- Add logging for every new retrieval, verifier, feedback, and calibration component.
- Keep configuration centralized in `configs/` whenever possible.
- Use environment variables for API keys and secrets.
- Add defensive error handling around external APIs.
- Add lightweight tests where practical for new utility modules.
- Keep all new code modular and easy to disable via config flags.

---

## Bash Permission Policy
Claude Code is explicitly allowed to run bash commands for normal repository work **without asking for permission each time**, provided the operations remain inside this project and are relevant to the approved tasks.

### Allowed Bash Operations
Claude Code may directly run commands for:

#### File and folder creation
- `mkdir -p ...`
- `touch ...`
- `cp ...`
- `mv ...`
- `cat > file <<'EOF' ... EOF`
- `echo ... > file`
- `echo ... >> file`

#### File editing and replacement
- `sed -i ...`
- `python - <<'PY' ... PY`
- `perl -0pi -e ...`
- overwriting existing project files when required for approved implementation work

#### File deletion and cleanup
- `rm -f ...`
- `rm -rf ...`
- deleting temporary files
- deleting obsolete duplicate files
- deleting cache files such as `__pycache__/`, `.pytest_cache/`, temporary logs, and generated temporary outputs

#### Search and inspection
- `ls`
- `find`
- `grep`
- `rg`
- `tree`
- `cat`
- `head`
- `tail`

#### Python environment and dependency workflow
- `pip install ...`
- `pip uninstall ...`
- `python script.py`
- `pytest`
- `streamlit run app.py`

#### Git-safe local development operations
- `git status`
- `git diff`
- `git add ...`
- `git restore ...`

---

## Pre-Approved Command Patterns
Claude Code may use these patterns without asking for confirmation:

### Create folders and clean caches
```bash
mkdir -p retrievers feedback training evaluation models/verification data/verification_dataset data/human_feedback
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

### Copy, move, and remove files
```bash
cp existing_file.py backup_existing_file.py
mv old_module.py new_module.py
rm -f obsolete_script.py
rm -rf temp_outputs/
```

### Create a new file with heredoc
```bash
cat > configs/verifier_thresholds.py <<'EOF'
# file contents
EOF
```

### Edit a file safely with Python
```bash
python - <<'PY'
from pathlib import Path
path = Path("app.py")
text = path.read_text()
text = text.replace("old", "new")
path.write_text(text)
PY
```

### Quick in-place replacement
```bash
sed -i 's/old_text/new_text/g' app.py
```

---

## Files Claude Code May Create
Claude Code may create any implementation files needed under these folders:

- `agents/`
- `configs/`
- `retrievers/`
- `src/`
- `training/`
- `evaluation/`
- `feedback/`
- `models/`
- `data/`
- `tests/`
- `ui/`
- `logs/`

Claude Code may also create:
- `.env.example`
- `requirements.txt` updates
- `README.md` updates
- helper scripts for evaluation, calibration, or migration

---

## Files Claude Code May Edit
Claude Code may edit these existing files without repeated approval when required for the approved roadmap:

- `app.py`
- `agents/verifier_agent.py`
- `agents/assembler_agent.py`
- `agents/summariser_agent.py`
- `agents/claim_verifier.py`
- `graph/workflow_graph.py`
- `graph/baseline_graph.py`
- `src/mabselector.py`
- `src/cost_calculator.py`
- `configs/prompts.py`
- any new config or helper module created during this implementation phase

---

## Files Claude Code May Delete
Claude Code may delete only the following without asking each time:

- temporary debug files
- temporary notebooks used only for testing
- duplicate obsolete modules already replaced by canonical versions
- cache folders like `__pycache__/`
- `.pyc` files
- generated temporary logs and scratch outputs

Claude Code should **not** delete core project files unless they are clearly obsolete duplicates and have been replaced safely.

---

## Implementation Priority Order
Claude Code should implement in this order unless a dependency requires adjustment:

1. Expanded Database Integration
2. Real-Time Human Feedback Loop
3. Confidence Calibration Study
4. Fine-Tuned Verification Models

Reasoning:
- Better retrieval improves the whole pipeline first.
- Human feedback creates labelled data.
- Calibration improves evaluation rigor.
- Fine-tuning should build on richer labels and better data.

---

## Deliverables Expected from Claude Code
For each major change, Claude Code should aim to provide:
- code implementation
- updated config files
- brief inline documentation/comments where needed
- bash commands used
- any required migration/setup instructions
- sample test commands
- notes on how the change affects evaluation and Chapter 5 reporting

---

## Preferred Output Style
When working on this repository, Claude Code should:
- be concise but technically complete
- make atomic edits when possible
- explain why a file is being created or changed
- preserve current behavior unless the task explicitly changes it
- surface risks before major architectural changes

---

## Summary for Claude Code
You are authorized to extend this project toward production readiness by implementing:
- fine-tuned verification models
- expanded multi-database academic retrieval
- real-time human feedback capture
- confidence calibration and ROC-based evaluation

You may create, edit, move, and delete relevant project files using bash and Python scripts without repeatedly asking for confirmation, as long as the work stays within this approved scope.
