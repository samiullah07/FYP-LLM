# agents/assembler_agent.py
"""
Assembler Agent for the Literature Review Pipeline.

Responsibility:
    This is the FINAL agent in the pipeline.
    It takes:
        1. The draft literature review (from Summariser Agent)
        2. The verified citation list (from Verifier Agent)

    And produces:
        1. A cleaned final review with hallucinated citations
           removed or rephrased.
        2. A structured log of all changes made.
        3. Final citation list containing only verified references.

    This directly addresses the IPR feedback:
        "how verifier feedback is fed back to which agent"
    Answer: verifier output -> assembler -> final clean review.

    The assembler uses the frozen prompt from configs/prompts.py
    to ensure fair, symmetric, reproducible output.
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from groq import Groq

from src.config import settings
from src.models import Citation
from configs.prompts import get_prompt


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

def _get_groq_client() -> Groq:
    return Groq(api_key=settings.groq_api_key)


# ---------------------------------------------------------------------------
# Helper: format citation status list for the assembler prompt
# ---------------------------------------------------------------------------

def _format_citation_status(citations: list[Citation]) -> str:
    """
    Format the list of Citation objects into a readable string
    for the assembler prompt.

    Example output:
        [VALID]       (Smith et al., 2023)
        [HALLUCINATED](Chen, 2026) — No matching paper found
        [PARTIAL]     (Jones et al., 2022) — Year mismatch

    Parameters
    ----------
    citations : list[Citation]
        Citation objects from the Verifier Agent.

    Returns
    -------
    str
        Formatted citation status string.
    """
    if not citations:
        return "No citations to verify."

    lines = []
    for cit in citations:
        if cit.valid is True and cit.error_reason is None:
            status = "VALID"
        elif cit.valid is True and cit.error_reason:
            status = "PARTIAL"
        else:
            status = "HALLUCINATED"

        reason = f" — {cit.error_reason}" if cit.error_reason else ""
        lines.append(f"  [{status:<12}] {cit.raw_reference}{reason}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: count changes made by assembler
# ---------------------------------------------------------------------------

def _count_changes(
    draft: str,
    final: str,
    hallucinated_citations: list[Citation],
    assembler_changes: list[dict] | None = None,
) -> dict:
    """
    Produce a change log comparing draft and final review.

    Parameters
    ----------
    draft               : original draft review text
    final               : cleaned final review text
    hallucinated_citations : list of hallucinated Citation objects
    assembler_changes    : detailed list of sentence changes (optional)

    Returns
    -------
    dict with change statistics including actual words removed
    """
    draft_words = len(draft.split())
    final_words = len(final.split())

    # If we have detailed assembler changes, use them for accurate word count
    if assembler_changes:
        words_removed = sum(
            len(c.get("original_sentence", "").split())
            for c in assembler_changes
            if c.get("new_sentence") == "DROPPED"
        )
    else:
        words_removed = max(0, draft_words - final_words)

    hallucinated_refs = [
        c.raw_reference
        for c in hallucinated_citations
        if c.valid is False
    ]

    return {
        "draft_word_count":       draft_words,
        "final_word_count":       final_words,
        "words_removed":          words_removed,
        "hallucinated_count":     len(hallucinated_refs),
        "hallucinated_refs":      hallucinated_refs,
        "likely_removed_refs":    hallucinated_refs,
        "review_shortened_by":    f"{words_removed} words",
        "sentences_dropped":      len([c for c in (assembler_changes or []) if c.get("new_sentence") == "DROPPED"]),
        "sentences_rewritten":     len([c for c in (assembler_changes or []) if c.get("new_sentence") != "DROPPED"]),
        "assembler_changes":       assembler_changes or [],
    }


# ---------------------------------------------------------------------------
# Core assembler function
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: parse sentences and find citations in each sentence
# ---------------------------------------------------------------------------

def _find_citations_in_sentence(sentence: str, citations: list[Citation]) -> list[Citation]:
    """Find which citations are referenced in a sentence."""
    found = []
    sentence_lower = sentence.lower()
    for cit in citations:
        if cit.raw_reference.lower() in sentence_lower:
            found.append(cit)
    return found


def _rewrite_with_hedge(sentence: str, citations: list[Citation]) -> str:
    """Rewrite a sentence with hedging language for PARTIAL citations."""
    import re
    for cit in citations:
        raw = cit.raw_reference
        match = re.search(r"([A-Za-z\s&]+)\s*\((\d{4})\)", raw)
        if match:
            author = match.group(1).strip()
            year = match.group(2)
            hedge = f"There is emerging evidence from {author} ({year}) suggests that"
            sentence = re.sub(
                r"\(?\s*" + re.escape(author) + r".*?" + year + r"\)?",
                hedge,
                sentence,
                count=1
            )
    return sentence


# ---------------------------------------------------------------------------
# Core assembler function — NOW actually uses citation statuses
# ---------------------------------------------------------------------------

def assemble_final_review(
    topic: str,
    draft_review: str,
    citations: list[Citation],
) -> dict:
    """
    Produce the final cleaned literature review by processing sentences
    based on citation status.

    Process:
        1. Split draft into sentences.
        2. For each sentence, find which citations it references.
        3. If citation is HALLUCINATED: DROP sentence.
        4. If citation is PARTIAL: REWRITE with hedging.
        5. If citation is VALID: Keep unchanged.
        6. Log all changes to assembler_changes list.
        7. Join sentences back into final review.

    Parameters
    ----------
    topic        : str
        The original research topic.
    draft_review : str
        Draft review text from the Summariser Agent.
    citations    : list[Citation]
        Verified citations from the Verifier Agent.

    Returns
    -------
    dict with final_review, changes, verified_refs, hallucinated_refs
    """
    import re

    # Step 1: Split into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', draft_review) if s.strip()]
    if not sentences:
        sentences = [draft_review]

    final_sentences = []
    assembler_changes = []
    words_removed = 0

    print(f"\n[AssemblerAgent] Processing {len(sentences)} sentences...")
    print(f"[AssemblerAgent] Citations to check: {len(citations)}")

    for i, sent in enumerate(sentences):
        sent_citations = _find_citations_in_sentence(sent, citations)

        if not sent_citations:
            final_sentences.append(sent)
            continue

        # Check statuses of citations in this sentence
        has_hallucinated = any(
                (getattr(c, "status", None) == "HALLUCINATED") or
                (c.valid is False and c.error_reason != "PARTIAL")
                for c in sent_citations)
        has_partial = any(
                (getattr(c, "status", None) == "PARTIAL") or
                (c.valid is True and c.error_reason)
                for c in sent_citations)

        # HALLUCINATED citations: DROP sentence entirely
        if has_hallucinated:
            words_removed += len(sent.split())
            for c in sent_citations:
                if c.valid is False:
                    assembler_changes.append({
                        "original_sentence": sent,
                        "new_sentence": "DROPPED",
                        "citation_status": c.error_reason or "HALLUCINATED",
                        "citation": c.raw_reference,
                    })
            print(f"  [Assembler] DROPPED sentence {i+1} (HALLUCINATED citation)")
            continue

        # PARTIAL citations: REWRITE with hedging
        if has_partial:
            new_sent = _rewrite_with_hedge(sent, sent_citations)
            words_before = len(sent.split())
            words_after = len(new_sent.split())
            words_removed += max(0, words_before - words_after)
            final_sentences.append(new_sent)
            for c in sent_citations:
                if c.valid is True and c.error_reason:
                    assembler_changes.append({
                        "original_sentence": sent,
                        "new_sentence": new_sent,
                        "citation_status": c.error_reason,
                        "citation": c.raw_reference,
                    })
            print(f"  [Assembler] REWRITTEN sentence {i+1} (PARTIAL citation)")
            continue

        # VALID citations: keep unchanged
        final_sentences.append(sent)

    final_review = " ".join(final_sentences)

    # Step 2: Build change log
    hallucinated = [c for c in citations if c.valid is False]
    verified = [c for c in citations if c.valid is True]

    changes = _count_changes(
        draft_review, final_review, hallucinated, assembler_changes
    )

    # Step 3: Separate valid and hallucinated refs
    verified_refs = [c.raw_reference for c in verified]
    hallucinated_refs = [c.raw_reference for c in hallucinated]

    print(f"[AssemblerAgent] Final review length : {len(final_review)} chars")
    print(f"[AssemblerAgent] Words removed       : {changes['words_removed']}")
    print(f"[AssemblerAgent] Sentences dropped    : {changes['sentences_dropped']}")
    print(f"[AssemblerAgent] Sentences rewritten : {changes['sentences_rewritten']}")
    print(f"[AssemblerAgent] Verified refs kept  : {len(verified_refs)}")
    # Use the actual count of dropped sentences (which correspond to hallucinated citations)
    print(f"[AssemblerAgent] Hallucinated removed: {changes['sentences_dropped']}")

    from configs.prompts import PROMPT_VERSION

    return {
        "final_review":      final_review,
        "changes":           changes,
        "verified_refs":     verified_refs,
        "hallucinated_refs": hallucinated_refs,
        "prompt_version":    PROMPT_VERSION,
    }

def save_assembler_log(
    result: dict,
    topic: str,
    out_dir: Path,
    run_id: str,
) -> Path:
    """
    Save assembler output as a structured JSON log file.

    Parameters
    ----------
    result  : dict returned by assemble_final_review()
    topic   : str research topic
    out_dir : Path directory to save log
    run_id  : str unique run identifier (e.g. timestamp)

    Returns
    -------
    Path to saved JSON log file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "run_id":            run_id,
        "topic":             topic,
        "prompt_version":    result["prompt_version"],
        "final_review":      result["final_review"],
        "changes":           result["changes"],
        "verified_refs":     result["verified_refs"],
        "hallucinated_refs": result["hallucinated_refs"],
    }

    out_path = out_dir / f"assembler_log_{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"[AssemblerAgent] Log saved to: {out_path}")
    return out_path