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
    Answer: verifier output → assembler → final clean review.

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
) -> dict:
    """
    Produce a simple change log comparing draft and final review.

    Parameters
    ----------
    draft               : original draft review text
    final               : cleaned final review text
    hallucinated_citations : list of hallucinated Citation objects

    Returns
    -------
    dict with change statistics
    """
    draft_words = len(draft.split())
    final_words = len(final.split())
    word_diff   = draft_words - final_words

    hallucinated_refs = [
        c.raw_reference
        for c in hallucinated_citations
        if c.valid is False
    ]

    removed_from_final = [
        ref for ref in hallucinated_refs
        if ref.split(",")[0].split("(")[-1].strip() not in final
    ]

    return {
        "draft_word_count":       draft_words,
        "final_word_count":       final_words,
        "words_removed":          max(0, word_diff),
        "hallucinated_count":     len(hallucinated_refs),
        "hallucinated_refs":      hallucinated_refs,
        "likely_removed_refs":    removed_from_final,
        "review_shortened_by":    f"{max(0, word_diff)} words",
    }


# ---------------------------------------------------------------------------
# Core assembler function
# ---------------------------------------------------------------------------

def assemble_final_review(
    topic: str,
    draft_review: str,
    citations: list[Citation],
) -> dict:
    """
    Produce the final cleaned literature review using the Assembler prompt.

    Process:
        1. Format citation status list (VALID / PARTIAL / HALLUCINATED).
        2. Call LLM with frozen assembler prompt.
        3. Return final review text + change log + verified refs only.

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
    dict with keys:
        final_review     : str   — cleaned review text
        changes          : dict  — change log statistics
        verified_refs    : list  — only valid citation strings
        hallucinated_refs: list  — removed citation strings
        prompt_version   : str   — prompt version used
    """
    client = _get_groq_client()

    # Step 1: Format citation status for the prompt
    citation_status_str = _format_citation_status(citations)

    print("\n[AssemblerAgent] Citation status for assembler:")
    print(citation_status_str)

    # Step 2: Build the assembler prompt from frozen template
    prompt_template = get_prompt("assembler")
    prompt = prompt_template.format(
        topic=topic,
        draft_review=draft_review,
        citation_status_list=citation_status_str,
    )

    print(f"\n[AssemblerAgent] Assembling final review...")
    print(f"[AssemblerAgent] Draft length : {len(draft_review)} chars")
    print(f"[AssemblerAgent] Citations    : {len(citations)} total")

    # Step 3: Call LLM
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert academic editor. "
                        "Follow the instructions precisely. "
                        "Return only the final review text and "
                        "reference list — no meta-commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=1500,
            temperature=0.2,  # low temperature = precise editing
        )
        final_review = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[AssemblerAgent] ERROR during LLM call: {e}")
        # Fallback: return draft as-is
        final_review = draft_review

    # Step 4: Build change log
    hallucinated = [c for c in citations if c.valid is False]
    changes = _count_changes(draft_review, final_review, hallucinated)

    # Step 5: Separate valid and hallucinated refs
    verified_refs = [
        c.raw_reference
        for c in citations
        if c.valid is True
    ]
    hallucinated_refs = [
        c.raw_reference
        for c in citations
        if c.valid is False
    ]

    print(f"[AssemblerAgent] Final review length : {len(final_review)} chars")
    print(f"[AssemblerAgent] Words removed       : {changes['words_removed']}")
    print(f"[AssemblerAgent] Verified refs kept  : {len(verified_refs)}")
    print(f"[AssemblerAgent] Hallucinated removed: {len(hallucinated_refs)}")

    from configs.prompts import PROMPT_VERSION

    return {
        "final_review":      final_review,
        "changes":           changes,
        "verified_refs":     verified_refs,
        "hallucinated_refs": hallucinated_refs,
        "prompt_version":    PROMPT_VERSION,
    }


# ---------------------------------------------------------------------------
# Helper: save assembler output to JSON log
# ---------------------------------------------------------------------------

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