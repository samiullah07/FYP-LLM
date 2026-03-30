# evaluation/cost_tracker.py
"""
Cost and latency tracker for the literature review pipeline.

Groq pricing (as of 2026):
    llama-3.3-70b-versatile: $0.59 per 1M input tokens
                              $0.79 per 1M output tokens
"""

# Groq pricing per 1M tokens (USD)
GROQ_PRICING = {
    "llama-3.3-70b-versatile": {
        "input":  0.59,
        "output": 0.79,
    },
    "llama-3.1-8b-instant": {
        "input":  0.05,
        "output": 0.08,
    },
    "mixtral-8x7b-32768": {
        "input":  0.24,
        "output": 0.24,
    },
}

# Average tokens per pipeline stage (estimated)
STAGE_TOKENS = {
    "planner":    {"input": 200,   "output": 150},
    "search":     {"input": 0,     "output": 0},    # no LLM call
    "summariser": {"input": 2000,  "output": 600},
    "verifier":   {"input": 500,   "output": 100},  # per citation
    "assembler":  {"input": 1500,  "output": 500},
}


def estimate_cost(
    model:           str,
    token_estimate:  int,
    num_citations:   int = 8,
) -> dict:
    """
    Estimate cost of one full pipeline run.

    Parameters
    ----------
    model          : LLM model name
    token_estimate : total tokens estimated from text length
    num_citations  : number of citations verified

    Returns
    -------
    dict with cost breakdown
    """
    pricing = GROQ_PRICING.get(model, GROQ_PRICING["llama-3.3-70b-versatile"])

    # Estimate input/output split (roughly 70/30)
    input_tokens  = int(token_estimate * 0.70)
    output_tokens = int(token_estimate * 0.30)

    # Add verification cost
    verifier_input  = STAGE_TOKENS["verifier"]["input"] * num_citations
    verifier_output = STAGE_TOKENS["verifier"]["output"] * num_citations
    input_tokens  += verifier_input
    output_tokens += verifier_output

    input_cost  = (input_tokens  / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost  = input_cost + output_cost

    return {
        "model":          model,
        "input_tokens":   input_tokens,
        "output_tokens":  output_tokens,
        "total_tokens":   input_tokens + output_tokens,
        "input_cost_usd": round(input_cost,  6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(total_cost,  6),
        "cost_per_1000_runs_usd": round(total_cost * 1000, 2),
    }


def format_cost_report(cost: dict, latency: float) -> str:
    """Format cost dict as a readable report string."""
    return (
        f"Cost Report:\n"
        f"  Model         : {cost['model']}\n"
        f"  Input tokens  : {cost['input_tokens']:,}\n"
        f"  Output tokens : {cost['output_tokens']:,}\n"
        f"  Total tokens  : {cost['total_tokens']:,}\n"
        f"  Cost (USD)    : ${cost['total_cost_usd']:.6f}\n"
        f"  Cost per 1000 : ${cost['cost_per_1000_runs_usd']:.2f}\n"
        f"  Latency       : {latency}s\n"
    )