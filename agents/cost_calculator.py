"""
Cost calculator for Groq API usage.
Provides model-specific per-token pricing and a function to compute cost
from actual token counts returned by the API.
"""

# Groq pricing per token (USD) — verified from groq.com/pricing
GROQ_PRICING = {
    "llama-3.3-70b-versatile": {
        "input": 0.00000059,   # $0.59 / 1M tokens
        "output": 0.00000079,  # $0.79 / 1M tokens
    },
    "llama-3.1-8b-instant": {
        "input": 0.00000005,   # $0.05 / 1M tokens
        "output": 0.00000008,  # $0.08 / 1M tokens
    },
    "llama3-8b-8192": {
        "input": 0.00000020,   # $0.20 / 1M tokens
        "output": 0.00000020,  # $0.20 / 1M tokens
    },
    # Fallback (use versatile pricing as default)
    "_default": {
        "input": 0.00000059,
        "output": 0.00000079,
    },
}


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
) -> float:
    """
    Compute actual cost (USD) for a Groq API call.

    Parameters
    ----------
    prompt_tokens : int
        Number of input tokens used.
    completion_tokens : int
        Number of output tokens generated.
    model : str
        Model name as used in the pipeline (e.g. "llama-3.3-70b-versatile").

    Returns
    -------
    float
        Estimated cost in USD, rounded to 9 decimal places.
    """
    prices = GROQ_PRICING.get(model, GROQ_PRICING["_default"])
    cost = (prompt_tokens * prices["input"]) + (completion_tokens * prices["output"])
    return round(cost, 9)


def format_cost(cost: float) -> str:
    """Format cost for display (5 decimal places as requested)."""
    return f"${cost:.5f} USD"
