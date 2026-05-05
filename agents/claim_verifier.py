#!/usr/bin/env python3

from typing import Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

@lru_cache(maxsize=1)
def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        print(f"Model load error: {str(e)}")
        return None


def compute_claim_similarity(claim_text: str, abstract: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    model = _get_model(model_name)
    if model is None:
        return 0.0  # Fallback score on model failure
    try:
        claim_vec = model.encode(claim_text, convert_to_numpy=True)
        abstract_vec = model.encode(abstract, convert_to_numpy=True)
        return cosine_similarity(claim_vec.reshape(1, -1), abstract_vec.reshape(1, -1))[0][0]
    except Exception as e:
        print(f"Similarity computation error: {str(e)}")
        return 0.0


def verify_claim_grounding(claim_text: str, paper_dict: Dict, threshold: float = 0.65) -> Dict:
    abstract = paper_dict.get("abstract")
    if not abstract:
        return {
            "verdict": "UNKNOWN",
            "score": 0.0,
            "reason": "no_abstract"
        }

    similarity = compute_claim_similarity(claim_text, abstract)

    if similarity >= threshold:
        return {
            "verdict": "GROUNDED",
            "score": round(similarity, 3),
            "reason": f"similarity={similarity:.3f} exceeds threshold"
        }
    elif similarity >= 0.35:
        return {
            "verdict": "WEAKLY_GROUNDED",
            "score": round(similarity, 3),
            "reason": f"similarity={similarity:.3f} below threshold"
        }
    else:
        return {
            "verdict": "UNGROUNDED",
            "score": round(similarity, 3),
            "reason": f"similarity={similarity:.3f} very low"
        }