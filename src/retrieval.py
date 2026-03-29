# src/retrieval.py
"""
Retrieval utilities for the literature review agent system.

This version uses a local SentenceTransformers model for embeddings,
so it does not depend on OpenAI quota.

It:
  - builds a FAISS index over Paper objects,
  - saves / loads the index and metadata,
  - supports semantic search over papers.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer  # local embeddings

from .config import settings
from .models import Paper


# Lazy-load global model to avoid repeated downloads
_embedding_model: SentenceTransformer | None = None


def _get_embeddings_model() -> SentenceTransformer:
    """
    Load or return a local embedding model.

    'all-MiniLM-L6-v2' is small, fast, and widely used for semantic similarity.[web:32]
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def build_vector_store(papers: List[Paper]) -> Tuple[faiss.IndexFlatIP, List[Paper]]:
    """
    Build a FAISS index from a list of Paper objects using local embeddings.
    """
    if not papers:
        raise ValueError("Cannot build vector store with an empty papers list.")

    texts = [f"{p.title}\n\n{p.abstract or ''}".strip() for p in papers]

    model = _get_embeddings_model()
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    # vectors: np.ndarray [n_papers, dim], already L2-normalised

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    return index, papers


def save_vector_store(index: faiss.IndexFlatIP, papers: List[Paper], base_path: Path) -> None:
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    faiss_path = base_path.with_suffix(".faiss")
    meta_path = base_path.with_suffix(".pkl")

    faiss.write_index(index, str(faiss_path))
    with open(meta_path, "wb") as f:
        pickle.dump(papers, f)


def load_vector_store(base_path: Path) -> Tuple[faiss.IndexFlatIP, List[Paper]]:
    base_path = Path(base_path)
    faiss_path = base_path.with_suffix(".faiss")
    meta_path = base_path.with_suffix(".pkl")

    if not faiss_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Vector store files not found at {faiss_path} and {meta_path}")

    index = faiss.read_index(str(faiss_path))
    with open(meta_path, "rb") as f:
        papers: List[Paper] = pickle.load(f)

    return index, papers


def search_similar_papers(
    index: faiss.IndexFlatIP,
    papers: List[Paper],
    query: str,
    k: int = 5,
) -> List[Paper]:
    """
    Retrieve top-k papers similar to a query using the local embedding model.
    """
    if not papers:
        return []

    model = _get_embeddings_model()
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    # q_vec: shape [1, dim], already normalised

    distances, indices = index.search(q_vec, k)
    idxs = indices[0]

    results: List[Paper] = []
    for i in idxs:
        if 0 <= i < len(papers):
            results.append(papers[i])

    return results


def default_index_path() -> Path:
    """
    Default base path for saving/loading the vector store.
    """
    return settings.data_dir / "processed" / "papers_index"