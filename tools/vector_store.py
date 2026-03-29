import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.logger import get_logger

logger = get_logger("VectorStore")
model = SentenceTransformer("all-MiniLM-L6-v2")

class SimpleVectorStore:

    def __init__(self):
        self.index = None
        self.papers = []
        self.dimension = 384

    def add_papers(self, papers: list[dict]):
        texts = [f"{p['title']}" for p in papers]
        vectors = model.encode(texts, convert_to_numpy=True).astype("float32")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(vectors)
        self.papers = papers
        logger.info(f"Indexed {len(papers)} papers")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.index:
            return []
        query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
        _, indices = self.index.search(query_vec, top_k)
        return [self.papers[i] for i in indices[0] if i < len(self.papers)]
