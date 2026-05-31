"""
Local embedding wrapper using sentence-transformers.
Downloads models automatically from HuggingFace on first use.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer


ModelName = Literal[
    "sentence-transformers/all-MiniLM-L6-v2",    # 384d, fast
    "sentence-transformers/all-mpnet-base-v2",    # 768d, balanced
    "sentence-transformers/all-roberta-large-v1", # 1024d, accurate
]

_MODEL_INFO = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "dimensions": 384,
        "description": "Lightweight, fast. Good for prototyping and real-time search.",
        "size_mb": 22,
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "dimensions": 768,
        "description": "Balanced quality/speed. Best general-purpose choice.",
        "size_mb": 420,
    },
    "sentence-transformers/all-roberta-large-v1": {
        "dimensions": 1024,
        "description": "High accuracy. Best for quality-sensitive applications.",
        "size_mb": 1300,
    },
}


class LocalEmbedder:
    """Wrapper around sentence-transformers for local embedding generation."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model: SentenceTransformer | None = None

        info = _MODEL_INFO.get(model_name, {})
        self.dimensions = info.get("dimensions", None)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"Loading model: {self.model_name} (downloading if needed)...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            if self.dimensions is None:
                self.dimensions = self._model.get_sentence_embedding_dimension()
            print(f"Model loaded. Dimensions: {self.dimensions}")
        return self._model

    def embed(self, texts: str | list[str]) -> np.ndarray:
        """Embed one or more texts. Returns float32 numpy array."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_with_timing(self, texts: list[str]) -> tuple[np.ndarray, float]:
        """Embed texts and return (embeddings, seconds_elapsed)."""
        start = time.perf_counter()
        embeddings = self.embed(texts)
        elapsed = time.perf_counter() - start
        return embeddings, elapsed

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts (range: -1 to 1)."""
        vecs = self.embed([text_a, text_b])
        return float(np.dot(vecs[0], vecs[1]))

    def most_similar(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Return top-k most similar candidates to query."""
        query_vec = self.embed(query)[0]
        candidate_vecs = self.embed(candidates)

        scores = candidate_vecs @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(candidates[i], float(scores[i])) for i in top_indices]

    @classmethod
    def model_info(cls) -> dict:
        return _MODEL_INFO

    def __repr__(self) -> str:
        return (
            f"LocalEmbedder(model='{self.model_name}', "
            f"dimensions={self.dimensions}, device='{self.device}')"
        )
