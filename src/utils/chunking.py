"""
Text chunking strategies for RAG pipelines.

Implements and compares four main approaches:
1. Fixed-size chunking
2. Recursive character splitting
3. Semantic chunking (embedding-based)
4. Overlap chunking
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Chunk:
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return len(self.text.split())

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(idx={self.index}, tokens≈{self.token_count}, '{preview}...')"


def fixed_size_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Split text into fixed-size character chunks with optional overlap.

    Simple but may break sentences mid-way. Good for initial prototyping.
    """
    metadata = metadata or {}
    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=idx,
                    start_char=start,
                    end_char=end,
                    metadata=metadata.copy(),
                )
            )
            idx += 1

        start = end - overlap if end < len(text) else end

    return chunks


def recursive_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    separators: list[str] | None = None,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Recursively split on separators in order of priority.

    Priority: double newline → newline → sentence → space → character
    This is the most robust general-purpose chunking strategy.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    metadata = metadata or {}
    raw_chunks = _recursive_split(text, chunk_size, separators)

    # Apply overlap by including tail of previous chunk
    final_chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        if i > 0 and overlap > 0:
            prev_text = raw_chunks[i - 1]
            overlap_text = prev_text[-overlap:] if len(prev_text) > overlap else prev_text
            chunk_text = overlap_text + chunk_text

        start_char = text.find(chunk_text[:30]) if chunk_text else 0
        final_chunks.append(
            Chunk(
                text=chunk_text.strip(),
                index=i,
                start_char=max(0, start_char),
                end_char=start_char + len(chunk_text),
                metadata=metadata.copy(),
            )
        )

    return final_chunks


def chunk_with_overlap(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Sliding window chunking with word-boundary awareness."""
    metadata = metadata or {}
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    idx = 0

    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        if chunk_text.strip():
            start_char = sum(len(w) + 1 for w in words[:start])
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=idx,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    metadata=metadata.copy(),
                )
            )
            idx += 1

        if end == len(words):
            break

    return chunks


def semantic_chunk(
    text: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
    similarity_threshold: float = 0.8,
    min_chunk_size: int = 100,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Split text based on semantic similarity between sentences.

    Splits where semantic similarity drops below threshold.
    Requires an embedding function as input.

    Args:
        text: Input text
        embed_fn: Function that takes list[str] and returns list[list[float]]
        similarity_threshold: Split when similarity drops below this value
        min_chunk_size: Minimum characters per chunk
        metadata: Optional metadata to attach to chunks
    """
    import numpy as np

    metadata = metadata or {}

    sentences = _split_into_sentences(text)
    if len(sentences) <= 1:
        return [Chunk(text=text.strip(), index=0, start_char=0, end_char=len(text), metadata=metadata)]

    embeddings = embed_fn(sentences)
    embeddings = np.array(embeddings)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms

    # Compute consecutive similarities
    sims = np.einsum("id,id->i", normalized[:-1], normalized[1:])

    # Find split points
    split_indices = [0]
    current_len = len(sentences[0])

    for i, sim in enumerate(sims):
        current_len += len(sentences[i + 1])
        if sim < similarity_threshold and current_len >= min_chunk_size:
            split_indices.append(i + 1)
            current_len = 0

    split_indices.append(len(sentences))

    chunks = []
    start_char = 0
    for idx in range(len(split_indices) - 1):
        chunk_sentences = sentences[split_indices[idx] : split_indices[idx + 1]]
        chunk_text = " ".join(chunk_sentences).strip()

        if chunk_text:
            end_char = start_char + len(chunk_text)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=idx,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=metadata.copy(),
                )
            )
            start_char = end_char + 1

    return chunks


def compare_strategies(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> dict:
    """
    Compare all chunking strategies on the same text.
    Returns summary statistics for each strategy.
    """
    strategies = {
        "fixed_size": fixed_size_chunk(text, chunk_size, overlap),
        "recursive": recursive_chunk(text, chunk_size, overlap),
        "with_overlap": chunk_with_overlap(text, chunk_size, overlap),
    }

    summary = {}
    for name, chunks in strategies.items():
        token_counts = [c.token_count for c in chunks]
        summary[name] = {
            "num_chunks": len(chunks),
            "avg_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
        }

    return summary


# ── Internal helpers ──────────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    pattern = r"(?<=[.!?])\s+"
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    separator = ""
    for sep in separators:
        if sep in text:
            separator = sep
            break

    if not separator:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(separator)
    result = []
    current = ""

    for part in parts:
        candidate = current + separator + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                result.append(current)
            if len(part) > chunk_size:
                result.extend(_recursive_split(part, chunk_size, separators[1:]))
                current = ""
            else:
                current = part

    if current:
        result.append(current)

    return result
