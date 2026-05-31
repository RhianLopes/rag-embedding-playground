"""
Vector quantization utilities for demonstrating float type tradeoffs.

Shows how float32 → float16 → int8 → binary affects:
- Memory footprint
- Cosine similarity preservation
- Computation speed
"""

from __future__ import annotations

import struct
import sys
from typing import Literal

import numpy as np


FloatType = Literal["float32", "float16", "int8", "binary"]

DTYPE_MAP = {
    "float32": np.float32,
    "float16": np.float16,
    "int8": np.int8,
}

BYTES_PER_ELEMENT = {
    "float32": 4,
    "float16": 2,
    "int8": 1,
    "binary": 0.125,  # 1 bit = 1/8 byte
}


def quantize_vectors(
    vectors: np.ndarray,
    target_type: FloatType,
) -> np.ndarray:
    """
    Quantize float32 vectors to a smaller representation.

    Args:
        vectors: Input float32 array of shape (N, D)
        target_type: Target quantization type

    Returns:
        Quantized array (same shape, different dtype for non-binary;
        shape (N, D//8) of uint8 for binary)
    """
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    if target_type == "float32":
        return vectors.copy()

    if target_type == "float16":
        return vectors.astype(np.float16)

    if target_type == "int8":
        return _quantize_to_int8(vectors)

    if target_type == "binary":
        return _quantize_to_binary(vectors)

    raise ValueError(f"Unknown target type: {target_type}")


def dequantize_vectors(
    vectors: np.ndarray,
    source_type: FloatType,
) -> np.ndarray:
    """Convert quantized vectors back to float32 for similarity computation."""
    if source_type in ("float32", "float16"):
        return vectors.astype(np.float32)

    if source_type == "int8":
        return _dequantize_from_int8(vectors)

    if source_type == "binary":
        return _dequantize_from_binary(vectors)

    raise ValueError(f"Unknown source type: {source_type}")


def memory_usage_bytes(vectors: np.ndarray, dtype: FloatType) -> int:
    """Calculate memory usage for storing vectors in given dtype."""
    n, d = vectors.shape
    return int(n * d * BYTES_PER_ELEMENT[dtype])


def memory_comparison_table(vectors: np.ndarray) -> dict:
    """Return a dict with memory usage and reduction ratios for all dtypes."""
    n, d = vectors.shape
    baseline = memory_usage_bytes(vectors, "float32")

    table = {}
    for dtype in ("float32", "float16", "int8", "binary"):
        mem = memory_usage_bytes(vectors, dtype)
        table[dtype] = {
            "bytes": mem,
            "mb": mem / (1024**2),
            "gb": mem / (1024**3),
            "reduction_ratio": baseline / mem,
            "reduction_pct": (1 - mem / baseline) * 100,
        }
    return table


def cosine_similarity_preservation(
    vectors: np.ndarray,
    dtype: FloatType,
    sample_size: int = 1000,
) -> dict:
    """
    Measure how well cosine similarity is preserved after quantization.

    Returns dict with mean absolute error, correlation, and min/max error.
    """
    if len(vectors) > sample_size:
        idx = np.random.choice(len(vectors), sample_size, replace=False)
        vectors = vectors[idx]

    # Original similarities (float32)
    original_normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    original_sims = original_normed @ original_normed.T

    # Quantized similarities
    quantized = quantize_vectors(vectors, dtype)
    dequantized = dequantize_vectors(quantized, dtype)
    quant_normed = dequantized / (np.linalg.norm(dequantized, axis=1, keepdims=True) + 1e-9)
    quant_sims = quant_normed @ quant_normed.T

    # Compute errors (upper triangle only, no diagonal)
    mask = np.triu(np.ones_like(original_sims, dtype=bool), k=1)
    errors = np.abs(original_sims[mask] - quant_sims[mask])

    return {
        "dtype": dtype,
        "mean_abs_error": float(errors.mean()),
        "max_abs_error": float(errors.max()),
        "correlation": float(np.corrcoef(original_sims[mask], quant_sims[mask])[0, 1]),
        "preservation_pct": float((1 - errors.mean()) * 100),
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _quantize_to_int8(vectors: np.ndarray) -> np.ndarray:
    """Linearly map float32 values to int8 range [-127, 127]."""
    # Per-vector min-max scaling
    min_vals = vectors.min(axis=1, keepdims=True)
    max_vals = vectors.max(axis=1, keepdims=True)
    scale = np.maximum(np.abs(min_vals), np.abs(max_vals)) / 127.0
    scale = np.where(scale == 0, 1.0, scale)
    return np.clip(np.round(vectors / scale), -127, 127).astype(np.int8)


def _dequantize_from_int8(vectors: np.ndarray) -> np.ndarray:
    """Approximate dequantization (loses scale info — for demo purposes)."""
    return vectors.astype(np.float32) / 127.0


def _quantize_to_binary(vectors: np.ndarray) -> np.ndarray:
    """Convert to binary: 1 if value > 0, else 0. Pack into uint8."""
    binary = (vectors > 0).astype(np.uint8)
    # Pack 8 bits into 1 byte
    n, d = binary.shape
    packed_dim = (d + 7) // 8
    result = np.zeros((n, packed_dim), dtype=np.uint8)
    for i in range(d):
        result[:, i // 8] |= binary[:, i] << (7 - (i % 8))
    return result


def _dequantize_from_binary(packed: np.ndarray) -> np.ndarray:
    """Unpack binary vectors back to float32 (0.0 or 1.0)."""
    n, packed_dim = packed.shape
    d = packed_dim * 8
    result = np.zeros((n, d), dtype=np.float32)
    for i in range(d):
        result[:, i] = ((packed[:, i // 8] >> (7 - (i % 8))) & 1).astype(np.float32)
    return result
