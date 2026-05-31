"""
Qdrant wrapper with educational helpers.
Simplifies common operations for learning purposes.
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)


class QdrantManager:
    """
    Educational wrapper around QdrantClient.

    Provides simplified APIs for common RAG operations while exposing
    the underlying client for advanced usage.
    """

    DISTANCE_MAP = {
        "cosine": Distance.COSINE,
        "dot": Distance.DOT,
        "euclidean": Distance.EUCLID,
        "manhattan": Distance.MANHATTAN,
    }

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: str | None = None,
    ):
        self.host = host
        self.port = port
        self.client = QdrantClient(host=host, port=port, api_key=api_key)
        print(f"Connected to Qdrant at {host}:{port}")
        self._print_version()

    def _print_version(self) -> None:
        try:
            existing = self.list_collections()
            print(f"Existing collections: {existing if existing else '(none)'}")
        except Exception as e:
            print(f"Warning: Could not fetch collections: {e}")

    def create_collection(
        self,
        name: str,
        dimensions: int,
        distance: str = "cosine",
        on_disk: bool = False,
        with_scalar_quantization: bool = False,
    ) -> None:
        """Create a collection. Deletes existing collection with same name first."""
        dist = self.DISTANCE_MAP.get(distance.lower(), Distance.COSINE)

        quantization_config = None
        if with_scalar_quantization:
            quantization_config = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )

        if self.client.collection_exists(name):
            self.client.delete_collection(name)

        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=dimensions,
                distance=dist,
                on_disk=on_disk,
            ),
            quantization_config=quantization_config,
        )
        print(f"Collection '{name}' created ({dimensions}d, distance={distance})")

    def upsert(
        self,
        collection_name: str,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]] | None = None,
        ids: list[str | int] | None = None,
        batch_size: int = 100,
    ) -> int:
        """Insert or update points. Returns number of points inserted."""
        n = len(vectors)
        if payloads is None:
            payloads = [{} for _ in range(n)]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]

        total_upserted = 0
        for i in range(0, n, batch_size):
            batch_vectors = vectors[i : i + batch_size]
            batch_payloads = payloads[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            points = [
                PointStruct(id=pid, vector=vec, payload=payload)
                for pid, vec, payload in zip(batch_ids, batch_vectors, batch_payloads)
            ]
            self.client.upsert(collection_name=collection_name, points=points)
            total_upserted += len(points)

        return total_upserted

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
        filter_payload: dict[str, Any] | None = None,
        with_payload: bool = True,
    ) -> list[dict[str, Any]]:
        """Semantic search. Returns list of {id, score, payload} dicts."""
        query_filter = None
        if filter_payload:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_payload.items()
            ]
            query_filter = Filter(must=conditions)

        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=with_payload,
        ).points

        return [
            {"id": r.id, "score": r.score, "payload": r.payload or {}}
            for r in results
        ]

    def delete_collection(self, name: str) -> None:
        self.client.delete_collection(name)
        print(f"Collection '{name}' deleted")

    def collection_info(self, name: str) -> dict:
        info = self.client.get_collection(name)
        return {
            "name": name,
            "points_count": info.points_count,
            "status": str(info.status),
            "vector_size": info.config.params.vectors.size if hasattr(info.config.params.vectors, "size") else None,
        }

    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.get_collections().collections]

    def __repr__(self) -> str:
        collections = self.list_collections()
        return f"QdrantManager({self.host}:{self.port}, collections={collections})"
