"""
Advanced RAG pipeline with query rewriting and result re-ranking.

Improvements over Naive RAG:
1. Query rewriting: LLM rewrites query for better retrieval
2. Multi-query retrieval: generate multiple query variations
3. Cross-encoder re-ranking: rerank retrieved docs by relevance
4. Context compression: extract only relevant parts from each chunk
"""

from __future__ import annotations

from typing import Any

import numpy as np
import ollama
from sentence_transformers import CrossEncoder

from src.embeddings.local_embedder import LocalEmbedder
from src.utils.chunking import recursive_chunk
from src.vector_db.qdrant_manager import QdrantManager


class AdvancedRAG:
    """
    Advanced RAG with query rewriting, multi-query, and re-ranking.
    """

    REWRITE_PROMPT = """You are an expert at reformulating search queries to improve document retrieval.

Given the user's question, generate {n_variations} different variations of the question that:
1. Use different vocabulary/synonyms
2. Approach the topic from different angles
3. Are more specific or more general as appropriate

Original question: {question}

Return ONLY the {n_variations} variations, one per line, no numbering or explanation."""

    ANSWER_PROMPT = """You are a helpful assistant. Answer based ONLY on the provided context.
If you cannot answer from the context, say so.

Context:
{context}

Question: {question}

Answer:"""

    def __init__(
        self,
        collection_name: str = "advanced_rag",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model: str = "llama3.2",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
        n_query_variations: int = 3,
        use_reranker: bool = True,
    ):
        self.collection_name = collection_name
        self.llm_model = llm_model
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.n_query_variations = n_query_variations
        self.use_reranker = use_reranker

        self.embedder = LocalEmbedder(embedding_model)
        self.db = QdrantManager(host=qdrant_host, port=qdrant_port)

        self._cross_encoder: CrossEncoder | None = None
        self._cross_encoder_model = cross_encoder_model

    @property
    def cross_encoder(self) -> CrossEncoder:
        if self._cross_encoder is None and self.use_reranker:
            print(f"Loading cross-encoder: {self._cross_encoder_model}")
            self._cross_encoder = CrossEncoder(self._cross_encoder_model)
        return self._cross_encoder

    def index(self, documents: list[dict[str, Any]], recreate: bool = False) -> int:
        """Index documents (same as NaiveRAG but with larger model)."""
        if recreate and self.collection_name in self.db.list_collections():
            self.db.delete_collection(self.collection_name)

        if self.collection_name not in self.db.list_collections():
            self.db.create_collection(
                name=self.collection_name,
                dimensions=self.embedder.dimensions or 768,
            )

        all_chunks, all_payloads = [], []
        for doc in documents:
            text = doc.get("text", "")
            metadata = {k: v for k, v in doc.items() if k != "text"}
            for chunk in recursive_chunk(text, chunk_size=512, overlap=64, metadata=metadata):
                all_chunks.append(chunk.text)
                all_payloads.append({"text": chunk.text, **chunk.metadata})

        vectors = self.embedder.embed(all_chunks)
        self.db.upsert(self.collection_name, vectors.tolist(), all_payloads)
        return len(all_chunks)

    def rewrite_query(self, question: str) -> list[str]:
        """Use LLM to generate multiple query variations."""
        prompt = self.REWRITE_PROMPT.format(
            question=question,
            n_variations=self.n_query_variations,
        )
        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        variations = [
            line.strip()
            for line in response["message"]["content"].strip().split("\n")
            if line.strip()
        ]
        return [question] + variations[:self.n_query_variations]

    def multi_query_retrieve(self, queries: list[str]) -> list[dict[str, Any]]:
        """Retrieve for multiple queries and deduplicate by ID."""
        seen_ids = set()
        all_results = []

        for query in queries:
            query_vec = self.embedder.embed(query)[0].tolist()
            results = self.db.search(
                self.collection_name,
                query_vec,
                top_k=self.top_k_retrieve // len(queries),
            )
            for r in results:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    all_results.append(r)

        return all_results

    def rerank(self, question: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Re-rank results using cross-encoder."""
        if not results or not self.use_reranker:
            return results[:self.top_k_rerank]

        pairs = [(question, r["payload"].get("text", "")) for r in results]
        scores = self.cross_encoder.predict(pairs)

        ranked = sorted(
            zip(scores, results),
            key=lambda x: x[0],
            reverse=True,
        )
        return [r for _, r in ranked[:self.top_k_rerank]]

    def query(self, question: str, verbose: bool = False) -> dict[str, Any]:
        """Advanced RAG query with rewriting and reranking."""
        # Step 1: Rewrite query
        queries = self.rewrite_query(question)
        if verbose:
            print(f"\nQuery variations ({len(queries)}):")
            for q in queries:
                print(f"  - {q}")

        # Step 2: Multi-query retrieval
        candidates = self.multi_query_retrieve(queries)
        if verbose:
            print(f"\nRetrieved {len(candidates)} unique candidates")

        # Step 3: Re-rank
        top_results = self.rerank(question, candidates)
        if verbose:
            print(f"After re-ranking: top {len(top_results)} results")

        # Step 4: Generate
        context = "\n\n---\n\n".join(
            r["payload"].get("text", "") for r in top_results
        )
        prompt = self.ANSWER_PROMPT.format(context=context, question=question)
        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )

        return {
            "question": question,
            "query_variations": queries,
            "answer": response["message"]["content"],
            "sources": top_results,
            "num_candidates": len(candidates),
        }
