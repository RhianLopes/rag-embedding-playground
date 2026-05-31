"""
Naive RAG pipeline — the simplest possible implementation.

Pipeline:
  Indexing:  Documents → Chunk → Embed → Store in Qdrant
  Querying:  Query → Embed → Search Qdrant → Build Prompt → Ollama → Response
"""

from __future__ import annotations

from typing import Any

import ollama

from src.embeddings.local_embedder import LocalEmbedder
from src.utils.chunking import recursive_chunk
from src.vector_db.qdrant_manager import QdrantManager


class NaiveRAG:
    """
    Minimal RAG implementation for learning purposes.
    Demonstrates the core pipeline without optimizations.
    """

    DEFAULT_PROMPT = """You are a helpful assistant. Answer the question based ONLY on the context provided below.
If the answer cannot be found in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    def __init__(
        self,
        collection_name: str = "naive_rag",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "llama3.2",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
    ):
        self.collection_name = collection_name
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.embedder = LocalEmbedder(embedding_model)
        self.db = QdrantManager(host=qdrant_host, port=qdrant_port)

    def index(self, documents: list[dict[str, Any]], recreate: bool = False) -> int:
        """
        Index documents into Qdrant.

        Args:
            documents: List of dicts with 'text' and optional metadata keys
            recreate: Whether to recreate the collection if it exists

        Returns:
            Total number of chunks indexed
        """
        if recreate:
            if self.collection_name in self.db.list_collections():
                self.db.delete_collection(self.collection_name)

        if self.collection_name not in self.db.list_collections():
            self.db.create_collection(
                name=self.collection_name,
                dimensions=self.embedder.dimensions or 384,
            )

        all_chunks = []
        all_payloads = []

        for doc in documents:
            text = doc.get("text", "")
            metadata = {k: v for k, v in doc.items() if k != "text"}

            chunks = recursive_chunk(
                text,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
                metadata=metadata,
            )

            for chunk in chunks:
                all_chunks.append(chunk.text)
                all_payloads.append({"text": chunk.text, **chunk.metadata})

        if not all_chunks:
            return 0

        print(f"Embedding {len(all_chunks)} chunks...")
        vectors = self.embedder.embed(all_chunks)

        self.db.upsert(
            collection_name=self.collection_name,
            vectors=vectors.tolist(),
            payloads=all_payloads,
        )

        print(f"Indexed {len(all_chunks)} chunks into '{self.collection_name}'")
        return len(all_chunks)

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Retrieve top-k relevant chunks for a query."""
        query_vector = self.embedder.embed(query)[0].tolist()
        return self.db.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=self.top_k,
        )

    def generate(self, query: str, context_results: list[dict]) -> str:
        """Generate an answer given a query and retrieved context."""
        context = "\n\n---\n\n".join(
            f"[Score: {r['score']:.3f}]\n{r['payload'].get('text', '')}"
            for r in context_results
        )

        prompt = self.DEFAULT_PROMPT.format(context=context, question=query)

        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    def query(self, question: str, verbose: bool = False) -> dict[str, Any]:
        """
        End-to-end RAG query: retrieve + generate.

        Returns dict with answer, sources, and metadata.
        """
        results = self.retrieve(question)

        if verbose:
            print(f"\nRetrieved {len(results)} chunks:")
            for i, r in enumerate(results):
                print(f"  [{i+1}] Score: {r['score']:.3f} | {r['payload'].get('text', '')[:80]}...")

        answer = self.generate(question, results)

        return {
            "question": question,
            "answer": answer,
            "sources": results,
            "num_sources": len(results),
        }
