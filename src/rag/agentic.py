"""
Agentic RAG — LLM decides when and what to retrieve.

Uses a simple agent loop pattern (without LangGraph dependency in src/).
For full LangGraph implementation, see: 04_rag_architectures/04_agentic_rag.ipynb

The agent:
1. Receives a question
2. Decides if it needs to retrieve (or if it can answer from context)
3. Retrieves and evaluates relevance
4. Rewrites query if retrieval was poor
5. Generates answer or loops back
"""

from __future__ import annotations

from typing import Any

import ollama

from src.embeddings.local_embedder import LocalEmbedder
from src.vector_db.qdrant_manager import QdrantManager


DECIDE_PROMPT = """You are an intelligent assistant. Given a question and the conversation history,
decide if you need to search for more information or if you can answer directly.

Question: {question}
Current context (if any): {context}

Respond with EXACTLY one of:
- "RETRIEVE: <search_query>" if you need to search (provide an optimized search query)
- "ANSWER: <your_answer>" if you can answer with the current context

Your decision:"""

EVALUATE_PROMPT = """Evaluate if the retrieved information is sufficient to answer the question.

Question: {question}
Retrieved information:
{retrieved}

Is this sufficient? Respond with:
- "SUFFICIENT" if you can answer the question
- "INSUFFICIENT: <reason> | REWRITE: <better_query>" if you need better information"""


class AgenticRAG:
    """
    Simple agentic RAG with a decide-retrieve-evaluate loop.

    The LLM acts as an agent deciding:
    - Whether retrieval is needed
    - Whether retrieved content is relevant
    - Whether to retry with a different query
    """

    def __init__(
        self,
        collection_name: str = "agentic_rag",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "llama3.2",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        max_iterations: int = 3,
        top_k: int = 5,
    ):
        self.collection_name = collection_name
        self.llm_model = llm_model
        self.max_iterations = max_iterations
        self.top_k = top_k

        self.embedder = LocalEmbedder(embedding_model)
        self.db = QdrantManager(host=qdrant_host, port=qdrant_port)

    def _retrieve(self, query: str) -> list[dict[str, Any]]:
        query_vec = self.embedder.embed(query)[0].tolist()
        return self.db.search(self.collection_name, query_vec, top_k=self.top_k)

    def _llm(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()

    def query(self, question: str, verbose: bool = False) -> dict[str, Any]:
        """Agentic RAG loop: decide → retrieve → evaluate → answer."""
        context = ""
        all_sources = []
        iteration_log = []

        for iteration in range(self.max_iterations):
            # Step 1: Agent decides whether to retrieve or answer
            decision = self._llm(
                DECIDE_PROMPT.format(question=question, context=context or "(none)")
            )

            if verbose:
                print(f"\n[Iteration {iteration + 1}] Agent decision: {decision[:100]}")

            iteration_log.append({"iteration": iteration + 1, "decision": decision})

            if decision.startswith("ANSWER:"):
                answer = decision[len("ANSWER:"):].strip()
                return {
                    "question": question,
                    "answer": answer,
                    "sources": all_sources,
                    "iterations": iteration + 1,
                    "log": iteration_log,
                    "terminated": "direct_answer",
                }

            elif decision.startswith("RETRIEVE:"):
                search_query = decision[len("RETRIEVE:"):].strip()

                # Step 2: Retrieve
                results = self._retrieve(search_query)
                all_sources.extend(results)

                retrieved_text = "\n\n".join(
                    r["payload"].get("text", "") for r in results
                )

                # Step 3: Evaluate retrieved content
                evaluation = self._llm(
                    EVALUATE_PROMPT.format(
                        question=question,
                        retrieved=retrieved_text[:2000],
                    )
                )

                if verbose:
                    print(f"  Retrieved {len(results)} docs. Evaluation: {evaluation[:100]}")

                if evaluation.startswith("SUFFICIENT"):
                    context = retrieved_text
                    break
                elif "REWRITE:" in evaluation:
                    rewrite_part = evaluation.split("REWRITE:")[-1].strip()
                    context = retrieved_text
                    question = rewrite_part  # Use rewritten query in next iteration
            else:
                # Fallback: treat as answer
                break

        # Final generation with accumulated context
        if context:
            final_prompt = f"""Answer this question using the context provided.

Context:
{context[:3000]}

Question: {question}

Answer:"""
            answer = self._llm(final_prompt)
        else:
            answer = "I could not find sufficient information to answer this question."

        return {
            "question": question,
            "answer": answer,
            "sources": all_sources,
            "iterations": len(iteration_log),
            "log": iteration_log,
            "terminated": "max_iterations",
        }
