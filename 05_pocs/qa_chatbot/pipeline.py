"""
Q&A Chatbot PoC — Full RAG Pipeline with Conversation History

Usage:
    uv run python 05_pocs/qa_chatbot/pipeline.py
"""

import sys
import time
from pathlib import Path

import ollama
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.embeddings.local_embedder import LocalEmbedder
from src.utils.chunking import recursive_chunk
from src.vector_db.qdrant_manager import QdrantManager

console = Console()

COLLECTION = "qa_chatbot"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
TOP_K = 5

SYSTEM_PROMPT = """Voce e um assistente tecnico especializado em Machine Learning, RAG e Embeddings.

Regras:
1. Responda APENAS com base no contexto fornecido
2. Se nao souber, diga: "Nao encontrei essa informacao no contexto"
3. Seja conciso e tecnico
4. Cite a fonte quando possivel

Contexto disponivel:
{context}
"""

USER_PROMPT = "Historico da conversa:\n{history}\n\nPergunta atual: {question}"


class QAChatbot:
    def __init__(self):
        self.embedder = LocalEmbedder(EMBEDDING_MODEL)
        self.db = QdrantManager()
        self.conversation_history = []
        self._ensure_indexed()

    def _ensure_indexed(self) -> None:
        """Index documents if not already indexed."""
        if COLLECTION in self.db.list_collections():
            info = self.db.collection_info(COLLECTION)
            if info["points_count"] and info["points_count"] > 0:
                console.print(f"[green]Using existing collection '{COLLECTION}' ({info['points_count']} chunks)[/green]")
                return

        console.print("[yellow]Indexing documents...[/yellow]")
        docs_dir = Path("data/sample_docs")
        if not docs_dir.exists():
            docs_dir = Path("../../data/sample_docs")

        all_chunks = []
        all_payloads = []

        for filepath in docs_dir.glob("*.md"):
            text = filepath.read_text(encoding="utf-8")
            for chunk in recursive_chunk(text, chunk_size=400, overlap=50):
                all_chunks.append(chunk.text)
                all_payloads.append({
                    "text": chunk.text,
                    "source": filepath.name,
                })

        self.db.create_collection(name=COLLECTION, dimensions=384, distance="cosine")
        vectors = self.embedder.embed(all_chunks)
        self.db.upsert(COLLECTION, vectors=vectors.tolist(), payloads=all_payloads)
        console.print(f"[green]Indexed {len(all_chunks)} chunks[/green]")

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve relevant chunks."""
        query_vec = self.embedder.embed(query)[0].tolist()
        return self.db.search(COLLECTION, query_vec, top_k=TOP_K)

    def chat(self, question: str) -> dict:
        """Process a question and return answer with sources."""
        # Retrieve context
        t0 = time.perf_counter()
        sources = self.retrieve(question)
        retrieve_time = time.perf_counter() - t0

        # Build context
        context = "\n\n---\n\n".join(
            f"[{r['payload'].get('source', 'unknown')}]\n{r['payload'].get('text', '')}"
            for r in sources
        )

        # Build conversation history
        history = ""
        if self.conversation_history:
            recent = self.conversation_history[-3:]  # last 3 turns
            history = "\n".join(
                f"Pergunta: {turn['question']}\nResposta: {turn['answer'][:200]}..."
                for turn in recent
            )

        # Generate response
        system = SYSTEM_PROMPT.format(context=context)
        user_msg = USER_PROMPT.format(history=history or "(inicio da conversa)", question=question)

        t0 = time.perf_counter()
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
        )
        generate_time = time.perf_counter() - t0

        answer = response["message"]["content"]

        # Update history
        self.conversation_history.append({"question": question, "answer": answer})

        return {
            "answer": answer,
            "sources": sources,
            "timings": {
                "retrieve_ms": retrieve_time * 1000,
                "generate_ms": generate_time * 1000,
            },
        }


def main():
    console.rule("[bold blue]Q&A Chatbot — RAG com Ollama + Qdrant[/bold blue]")
    console.print("[dim]Type 'exit' or 'quit' to stop | 'clear' to reset history[/dim]\n")

    try:
        chatbot = QAChatbot()
    except Exception as e:
        console.print(f"[red]Error initializing chatbot: {e}[/red]")
        console.print("[yellow]Make sure Docker containers are running: docker compose up -d[/yellow]")
        return

    while True:
        try:
            question = Prompt.ask("\n[bold cyan]You[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            break

        if not question.strip():
            continue

        if question.lower() in ("exit", "quit", "sair"):
            break

        if question.lower() == "clear":
            chatbot.conversation_history = []
            console.print("[green]Conversation history cleared[/green]")
            continue

        with console.status("[cyan]Thinking...[/cyan]"):
            try:
                result = chatbot.chat(question)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                continue

        # Display answer
        console.print()
        console.print(Panel(
            Markdown(result["answer"]),
            title="[bold green]Assistant[/bold green]",
            border_style="green",
        ))

        # Display sources and timing
        sources_info = " | ".join(
            f"{r['payload'].get('source', '?')} ({r['score']:.3f})"
            for r in result["sources"][:3]
        )
        t = result["timings"]
        console.print(f"[dim]Sources: {sources_info}[/dim]")
        console.print(f"[dim]Timing: retrieve={t['retrieve_ms']:.0f}ms | generate={t['generate_ms']:.0f}ms[/dim]")

    console.print("\n[dim]Goodbye![/dim]")


if __name__ == "__main__":
    main()
