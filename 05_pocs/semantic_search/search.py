"""
Semantic Search PoC — Search CLI

Usage:
    uv run python 05_pocs/semantic_search/search.py "como funciona HNSW?"
    uv run python 05_pocs/semantic_search/search.py "embeddings de texto" --top-k 5
"""

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.embeddings.local_embedder import LocalEmbedder
from src.vector_db.qdrant_manager import QdrantManager

console = Console()

COLLECTION = "semantic_search"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def search(query: str, top_k: int = 5, score_threshold: float = 0.0) -> list[dict]:
    """Search for relevant documents."""
    embedder = LocalEmbedder(EMBEDDING_MODEL)
    db = QdrantManager()

    if COLLECTION not in db.list_collections():
        console.print(f"[red]Collection '{COLLECTION}' not found.[/red]")
        console.print("[yellow]Run ingestion first: python 05_pocs/semantic_search/ingest.py[/yellow]")
        sys.exit(1)

    t0 = time.perf_counter()
    query_vector = embedder.embed(query)[0].tolist()
    embed_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    results = db.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        top_k=top_k,
        score_threshold=score_threshold if score_threshold > 0 else None,
    )
    search_time = (time.perf_counter() - t0) * 1000

    return results, embed_time, search_time


def display_results(query: str, results: list, embed_time: float, search_time: float) -> None:
    """Display search results in a rich table."""
    console.print()
    console.print(Panel(f"[bold cyan]{query}[/bold cyan]", title="Search Query", border_style="blue"))
    console.print(f"[dim]Embed: {embed_time:.1f}ms | Search: {search_time:.1f}ms | Total: {embed_time+search_time:.1f}ms[/dim]\n")

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold blue")
    table.add_column("#", style="dim", width=4)
    table.add_column("Score", style="green", width=8)
    table.add_column("Source", style="cyan", width=25)
    table.add_column("Text", no_wrap=False)

    for i, r in enumerate(results, 1):
        payload = r["payload"]
        text = payload.get("text", "")
        preview = text[:200] + "..." if len(text) > 200 else text

        table.add_row(
            str(i),
            f"{r['score']:.4f}",
            payload.get("filename", "unknown"),
            preview,
        )

    console.print(table)
    console.print(f"\n[dim]Found {len(results)} results[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic search over indexed documents")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum score threshold")
    args = parser.parse_args()

    results, embed_time, search_time = search(args.query, top_k=args.top_k, score_threshold=args.threshold)
    display_results(args.query, results, embed_time, search_time)
