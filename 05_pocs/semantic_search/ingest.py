"""
Semantic Search PoC — Ingestion Pipeline

Usage:
    uv run python 05_pocs/semantic_search/ingest.py --docs data/sample_docs/
    uv run python 05_pocs/semantic_search/ingest.py --docs path/to/your/docs/ --recreate
"""

import argparse
import sys
from pathlib import Path
from typing import Iterator

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.embeddings.local_embedder import LocalEmbedder
from src.utils.chunking import recursive_chunk
from src.vector_db.qdrant_manager import QdrantManager

console = Console()

COLLECTION = "semantic_search"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def load_documents(docs_dir: Path) -> Iterator[dict]:
    """Load text documents from a directory."""
    supported = {".md", ".txt", ".rst"}
    files = [f for f in docs_dir.iterdir() if f.suffix.lower() in supported]

    console.print(f"[blue]Found {len(files)} documents in {docs_dir}[/blue]")

    for filepath in files:
        try:
            text = filepath.read_text(encoding="utf-8")
            yield {
                "text": text,
                "filename": filepath.name,
                "title": filepath.stem.replace("_", " ").title(),
                "path": str(filepath),
                "size_chars": len(text),
            }
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load {filepath}: {e}[/yellow]")


def ingest(docs_dir: Path, recreate: bool = False) -> int:
    """Ingest documents into Qdrant for semantic search."""
    console.rule("[bold blue]Semantic Search Ingestion Pipeline[/bold blue]")

    # Initialize components
    console.print("[cyan]Loading embedding model...[/cyan]")
    embedder = LocalEmbedder(EMBEDDING_MODEL)
    db = QdrantManager()

    # Create or recreate collection
    if recreate and COLLECTION in db.list_collections():
        db.delete_collection(COLLECTION)

    if COLLECTION not in db.list_collections():
        db.create_collection(
            name=COLLECTION,
            dimensions=384,
            distance="cosine",
            with_scalar_quantization=False,
        )

    # Load and chunk documents
    all_chunks = []
    all_payloads = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        docs = list(load_documents(docs_dir))
        task = progress.add_task("Chunking documents...", total=len(docs))

        for doc in docs:
            chunks = recursive_chunk(
                doc["text"],
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
                metadata={
                    "filename": doc["filename"],
                    "title": doc["title"],
                },
            )
            for chunk in chunks:
                all_chunks.append(chunk.text)
                all_payloads.append({
                    "text": chunk.text,
                    "title": doc["title"],
                    "filename": doc["filename"],
                    "chunk_idx": chunk.index,
                })
            progress.advance(task)

    console.print(f"\n[green]Total chunks: {len(all_chunks)}[/green]")

    # Embed and index
    console.print("[cyan]Embedding chunks...[/cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Embedding {len(all_chunks)} chunks...", total=None)
        vectors = embedder.embed(all_chunks)

    total = db.upsert(
        collection_name=COLLECTION,
        vectors=vectors.tolist(),
        payloads=all_payloads,
    )

    console.print(f"\n[bold green]✅ Ingestion complete![/bold green]")
    console.print(f"   Collection: {COLLECTION}")
    console.print(f"   Chunks indexed: {total}")
    console.print(f"   Dimensions: 384 (all-MiniLM-L6-v2)")
    console.print(f"\n[dim]Access Qdrant Web UI: http://localhost:6333/dashboard[/dim]")

    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents for semantic search")
    parser.add_argument("--docs", type=Path, default=Path("data/sample_docs"),
                        help="Directory with documents to ingest")
    parser.add_argument("--recreate", action="store_true",
                        help="Recreate collection if it exists")
    args = parser.parse_args()

    if not args.docs.exists():
        console.print(f"[red]Error: Directory {args.docs} does not exist[/red]")
        sys.exit(1)

    ingest(args.docs, recreate=args.recreate)
