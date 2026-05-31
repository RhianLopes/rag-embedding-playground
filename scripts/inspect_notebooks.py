import json

paths = [
    "01_embeddings/04_distance_metrics.ipynb",
    "03_rag_fundamentals/02_chunking_strategies.ipynb",
    "03_rag_fundamentals/03_retrieval_strategies.ipynb",
    "03_rag_fundamentals/04_generation_prompts.ipynb",
    "05_pocs/semantic_search/demo.ipynb",
    "02_vector_databases/01_qdrant_intro.ipynb",
]

for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"\n=== {path} ({len(nb['cells'])} cells) ===")
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        preview = src[:100].replace("\n", " ")
        print(f"  [{i}] {c['cell_type'][:4]}: {preview}")
