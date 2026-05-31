# Vector Databases: A Comprehensive Overview

## What is a Vector Database?

A vector database is a specialized database system designed to store, index, and query
high-dimensional vector embeddings efficiently. Unlike traditional relational databases that
store structured data and use exact match queries, vector databases enable **approximate
nearest neighbor (ANN) search** — finding the most similar vectors to a query vector.

## Why Vector Databases?

As AI applications increasingly rely on embeddings (dense vector representations of text,
images, audio, etc.), the need for efficient similarity search has grown dramatically.

### The Challenge of Naive Similarity Search

For N vectors of dimension D:
- **Brute force search**: O(N × D) — works for thousands, breaks at millions
- **Vector database ANN**: O(log N × D) — scales to billions

## Core Concepts

### Collections / Indexes
The top-level organizational unit. Similar to a "table" in SQL databases, but stores vectors
with optional metadata (payload).

### Points / Documents
Individual records in a collection. Each point has:
- **ID**: Unique identifier (UUID or integer)
- **Vector**: The embedding (array of floats)
- **Payload**: Optional metadata (JSON-like dictionary)

### Distance Metrics

| Metric | Formula | Best For |
|--------|---------|---------|
| Cosine | `1 - (A·B)/(|A||B|)` | Text embeddings (direction matters) |
| Dot Product | `-(A·B)` | When vectors are normalized |
| Euclidean (L2) | `sqrt(Σ(A_i - B_i)²)` | When magnitude matters |

### Indexing Algorithms

**HNSW (Hierarchical Navigable Small World)**
- Most popular ANN algorithm
- Builds a multi-layer graph where each layer is a "shortcut" to the next
- Excellent recall/speed tradeoff
- Key parameters:
  - `m`: Max connections per node (16-64 typical)
  - `ef_construct`: Build-time candidates (100-500 typical)
  - `ef`: Query-time candidates

**IVF (Inverted File Index)**
- Divides vectors into clusters (Voronoi cells)
- Fast but requires training
- Common in Faiss

**LSH (Locality Sensitive Hashing)**
- Hash-based approach
- Very fast but lower recall

## Popular Vector Databases (2025)

| Database | Type | Best For | License |
|----------|------|---------|---------|
| **Qdrant** | Dedicated | Production RAG, Rust-based, excellent filtering | Apache 2.0 |
| **Chroma** | Embedded/Server | Development, simple APIs | Apache 2.0 |
| **Weaviate** | Dedicated | GraphQL API, modules | BSD 3-Clause |
| **Pinecone** | Managed cloud | Serverless, no infra | Commercial |
| **Milvus** | Dedicated | Large scale, GPU support | Apache 2.0 |
| **pgvector** | Extension | Already using PostgreSQL | MIT |
| **Faiss** | Library | Research, offline batch | MIT |

## Qdrant Deep Dive

Qdrant is written in Rust, offering excellent performance and memory safety. It's the
recommended choice for this playground.

### Key Features
- **Filtering**: Combine vector search with metadata filters (unlike many competitors)
- **Sparse vectors**: Support for BM25/SPLADE alongside dense vectors (hybrid search)
- **Quantization**: Scalar (int8) and product quantization built-in
- **Payload indexing**: Index payload fields for fast filtering
- **Collections**: Organize vectors into separate namespaces
- **Snapshots**: Built-in backup and restore

### Qdrant REST API Examples

```bash
# Create a collection
curl -X PUT http://localhost:6333/collections/my_collection \
  -H 'Content-Type: application/json' \
  -d '{"vectors": {"size": 384, "distance": "Cosine"}}'

# Insert points
curl -X PUT http://localhost:6333/collections/my_collection/points \
  -H 'Content-Type: application/json' \
  -d '{"points": [{"id": 1, "vector": [...], "payload": {"text": "hello"}}]}'

# Search
curl -X POST http://localhost:6333/collections/my_collection/points/search \
  -H 'Content-Type: application/json' \
  -d '{"vector": [...], "limit": 5, "with_payload": true}'
```

## Quantization in Vector Databases

### Why Quantize?

Memory usage for 1M vectors at 768 dimensions:
- float32: 1M × 768 × 4 bytes = **3.07 GB**
- float16: 1M × 768 × 2 bytes = **1.54 GB**
- int8:    1M × 768 × 1 byte  = **768 MB**
- binary:  1M × 768 / 8 bytes = **96 MB**

### Scalar Quantization (int8)
Linearly maps float32 values to int8 range [-128, 127].
- Memory: 4x reduction
- Speed: ~2x faster on CPU
- Recall: ~97-99% compared to float32

### Product Quantization (PQ)
Divides vectors into sub-vectors and quantizes each independently.
- Memory: 8-32x reduction
- Speed: Faster
- Recall: Lower than scalar (~90-95%)

## Hybrid Search

Combining dense (semantic) and sparse (keyword) retrieval:

```
hybrid_score = α × dense_score + (1 - α) × sparse_score
```

- **Dense**: Good for semantic similarity ("what does this mean?")
- **Sparse (BM25)**: Good for exact keyword matching ("find this term")
- **Hybrid**: Best of both worlds for RAG

Qdrant supports both dense and sparse vectors natively since version 1.7.
