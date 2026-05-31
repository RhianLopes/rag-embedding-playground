# RAG Fundamentals: Retrieval-Augmented Generation

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Model (LLM)
responses by providing relevant context retrieved from an external knowledge base. Instead of
relying solely on the LLM's parametric knowledge (learned during training), RAG fetches
relevant documents at inference time.

## Why RAG?

LLMs have several limitations that RAG addresses:

| Problem | RAG Solution |
|---------|-------------|
| Knowledge cutoff date | Query up-to-date external docs |
| Hallucination | Ground responses in retrieved facts |
| No source attribution | Cite retrieved documents |
| Context length limits | Retrieve only relevant snippets |
| Domain-specific knowledge | Index proprietary documents |
| Cost of fine-tuning | No retraining needed |

## The RAG Pipeline

### Phase 1: Indexing (Offline)

```
Documents → Chunking → Embedding → Vector Store
```

1. **Load**: Read documents (PDF, HTML, Markdown, etc.)
2. **Chunk**: Split into manageable pieces (256-512 tokens typical)
3. **Embed**: Convert each chunk to a dense vector using an embedding model
4. **Store**: Index vectors in a vector database with metadata

### Phase 2: Retrieval & Generation (Online)

```
User Query → Query Embedding → Vector Search → Context Assembly → LLM → Answer
```

1. **Query**: User asks a question
2. **Embed**: Convert query to vector using same embedding model
3. **Retrieve**: Find top-k most similar chunks
4. **Assemble**: Build a prompt with retrieved context
5. **Generate**: LLM produces an answer grounded in the context

## Chunking Strategies

### Fixed-Size Chunking
```python
chunks = [text[i:i+512] for i in range(0, len(text), 512)]
```
Simple but may split sentences mid-thought. Add overlap (e.g., 50 chars) to mitigate.

### Recursive Character Splitting
Split on: `\n\n` → `\n` → `. ` → ` ` until under size limit.
Best general-purpose choice (used by LangChain's `RecursiveCharacterTextSplitter`).

### Semantic Chunking
Use an embedding model to detect topic shifts. Split where semantic similarity drops.
Better quality, higher cost.

### Document-Aware Chunking
Respect document structure: split on headings, paragraphs, list items.
Best for structured documents like Markdown, HTML.

## Retrieval Strategies

### Dense Retrieval
Uses embedding similarity (cosine, dot product). Best for semantic understanding.
```python
results = collection.query(query_embeddings=[query_vec], n_results=5)
```

### Sparse Retrieval (BM25)
Classic keyword-based. Best for exact term matching.
```python
from rank_bm25 import BM25Okapi
bm25 = BM25Okapi(tokenized_corpus)
scores = bm25.get_scores(query_tokens)
```

### Hybrid Retrieval
Combine dense + sparse scores:
```
final_score = α * dense_score + (1-α) * sparse_score
```
Best of both worlds. Typical α = 0.5-0.7 in favor of dense.

## The RAG Prompt Template

```
You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer cannot be found in the context, say "I don't know."

Context:
{retrieved_chunks}

Question: {user_question}

Answer:
```

## Evaluation Metrics (RAGAs)

| Metric | Measures | Formula Concept |
|--------|---------|----------------|
| Faithfulness | Are claims supported by context? | Supported claims / Total claims |
| Answer Relevancy | Does answer address the question? | Similarity of generated Q to original Q |
| Context Precision | How relevant are retrieved docs? | Relevant retrieved / Total retrieved |
| Context Recall | Did we retrieve all relevant info? | Retrieved relevant / Total relevant |

## Common Failure Modes

### Retrieval Failures
- **Wrong chunks**: Embedding model doesn't capture domain semantics
- **Missing context**: Relevant info split across chunk boundaries
- **Too much noise**: Retrieved irrelevant chunks dilute signal

### Generation Failures
- **Ignoring context**: LLM uses parametric knowledge instead of retrieved context
- **Hallucination within context**: LLM extrapolates beyond what context says
- **Context confusion**: Multiple contradictory chunks confuse the LLM

## Best Practices

1. **Chunk size matters**: 256-512 tokens is the sweet spot for most use cases
2. **Add overlap**: 10-20% overlap between chunks preserves context at boundaries
3. **Include metadata**: Chunk source, date, section heading in payload
4. **Embed queries and documents similarly**: Use same model for both
5. **Re-rank results**: Use a cross-encoder to re-score top-k candidates
6. **Evaluate continuously**: Use RAGAs or custom metrics to track quality
7. **Monitor and iterate**: Track which queries fail and improve accordingly
