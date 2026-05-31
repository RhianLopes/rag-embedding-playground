# Módulo 03 — RAG Fundamentals

> Construa um sistema RAG do zero: chunking, embedding, retrieval e geração com Ollama.

## Pré-requisitos

```bash
# Qdrant e Ollama devem estar rodando
docker compose up -d

# Verificar Ollama
curl http://localhost:11434/api/tags

# Baixar modelo de chat (se ainda não baixou)
docker exec ollama ollama pull llama3.2
```

## Notebooks

| # | Notebook | O que você vai aprender |
|---|----------|------------------------|
| 01 | [Naive RAG](01_naive_rag.ipynb) | Pipeline mínimo end-to-end: ingest → chunk → embed → retrieve → generate |
| 02 | [Chunking Strategies](02_chunking_strategies.ipynb) | Fixed-size, recursive, semantic — comparação prática |
| 03 | [Retrieval Strategies](03_retrieval_strategies.ipynb) | Dense, sparse (BM25), hybrid — quando usar cada uma |
| 04 | [Generation & Prompts](04_generation_prompts.ipynb) | Prompt templates, context stuffing, grounding, citation |

## O Pipeline RAG

```
┌─────────────────── INDEXING (offline) ───────────────────┐
│                                                            │
│  Documentos → Chunking → Embedding → Qdrant               │
│                                                            │
└────────────────────────────────────────────────────────────┘

┌─────────────────── QUERYING (online) ────────────────────┐
│                                                            │
│  Pergunta → Embedding → Busca no Qdrant → Contexto        │
│                              ↓                            │
│                       Prompt + Contexto → Ollama → Resp.  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## Decisão de Chunking

```
Tipo de documento?
├── Estruturado (Markdown, HTML, código)
│   └── Document-aware chunking (respeita seções/headers)
├── Prosa longa (PDFs, artigos)
│   └── RecursiveCharacterTextSplitter (chunk_size=512, overlap=64)
├── Conversacional (chat logs, Q&A)
│   └── Semantic chunking (divide por mudança de tópico)
└── Curto e denso (tweets, títulos)
    └── Sem chunking necessário
```

## Referências

- [LangChain Text Splitters](https://python.langchain.com/docs/how_to/#text-splitters)
- [docs/rag/chunking_guide.md](../docs/rag/chunking_guide.md)
- [data/sample_docs/rag_fundamentals.md](../data/sample_docs/rag_fundamentals.md)
