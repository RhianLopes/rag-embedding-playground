# RAG & Embeddings Playground

> Um playground de aprendizagem estruturada e prática sobre Embeddings e RAG —
> do zero até arquiteturas avançadas, 100% local, zero custo com APIs.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![Qdrant](https://img.shields.io/badge/Qdrant-latest-red)](https://qdrant.tech)
[![Ollama](https://img.shields.io/badge/Ollama-local-green)](https://ollama.com)
[![uv](https://img.shields.io/badge/uv-package_manager-purple)](https://docs.astral.sh/uv/)

---

## O que você vai aprender

```
Embeddings → Dimensões → Float Types → Vector DBs → RAG → Arquiteturas → PoCs
```

| Módulo | Tópico | Conceitos Chave |
|--------|--------|-----------------|
| [01 Embeddings](01_embeddings/) | Fundamentos | O que são, geometria vetorial, visualização |
| [02 Vector DBs](02_vector_databases/) | Qdrant | Collections, HNSW, quantização, hybrid search |
| [03 RAG Fundamentos](03_rag_fundamentals/) | Pipeline RAG | Chunking, retrieval, geração, prompts |
| [04 Arquiteturas](04_rag_architectures/) | Evolução RAG | Naive → Advanced → Modular → Agentic → GraphRAG |
| [05 PoCs](05_pocs/) | Projetos práticos | Semantic search, Q&A chatbot, avaliação |
| [06 Avaliação](06_evaluation/) | RAGAs | Faithfulness, relevância, precisão, recall |

---

## Stack Local (Zero Custo)

| Componente | Ferramenta | Porta |
|------------|-----------|-------|
| Vector DB | Qdrant (Docker) | 6333 (REST), 6334 (gRPC) |
| LLMs | Ollama (Docker) | 11434 |
| Embeddings | sentence-transformers | local |
| Package manager | uv | - |
| Notebooks | JupyterLab | 8888 |

---

## Quick Start

### Pré-requisitos
- [Docker Desktop](https://docs.docker.com/desktop/) instalado e rodando
- [uv](https://docs.astral.sh/uv/getting-started/installation/) instalado

### 1. Subir a infraestrutura

```bash
docker compose up -d

# Verificar Qdrant (deve retornar JSON com versão)
curl http://localhost:6333/

# Web UI: http://localhost:6333/dashboard
```

### 2. Baixar modelos Ollama

```bash
# Chat LLM (~2GB)
docker exec ollama ollama pull llama3.2

# Embedding model (~270MB)
docker exec ollama ollama pull nomic-embed-text
```

### 3. Configurar ambiente Python

```bash
# Copiar variáveis de ambiente
cp .env.example .env

# Instalar dependências com uv
uv sync

# Configurar nbstripout (git limpo para notebooks)
uv run nbstripout --install --attributes .gitattributes
```

### 4. Iniciar JupyterLab

```bash
uv run jupyter lab
```

Abra `00_quickstart.ipynb` para testar toda a stack de uma vez.

---

## Learning Path

### Iniciante — "O que é um embedding?"
```
01_embeddings/01_what_are_embeddings.ipynb
01_embeddings/04_distance_metrics.ipynb
02_vector_databases/01_qdrant_intro.ipynb
03_rag_fundamentals/01_naive_rag.ipynb
```

### Intermediário — "Como funciona na prática?"
```
01_embeddings/02_vector_dimensions.ipynb
01_embeddings/03_float_types.ipynb
02_vector_databases/02_hnsw_indexing.ipynb
03_rag_fundamentals/02_chunking_strategies.ipynb
03_rag_fundamentals/03_retrieval_strategies.ipynb
04_rag_architectures/01_naive_rag_arch.ipynb
04_rag_architectures/02_advanced_rag.ipynb
```

### Avançado — "Arquiteturas e produção"
```
04_rag_architectures/03_modular_rag.ipynb
04_rag_architectures/04_agentic_rag.ipynb
04_rag_architectures/05_graphrag.ipynb
05_pocs/semantic_search/demo.ipynb
05_pocs/qa_chatbot/demo.ipynb
06_evaluation/01_ragas_metrics.ipynb
```

---

## Estrutura do Repositório

```
rag-embedding-playground/
├── 00_quickstart.ipynb          ← Smoke test de todo o stack
│
├── 01_embeddings/               ← Fundamentos de Embeddings
├── 02_vector_databases/         ← Qdrant deep dive
├── 03_rag_fundamentals/         ← RAG do zero
├── 04_rag_architectures/        ← Evolução das arquiteturas
├── 05_pocs/                     ← Projetos completos
├── 06_evaluation/               ← Avaliação com RAGAs
│
├── src/                         ← Módulos Python reutilizáveis
│   ├── embeddings/              ← Wrappers sentence-transformers
│   ├── rag/                     ← Pipelines RAG
│   ├── vector_db/               ← Wrapper Qdrant
│   └── utils/                   ← Chunking e utilitários
│
├── docs/                        ← Documentação de referência
│   ├── embeddings/              ← Dimensões, float types, modelos
│   └── rag/                     ← Arquiteturas, chunking, avaliação
│
├── data/
│   └── sample_docs/             ← Documentos de exemplo para PoCs
│
├── infrastructure/
│   ├── qdrant/config.yaml       ← Configuração Qdrant
│   └── ollama/pull_models.sh   ← Script para baixar modelos
│
├── docker-compose.yml           ← Qdrant + Ollama
├── pyproject.toml               ← Dependências (uv)
└── .env.example                 ← Variáveis de ambiente
```

---

## Documentação de Referência

- [Float Types & Precisão](docs/embeddings/float_types.md)
- [Dimensões por Modelo](docs/embeddings/dimensions.md)
- [Comparação de Modelos](docs/embeddings/models_overview.md)
- [Arquiteturas RAG](docs/rag/architectures.md)
- [Guia de Chunking](docs/rag/chunking_guide.md)
- [Guia de Avaliação RAG](docs/rag/evaluation_guide.md)

---

## Modelos Embedding — Referência Rápida

| Modelo | Dimensões | Tamanho | Velocidade | Qualidade |
|--------|-----------|---------|-----------|-----------|
| all-MiniLM-L6-v2 | 384 | 22MB | ⚡⚡⚡ | Boa |
| all-mpnet-base-v2 | 768 | 420MB | ⚡⚡ | Muito boa |
| all-roberta-large-v1 | 1024 | 1.3GB | ⚡ | Excelente |
| nomic-embed-text (Ollama) | 768 | 270MB | ⚡⚡ | Muito boa |
| mxbai-embed-large (Ollama) | 1024 | 670MB | ⚡ | Excelente |

---

*Construído com curiosidade e café.*
