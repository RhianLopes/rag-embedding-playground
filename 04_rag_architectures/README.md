# Módulo 04 — RAG Architectures

> A evolução das arquiteturas RAG: do pipeline ingênuo ao agente autônomo.

## Linha do Tempo

```
2023          2024                    2025
  │             │                       │
Naive RAG → Advanced RAG → Modular RAG → Agentic RAG → GraphRAG
  │             │                       │
 Simples    Query rewriting         Auto-decisão
            Re-ranking              de retrieval
            Compressão              
```

## Notebooks

| # | Notebook | Complexidade | O que você vai aprender |
|---|----------|-------------|------------------------|
| 01 | [Naive RAG Arch](01_naive_rag_arch.ipynb) | ⭐ | Baseline: index → top-k → generate |
| 02 | [Advanced RAG](02_advanced_rag.ipynb) | ⭐⭐ | Query rewriting, re-ranking, compressão |
| 03 | [Modular RAG](03_modular_rag.ipynb) | ⭐⭐⭐ | Componentes independentes, Haystack-style |
| 04 | [Agentic RAG](04_agentic_rag.ipynb) | ⭐⭐⭐⭐ | LangGraph: agente decide quando recuperar |
| 05 | [GraphRAG](05_graphrag.ipynb) | ⭐⭐⭐⭐⭐ | Knowledge graph: entidades e relações |

## Arquiteturas em Resumo

### Naive RAG
```
Query → Embed → Top-K → Prompt → LLM → Resposta
```
Simples, funciona bem para casos básicos. Falha em queries complexas.

### Advanced RAG
```
Query → Rewrite → Embed → Top-K → Re-rank → Compress → Prompt → LLM
```
Adiciona inteligência em cada etapa. Melhor precisão, mais latência.

### Modular RAG
```
[Módulos plugáveis]: Router → Retriever → Re-ranker → Generator → Validator
```
Cada componente é independente e intercambiável. Fácil de testar e melhorar.

### Agentic RAG
```
Query → Agent → [decide] → Retrieve? → Reflect? → Rewrite? → Resposta
                  ↑__________________________________|
```
O LLM decide autonomamente quando e o que buscar. Mais flexível, mais custoso.

### GraphRAG
```
Docs → Extração de Entidades → Knowledge Graph → Community Detection
                                      ↓
Query → Graph Search → Subgrafo relevante → LLM → Resposta
```
Melhor para queries sobre relacionamentos e síntese de informações distribuídas.

## Quando usar cada arquitetura?

| Caso de Uso | Arquitetura Recomendada |
|-------------|------------------------|
| MVP / Prototipagem | Naive RAG |
| Melhorar precisão iterativamente | Advanced RAG |
| Sistema de produção modular | Modular RAG |
| Queries complexas, multi-step | Agentic RAG |
| Análise de documentos relacionados | GraphRAG |

## Referências

- [docs/rag/architectures.md](../docs/rag/architectures.md)
- [Survey: RAG Meets LLMs (2024)](https://arxiv.org/abs/2312.10997)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
