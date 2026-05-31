# Módulo 06 — Avaliação de RAG

> Como medir objetivamente a qualidade de um sistema RAG com RAGAs.

## Por que avaliar?

Sem métricas objetivas, é impossível saber se uma mudança no pipeline melhorou ou piorou o sistema.
RAGAs fornece 4 métricas fundamentais que cobrem os principais pontos de falha.

## Notebooks

| # | Notebook | O que você vai aprender |
|---|----------|------------------------|
| 01 | [RAGAs Metrics](01_ragas_metrics.ipynb) | faithfulness, answer_relevancy, context_precision, context_recall |

## As 4 Métricas RAGAs

```
┌────────────────────────────────────────────────────────────────┐
│                    SISTEMA RAG                                  │
│                                                                 │
│  Query ──→ Retriever ──→ Generator ──→ Resposta                 │
│               │               │                                 │
│    context_precision    faithfulness                            │
│    context_recall       answer_relevancy                        │
└────────────────────────────────────────────────────────────────┘
```

| Métrica | O que mede | Escala | Falha quando |
|---------|-----------|--------|-------------|
| **Faithfulness** | A resposta é suportada pelo contexto? | 0-1 | LLM alucina além do contexto |
| **Answer Relevancy** | A resposta responde a pergunta? | 0-1 | Resposta vaga ou off-topic |
| **Context Precision** | Os docs recuperados são relevantes? | 0-1 | Recupera ruído |
| **Context Recall** | Recuperou toda informação necessária? | 0-1 | Perde docs relevantes |

## Interpretação

```
RAGAs Score = média(faithfulness, answer_relevancy, context_precision, context_recall)

Score > 0.8 → Sistema bom para produção
Score 0.6-0.8 → Precisa melhorias em algumas áreas
Score < 0.6 → Problemas sérios no pipeline
```

## Diagnóstico por Métrica

| Métrica baixa | Causa provável | Solução |
|---------------|---------------|---------|
| Faithfulness | LLM ignora contexto | Reforçar prompt: "responda APENAS com base no contexto" |
| Answer Relevancy | Recupera docs irrelevantes | Melhorar embedding model ou query rewriting |
| Context Precision | Chunks muito grandes/mistos | Refinar estratégia de chunking |
| Context Recall | top-k muito pequeno | Aumentar k ou melhorar retrieval |

## Pré-requisitos

```bash
# RAGAs usa um LLM para avaliar (usamos Ollama)
docker exec ollama ollama pull llama3.2

# Instalar ragas
uv sync
```

## Referências

- [Documentação RAGAs](https://docs.ragas.io/)
- [docs/rag/evaluation_guide.md](../docs/rag/evaluation_guide.md)
- [05_pocs/evaluation_ragas/evaluate.ipynb](../05_pocs/evaluation_ragas/evaluate.ipynb)
