# PoC: RAG Evaluation com RAGAs

Avaliacao sistematica da qualidade de um sistema RAG usando as 4 metricas fundamentais do RAGAs.

## O que e RAGAs?

RAGAs (Retrieval Augmented Generation Assessment) e um framework de avaliacao que mede:

| Metrica | O que mede | Como calcular |
|---------|-----------|---------------|
| **Faithfulness** | A resposta e suportada pelo contexto? | LLM extrai afirmacoes → verifica no contexto |
| **Answer Relevancy** | A resposta responde a pergunta? | Gera perguntas alternativas → similaridade |
| **Context Precision** | Os docs recuperados sao relevantes? | Classifica cada doc como relevante/irrelevante |
| **Context Recall** | Recuperou toda a informacao necessaria? | Compara com ground truth |

## Como usar

```bash
# 1. Garantir que Ollama esta rodando com llama3.2
docker exec ollama ollama pull llama3.2

# 2. Abrir o notebook
uv run jupyter lab 05_pocs/evaluation_ragas/evaluate.ipynb
```

## O que voce vai aprender

1. Como montar um dataset de avaliacao (perguntas + respostas esperadas)
2. Como executar RAGAs com Ollama local
3. Como interpretar os scores
4. Como usar os resultados para melhorar o sistema

## Referencia

- [Documentacao RAGAs](https://docs.ragas.io/)
- [06_evaluation/01_ragas_metrics.ipynb](../../06_evaluation/01_ragas_metrics.ipynb)
