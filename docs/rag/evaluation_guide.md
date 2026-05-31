# Guia de Avaliação de Sistemas RAG

## Por que Avaliar?

Sem métricas objetivas, é impossível saber se uma mudança no pipeline melhorou ou piorou o sistema.
Avaliação sistemática permite:

1. Identificar o elo mais fraco do pipeline
2. Medir impacto de mudanças antes do deploy
3. Estabelecer baselines e metas de qualidade
4. Detectar regressões

---

## As 4 Métricas RAGAs

### Faithfulness (Fidelidade)

**O que mede:** A resposta é factualmente suportada pelo contexto?

```
Faithfulness = afirmações_suportadas_pelo_contexto / total_afirmações_na_resposta
```

**Interpretação:**
- `1.0` = LLM usou APENAS o contexto
- `0.0` = LLM ignorou completamente o contexto

**Quando é baixo:**
- LLM usando conhecimento paramétrico em vez do contexto
- Prompt não instrui adequadamente a usar apenas o contexto
- LLM "completa" informações além do que está nos chunks

**Como melhorar:**
```python
# Prompt mais restritivo
prompt = """Responda APENAS com base no contexto abaixo.
Se a informação não estiver no contexto, responda: "Não encontrei essa informação."
NÃO use conhecimento externo.

Contexto:
{context}

Pergunta: {question}"""
```

---

### Answer Relevancy (Relevância da Resposta)

**O que mede:** A resposta realmente responde à pergunta feita?

```
Answer Relevancy = cosine_similarity(
    embed(pergunta_original), 
    mean(embed(perguntas_geradas_a_partir_da_resposta))
)
```

**Interpretação:**
- `1.0` = resposta diretamente responsiva à pergunta
- `0.0` = resposta completamente off-topic

**Quando é baixo:**
- Retrieval retornou docs irrelevantes
- LLM divagou ou respondeu pergunta diferente
- Contexto insuficiente para a pergunta

**Como melhorar:**
- Melhorar embedding model
- Adicionar query rewriting
- Aumentar top-k na busca

---

### Context Precision (Precisão do Contexto)

**O que mede:** Qual percentual dos documentos recuperados é relevante?

```
Context Precision = documentos_relevantes_no_top_k / k
```

**Interpretação:**
- `1.0` = todos os docs recuperados são relevantes
- `0.0` = nenhum doc relevante no top-k

**Quando é baixo:**
- Embedding model pouco preciso para o domínio
- Chunking muito grande (chunks misturando tópicos)
- Threshold de score muito baixo

**Como melhorar:**
- Cross-encoder re-ranking
- Aumentar score threshold
- Chunks menores e mais focados

---

### Context Recall (Recall do Contexto)

**O que mede:** O pipeline recuperou TODA a informação necessária para responder?

```
Context Recall = afirmações_do_ground_truth_encontradas_no_contexto / 
                 total_afirmações_do_ground_truth
```

**Interpretação:**
- `1.0` = contexto contém tudo que é necessário
- `0.0` = contexto não contém nada do necessário

**Quando é baixo:**
- k muito pequeno (poucas chunks recuperadas)
- Informação está em chunk não recuperado
- Chunking dividiu informação importante

**Como melhorar:**
- Aumentar top-k
- Hybrid search (BM25 + dense)
- Overlap maior entre chunks

---

## RAGAs Score Consolidado

```python
ragas_score = mean(faithfulness, answer_relevancy, context_precision, context_recall)
```

| Score | Interpretação |
|-------|--------------|
| > 0.9 | Excelente — sistema de produção |
| 0.8-0.9 | Bom — pequenas melhorias possíveis |
| 0.6-0.8 | Adequado — precisa de atenção |
| < 0.6 | Problemático — revisão necessária |

---

## Como Montar um Dataset de Avaliação

### Opção 1: Manual (mais confiável)

```python
eval_dataset = [
    {
        "question": "O que é HNSW?",
        "ground_truth": "HNSW é um algoritmo de indexação para busca ANN...",
    },
    ...
]
```

### Opção 2: Gerado por LLM (escala melhor)

```python
GENERATE_QA_PROMPT = """Com base no documento abaixo, gere {n} pares 
pergunta-resposta de alta qualidade.
Formato: {"question": "...", "answer": "..."}

Documento:
{document}"""

# Gerar QA pairs automaticamente
for doc in documents:
    qa_pairs = llm.generate(GENERATE_QA_PROMPT.format(n=5, document=doc))
    eval_dataset.extend(parse_qa_pairs(qa_pairs))
```

**Recomendação:** Pelo menos 50 questões por domínio, revisadas por humano.

---

## Usando RAGAs com Ollama Local

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

# Dataset
data = {
    "question": ["pergunta 1", "pergunta 2"],
    "answer": ["resposta do RAG 1", "resposta do RAG 2"],
    "contexts": [["chunk1", "chunk2"], ["chunk3"]],
    "ground_truth": ["ground truth 1", "ground truth 2"],
}
dataset = Dataset.from_dict(data)

# LLM e embeddings locais
llm = ChatOllama(model="llama3.2", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Avaliar
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=llm,
    embeddings=embeddings,
)

print(results)
# {faithfulness: 0.87, answer_relevancy: 0.91, ...}
```

---

## Ciclo de Melhoria Contínua

```
1. Estabelecer baseline com RAGAs
   ↓
2. Identificar a métrica mais baixa
   ↓
3. Diagnosticar causa raiz
   │
   ├── Faithfulness baixo → Melhorar prompt
   ├── Answer Relevancy baixo → Melhorar retrieval
   ├── Context Precision baixo → Adicionar re-ranking
   └── Context Recall baixo → Aumentar k ou melhorar chunking
   ↓
4. Implementar melhoria
   ↓
5. Re-avaliar com RAGAs
   ↓
6. Repetir até atingir score alvo
```

---

## Outras Métricas Úteis

| Métrica | O que mede | Quando usar |
|---------|-----------|------------|
| MRR (Mean Reciprocal Rank) | Rank do primeiro doc relevante | Qualidade de retrieval |
| Hit@k | Percentual de queries com doc correto no top-k | Avaliação de retrieval |
| BLEU/ROUGE | Similaridade textual com ground truth | Quando GT é muito específico |
| Latência P50/P95/P99 | Velocidade do sistema | Produção |
| Tokens de contexto | Custo com LLMs pagos | Otimização de custo |

---

## Referências

- [Notebook RAGAs](../../06_evaluation/01_ragas_metrics.ipynb)
- [PoC Avaliação](../../05_pocs/evaluation_ragas/evaluate.ipynb)
- [RAGAs Documentation](https://docs.ragas.io/)
- [MTEB Benchmark](https://huggingface.co/spaces/mteb/leaderboard)
