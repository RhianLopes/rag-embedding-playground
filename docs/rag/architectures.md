# Arquiteturas RAG: Guia de Referência

## Evolução das Arquiteturas

```
2023                2024                    2025
  │                   │                       │
Naive RAG ──→ Advanced RAG ──→ Modular RAG ──→ Agentic RAG
                                                     │
                                               GraphRAG (paralelo)
```

---

## 1. Naive RAG

### Diagrama

```
INDEXING:
Documentos → Chunking → Embedding → Vector DB

QUERYING:
Query → Embedding → ANN Search(top-k) → Prompt → LLM → Resposta
```

### Características

| Aspecto | Detalhe |
|---------|---------|
| Latência | ~2-3s total |
| Qualidade | Boa para queries simples |
| Complexidade | Mínima |
| Casos de uso | MVP, prototipagem, queries claras |

### Limitações

- Sem otimização de query
- Sem re-ranking
- Qualidade depende muito do chunking
- Falha em queries ambíguas ou multi-step

### Código Mínimo

```python
from src.rag.naive import NaiveRAG

rag = NaiveRAG(collection_name="docs", chunk_size=512, top_k=5)
rag.index([{"text": doc_text, "source": "doc.pdf"}])
result = rag.query("como funciona X?")
```

---

## 2. Advanced RAG

### Diagrama

```
Query
  ↓ [PRE-RETRIEVAL]
  ├── Query Rewriting (LLM reformula)
  ├── HyDE (gera documento hipotético)
  └── Multi-query (N variações)
  ↓
  Retrieval (top-K maior)
  ↓ [POST-RETRIEVAL]
  ├── Cross-Encoder Re-ranking
  ├── Context Compression
  └── MMR (diversidade)
  ↓
LLM → Resposta
```

### Características

| Aspecto | Detalhe |
|---------|---------|
| Latência | ~4-8s total |
| Qualidade | Muito boa (85-92% em benchmarks) |
| Complexidade | Moderada |
| Casos de uso | Produção, qualidade prioritária |

### Técnicas Chave

**Query Rewriting:**
```python
# LLM reformula a query para melhorar o retrieval
variantes = ["Como funciona X?", "X, definição e funcionamento", 
             "Processo de X explicado"]
```

**HyDE (Hypothetical Document Embeddings):**
```python
# Gerar um documento hipotético e usar seu embedding
# O embedding de um "resposta" se aproxima mais do espaço de documentos
hyp_doc = llm.generate("Escreva um parágrafo que responde: " + query)
query_vector = embed(hyp_doc)  # em vez de embed(query)
```

**Cross-Encoder Re-ranking:**
```python
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = cross_encoder.predict([(query, doc) for doc in candidates])
reranked = sorted(zip(scores, candidates), reverse=True)
```

---

## 3. Modular RAG

### Diagrama

```
Pipeline com componentes intercambiáveis:

[Router] → [Retriever] → [Reranker] → [Generator] → [Validator]
    │           │             │             │
  Dense       BM25          CE          Ollama
  BM25       Hybrid         ID          LangChain
  Hybrid     ...           MMR          ...
```

### Características

| Aspecto | Detalhe |
|---------|---------|
| Latência | Configurável (depende dos componentes) |
| Qualidade | Alta (componentes otimizados independentemente) |
| Complexidade | Alta (arquitetura mais elaborada) |
| Casos de uso | Sistemas de produção, A/B testing |

### Interface Padrão

```python
class BaseRetriever(ABC):
    def retrieve(self, query: str, top_k: int) -> List[Document]: ...

class BaseReranker(ABC):
    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]: ...

class BaseGenerator(ABC):
    def generate(self, query: str, docs: List[Document]) -> str: ...
```

---

## 4. Agentic RAG

### Diagrama

```
Query
  ↓
[AGENTE — LLM]
  ├── Decisão: Precisa buscar?
  │     ├── Não → Responder diretamente
  │     └── Sim → Formular query de busca
  │                     ↓
  │               [RETRIEVER]
  │                     ↓
  │             [AGENTE — Avaliar]
  │                     ├── Suficiente → Gerar resposta
  │                     └── Insuficiente → Reformular query ↑ (loop)
  └── MAX_ITERATIONS → Responder com contexto disponível
```

### Características

| Aspecto | Detalhe |
|---------|---------|
| Latência | ~10-30s (múltiplas iterações) |
| Qualidade | Máxima para queries complexas |
| Complexidade | Alta (gerenciamento de estado) |
| Casos de uso | Queries multi-step, raciocínio complexo |

### Frameworks

- **LangGraph**: grafos de estado para fluxo de agentes
- **AutoGPT-style**: loop decide-act-observe
- **ReAct**: Reason + Act alternados

---

## 5. GraphRAG

### Diagrama

```
Documentos
    ↓ [Extração] LLM extrai entidades e relações
Knowledge Graph
    ↓
    ├── [Graph Search] Subgrafo relevante + neighbors
    └── [Vector Search] Chunks textuais similares
         ↓
    [Score Fusion] Combinar resultados
         ↓
    LLM → Resposta com raciocínio sobre relações
```

### Características

| Aspecto | Detalhe |
|---------|---------|
| Indexação | Cara (LLM para extração) |
| Latência de query | ~5-15s |
| Qualidade para relações | Excelente |
| Casos de uso | Documentos com muitas entidades, queries relacionais |

### Quando GraphRAG supera RAG tradicional

```
Queries de relacionamento: "Qual a diferença entre X e Y?"
Queries multi-hop: "Quem trabalhou com X no projeto Y?"
Síntese: "Resuma todas as menções de Z nos documentos"
```

---

## Comparação Final

| Arquitetura | Latência | Qualidade | Complexidade | Custo/query |
|-------------|---------|-----------|-------------|-------------|
| Naive RAG | ~2s | ⭐⭐⭐ | ⭐ | Baixo |
| Advanced RAG | ~5s | ⭐⭐⭐⭐ | ⭐⭐⭐ | Médio |
| Modular RAG | ~3-8s | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Médio |
| Agentic RAG | ~15s | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Alto |
| GraphRAG | ~10s | ⭐⭐⭐⭐⭐* | ⭐⭐⭐⭐⭐ | Alto |

*Excelente para queries relacionais, moderado para queries factuais simples.

---

## Referências

- [Notebook 01: Naive RAG Arch](../../04_rag_architectures/01_naive_rag_arch.ipynb)
- [Notebook 02: Advanced RAG](../../04_rag_architectures/02_advanced_rag.ipynb)
- [Notebook 04: Agentic RAG](../../04_rag_architectures/04_agentic_rag.ipynb)
- [Survey: RAG Meets LLMs (2024)](https://arxiv.org/abs/2312.10997)
