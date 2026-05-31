# Dimensões de Embeddings: Guia de Referência

## O que são Dimensões?

O número de dimensões (D) é o tamanho do vetor de embedding.
Um modelo 384d produz vetores com 384 números por texto.

**Trade-off fundamental:**
```
Mais dimensões → Mais expressividade semântica
Mais dimensões → Mais memória e tempo de busca
```

---

## Modelos Open-Source por Dimensão

### 384 dimensões — Rápido

| Modelo | Parâmetros | Tamanho | MTEB Score | Uso típico |
|--------|-----------|---------|-----------|------------|
| all-MiniLM-L6-v2 | 22M | 22MB | 56.3 | Prototipagem, tempo real |
| all-MiniLM-L12-v2 | 33M | 33MB | 56.5 | Mais preciso que L6 |
| paraphrase-MiniLM-L3-v2 | 17M | 17MB | 50.7 | Ultra-rápido |

**Memória (float32):**
- 100K docs: 153 MB
- 1M docs: 1.53 GB
- 10M docs: 15.3 GB

---

### 768 dimensões — Balanceado (recomendado)

| Modelo | Parâmetros | Tamanho | MTEB Score | Uso típico |
|--------|-----------|---------|-----------|------------|
| all-mpnet-base-v2 | 110M | 420MB | 57.8 | **Produção geral** |
| paraphrase-multilingual-mpnet | 278M | 1.1GB | 53.7 | Multilíngue |
| all-distilroberta-v1 | 82M | 330MB | 55.9 | Rápido e bom |
| nomic-embed-text (Ollama) | 137M | 270MB | 62.0 | Via Ollama |

**Memória (float32):**
- 100K docs: 307 MB
- 1M docs: 3.07 GB
- 10M docs: 30.7 GB

---

### 1024 dimensões — Alta qualidade

| Modelo | Parâmetros | Tamanho | MTEB Score | Uso típico |
|--------|-----------|---------|-----------|------------|
| all-roberta-large-v1 | 355M | 1.3GB | 60.0 | Qualidade prioritária |
| mxbai-embed-large (Ollama) | 335M | 670MB | 64.6 | Via Ollama, excelente |
| e5-large-v2 | 335M | 1.3GB | 62.2 | Microsoft E5 |

**Memória (float32):**
- 100K docs: 409 MB
- 1M docs: 4.09 GB
- 10M docs: 40.9 GB

---

### 1536 dimensões — APIs proprietárias

| Modelo | Provider | MTEB Score | Custo |
|--------|---------|-----------|-------|
| text-embedding-3-small | OpenAI | 62.3 | Baixo |
| text-embedding-ada-002 | OpenAI (deprecated) | 61.0 | Baixo |

**Memória (float32):**
- 1M docs: 6.14 GB

---

### 3072 dimensões — State-of-the-art proprietário

| Modelo | Provider | MTEB Score | Custo |
|--------|---------|-----------|-------|
| text-embedding-3-large | OpenAI | 64.6 | Alto |

**Memória (float32):**
- 1M docs: 12.3 GB

---

## Comparação de Memória

```
             1M documentos (float32):
             
384d  ■■                                   1.53 GB
768d  ■■■■                                 3.07 GB
1024d ■■■■■■                               4.09 GB
1536d ■■■■■■■■■                            6.14 GB
3072d ■■■■■■■■■■■■■■■■■■■■               12.29 GB
```

---

## Impacto na Velocidade de Busca (HNSW)

Tempo de busca cresce aproximadamente `O(D)`:

| Dimensões | Tempo relativo (busca) |
|-----------|----------------------|
| 384 | 1.0x (baseline) |
| 768 | 1.8x |
| 1024 | 2.4x |
| 1536 | 3.5x |
| 3072 | 6.8x |

*Valores aproximados, dependem do hardware e número de documentos.*

---

## Dimensões Truncadas (OpenAI text-embedding-3)

`text-embedding-3-large` suporta truncação de dimensões:

```python
import openai

# Embeddings com dimensões reduzidas (mais baratos e rápidos)
response = openai.embeddings.create(
    model="text-embedding-3-large",
    input="exemplo de texto",
    dimensions=256,  # truncar de 3072 para 256!
)
```

**Insight interessante:** `text-embedding-3-large` truncado para 256d supera `ada-002` em 1536d!

---

## Guia de Decisão

```
Escala do projeto?
│
├── Prototipagem / < 100K docs
│   └── all-MiniLM-L6-v2 (384d) — mais rápido para iterar
│
├── Produção / 100K - 10M docs
│   └── all-mpnet-base-v2 (768d) — melhor equilíbrio
│       com int8 quantization: 768 MB / 1M docs
│
├── Qualidade crítica / domínio específico
│   └── mxbai-embed-large via Ollama (1024d)
│       ou all-roberta-large-v1
│
└── Multilíngue
    └── paraphrase-multilingual-mpnet-base-v2 (768d)
        suporta 50+ línguas
```

---

## Referências

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Notebook 02: Vector Dimensions](../../01_embeddings/02_vector_dimensions.ipynb)
- [Notebook 05: Models Comparison](../../01_embeddings/05_models_comparison.ipynb)
