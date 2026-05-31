# Modelos de Embedding: Visão Geral Comparativa

## Modelos Locais (sentence-transformers)

### Como instalar e usar

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode(
    ["texto 1", "texto 2"],
    normalize_embeddings=True,  # normalizar para cosine similarity
    batch_size=32,
    show_progress_bar=True,
)
# shape: (2, 768) — float32
```

### Tabela de Modelos Recomendados

| Modelo | Dims | Params | MTEB | Velocidade | RAM modelo | Ideal para |
|--------|------|--------|------|-----------|-----------|------------|
| all-MiniLM-L6-v2 | 384 | 22M | 56.3 | ⚡⚡⚡ | 22MB | Prototipagem, real-time |
| all-MiniLM-L12-v2 | 384 | 33M | 56.5 | ⚡⚡ | 33MB | Melhor que L6 sem custo |
| all-mpnet-base-v2 | 768 | 110M | 57.8 | ⚡⚡ | 420MB | **Padrão produção** |
| all-distilroberta-v1 | 768 | 82M | 55.9 | ⚡⚡ | 330MB | Alternativa leve ao mpnet |
| all-roberta-large-v1 | 1024 | 355M | 60.0 | ⚡ | 1.3GB | Qualidade máxima local |
| paraphrase-multilingual-mpnet | 768 | 278M | 53.7 | ⚡ | 1.1GB | Multilíngue (50+ langs) |
| multi-qa-mpnet-base-dot-v1 | 768 | 110M | 57.0 | ⚡⚡ | 420MB | Específico para Q&A |

---

## Modelos via Ollama (local, sem código Python)

```bash
# Baixar modelo
ollama pull nomic-embed-text
ollama pull mxbai-embed-large

# Usar via API
curl http://localhost:11434/api/embeddings \
  -d '{"model": "nomic-embed-text", "prompt": "texto aqui"}'
```

```python
import ollama

response = ollama.embeddings(
    model="nomic-embed-text",
    prompt="texto para embedar",
)
vector = response["embedding"]  # list of floats
```

| Modelo Ollama | Dims | MTEB | Tamanho | Vantagem |
|--------------|------|------|---------|---------|
| nomic-embed-text | 768 | 62.0 | 274MB | Excelente qualidade, gratuito |
| mxbai-embed-large | 1024 | 64.6 | 670MB | Melhor qualidade, mais pesado |
| all-minilm | 384 | 56.3 | 45MB | Ultra-rápido |

---

## Ranking por Caso de Uso

### Para RAG Geral

1. **nomic-embed-text** (Ollama) — melhor qualidade sem custo de API
2. **all-mpnet-base-v2** — mais controle, bem documentado
3. **all-MiniLM-L6-v2** — quando latência é crítica

### Para Código

| Modelo | Especialidade |
|--------|--------------|
| microsoft/codebert-base | Código geral |
| nomic-ai/nomic-embed-text | Código + texto misturado |
| jinaai/jina-embeddings-v2-base-code | Específico para código |

### Para Domínio Biomédico

| Modelo | Especialidade |
|--------|--------------|
| allenai/scibert_scivocab_uncased | Artigos científicos |
| dmis-lab/biobert-base-cased-v1.2 | Biomédico |

### Para Documentos Longos

| Modelo | Max tokens |
|--------|-----------|
| all-mpnet-base-v2 | 384 tokens |
| jina-embeddings-v2-base-en | 8192 tokens |
| nomic-embed-text | 8192 tokens |

---

## Como Escolher

```
1. Qual é a língua dos documentos?
   └── Apenas português/inglês → qualquer modelo acima
       Múltiplos idiomas → paraphrase-multilingual-*

2. Qual é o tamanho médio dos documentos?
   └── < 512 tokens → qualquer modelo
       > 512 tokens → nomic-embed-text ou jina-v2 (8K tokens)

3. Qual é a prioridade?
   └── Velocidade → all-MiniLM-L6-v2 (384d)
       Equilíbrio → all-mpnet-base-v2 (768d)
       Qualidade → mxbai-embed-large via Ollama (1024d)

4. Tem GPU?
   └── Sim → modelos maiores ficam mais rápidos
       Não → ficar em 384d ou 768d
```

---

## Benchmarks de Velocidade (CPU, MacBook M2)

```
Throughput (textos/segundo, batch=32):

all-MiniLM-L6-v2  (384d):  ████████████████  ~2000 t/s
all-mpnet-base-v2 (768d):  ████████          ~700 t/s
all-roberta-large (1024d): ████              ~300 t/s
```

---

## Referências

- [SBERT Model Hub](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Ollama Embedding Models](https://ollama.com/search?c=embedding)
- [Notebook Comparação](../../01_embeddings/05_models_comparison.ipynb)
