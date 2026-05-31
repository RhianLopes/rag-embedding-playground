# Módulo 01 — Embeddings Fundamentals

> Entenda o que são embeddings, como funcionam geometricamente e por que dimensões e tipos de float importam.

## Notebooks

| # | Notebook | O que você vai aprender |
|---|----------|------------------------|
| 01 | [O que são Embeddings](01_what_are_embeddings.ipynb) | Conceito, intuição geométrica, visualização 2D/3D com PCA e UMAP |
| 02 | [Dimensões Vetoriais](02_vector_dimensions.ipynb) | 384 vs 768 vs 1024 — tradeoffs de memória, velocidade e qualidade |
| 03 | [Tipos de Float](03_float_types.ipynb) | float32, float16, int8, binary — precisão vs tamanho |
| 04 | [Métricas de Distância](04_distance_metrics.ipynb) | Cosine, dot product, euclidean — quando usar cada uma |
| 05 | [Comparação de Modelos](05_models_comparison.ipynb) | MiniLM vs MPNet vs RoBERTa em benchmarks práticos |

## Conceitos Chave

### O que é um embedding?
Um embedding é uma representação densa de dados em um espaço vetorial de alta dimensão.
Textos semanticamente similares ficam **próximos** nesse espaço; textos diferentes ficam **distantes**.

```
"O gato dorme"  → [0.12, -0.34, 0.89, ...]  (384 números)
"O felino repousa" → [0.11, -0.31, 0.91, ...] ← muito próximo!
"A economia cresceu" → [-0.45, 0.67, -0.23, ...] ← distante
```

### Por que dimensões importam?

| Dimensões | Memória (1M vetores) | Tempo de busca | Qualidade semântica |
|-----------|---------------------|----------------|---------------------|
| 384 | 1.53 GB | ⚡⚡⚡ | Boa |
| 768 | 3.07 GB | ⚡⚡ | Muito boa |
| 1024 | 4.09 GB | ⚡ | Excelente |
| 3072 | 12.3 GB | 🐌 | State-of-the-art |

### Por que float types importam?

| Tipo | Bits/dim | 1M vetores 768d | Precisão relativa |
|------|----------|-----------------|-------------------|
| float32 | 32 | 3.07 GB | 100% (baseline) |
| float16 | 16 | 1.54 GB | ~99% |
| int8 | 8 | 768 MB | ~97% |
| binary | 1 | 96 MB | ~60-80% |

## Pré-requisitos

```bash
# Garantir que o ambiente está instalado
uv sync

# Verificar modelos disponíveis (baixados automaticamente pelo sentence-transformers)
python -c "from sentence_transformers import SentenceTransformer; print('OK')"
```

## Referências

- [Documentação sentence-transformers](https://www.sbert.net/)
- [Leaderboard MTEB](https://huggingface.co/spaces/mteb/leaderboard)
- [docs/embeddings/float_types.md](../docs/embeddings/float_types.md)
- [docs/embeddings/dimensions.md](../docs/embeddings/dimensions.md)
