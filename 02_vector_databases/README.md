# Módulo 02 — Vector Databases (Qdrant)

> Aprenda a trabalhar com o Qdrant: collections, indexação HNSW, quantização e busca híbrida.

## Pré-requisitos

```bash
# Qdrant deve estar rodando
docker compose up -d qdrant

# Verificar
curl http://localhost:6333/    # deve retornar {"title":"qdrant",...}
# Web UI: http://localhost:6333/dashboard
```

## Notebooks

| # | Notebook | O que você vai aprender |
|---|----------|------------------------|
| 01 | [Qdrant Intro](01_qdrant_intro.ipynb) | Collections, points, payloads, CRUD básico |
| 02 | [HNSW Indexing](02_hnsw_indexing.ipynb) | Parâmetros m e ef_construct, accuracy vs speed |
| 03 | [Quantização](03_quantization.ipynb) | Scalar quantization (int8) no Qdrant |
| 04 | [Filtering & Hybrid Search](04_filtering_search.ipynb) | Payload filters + busca híbrida (dense + sparse) |

## Conceitos Chave

### Collection
Equivalente a uma "tabela" num banco relacional, mas para vetores.

```python
client.create_collection(
    collection_name="minha_colecao",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
```

### Point
Registro individual: ID + vetor + payload (metadata JSON).

```python
client.upsert(
    collection_name="minha_colecao",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],  # 384 floats
            payload={"texto": "...", "fonte": "doc.pdf", "pagina": 3}
        )
    ]
)
```

### HNSW — Os parâmetros que importam

```
m=16           → Conexões por nó no grafo (↑ = melhor recall, mais RAM)
ef_construct=100 → Candidatos avaliados durante build (↑ = melhor qualidade, mais lento)
ef=128         → Candidatos avaliados durante query (↑ = melhor recall, mais lento)
```

### Quantização Scalar (int8)

```
float32 (4 bytes) → int8 (1 byte)
Redução: 4x de memória, ~2x mais rápido, ~1-3% de perda de recall
```

## Referências

- [Documentação Qdrant](https://qdrant.tech/documentation/)
- [Python Client API](https://python-client.qdrant.tech/)
- [HNSW Paper (Malkov, 2018)](https://arxiv.org/abs/1603.09320)
