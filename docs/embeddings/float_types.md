# Float Types em Embeddings: Guia de Referência

## Visão Geral

Cada vetor de embedding é armazenado como um array de números de ponto flutuante.
A escolha do tipo de float impacta diretamente **memória**, **velocidade** e **qualidade**.

---

## Tabela Comparativa

| Tipo | Bits/dim | Bytes/dim | Faixa de valores | Precisão decimal | Memória (1M × 768d) |
|------|----------|-----------|-----------------|-----------------|---------------------|
| **float32** | 32 | 4 | ±3.4 × 10^38 | ~7 dígitos | 3.07 GB |
| **float16** | 16 | 2 | ±65,504 | ~3 dígitos | 1.54 GB |
| **int8** | 8 | 1 | -128 a +127 | inteiro | 768 MB |
| **binary** | 1 | 0.125 | 0 ou 1 | 1 bit | 96 MB |

---

## float32 — O Padrão

```python
import numpy as np
vec = np.array([0.123456789, -0.987654321, ...], dtype=np.float32)
# 4 bytes por valor
# Precisão: 7 casas decimais
# Padrão IEEE 754 single precision
```

**Quando usar:**
- Desenvolvimento e experimentação
- Quando qualidade máxima é prioritária
- Collections pequenas (< 100K docs)
- Baseline para comparação

**Representação binária:**
```
Sign(1) | Exponent(8) | Mantissa(23) = 32 bits total
```

---

## float16 — Para GPU

```python
vec_f16 = vec.astype(np.float16)
# 2 bytes por valor (50% de redução)
# Precisão: 3 casas decimais
# Faixa menor: máximo ~65K
```

**Quando usar:**
- Hardware GPU disponível (otimizado em CUDA)
- Modelos de deep learning em inferência
- Redução de memória com mínima perda de qualidade

**Perda de qualidade:**
- Cosine similarity: erro médio < 0.001
- Correlação com float32: > 0.9999

**Atenção:** float16 pode ter **overflow** para valores > 65,504. Para embeddings normalizados (norma = 1), todos os valores ficam em [-1, 1], então é seguro.

---

## int8 — Para Produção CPU

```python
# Quantização linear: float32 → int8
def quantize_int8(vector):
    scale = np.max(np.abs(vector)) / 127.0
    return np.clip(np.round(vector / scale), -127, 127).astype(np.int8)
```

**Quando usar:**
- **Produção em CPU** — caso de uso mais comum
- Qdrant Scalar Quantization
- Redução de custos de infraestrutura
- Coleções com > 1M documentos

**Perda de qualidade:**
- Cosine similarity: erro médio ~0.01-0.03
- Recall@10 vs float32: 97-99%
- Aceleração de busca: ~2x mais rápido

**No Qdrant:**
```python
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType

quantization_config = ScalarQuantization(
    scalar=ScalarQuantizationConfig(
        type=ScalarType.INT8,
        quantile=0.99,  # ignorar 1% dos valores extremos
        always_ram=True,
    )
)
```

---

## binary — Para Larga Escala

```python
# Binarizar: 1 se valor > 0, senão 0
binary_vec = (vector > 0).astype(np.uint8)
# Compactar 8 bits em 1 byte
packed = np.packbits(binary_vec)
```

**Quando usar:**
- Busca aproximada em bilhões de documentos
- Pré-filtragem antes de reranking com float32
- Quando memória é crítica e recall moderado é aceitável

**Perda de qualidade:**
- Cosine similarity: erro médio ~0.05-0.15
- Recall@10 vs float32: 60-80%
- Aceleração: 10-40x (operações com bitwise)

**NÃO usar para:**
- RAG de produção com qualidade relevante
- Domínios com textos muito similares entre si

---

## Comparação de Recall

Usando `all-mpnet-base-v2` (768d) em dataset BEIR:

| Tipo | Recall@10 | Redução de memória | Velocidade relativa |
|------|-----------|-------------------|---------------------|
| float32 | 100% (baseline) | 0% | 1x |
| float16 | 99.8% | 50% | 1.5-2x (GPU) |
| int8 | 97-99% | 75% | 2x |
| binary | 60-80% | 97% | 10-40x |

---

## Guia de Decisão

```
Qual é o requisito?
│
├── Desenvolvimento/pesquisa
│   └── float32 (qualidade máxima, simplicidade)
│
├── Produção com GPU
│   └── float16 (50% RAM, quase sem perda)
│
├── Produção com CPU (> 1M docs)
│   └── int8 Scalar Quantization no Qdrant
│       → rescore=True para manter qualidade
│
└── Escala extrema (> 100M docs) ou pré-filtragem
    └── binary (aceitar recall ~70%)
```

---

## Referências

- [Qdrant Quantization Docs](https://qdrant.tech/documentation/guides/quantization/)
- [Notebook 03: Float Types](../../01_embeddings/03_float_types.ipynb)
- [Notebook 03: Quantização Qdrant](../../02_vector_databases/03_quantization.ipynb)
