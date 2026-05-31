# Guia de Chunking: Como Dividir Documentos para RAG

## Por que chunking importa?

> O chunking é a primeira e mais impactante decisão no pipeline RAG.
> Chunks ruins causam retrieval ruim, que causa respostas ruins — independente do LLM.

**Regra geral:** chunk_size = 256-512 tokens, overlap = 10-20%

---

## Estratégias

### 1. Fixed-Size Chunking

```python
def fixed_size_chunk(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

**Pros:** Simples, previsível, sem dependências  
**Contras:** Corta no meio de frases, ignora estrutura  
**Quando usar:** Prototipagem rápida, textos sem estrutura clara

---

### 2. Recursive Character Splitting (Recomendado)

```python
# LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""],  # prioridade
)
chunks = splitter.split_text(text)
```

**Pros:** Respeita estrutura natural do texto, muito robusto  
**Contras:** Chunks de tamanho variável  
**Quando usar:** **Maioria dos casos** — PDF, artigos, documentação

---

### 3. Semantic Chunking

```python
# Dividir onde a similaridade entre frases cai
def semantic_chunk(text, model, threshold=0.8):
    sentences = split_into_sentences(text)
    embeddings = model.encode(sentences, normalize_embeddings=True)
    
    # Similaridade entre frases consecutivas
    similarities = embeddings[:-1] @ embeddings[1:].T
    diag_sims = np.diag(similarities)
    
    # Dividir onde similaridade cai abaixo do threshold
    split_points = np.where(diag_sims < threshold)[0]
    ...
```

**Pros:** Respeita mudanças de tópico, melhor qualidade  
**Contras:** Requer embedding model, mais lento  
**Quando usar:** Documentos conversacionais, sem estrutura clara

---

### 4. Document-Aware Chunking

```python
# Dividir nos headings do Markdown
import re

def markdown_chunk(text, max_chunk_size=1000):
    # Extrair seções por heading
    sections = re.split(r'\n(?=#{1,3} )', text)
    chunks = []
    for section in sections:
        if len(section) <= max_chunk_size:
            chunks.append(section)
        else:
            # Subdividir seções grandes
            chunks.extend(recursive_chunk(section, max_chunk_size))
    return chunks
```

**Pros:** Preserva contexto semântico completo por seção  
**Contras:** Específico para formato de documento  
**Quando usar:** Markdown, HTML estruturado, documentos técnicos

---

## Parâmetros Recomendados

### chunk_size

| Caso de uso | chunk_size (chars) | ~tokens |
|-------------|-------------------|---------|
| Busca precisa (Q&A) | 256-512 | 64-128 |
| Contexto balanceado | 512-1024 | 128-256 |
| Síntese/Sumarização | 1024-2048 | 256-512 |
| Documentos técnicos | 512-768 | 128-192 |

**Regra prática:** Use 512 caracteres (≈ 128 tokens) como ponto de partida.

### overlap

| Situação | Overlap recomendado |
|----------|---------------------|
| Textos com frases independentes | 10% (50 chars) |
| Textos com contexto contínuo | 20% (100 chars) |
| Contexto muito dependente | 30% (150 chars) |

**Regra prática:** Overlap de 10-15% do chunk_size.

---

## Impacto na Qualidade do RAG

Experimento com `all-mpnet-base-v2` em dataset de artigos técnicos:

| Estratégia | Hit@1 | Hit@5 | MRR |
|-----------|-------|-------|-----|
| Fixed 256 sem overlap | 0.61 | 0.82 | 0.68 |
| Fixed 512 com overlap | 0.68 | 0.87 | 0.74 |
| Recursive 512/64 | **0.75** | **0.91** | **0.81** |
| Semantic threshold=0.8 | 0.74 | 0.90 | 0.80 |
| Document-aware (markdown) | 0.77 | 0.92 | 0.83 |

---

## Decisão de Chunking

```
Tipo de documento?
│
├── Markdown / RST / HTML
│   └── Document-aware (headings como separadores)
│       Fallback: Recursive com "\n##" como separador
│
├── PDF convertido para texto
│   └── Recursive (512 chars, 64 overlap)
│       Cuidado com quebras de página e cabeçalhos
│
├── Código-fonte
│   └── Nunca cortar no meio de uma função/classe
│       Usar AST parsing ou regex por bloco
│
├── Conversas / Diálogos
│   └── Por turno (cada mensagem = 1 chunk)
│       Ou janela deslizante de N turnos
│
├── Documentos jurídicos
│   └── Por cláusula/artigo (document-aware)
│       Overlap maior para contexto de artigos relacionados
│
└── Sem estrutura clara
    └── Semantic chunking (mais lento, melhor qualidade)
```

---

## Metadados nos Chunks

Sempre incluir metadados para filtragem e rastreabilidade:

```python
payload = {
    "text": chunk.text,
    "source": "documento.pdf",
    "page": 5,
    "section": "Introdução",
    "chunk_idx": 3,
    "total_chunks": 15,
    "created_at": "2025-05-31",
}
```

---

## Referências

- [Notebook 02: Chunking Strategies](../../03_rag_fundamentals/02_chunking_strategies.ipynb)
- [LangChain Text Splitters](https://python.langchain.com/docs/how_to/#text-splitters)
- [src/utils/chunking.py](../../src/utils/chunking.py)
