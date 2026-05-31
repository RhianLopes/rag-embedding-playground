"""
Complete rebuild of all broken notebooks.
Writes each notebook from scratch with correct cell types and order.
"""
import json, os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"}
        },
        "cells": cells
    }

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.strip()}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.strip()}

def save(notebook, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"Rebuilt: {path}")

# ===========================================================================
# 04 — Distance Metrics
# ===========================================================================
def rebuild_distance_metrics():
    cells = [
        md("""
# 04 — Métricas de Distância: Cosine, Dot Product e Euclidiana

## Por que a métrica importa?

Você tem dois embeddings e quer saber o quão parecidos são os textos. Como calcular essa "parecença"?
A fórmula matemática que você escolhe — a **métrica de distância** — impacta diretamente a qualidade dos resultados.

As três métricas mais usadas em busca vetorial:

| Métrica | O que mede | Range |
|---------|-----------|-------|
| **Cosine** | Ângulo entre os vetores | -1 a 1 (1 = idêntico) |
| **Dot Product** | Projeção de um vetor no outro | -∞ a +∞ |
| **Euclidiana (L2)** | Distância em linha reta | 0 a +∞ (0 = idêntico) |

**Spoiler:** para RAG com texto, a resposta quase sempre é Cosine (ou Dot Product com vetores normalizados, que é equivalente). Este notebook explica por quê — e os casos onde a escolha muda.
"""),
        code("""
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Modelo carregado:", model.get_sentence_embedding_dimension(), "dimensões")
"""),
        md("""
## 4.1 Similaridade de Cosseno

A similaridade de cosseno mede o **ângulo** entre dois vetores — ignora completamente o comprimento (norma).

**Fórmula:** `cos(θ) = (A · B) / (|A| × |B|)`

**Intuição geométrica:** dois vetores saindo da origem. Se apontam para a mesma direção (θ = 0°), cos = 1.
Se perpendiculares (θ = 90°), cos = 0. Se opostos (θ = 180°), cos = -1.

**Por que ignorar o comprimento?** Frases mais longas tendem a ter vetores com norma maior — não porque são
"mais importantes", mas porque têm mais tokens. Se usarmos distância euclidiana, frases longas pareceriam
sempre "mais distantes", mesmo sendo semanticamente idênticas a frases curtas. O cosseno elimina esse viés.
"""),
        code("""
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def dot_product(a, b):
    return np.dot(a, b)

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Frases de teste
frases = [
    "O gato dorme no sofá",           # 0
    "O felino repousa no divã",        # 1 — similar a 0
    "Machine learning usa gradiente",  # 2 — diferente
]
embs = model.encode(frases, normalize_embeddings=True)

print("Similaridade COSINE:")
for i in range(len(frases)):
    for j in range(i+1, len(frases)):
        sim = cosine_similarity(embs[i], embs[j])
        print(f"  '{frases[i][:30]}' × '{frases[j][:30]}' = {sim:.3f}")
"""),
        md("""
### O que o cosseno nos diz?

- **~0.58** entre "gato dorme" e "felino repousa": alta similaridade, sem nenhuma palavra em comum. O modelo entendeu que ambas descrevem a mesma situação.
- **~0.13** entre texto de animal e texto de ML: pouca relação semântica.

Esse poder de capturar semântica sem sobreposição de palavras é o que torna embeddings + cosseno tão úteis para RAG.
"""),
        md("""
## 4.2 Dot Product (Produto Escalar)

**Fórmula:** `A · B = Σ(aᵢ × bᵢ)`

**Relação com cosseno:** quando os vetores estão **normalizados** (norma = 1), dot product e cosine são **matematicamente idênticos**.

A maioria dos modelos modernos (incluindo all-MiniLM-L6-v2 com `normalize_embeddings=True`) retorna vetores normalizados. Nesses casos, você pode usar qualquer um — o resultado é o mesmo.

**Quando divergem:** se os vetores NÃO estão normalizados. Nesse caso, o dot product favorece vetores com norma maior (frases mais longas), o que geralmente é indesejado para texto.

**Vantagem computacional:** o dot product é ligeiramente mais rápido (sem a divisão pelas normas). Para bilhões de operações, isso importa.
"""),
        code("""
# Demonstração: com vetores normalizados, cosine == dot product
print("Com vetores NORMALIZADOS (norma=1):")
print(f"  Cosine = {cosine_similarity(embs[0], embs[1]):.6f}")
print(f"  Dot    = {dot_product(embs[0], embs[1]):.6f}")
print(f"  Diferença: {abs(cosine_similarity(embs[0], embs[1]) - dot_product(embs[0], embs[1])):.8f}")

# Sem normalização — agora divergem
embs_raw = model.encode(frases, normalize_embeddings=False)
print("\nCom vetores NÃO normalizados:")
print(f"  Cosine = {cosine_similarity(embs_raw[0], embs_raw[1]):.6f}")
print(f"  Dot    = {dot_product(embs_raw[0], embs_raw[1]):.6f}")
print(f"  Diferença: {abs(cosine_similarity(embs_raw[0], embs_raw[1]) - dot_product(embs_raw[0], embs_raw[1])):.4f}")
"""),
        md("""
## 4.3 Distância Euclidiana (L2)

**Fórmula:** `d(A, B) = √(Σ(aᵢ - bᵢ)²)`

É a "distância em linha reta" entre dois pontos no espaço vetorial.

**Quando faz sentido:** quando a *posição absoluta* no espaço importa — coordenadas GPS, pixels de imagem, dados físicos.

**Por que geralmente NÃO é ideal para texto:**
1. Afetada pela norma: frases mais longas têm vetores maiores, logo distâncias maiores
2. Em alta dimensão, as distâncias euclidianas ficam todas parecidas (maldição da dimensionalidade)
3. Não invariante a escala

**Exceção:** embeddings de imagem treinados com triplet loss (FaceNet, por exemplo) são otimizados para distância euclidiana.
"""),
        code("""
# Visualização 2D: por que cosine vs euclidean faz diferença

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Vetores 2D para ilustração
A = np.array([2.0, 1.0])   # vetor curto
B = np.array([4.0, 2.0])   # vetor longo, mesma direção de A
C = np.array([1.0, 2.0])   # vetor diferente

for ax, title, normalize in [(axes[0], "Vetores ORIGINAIS", False), (axes[1], "Vetores NORMALIZADOS", True)]:
    vecs = {"A": A.copy(), "B": B.copy(), "C": C.copy()}
    if normalize:
        vecs = {k: v/np.linalg.norm(v) for k, v in vecs.items()}

    colors = {"A": "blue", "B": "red", "C": "green"}
    for name, v in vecs.items():
        ax.annotate("", xy=v, xytext=(0,0), arrowprops=dict(arrowstyle="->", color=colors[name], lw=2))
        ax.text(v[0]+0.05, v[1]+0.05, f"{name} {tuple(v.round(2))}", color=colors[name], fontsize=10)

    # Mostrar métricas
    a, b, c_ = vecs["A"], vecs["B"], vecs["C"]
    ax.set_title(f"{title}\nCos(A,B)={cosine_similarity(a,b):.2f} | Euc(A,B)={euclidean_distance(a,b):.2f}\n"
                 f"Cos(A,C)={cosine_similarity(a,c_):.2f} | Euc(A,C)={euclidean_distance(a,c_):.2f}")
    ax.set_xlim(-0.5, 5); ax.set_ylim(-0.5, 3)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.grid(True, alpha=0.3)

plt.suptitle("A e B têm a mesma direção (só diferem em escala).\nCosine vê isso; Euclidiana não.", fontsize=12)
plt.tight_layout()
plt.show()
"""),
        md("""
### Conclusão da visualização

No gráfico da esquerda (vetores originais):
- **A** e **B** apontam para a mesma direção — semanticamente idênticos
- Cosine(A, B) = 1.0 ✓ — captura que são iguais
- Euclidean(A, B) = grande ✗ — "acha" que são distantes por causa da escala

Isso ilustra por que **cosseno é a escolha correta para texto**: ele é invariante à norma do vetor. Documentos curtos e longos sobre o mesmo assunto ficam igualmente próximos.
"""),
        md("""
## 4.4 Configurando Métricas no Qdrant

No Qdrant, a métrica de distância é configurada na criação da coleção — não pode ser mudada depois sem reindexar.
"""),
        code("""
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    client = QdrantClient(host="localhost", port=6333)
    client.get_collections()
    QDRANT_OK = True
    print("Qdrant conectado")
except Exception as e:
    QDRANT_OK = False
    print(f"Qdrant offline ({e}) — mostrando só o código de configuração")

if QDRANT_OK:
    for metric_name, metric in [("COSINE", Distance.COSINE), ("DOT", Distance.DOT), ("EUCLID", Distance.EUCLID)]:
        col = f"demo_{metric_name.lower()}"
        if client.collection_exists(col):
            client.delete_collection(col)
        client.create_collection(col, vectors_config=VectorParams(size=384, distance=metric))
        print(f"Coleção '{col}' criada com {metric_name}")
"""),
        md("""
## Resumo: Qual métrica escolher?

```
Tipo de dado?
├── Texto semântico (maioria dos casos RAG)
│   └── Vetores normalizados? → COSINE ou DOT (equivalentes)
│       Vetores não normalizados? → COSINE (mais seguro)
├── Imagens (FaceNet, ArcFace)
│   └── EUCLID (modelos treinados para isso)
└── Ranking com magnitude (DPR, ColBERT)
    └── DOT não normalizado (magnitude = relevância)
```

| Métrica | Fórmula | Quando usar |
|---------|---------|-------------|
| **Cosine** | cos(θ) = A·B / (\|A\|\|B\|) | **Padrão para RAG com texto** |
| **Dot Product** | A · B = Σaᵢbᵢ | Texto normalizado (= cosine) ou ranking com magnitude |
| **Euclidiana** | √Σ(aᵢ-bᵢ)² | Imagens, dados físicos, quando posição absoluta importa |

**Para o seu sistema RAG:** use `Distance.COSINE` no Qdrant. Você estará correto em 95%+ dos casos.

**Próximos passos:**
- [05 — Comparação de Modelos](05_models_comparison.html): qual modelo de embedding escolher?
- [01 — Qdrant Intro](../02_vector_databases/01_qdrant_intro.html): como criar coleções com cada métrica
"""),
    ]
    save(nb(cells), "01_embeddings/04_distance_metrics.ipynb")

# ===========================================================================
# 02 — Chunking Strategies
# ===========================================================================
def rebuild_chunking():
    cells = [
        md("""
# 02 — Estratégias de Chunking

## Por que chunking é um dos problemas mais importantes do RAG?

Na prática, chunking mal feito é responsável por boa parte das falhas de sistemas RAG.
Um LLM excelente com chunks ruins gera respostas ruins. Chunks bons com um LLM mediano ainda funcionam razoavelmente.

**O desafio:** dividir documentos em pedaços que sejam:
- Pequenos o suficiente para caber no contexto do LLM (tipicamente 512-2048 tokens)
- Grandes o suficiente para conter uma ideia **completa**
- Que não quebrem no meio de conceitos importantes
- Similares em tamanho (para evitar que um chunk domine os resultados)

**Estratégias que vamos comparar:**

| Estratégia | Complexidade | Quando usar |
|-----------|-------------|-------------|
| Fixed-size | Baixa | Protótipos, textos uniformes |
| Recursive split | Média | **Recomendado como padrão** |
| Semantic chunking | Alta | Documentos longos e heterogêneos |
"""),
        code("""
# Setup
import re
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Texto de exemplo — artigo técnico sobre RAG
texto_exemplo = \"\"\"
RAG (Retrieval-Augmented Generation) é uma técnica que combina recuperação de informação com geração de texto.

O processo de indexação converte documentos em embeddings vetoriais que são armazenados num banco de dados vetorial.
Durante a consulta, a query do usuário é convertida em embedding e os documentos mais similares são recuperados.

HNSW (Hierarchical Navigable Small World) é o algoritmo de indexação mais usado em bancos vetoriais.
Ele constrói um grafo hierárquico onde nós são conectados a vizinhos próximos em múltiplas camadas.
A busca começa na camada superior (poucos nós, conexões longas) e desce até encontrar os vizinhos mais próximos.

Embeddings são representações densas de texto em espaços de alta dimensão.
Modelos como all-MiniLM-L6-v2 geram vetores de 384 dimensões treinados para capturar semântica.
Textos semanticamente similares ficam próximos no espaço vetorial, permitindo busca por similaridade.

Chunking é o processo de dividir documentos longos em pedaços menores para indexação.
A escolha da estratégia de chunking afeta diretamente a qualidade do retrieval.
Chunks muito pequenos perdem contexto; chunks muito grandes são menos precisos na busca.
\"\"\"

print(f"Texto de exemplo: {len(texto_exemplo)} caracteres, {len(texto_exemplo.split())} palavras")
"""),
        md("""
## 2.1 Fixed-Size Chunking

A estratégia mais simples: divide o texto a cada N caracteres, com overlap de M caracteres.

**Como funciona:** corta o texto mecanicamente, sem considerar estrutura.

**Overlap:** os últimos M caracteres do chunk anterior são repetidos no início do próximo.
Isso evita que informações que "cruzam" a fronteira se percam completamente.

**Vantagem:** simples, previsível, sem dependências extras.

**Desvantagem:** pode cortar no meio de uma frase ou ideia:
```
Chunk 1: "...O algoritmo HNSW usa uma estrutura hierárquica de"
Chunk 2: "grafos onde nós são conectados a vizinhos próximos..."
```
Chunk 1 termina no meio de uma frase — se recuperado isoladamente, perde o contexto.
"""),
        code("""
def fixed_size_chunk(text, chunk_size=500, overlap=50):
    \"\"\"Divide texto em pedaços de tamanho fixo com overlap.\"\"\"
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

chunks_fixed = fixed_size_chunk(texto_exemplo, chunk_size=300, overlap=50)

print(f"Fixed-size (300 chars, overlap=50): {len(chunks_fixed)} chunks")
for i, c in enumerate(chunks_fixed):
    print(f"  Chunk {i}: {len(c)} chars | '{c[:80].strip()}...'")
"""),
        md("""
### Observação: onde corta?

Note que o fixed-size pode cortar no meio de uma frase.
O texto sobre HNSW provavelmente ficou dividido em pontos arbitrários — não em fronteiras semânticas.

Para textos com estrutura clara (parágrafos, seções), isso é problemático.
Para textos uniformes (transcrições de fala, logs), funciona razoavelmente.
"""),
        md("""
## 2.2 Recursive Character Split

Uma evolução do fixed-size: em vez de cortar sempre no mesmo ponto, tenta **respeitar a estrutura** do texto.

**Como funciona:** define uma hierarquia de separadores:
1. `\\n\\n` (parágrafo) — divide aqui se possível
2. `\\n` (linha) — se o chunk ainda for grande, divide por linha
3. `. ` (frase) — se ainda for grande, divide por frase
4. ` ` (palavra) — último recurso

O algoritmo aplica esses separadores recursivamente até que todos os chunks estejam abaixo do tamanho máximo.

**Resultado:** chunks que respeitam parágrafos e frases — muito mais coerentes semanticamente.

**Este é o padrão recomendado para a maioria dos casos.** LangChain e LlamaIndex usam isso como default.
"""),
        code("""
def recursive_chunk(text, chunk_size=500, overlap=50, separators=None):
    \"\"\"Divide texto respeitando hierarquia de separadores.\"\"\"
    if separators is None:
        separators = ["\\n\\n", "\\n", ". ", " ", ""]

    def split_text(text, separators):
        if not separators or len(text) <= chunk_size:
            return [text] if text.strip() else []

        sep = separators[0]
        parts = text.split(sep) if sep else list(text)

        chunks = []
        current = ""
        for part in parts:
            test = current + (sep if current else "") + part
            if len(test) <= chunk_size:
                current = test
            else:
                if current.strip():
                    chunks.append(current.strip())
                if len(part) > chunk_size:
                    # Parte ainda grande: recursão com próximo separador
                    sub_chunks = split_text(part, separators[1:])
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part
        if current.strip():
            chunks.append(current.strip())
        return chunks

    raw_chunks = split_text(text, separators)

    # Adiciona overlap
    result = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and overlap > 0:
            prev_end = raw_chunks[i-1][-overlap:]
            chunk = prev_end + " " + chunk
        result.append(chunk)
    return result

chunks_recursive = recursive_chunk(texto_exemplo, chunk_size=300, overlap=50)

print(f"Recursive split (300 chars, overlap=50): {len(chunks_recursive)} chunks")
for i, c in enumerate(chunks_recursive):
    print(f"  Chunk {i}: {len(c)} chars | '{c[:80].strip()}...'")
"""),
        md("""
### Diferença chave em relação ao fixed-size

O recursive split tende a terminar chunks em pontos "naturais" do texto — fim de parágrafo, fim de frase.
Compare com o fixed-size: o conteúdo dos chunks é mais coerente e cada um tende a representar uma ideia completa.

Isso impacta diretamente o retrieval: se o chunk contém uma ideia completa, o embedding captura melhor seu significado.
"""),
        md("""
## 2.3 Semantic Chunking

A estratégia mais sofisticada: divide onde o **significado muda**, não onde o texto tem quebras.

**Como funciona:**
1. Divide o texto em sentenças
2. Computa o embedding de cada sentença (ou janela de sentenças)
3. Calcula a similaridade entre sentenças consecutivas
4. Quando a similaridade cai abruptamente (mudança de tópico), cria um novo chunk

**Vantagem:** captura fronteiras temáticas reais — cada chunk trata de um assunto coeso.

**Custo:** precisa embedar todas as sentenças antes de chunkar. Para documentos longos, isso pode ser 3-5x mais lento que os outros métodos.

**Quando vale:** documentos longos e heterogêneos onde mudanças de tópico são frequentes e importantes.
"""),
        code("""
def semantic_chunk(text, model, threshold=0.7, min_chunk_size=100):
    \"\"\"Divide texto em pontos de mudança semântica.\"\"\"
    # Divide em sentenças
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\\s+', text) if s.strip()]
    if len(sentences) <= 1:
        return [text]

    # Embeda cada sentença
    embs = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)

    # Calcula similaridade entre sentenças consecutivas
    similarities = []
    for i in range(len(sentences) - 1):
        sim = float(np.dot(embs[i], embs[i+1]))
        similarities.append(sim)

    # Identifica pontos de quebra (baixa similaridade = mudança de tópico)
    chunks = []
    current_chunk = sentences[0]

    for i, sim in enumerate(similarities):
        if sim < threshold and len(current_chunk) >= min_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentences[i+1]
        else:
            current_chunk += " " + sentences[i+1]

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

chunks_semantic = semantic_chunk(texto_exemplo, model, threshold=0.65)

print(f"Semantic chunking (threshold=0.65): {len(chunks_semantic)} chunks")
for i, c in enumerate(chunks_semantic):
    print(f"  Chunk {i}: {len(c)} chars")
    print(f"    '{c[:100].strip()}...'")
    print()
"""),
        md("""
### O que o semantic chunking encontrou?

Se funcionou bem, cada chunk deve corresponder a um dos tópicos do texto:
- Chunk sobre RAG em geral
- Chunk sobre HNSW
- Chunk sobre Embeddings
- Chunk sobre Chunking

Compare com os chunks fixed-size: os limites faziam sentido semântico aqui?

**Limitação importante:** o threshold (0.65) precisa ser ajustado para cada tipo de documento.
Um threshold muito alto = chunks minúsculos. Muito baixo = chunks enormes.
"""),
        md("""
## 2.4 Comparação: Impacto na Qualidade de Retrieval

Métricas de texto não são suficientes — precisamos medir o impacto no que realmente importa: **o sistema encontra o chunk certo para cada query?**
"""),
        code("""
# Compara retrieval quality das 3 estratégias
queries_com_resposta = [
    ("Como funciona o HNSW?", "HNSW"),
    ("O que é um embedding?", "Embeddings"),
    ("Como o RAG funciona?", "RAG"),
    ("Por que chunking importa?", "Chunking"),
]

results = {}
for strategy_name, chunks in [
    ("Fixed-size", chunks_fixed),
    ("Recursive", chunks_recursive),
    ("Semantic", chunks_semantic),
]:
    chunk_embs = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    hits = 0
    for query, expected_topic in queries_com_resposta:
        q_emb = model.encode(query, normalize_embeddings=True)
        scores = chunk_embs @ q_emb
        best_chunk = chunks[np.argmax(scores)]
        # Verifica se o chunk mais relevante contém o tópico esperado
        if expected_topic.lower() in best_chunk.lower():
            hits += 1
    results[strategy_name] = hits / len(queries_com_resposta)
    print(f"{strategy_name}: {hits}/{len(queries_com_resposta)} queries com chunk correto como top-1")

print("\nRecall@1 por estratégia:")
for name, recall in sorted(results.items(), key=lambda x: -x[1]):
    bar = "█" * int(recall * 20)
    print(f"  {name:12s}: {bar} {recall:.0%}")
"""),
        md("""
### O que os números revelam?

A diferença entre estratégias mostra por que chunking merece atenção:
- **Fixed-size** frequentemente corta no meio de ideias, misturando tópicos num mesmo chunk
- **Recursive** respeita parágrafos, gerando chunks mais coesos — melhor recall
- **Semantic** encontra fronteiras de tópico reais, mas depende do threshold escolhido

**Insight chave:** invista tempo escolhendo a estratégia de chunking certa *antes* de otimizar o modelo de embedding ou o LLM. Um bom chunking com um modelo simples supera um modelo excelente com chunks ruins.

## Resumo

| Estratégia | Recall típico | Custo | Recomendação |
|-----------|--------------|-------|--------------|
| Fixed-size | Baixo-médio | O(1) | Só para protótipos |
| **Recursive split** | **Médio-alto** | **O(N)** | **Padrão recomendado** |
| Semantic | Alto | O(N×M) | Documentos longos e heterogêneos |

**Parâmetros para começar:**
- `chunk_size`: 256-512 caracteres (ou tokens)
- `overlap`: 10-20% do chunk_size (garante contexto nas fronteiras)

**Próximos passos:**
- [03 — Retrieval Strategies](03_retrieval_strategies.html): você tem bons chunks — como encontrá-los eficientemente?
"""),
    ]
    save(nb(cells), "03_rag_fundamentals/02_chunking_strategies.ipynb")

# ===========================================================================
# 03 — Retrieval Strategies
# ===========================================================================
def rebuild_retrieval():
    cells = [
        md("""
# 03 — Estratégias de Retrieval

## Por que retrieval importa tanto?

Uma resposta de RAG só pode ser tão boa quanto os documentos recuperados.
Se o retrieval falha em encontrar o chunk relevante, o LLM não tem como responder — não importa quão bom seja o modelo.

**Falhas comuns do retrieval simples (dense only):**
- Query usa termos técnicos exatos que o modelo de embedding não captura bem
- Documentos usam sinônimos ou abreviações que o embedding aproxima mal
- Queries muito curtas ("erro 404") têm embeddings pouco informativos
- Keywords raras ou nomes próprios específicos

**A solução:** combinar tipos de retrieval para cobrir pontos cegos de cada um.

| Estratégia | Captura | Falha em |
|-----------|---------|----------|
| Dense (embedding) | Semântica, sinônimos | Keywords exatas, termos raros |
| Sparse (BM25) | Keywords exatas | Sinônimos, contexto |
| **Hybrid (RRF)** | **Ambos** | **Quase nada — padrão de produção** |
| MMR | Relevância + diversidade | Quando você quer só os mais similares |
"""),
        code("""
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import math

model = SentenceTransformer("all-MiniLM-L6-v2")

# Base de documentos para demonstração
documentos = [
    "HNSW é um algoritmo de indexação vetorial que usa grafos hierárquicos para busca eficiente",
    "O algoritmo Hierarchical Navigable Small World permite busca aproximada em O(log N)",
    "Embeddings são vetores densos que representam o significado semântico de textos",
    "A similaridade de cosseno mede o ângulo entre dois vetores de embedding",
    "RAG combina retrieval de documentos com geração de texto para responder perguntas",
    "Chunking divide documentos longos em pedaços menores para indexação vetorial",
    "BM25 é um algoritmo de ranking baseado em frequência de termos para busca textual",
    "Qdrant é um banco de dados vetorial open-source otimizado para busca semântica",
    "Quantização reduz o uso de memória dos vetores com perda mínima de qualidade",
    "Retrieval-Augmented Generation melhora LLMs com conhecimento externo atualizado",
]

# Pré-computar embeddings dos documentos
doc_embs = model.encode(documentos, normalize_embeddings=True)
print(f"Base: {len(documentos)} documentos indexados")
print(f"Dimensão dos embeddings: {doc_embs.shape[1]}d")
"""),
        md("""
## 3.1 Dense Retrieval (Embedding-based)

O retrieval denso é o que você já conhece: embedding da query → similaridade cosine → top-K.

**Ponto forte:** entende *semântica*. "cachorro" e "cão" são similares.
"como resolver problema X" encontra documentos sobre "soluções para X" sem usar exatamente essas palavras.

**Ponto fraco:** para queries muito específicas com termos técnicos exatos (nomes de erro, siglas, nomes próprios incomuns),
o embedding pode não capturar a especificidade necessária.

**Também falha com negação:** "o que NÃO é machine learning?" — o embedding de
"NÃO machine learning" fica próximo de embeddings de ML, porque o modelo entende o conceito, não a negação.
"""),
        code("""
def dense_retrieve(query, top_k=3):
    q_emb = model.encode(query, normalize_embeddings=True)
    scores = doc_embs @ q_emb
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(i, documentos[i], float(scores[i])) for i in top_indices]

# Teste com diferentes tipos de query
test_queries = [
    "como funciona a busca em grafos hierárquicos",  # semântica — dense deve ir bem
    "BM25 TF-IDF",                                   # keyword exata — dense pode falhar
    "RAG",                                            # sigla — pode ser ambíguo
]

print("=== DENSE RETRIEVAL ===")
for query in test_queries:
    results = dense_retrieve(query, top_k=2)
    print(f"\nQuery: '{query}'")
    for rank, (idx, doc, score) in enumerate(results, 1):
        print(f"  #{rank} (score={score:.3f}): {doc[:70]}...")
"""),
        md("""
### Análise dos resultados

Para a query semântica ("busca em grafos hierárquicos"), o dense provavelmente encontrou documentos sobre HNSW — mesmo sem essas palavras na query.

Para "BM25 TF-IDF" — uma keyword muito específica — o dense pode não ter performado tão bem, porque o embedding de "BM25 TF-IDF" pode ser próximo de documentos genéricos sobre busca.

Isso motiva o BM25.
"""),
        md("""
## 3.2 Sparse Retrieval (BM25)

BM25 é uma versão melhorada do TF-IDF — o algoritmo clássico de busca textual.

**Como funciona:** pondera palavras por:
- **TF (Term Frequency):** quantas vezes a palavra aparece no documento
- **IDF (Inverse Document Frequency):** palavras raras recebem mais peso que palavras comuns
- **Normalização por tamanho:** documentos longos não ganham vantagem injusta

**Ponto forte:** encontra documentos que contêm exatamente as palavras da query.
Para queries como "CUDA out of memory error", BM25 vai direto nos documentos com essas palavras.

**Ponto fraco:** completamente cego a semântica.
"cachorro" e "cão" são palavras totalmente diferentes para o BM25.
"""),
        code("""
class BM25:
    \"\"\"Implementação simples de BM25 para demonstração.\"\"\"
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.tokenized = [doc.lower().split() for doc in docs]
        self.avgdl = sum(len(d) for d in self.tokenized) / len(self.tokenized)

        # Compute IDF
        self.idf = {}
        N = len(docs)
        for doc in self.tokenized:
            for word in set(doc):
                self.idf[word] = self.idf.get(word, 0) + 1
        self.idf = {w: math.log((N - df + 0.5) / (df + 0.5) + 1)
                    for w, df in self.idf.items()}

    def score(self, query, doc_idx):
        query_tokens = query.lower().split()
        doc = self.tokenized[doc_idx]
        dl = len(doc)
        score = 0
        tf_map = defaultdict(int)
        for w in doc:
            tf_map[w] += 1
        for term in query_tokens:
            if term not in self.idf:
                continue
            tf = tf_map.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf[term] * numerator / denominator
        return score

    def retrieve(self, query, top_k=3):
        scores = [(i, self.score(query, i)) for i in range(len(self.docs))]
        scores.sort(key=lambda x: -x[1])
        return [(idx, self.docs[idx], sc) for idx, sc in scores[:top_k]]

bm25 = BM25(documentos)

print("=== SPARSE RETRIEVAL (BM25) ===")
for query in test_queries:
    results = bm25.retrieve(query, top_k=2)
    print(f"\nQuery: '{query}'")
    for rank, (idx, doc, score) in enumerate(results, 1):
        print(f"  #{rank} (score={score:.3f}): {doc[:70]}...")
"""),
        md("""
### Dense vs Sparse: cada um tem seu ponto cego

Compare os resultados das duas estratégias:

- Para queries semânticas: dense tende a ir melhor (encontra sinônimos e contexto)
- Para keywords específicas ("BM25 TF-IDF"): sparse vai melhor (match exato de termos)

Nenhuma estratégia domina a outra em todos os casos. Isso motiva a combinação das duas.
"""),
        md("""
## 3.3 Hybrid Search com RRF (Reciprocal Rank Fusion)

Hybrid search combina os rankings de dense e sparse para cobrir os pontos cegos de cada um.

**RRF** é o algoritmo de fusão mais popular:

```
score_rrf(doc) = 1/(k + rank_dense) + 1/(k + rank_sparse)
```

Onde `k=60` é uma constante empírica. Um documento que aparece bem nos dois rankings tem score máximo.
Um documento que aparece só em um ainda contribui com algum score.

**Por que não somar os scores diretamente?**
Scores de sistemas diferentes têm escalas incompatíveis (cosine: 0-1, BM25: 0-∞).
RRF usa apenas as *posições* (ranks), não os valores absolutos — isso é escala-invariante.
"""),
        code("""
def hybrid_retrieve(query, top_k=3, k_rrf=60, alpha=0.5):
    \"\"\"
    Hybrid search com RRF.
    alpha=1.0 -> só dense, alpha=0.0 -> só sparse, alpha=0.5 -> equilíbrio
    \"\"\"
    # Dense rankings
    q_emb = model.encode(query, normalize_embeddings=True)
    dense_scores = doc_embs @ q_emb
    dense_ranking = {idx: rank+1 for rank, idx in enumerate(np.argsort(dense_scores)[::-1])}

    # Sparse rankings
    bm25_scores = [(i, bm25.score(query, i)) for i in range(len(documentos))]
    bm25_scores.sort(key=lambda x: -x[1])
    sparse_ranking = {idx: rank+1 for rank, (idx, _) in enumerate(bm25_scores)}

    # RRF fusion
    rrf_scores = {}
    for doc_idx in range(len(documentos)):
        dense_rrf = alpha / (k_rrf + dense_ranking.get(doc_idx, len(documentos)))
        sparse_rrf = (1-alpha) / (k_rrf + sparse_ranking.get(doc_idx, len(documentos)))
        rrf_scores[doc_idx] = dense_rrf + sparse_rrf

    top = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
    return [(idx, documentos[idx], score) for idx, score in top]

print("=== HYBRID SEARCH (RRF, alpha=0.5) ===")
for query in test_queries:
    results = hybrid_retrieve(query, top_k=2)
    print(f"\nQuery: '{query}'")
    for rank, (idx, doc, score) in enumerate(results, 1):
        print(f"  #{rank} (rrf={score:.4f}): {doc[:70]}...")
"""),
        md("""
### Comparação lado a lado

Vamos comparar as 3 estratégias nas mesmas queries para ver onde cada uma vai bem e onde falha:
"""),
        code("""
print("=== COMPARAÇÃO FINAL ===")
print(f"{'Query':<45} {'Dense':^20} {'Sparse':^20} {'Hybrid':^20}")
print("-" * 105)

for query in test_queries:
    d = dense_retrieve(query, 1)[0][1][:35]
    s = bm25.retrieve(query, 1)[0][1][:35]
    h = hybrid_retrieve(query, 1)[0][1][:35]
    print(f"{query:<45} {d:<20} {s:<20} {h:<20}")
"""),
        md("""
### O que a comparação mostra?

O Hybrid (RRF) tende a ser mais robusto: ele "herda" os acertos de cada estratégia e raramente falha em ambas ao mesmo tempo.

**Regra prática:**
- Hybrid com alpha=0.5 é um excelente padrão para começar
- Se suas queries são muito técnicas com keywords específicas: alpha=0.3 (mais peso no sparse)
- Se suas queries são linguagem natural: alpha=0.7 (mais peso no dense)
"""),
        md("""
## 3.4 MMR (Max Marginal Relevance)

Um problema sutil do retrieval padrão: os top-K resultados podem ser muito similares entre si.

Se você busca "como funciona gradient descent?" e os 5 chunks mais relevantes são parágrafos consecutivos
do mesmo artigo, você está desperdiçando espaço de contexto com informação redundante.

**MMR** resolve isso balanceando **relevância** com **diversidade**:

```
MMR = argmax[λ × sim(doc, query) - (1-λ) × max(sim(doc, já_selecionados))]
```

- λ = 1.0: só relevância (igual ao retrieval padrão)
- λ = 0.5: equilíbrio — bom para cobrir múltiplos ângulos de uma questão
"""),
        code("""
def mmr_retrieve(query, top_k=3, lambda_param=0.5):
    \"\"\"Max Marginal Relevance: balanceia relevância com diversidade.\"\"\"
    q_emb = model.encode(query, normalize_embeddings=True)
    query_scores = doc_embs @ q_emb

    selected = []
    candidates = list(range(len(documentos)))

    while len(selected) < top_k and candidates:
        best_idx = None
        best_score = -np.inf

        for idx in candidates:
            relevance = float(query_scores[idx])
            # Penaliza por similaridade com docs já selecionados
            if selected:
                sim_to_selected = max(float(doc_embs[idx] @ doc_embs[s]) for s in selected)
            else:
                sim_to_selected = 0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        candidates.remove(best_idx)

    return [(idx, documentos[idx], float(query_scores[idx])) for idx in selected]

query = "como funciona a busca vetorial"
print("=== MMR vs DENSE para: '{}' ===".format(query))
print("\nDense (pode ter redundância):")
for rank, (idx, doc, score) in enumerate(dense_retrieve(query, 4), 1):
    print(f"  #{rank}: {doc}")

print("\nMMR (mais diverso):")
for rank, (idx, doc, score) in enumerate(mmr_retrieve(query, 4), 1):
    print(f"  #{rank}: {doc}")
"""),
        md("""
## Resumo

| Estratégia | Recall semântico | Recall keyword | Diversidade | Complexidade |
|-----------|:---:|:---:|:---:|:---:|
| Dense | ★★★★★ | ★★☆☆☆ | ★★☆☆☆ | Baixa |
| Sparse (BM25) | ★★☆☆☆ | ★★★★★ | ★★☆☆☆ | Baixa |
| **Hybrid RRF** | **★★★★★** | **★★★★★** | **★★☆☆☆** | **Média** |
| MMR | ★★★★☆ | ★★☆☆☆ | ★★★★★ | Média |

**Para o seu sistema RAG:**
- Comece com **Hybrid RRF (alpha=0.5)** — cobre os pontos cegos de cada estratégia
- Use **MMR** quando suas queries pedem cobertura de múltiplos aspectos (sumarização, Q&A amplo)
- Use **Dense puro** apenas quando seus documentos são todos linguagem natural sem keywords técnicas

**Próximos passos:**
- [04 — Generation Prompts](04_generation_prompts.html): com os documentos certos em mãos, como pedir ao LLM para gerar a melhor resposta?
"""),
    ]
    save(nb(cells), "03_rag_fundamentals/03_retrieval_strategies.ipynb")

# ===========================================================================
# 04 — Generation Prompts
# ===========================================================================
def rebuild_prompts():
    cells = [
        md("""
# 04 — Prompt Engineering para RAG

## O prompt é o último quilômetro

Você indexou bem, fez retrieval eficiente, recuperou os documentos certos.
Agora vem o último passo: pedir ao LLM que gere uma resposta.

O prompt determina:
- **Fidelidade:** o LLM usa apenas o contexto fornecido ou "inventa" informações?
- **Formato:** a resposta é curta e direta ou longa e detalhada?
- **Atribuição:** o LLM cita as fontes?
- **Honestidade:** quando a resposta não está no contexto, o LLM admite ou alucina?

**Alucinação** é o maior risco de sistemas RAG. Um LLM bem instruído diz
"não encontrei essa informação nos documentos". Um LLM mal instruído inventa uma resposta convincente mas falsa.

Este notebook mostra a evolução do prompt: do básico ao production-grade.

> **Pré-requisito:** Ollama rodando com Llama 3.2 (`docker compose up -d` ou `ollama pull llama3.2`)
"""),
        code("""
import httpx, json

# Verificar se Ollama está disponível
try:
    r = httpx.get("http://localhost:11434/api/tags", timeout=3)
    models = [m["name"] for m in r.json().get("models", [])]
    LLM_OK = True
    print(f"Ollama disponível. Modelos: {models}")
    LLM_MODEL = "llama3.2" if any("llama3.2" in m for m in models) else models[0] if models else None
    print(f"Usando modelo: {LLM_MODEL}")
except Exception as e:
    LLM_OK = False
    LLM_MODEL = None
    print(f"Ollama não disponível ({e})")
    print("Os prompts serão mostrados mas não executados.")
"""),
        code("""
def gerar(prompt, model=LLM_MODEL, max_tokens=400):
    \"\"\"Chama o Ollama e retorna a resposta.\"\"\"
    if not LLM_OK:
        print("[Ollama offline — resposta simulada]")
        return "[Ollama não disponível]"
    try:
        r = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"num_predict": max_tokens, "temperature": 0.1}},
            timeout=60
        )
        return r.json()["response"]
    except Exception as e:
        return f"[Erro: {e}]"

# Contexto de exemplo — simula o que o retrieval retornaria
contexto_chunks = [
    \"\"\"HNSW (Hierarchical Navigable Small World) é um algoritmo de indexação vetorial.
Ele constrói um grafo hierárquico onde cada nó é conectado a seus vizinhos mais próximos.
A busca começa nas camadas superiores (poucos nós, conexões longas) e desce progressivamente.
Isso permite busca aproximada em O(log N) com ~95-99% de recall.\"\"\",

    \"\"\"Parâmetros do HNSW:
- m: número de conexões por nó (padrão=16). Mais conexões = melhor recall, mais memória.
- ef_construct: candidatos considerados durante indexação (padrão=100).
- ef: candidatos na busca (padrão=128). Aumentar melhora recall sem precisar reindexar.\"\"\",

    \"\"\"Qdrant é um banco de dados vetorial open-source escrito em Rust.
Suporta HNSW nativo, quantização escalar e por produto.
Permite combinar busca vetorial com filtros em metadados (payload) numa única query.\"\"\",
]

pergunta = "Quais são os parâmetros do HNSW e como eles afetam a performance?"
print("Pergunta:", pergunta)
print(f"\n{len(contexto_chunks)} chunks recuperados")
"""),
        md("""
## 4.1 Prompt Básico

O prompt mais simples possível: forneça o contexto e faça a pergunta.

**Problema:** sem instrução explícita, o LLM tende a "completar o texto de forma coerente" —
o que significa potencialmente inventar informações que não estão nos documentos.
"""),
        code("""
PROMPT_BASICO = \"\"\"Contexto:
{contexto}

Pergunta: {pergunta}
Resposta:\"\"\"

ctx = "\\n\\n".join(contexto_chunks)
prompt = PROMPT_BASICO.format(contexto=ctx, pergunta=pergunta)
print("=== PROMPT BÁSICO ===")
print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
print()
resposta_basica = gerar(prompt)
print("RESPOSTA:", resposta_basica)
"""),
        md("""
## 4.2 Prompt com Grounding Explícito

A solução para reduzir alucinação: instrua **explicitamente** o LLM a responder apenas com base no contexto.

A adição de frases como *"Responda SOMENTE com base nas informações fornecidas"* e
*"Se a resposta não estiver no contexto, diga 'Não encontrei essa informação'"* reduz drasticamente a taxa de alucinação.

**Por que funciona?** LLMs são treinados para seguir instruções.
Quando você explicita o comportamento esperado, o modelo ajusta seu output.
Instruções precisas ("SOMENTE com base no contexto") são mais efetivas que instruções vagas ("use o contexto").
"""),
        code("""
PROMPT_GROUNDED = \"\"\"Você é um assistente especializado. Responda SOMENTE com base nos documentos abaixo.
Se a informação solicitada não estiver nos documentos, responda exatamente:
"Não encontrei essa informação nos documentos fornecidos."
Não adicione nenhuma informação que não esteja explicitamente nos documentos.

DOCUMENTOS:
{contexto}

PERGUNTA: {pergunta}

RESPOSTA:\"\"\"

prompt = PROMPT_GROUNDED.format(contexto=ctx, pergunta=pergunta)
print("=== PROMPT COM GROUNDING ===")
resposta_grounded = gerar(prompt)
print("RESPOSTA:", resposta_grounded)
"""),
        md("""
## 4.3 Prompt com Citações

Para sistemas onde rastreabilidade é importante, pedir ao LLM que cite as fontes muda tudo.

**Por que citações são importantes:**
1. **Verificação:** o usuário pode confirmar na fonte original
2. **Confiança:** respostas com fontes parecem (e geralmente são) mais confiáveis
3. **Debug:** quando o sistema erra, você sabe qual chunk causou o problema
4. **Compliance:** em domínios regulados, rastreabilidade pode ser requisito
"""),
        code("""
PROMPT_COM_CITACOES = \"\"\"Você é um assistente especializado. Responda SOMENTE com base nos documentos numerados abaixo.
Cite as fontes usando [1], [2], etc. ao final de cada afirmação.
Se a informação não estiver nos documentos, diga: "Não encontrei essa informação."

{docs_numerados}

PERGUNTA: {pergunta}

RESPOSTA (com citações):\"\"\"

docs_numerados = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(contexto_chunks))
prompt = PROMPT_COM_CITACOES.format(docs_numerados=docs_numerados, pergunta=pergunta)
print("=== PROMPT COM CITAÇÕES ===")
resposta_citacoes = gerar(prompt)
print("RESPOSTA:", resposta_citacoes)
"""),
        md("""
## 4.4 Teste de Alucinação

O teste mais importante: **o que acontece quando você pergunta algo que NÃO está nos documentos?**

Um sistema RAG bem construído deve dizer "não sei" de forma honesta.
Um sistema mal construído vai inventar uma resposta plausível — e isso é perigoso.
"""),
        code("""
pergunta_fora = "Qual é o preço do plano enterprise do Qdrant?"

print("=== TESTE DE ALUCINAÇÃO ===")
print(f"Pergunta: '{pergunta_fora}'")
print(f"(A resposta NÃO está nos documentos)\n")

print("--- Prompt Básico (propenso a alucinar) ---")
prompt_basico = PROMPT_BASICO.format(contexto=ctx, pergunta=pergunta_fora)
r1 = gerar(prompt_basico, max_tokens=150)
print(r1)

print("\n--- Prompt Grounded (deve recusar) ---")
prompt_grounded = PROMPT_GROUNDED.format(contexto=ctx, pergunta=pergunta_fora)
r2 = gerar(prompt_grounded, max_tokens=150)
print(r2)
"""),
        md("""
### O que o teste revela?

A diferença entre os prompts é dramática em cenários de alucinação:

- **Prompt básico:** frequentemente gera respostas inventadas que *parecem* corretas
- **Prompt grounded:** força o LLM a admitir quando não sabe

**Insight crítico:** testar com perguntas "fora do escopo" é tão importante quanto testar com perguntas que o sistema deveria saber responder. Sistemas RAG em produção sempre recebem queries para as quais não têm resposta.

**Como testar sistematicamente:**
1. Conjunto A: queries com resposta conhecida (mede acurácia)
2. Conjunto B: queries cujas respostas não estão nos documentos (mede taxa de alucinação)

Ambos os conjuntos são necessários para uma avaliação completa.
"""),
        md("""
## Resumo: Evolução do Prompt

| Versão | Alucinação | Rastreabilidade | Complexidade |
|--------|:---:|:---:|:---:|
| Básico | Alta | Nenhuma | Mínima |
| Com grounding | Baixa | Nenhuma | Baixa |
| Com citações | Baixa | Alta | Média |

**Template de produção recomendado:**
```
Você é um assistente especializado. Responda APENAS com base nos documentos abaixo.
Se a resposta não estiver nos documentos, diga: "Não encontrei essa informação."

DOCUMENTOS:
[1] {doc1}
[2] {doc2}

PERGUNTA: {query}

RESPOSTA (cite as fontes como [1], [2] etc.):
```

**Próximos passos:**
- [01 — Naive RAG Architecture](../04_rag_architectures/01_naive_rag_arch.html): como essas peças se combinam numa arquitetura de produção?
"""),
    ]
    save(nb(cells), "03_rag_fundamentals/04_generation_prompts.ipynb")

# ===========================================================================
# 01 — Qdrant Intro (fix ordering and duplicates)
# ===========================================================================
def rebuild_qdrant_intro():
    cells = [
        md("""
# 01 — Introdução ao Qdrant

## Por que um banco de dados vetorial?

Você já sabe como gerar embeddings. Mas onde armazena 1 milhão deles e como faz buscas eficientes?

A resposta ingênua seria: numpy array + busca linear. Problema: busca linear em 1M vetores de 768d leva **segundos**. Para um sistema RAG em produção, você precisa de resultados em **milissegundos**.

**Bancos de dados vetoriais** como o Qdrant resolvem isso com índices especializados (HNSW) que permitem busca aproximada em O(log N) — tipicamente 5-20ms para milhões de vetores.

Mas o Qdrant faz mais que só armazenar vetores:
- **Payloads:** metadados JSON junto com cada vetor (título, data, categoria, URL...)
- **Filtros:** combina busca vetorial com filtros nos metadados em uma única query
- **CRUD completo:** criar, ler, atualizar, deletar pontos e coleções
- **Quantização nativa:** reduz memória em 4-32x com configuração simples

Este notebook cobre as operações fundamentais que você vai usar em todo projeto RAG.

> **Pré-requisito:** Qdrant rodando (`docker compose up -d` no diretório infrastructure/)
"""),
        code("""
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range
)
from sentence_transformers import SentenceTransformer
import uuid

# Conectar ao Qdrant
try:
    client = QdrantClient(host="localhost", port=6333)
    client.get_collections()
    print("Qdrant conectado")
except Exception as e:
    print(f"Erro: {e}")
    print("Certifique-se que o Qdrant está rodando: docker compose up -d")
    raise

model = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION = "artigos_tech"
DIMS = model.get_sentence_embedding_dimension()
print(f"Modelo: {DIMS}d")
"""),
        md("""
## 1.1 Criando uma Coleção

Uma **coleção** no Qdrant é análoga a uma tabela num banco relacional — é onde você armazena vetores com as mesmas características (dimensão, métrica de distância).

Parâmetros fundamentais:
- **`size`**: dimensão dos vetores (deve ser igual à saída do modelo)
- **`distance`**: métrica de distância — use `COSINE` para texto na maioria dos casos

**Por que a distância é configurada na coleção?** O Qdrant otimiza o índice HNSW para uma métrica específica. Mudar depois exigiria reindexar tudo.
"""),
        code("""
# Recriar collection (limpa se já existia)
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
    print(f"Coleção '{COLLECTION}' anterior removida")

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=DIMS, distance=Distance.COSINE)
)
print(f"Coleção '{COLLECTION}' criada ({DIMS}d, COSINE)")
info = client.get_collection(COLLECTION)
print(f"Status: {info.status}, pontos: {info.points_count}")
"""),
        md("""
## 1.2 Inserindo Documentos (Upsert)

Cada documento no Qdrant é chamado de **Point** e tem três componentes:
- **`id`**: identificador único (int ou UUID)
- **`vector`**: o embedding do documento
- **`payload`**: metadados em JSON livre — qualquer coisa que você queira filtrar depois

O **payload** é uma das features mais poderosas do Qdrant.
Diferente de bancos vetoriais simples, você pode armazenar informações ricas e depois filtrar por elas sem sair do banco.

`upsert` (update + insert): se o ID já existir, atualiza. Se não existir, cria. Ideal para reindexação incremental.
"""),
        code("""
# Documentos com metadados ricos
artigos = [
    {"id": 1, "texto": "HNSW é um algoritmo de indexação vetorial hierárquico para busca aproximada eficiente",
     "categoria": "algoritmos", "ano": 2020, "score_qualidade": 9.5},
    {"id": 2, "texto": "Qdrant é um banco de dados vetorial open-source escrito em Rust com suporte a payload filtering",
     "categoria": "ferramentas", "ano": 2021, "score_qualidade": 9.0},
    {"id": 3, "texto": "Embeddings densos representam o significado semântico de textos como vetores de alta dimensão",
     "categoria": "conceitos", "ano": 2019, "score_qualidade": 8.5},
    {"id": 4, "texto": "RAG combina retrieval de documentos com geração de texto para responder perguntas",
     "categoria": "arquiteturas", "ano": 2020, "score_qualidade": 9.2},
    {"id": 5, "texto": "Quantização escalar reduz embeddings de float32 para int8, economizando 75% de memória",
     "categoria": "otimização", "ano": 2022, "score_qualidade": 8.8},
    {"id": 6, "texto": "BM25 é o algoritmo clássico de busca textual baseado em frequência de termos",
     "categoria": "algoritmos", "ano": 1994, "score_qualidade": 7.5},
    {"id": 7, "texto": "Hybrid search combina dense retrieval com BM25 para melhor cobertura semântica e keyword",
     "categoria": "arquiteturas", "ano": 2023, "score_qualidade": 9.8},
]

# Gerar embeddings e upsert em batch
textos = [a["texto"] for a in artigos]
embeddings = model.encode(textos, normalize_embeddings=True)

points = [
    PointStruct(
        id=a["id"],
        vector=embeddings[i].tolist(),
        payload={"texto": a["texto"], "categoria": a["categoria"],
                 "ano": a["ano"], "score_qualidade": a["score_qualidade"]}
    )
    for i, a in enumerate(artigos)
]

client.upsert(collection_name=COLLECTION, points=points)
info = client.get_collection(COLLECTION)
print(f"Upsert concluído. Total de pontos: {info.points_count}")
"""),
        md("""
## 1.3 Busca Semântica

A operação mais importante: dado um vetor de query, encontrar os K vetores mais similares.

O Qdrant usa HNSW internamente para fazer essa busca em O(log N).

Parâmetros:
- **`query_vector`**: embedding da query do usuário
- **`limit`**: quantos resultados retornar (top-K)
- **`score_threshold`**: mínimo de similaridade para incluir no resultado (opcional)

O resultado inclui o **score** de cada ponto — a similaridade cosine com a query.
Score > 0.8: muito relevante. Score < 0.3: provavelmente irrelevante.
"""),
        code("""
def buscar(query, limit=3, score_threshold=None):
    q_vec = model.encode(query, normalize_embeddings=True).tolist()
    resultados = client.search(
        collection_name=COLLECTION,
        query_vector=q_vec,
        limit=limit,
        score_threshold=score_threshold
    )
    return resultados

# Teste
query = "como funciona a indexação para busca vetorial eficiente"
print(f"Query: '{query}'\n")
for r in buscar(query, limit=3):
    print(f"Score {r.score:.3f} | {r.payload['texto'][:70]}...")
    print(f"         categoria={r.payload['categoria']}, ano={r.payload['ano']}")
"""),
        md("""
### Interpretando os scores

Um score de 0.7+ indica alta relevância semântica. Note que o sistema encontra documentos
semanticamente similares à query mesmo sem compartilhar palavras exatas com ela —
"indexação para busca vetorial eficiente" encontra documentos sobre HNSW e Qdrant porque compartilham o *conceito*.

Isso é o poder fundamental dos embeddings para RAG.
"""),
        md("""
## 1.4 Operações CRUD

O Qdrant suporta operações completas de CRUD sobre os pontos — útil para manter a base atualizada sem precisar reindexar tudo.
"""),
        code("""
# READ: buscar por ID
pontos = client.retrieve(collection_name=COLLECTION, ids=[1, 2], with_payload=True)
print("READ por ID:")
for p in pontos:
    print(f"  ID {p.id}: {p.payload['texto'][:60]}...")

# UPDATE: atualizar payload sem precisar re-embedar
client.set_payload(
    collection_name=COLLECTION,
    payload={"verificado": True, "revisor": "equipe_ML"},
    points=[1, 2]
)
ponto_atualizado = client.retrieve(collection_name=COLLECTION, ids=[1], with_payload=True)[0]
print(f"\nUPDATE — novo payload do ponto 1: verificado={ponto_atualizado.payload.get('verificado')}")

# DELETE: remover ponto específico
client.delete(collection_name=COLLECTION, points_selector=[6])
print(f"\nDELETE ponto 6 (BM25 antigo)")
print(f"Total após delete: {client.get_collection(COLLECTION).points_count} pontos")
"""),
        md("""
## 1.5 Filtros por Payload: O Diferencial do Qdrant

Esta é a feature que separa o Qdrant de uma busca linear simples.

Você pode combinar busca vetorial com filtros nos metadados em **uma única operação eficiente**.
O Qdrant executa isso de forma otimizada — não é busca vetorial seguida de filtro (que seria lento).

**Estrutura de filtros:**
- `must`: AND — todos os filtros devem ser verdadeiros
- `should`: OR — pelo menos um deve ser verdadeiro
- `must_not`: NOT — nenhum desses deve ser verdadeiro

Isso permite queries complexas:
"artigos de algoritmos OU arquiteturas, publicados após 2019, com score de qualidade acima de 9.0"
"""),
        code("""
q_vec = model.encode("indexação e busca eficiente", normalize_embeddings=True).tolist()

# Filtro: só categoria "algoritmos" ou "arquiteturas", ano >= 2020
filtro = Filter(
    must=[
        FieldCondition(key="ano", range=Range(gte=2020))
    ],
    should=[
        FieldCondition(key="categoria", match=MatchValue(value="algoritmos")),
        FieldCondition(key="categoria", match=MatchValue(value="arquiteturas")),
    ]
)

resultados = client.search(
    collection_name=COLLECTION,
    query_vector=q_vec,
    query_filter=filtro,
    limit=3,
    with_payload=True
)

print("Busca semântica + filtro (algoritmos/arquiteturas, ano >= 2020):")
for r in resultados:
    print(f"  Score {r.score:.3f} | {r.payload['categoria']:12s} | {r.payload['ano']} | {r.payload['texto'][:55]}...")
"""),
        md("""
### Por que filtros no banco vetorial são essenciais para RAG?

Imagine um sistema RAG para uma empresa com documentos de vários departamentos.
Quando um funcionário do RH faz uma pergunta, você não quer que o sistema retorne
documentos confidenciais de finanças — mesmo que sejam semanticamente similares.

Com filtros de payload:
```python
filtro = Filter(must=[FieldCondition(key="department", match=MatchValue(value="HR"))])
```

Outra aplicação comum: filtrar por data.
"Mostre apenas documentos dos últimos 6 meses" — adicione `created_at` ao payload e filtre na query.
"""),
        code("""
# Scroll: paginar todos os pontos sem query (útil para auditoria)
all_points, next_cursor = client.scroll(
    collection_name=COLLECTION, limit=10, with_payload=True
)
print(f"Scroll — {len(all_points)} pontos:")
for p in all_points:
    print(f"  ID {p.id}: {p.payload['categoria']:12s} | {p.payload['texto'][:55]}...")
"""),
        md("""
## Resumo

| Operação | Método Qdrant | Uso típico |
|----------|--------------|-----------|
| Criar índice | `create_collection` | Uma vez, na inicialização |
| Indexar docs | `upsert` | Indexação inicial e incremental |
| Buscar | `search` | Cada query do usuário |
| Buscar + filtrar | `search` + `query_filter` | RAG multi-tenant, filtros temporais |
| Ler por ID | `retrieve` | Debug, auditoria |
| Deletar | `delete` | Remover docs desatualizados |
| Atualizar metadata | `set_payload` | Sem precisar re-embedar |
| Paginar todos | `scroll` | Exportação, auditoria, re-indexação |

**Próximos passos:**
- [02 — HNSW Indexing](02_hnsw_indexing.html): como o Qdrant faz buscas tão rápidas internamente?
- [03 — Quantization](03_quantization.html): como comprimir o índice para economizar memória
"""),
    ]
    save(nb(cells), "02_vector_databases/01_qdrant_intro.ipynb")


if __name__ == "__main__":
    rebuild_distance_metrics()
    rebuild_chunking()
    rebuild_retrieval()
    rebuild_prompts()
    rebuild_qdrant_intro()
    print("\nAll notebooks rebuilt successfully!")
