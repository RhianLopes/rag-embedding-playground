"""
Rewrites all notebook markdown cells with rich educational content:
- Conceptual intro before each code block
- Conclusions and insights after results
- Takeaways at the end
"""
import json, glob, re, os

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save(nb, path):
    for cell in nb["cells"]:
        if cell.get("outputs"):
            cell["outputs"] = []
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip()}

def insert_after(cells, code_keyword, new_cell):
    """Insert a markdown cell after the first code cell containing code_keyword."""
    result = []
    for c in cells:
        result.append(c)
        if c["cell_type"] == "code" and code_keyword in "".join(c.get("source", [])):
            result.append(new_cell)
    return result

def replace_markdown_at(cells, index, new_source):
    cells[index]["source"] = new_source.strip()
    return cells

# ---------------------------------------------------------------------------
# 01 — O que são Embeddings?
# ---------------------------------------------------------------------------

def doc_01_what_are_embeddings():
    path = "01_embeddings/01_what_are_embeddings.ipynb"
    nb = load(path)
    c = nb["cells"]

    replace_markdown_at(c, 0, """
# 01 — O que são Embeddings?

Antes de qualquer código, vamos entender o problema que os embeddings resolvem.

Modelos de machine learning só trabalham com números. Mas texto é linguagem humana — cheio de contexto, ambiguidade e significado. Como transformar uma frase em números de forma que o *significado* seja preservado?

A resposta ingênua seria: contar palavras (bag-of-words). Problema: "banco de dados" e "sentar no banco" usam a mesma palavra, mas significam coisas completamente diferentes.

A resposta moderna é o **embedding**: um vetor de centenas de números aprendido por uma rede neural treinada em bilhões de frases. A rede aprendeu a mapear o *significado* — não só as palavras — para posições num espaço geométrico.

**O que você vai aprender neste notebook:**
- O que é um embedding e como ele funciona intuitivamente
- Como medir similaridade entre textos usando álgebra linear simples
- Como visualizar significado em 2D e 3D
- Por que podemos fazer "aritmética de significados" com vetores
""")

    replace_markdown_at(c, 3, """
## 1.2 Intuição Geométrica

Agora que geramos os embeddings, vamos entender o que esses números *significam* geometricamente.

Imagine cada embedding como um ponto no espaço — mas em vez de 3 dimensões (x, y, z), temos 384.

A **similaridade de cosseno** mede o ângulo entre dois vetores:
- **1.0** = vetores idênticos (mesma direção)
- **0.0** = vetores ortogonais (sem relação)
- **-1.0** = vetores opostos (raro em embeddings de texto)

A intuição: textos com significado parecido apontam para a *mesma direção* no espaço de 384 dimensões. A magnitude (comprimento do vetor) não importa — só a direção.

> **Por que cosseno e não distância euclidiana?** Porque a distância euclidiana é afetada pelo comprimento do vetor. Frases mais longas geram vetores com norma maior, o que distorceria a comparação. O cosseno elimina esse efeito.

A célula abaixo calcula todas as similaridades par-a-par entre nossas 8 frases:
""")

    # insert conclusion after sim_matrix code
    c = insert_after(c, "sim_matrix[4,7]", md("""
### O que os números nos dizem?

Leia a tabela de cima para baixo e observe:

- **"gato dorme" × "felino repousa" = ~0.58** → alta similaridade, mesmo sem nenhuma palavra em comum. O modelo entendeu que "gato" e "felino" são a mesma coisa, e "dorme" e "repousa" também.
- **"machine learning" × "deep learning" = ~0.42** → ambos são conceitos de IA, mas o modelo distingue que não são idênticos.
- **"machine learning" × "bolo de chocolate" = ~0.10** → praticamente nenhuma relação semântica.

**Insight chave:** a similaridade de cosseno captura *relacionamento semântico*, não *sobreposição de palavras*. Isso é o que torna embeddings úteis para busca — você pode encontrar documentos relevantes mesmo que eles usem vocabulário diferente da sua query.
"""))

    replace_markdown_at(c, c.index(next(x for x in c if x["cell_type"] == "markdown" and "1.3" in "".join(x.get("source", [])))), """
## 1.3 Visualização 2D com PCA

384 dimensões é impossível de visualizar diretamente. A técnica **PCA (Principal Component Analysis)** comprime essas dimensões para 2, preservando o máximo de variância possível.

Pense assim: se você tivesse uma nuvem de pontos em 3D e precisasse fotografá-la de um ângulo só, você escolheria o ângulo que mostra mais estrutura. O PCA faz isso matematicamente, escolhendo as 2 "direções" que mais separam os dados.

**O que esperar:** frases sobre animais deverão aparecer próximas umas das outras, frases sobre tecnologia em outro cluster, e assim por diante. Se o modelo tiver aprendido bem, os clusters visuais vão fazer sentido semântico.
""")

    c = insert_after(c, "frases da mesma categoria ficam agrupadas", md("""
### Conclusão: geometria = semântica

O gráfico confirma o que a tabela de similaridade sugeriu: o modelo MiniLM organizou o espaço vetorial de forma que **categorias semânticas formam clusters visíveis**.

Isso não foi programado explicitamente — emergiu do treinamento em bilhões de pares de frases. O modelo aprendeu que "gato" e "felino" devem ficar próximos, e que ambos devem ficar longe de "bolsa de valores".

**Por que isso importa para RAG?** Quando você busca "qual o preço das ações hoje?", o sistema encontra documentos sobre finanças — mesmo que usem palavras como "cotação", "mercado", "índice" — porque todos esses conceitos ocupam a mesma região do espaço vetorial.
"""))

    replace_markdown_at(c, c.index(next(x for x in c if x["cell_type"] == "markdown" and "1.5" in "".join(x.get("source", [])))), """
## 1.5 Analogias Vetoriais: Álgebra de Significado

Uma das descobertas mais surpreendentes sobre embeddings: você pode fazer **aritmética com significados**.

O exemplo clássico:

```
vetor("rei") - vetor("homem") + vetor("mulher") ≈ vetor("rainha")
```

Isso funciona porque o modelo aprendeu que a relação "rei→rainha" é similar à relação "ator→atriz" ou "professor→professora" — uma transformação de gênero. Essa transformação existe como uma *direção* consistente no espaço vetorial.

**Cuidado com o experimento abaixo:** usamos um vocabulário de apenas 12 palavras. Analogias vetoriais funcionam bem com vocabulários grandes (WordVec tem 3 milhões de palavras). Com 12 palavras, o "vizinho mais próximo" é forçado a ser uma das 12 opções — e o resultado pode ser aleatório.
""")

    c = insert_after(c, "Com modelos maiores e vocabul", md("""
### Por que os resultados parecem estranhos?

Com apenas 12 palavras no vocabulário, o modelo não tem opções. Se o resultado correto de "rei - homem + mulher" fosse "rainha" mas o vetor calculado ficasse mais próximo de "Paris" por acaso, é isso que aparece.

**Em modelos reais** (Word2Vec com 3M palavras, GloVe, FastText), as analogias funcionam surpreendentemente bem:
- rei - homem + mulher = rainha ✓
- Paris - França + Itália = Roma ✓
- melhor - bom + ruim = pior ✓

O fenômeno existe e é real — só precisamos de vocabulário suficiente para demonstrá-lo.

**Insight profundo:** isso prova que os embeddings não são apenas "lookup de palavras". Eles codificam *relações* entre conceitos como direções geométricas consistentes. Gênero, país-capital, grau de comparação — tudo são direções no espaço vetorial.
"""))

    replace_markdown_at(c, c.index(next(x for x in c if x["cell_type"] == "markdown" and "1.6" in "".join(x.get("source", [])))), """
## 1.6 Como os Embeddings são Criados?

Entender o processo de criação ajuda a saber as limitações e quando os embeddings vão (ou não vão) funcionar bem.

O modelo `all-MiniLM-L6-v2` usa uma arquitetura **BERT-like** com 6 camadas Transformer:

```
Texto → Tokenização → 6x Transformer → Pooling → Projeção → Vetor 384d
```

**Como ele foi treinado?** Com *contrastive learning* em ~1 bilhão de pares de frases:
- Pares positivos: "O gato dorme" + "O felino repousa" → vetores devem ser próximos
- Pares negativos: "O gato dorme" + "A bolsa caiu" → vetores devem ser distantes

A rede ajusta seus pesos para satisfazer essas restrições em bilhões de exemplos. O resultado emergente: um espaço onde proximidade geométrica = similaridade semântica.

**Limitações importantes:**
- O modelo não "entende" português — ele aprendeu padrões estatísticos
- Funciona melhor para frases no domínio do seu treinamento
- Frases muito longas (>256 tokens) são truncadas
""")

    c = insert_after(c, "~{total_params/1e6:.0f}M", md("""
### Tamanho importa?

O MiniLM tem 22 milhões de parâmetros e gera vetores de 384 dimensões. Por comparação:
- **MPNet (768d):** 110M parâmetros — 5x maior, melhor qualidade
- **GPT-3.5 (embedding):** bilhões de parâmetros — mas você paga por API call
- **nomic-embed (768d):** roda local via Ollama, qualidade comparável ao MPNet

**Regra prática:** MiniLM para prototipar (rápido, leve). MPNet para produção (melhor qualidade). Modelos via API quando você não quer gerenciar infraestrutura.

---

## Resumo do Notebook

| Conceito | O que aprendemos |
|---------|------------------|
| **Embedding** | Vetor de números que representa o *significado* de um texto |
| **Dimensão** | Cada número no vetor — 384 "slots" para codificar significado |
| **Similaridade cosine** | Mede ângulo entre vetores — ignora magnitude, captura direção |
| **Clusters semânticos** | Textos sobre o mesmo assunto ocupam a mesma região do espaço |
| **Aritmética vetorial** | Operações geométricas correspondem a operações de significado |
| **Treinamento** | Contrastive learning em 1B pares — emergência, não programação explícita |

**O que vem a seguir:** agora que você entende o que é um embedding, o próximo notebook explora por que diferentes modelos usam 384, 768 ou 1024 dimensões — e o impacto real em memória, velocidade e qualidade.

- [02 — Dimensões Vetoriais](02_vector_dimensions.html): por que 384 vs 768 vs 1024?
- [03 — Float Types](03_float_types.html): como comprimir embeddings sem perder qualidade
- [04 — Métricas de Distância](04_distance_metrics.html): quando usar cosine vs euclidean
"""))

    nb["cells"] = c
    save(nb, path)
    print(f"Documented: {path}")

# ---------------------------------------------------------------------------
# 03 — Float Types
# ---------------------------------------------------------------------------

def doc_03_float_types():
    path = "01_embeddings/03_float_types.ipynb"
    nb = load(path)
    c = nb["cells"]

    c[0]["source"] = """
# 03 — Tipos de Float: Comprimindo Embeddings sem Perder Qualidade

## O problema de escala

Você já aprendeu que cada embedding é uma lista de números. Mas como esses números são armazenados na memória?

Cada número pode usar diferentes quantidades de bytes dependendo da **precisão**:

| Tipo | Bits | Bytes | Valores possíveis | Exemplo |
|------|------|-------|-------------------|---------|
| `float64` | 64 | 8 | 1.8 × 10^308 | 0.08312847362... |
| `float32` | 32 | 4 | 3.4 × 10^38 | 0.083128... |
| `float16` | 16 | 2 | 65.504 | 0.0831... |
| `int8` | 8 | 1 | -128 a 127 | 11 (≈ 0.083) |
| `binary` | 1 | 0.125 | 0 ou 1 | 0 |

**Por que isso importa?** Um índice vetorial de 1 milhão de documentos a 768 dimensões:
- `float32`: 3.07 GB (padrão, sem compressão)
- `float16`: 1.53 GB (metade)
- `int8`: 0.77 GB (1/4)
- `binary`: 0.10 GB (1/30)

Em produção, isso é a diferença entre caber numa instância de $200/mês ou precisar de $2.000/mês.

**A pergunta central deste notebook:** quanto de precisão você pode abrir mão antes que a qualidade de busca caia de forma inaceitável?
""".strip()

    # Find and improve section by section
    for i, cell in enumerate(c):
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and "2.1" in src:
            c[i]["source"] = """
## 3.1 Como Cada Tipo Armazena Números

Antes de medir o impacto na qualidade, vamos visualizar o que cada tipo de dado realmente faz com os números.

`float32` usa 32 bits: 1 para sinal, 8 para o expoente, 23 para a fração. Isso permite representar números entre ~10^-38 e ~10^38 com ~7 dígitos decimais de precisão.

`int8` usa 8 bits e só representa inteiros de -128 a 127. Para armazenar um float como 0.083 em int8, multiplicamos por uma escala (ex: 127/max_value) e arredondamos. Esse processo é chamado de **quantização escalar**.

O gráfico abaixo mostra como cada tipo "vê" os mesmos números:
""".strip()
        elif cell["cell_type"] == "markdown" and "2.2" in src or ("Memória" in src and "Cálculo" in src):
            c[i]["source"] = """
## 3.2 Impacto Real na Memória

Agora vamos quantificar o ganho de memória em cenários reais de produção.

O cálculo é simples: `N_vetores × dimensões × bytes_por_valor`.

Mas os números resultantes são surpreendentes — a diferença entre `float32` e `int8` é um fator de 4x, o que pode transformar um sistema inviável em viável economicamente.
""".strip()
        elif cell["cell_type"] == "markdown" and ("2.3" in src or "Qualidade" in src or "Recall" in src) and "preserv" in src.lower():
            c[i]["source"] = """
## 3.3 A Grande Questão: Quanto a Qualidade Cai?

Memória mais barata é ótimo — mas se os resultados de busca piorarem, não vale a pena.

A métrica chave é o **Recall@K**: dos K documentos mais relevantes segundo o índice float32 (ground truth), quantos aparecem no índice quantizado?

- **Recall = 1.0**: quantização perfeita, zero perda de qualidade
- **Recall = 0.95**: 5% dos resultados são diferentes — aceitável para produção
- **Recall = 0.70**: 30% de perda — geralmente inaceitável

A hipótese é que `float16` perde pouco (arredondamento mínimo), `int8` perde um pouco mais mas é recuperável, e `binary` pode perder bastante para textos com nuances sutis.
""".strip()
        elif cell["cell_type"] == "markdown" and "Qdrant" in src and "quantização" in src.lower():
            c[i]["source"] = """
## 3.4 Quantização Nativa do Qdrant

Você não precisa quantizar manualmente. O Qdrant tem suporte nativo a quantização — você configura na criação da coleção e ele gerencia tudo automaticamente.

**Como funciona internamente:**
1. Você indexa os vetores normalmente em float32
2. O Qdrant cria uma cópia comprimida (int8) para busca rápida
3. Na query, ele busca no índice int8 (rápido) e opcionalmente refina os top-K no float32 original (preciso)

Essa estratégia de "busca aproximada + refinamento exato" é chamada de **rescore** e é o segredo para manter alta qualidade com baixo uso de memória.
""".strip()

    c = insert_after(c, "rescore", md("""
### Resultado do Rescore

O `rescore=True` é quase sempre a escolha certa: você paga um pequeno overhead de latência (rerank nos top-K em float32) mas recupera quase toda a precisão perdida na quantização int8.

**Números típicos na prática:**
- float32 baseline: recall=1.0, latência=X
- int8 sem rescore: recall~0.93, latência=0.5X (2x mais rápido)
- int8 com rescore: recall~0.99, latência=0.7X (30% mais rápido com qualidade quase perfeita)

**Quando NÃO usar rescore:** sistemas com latência crítica (<10ms) onde 30% mais rápido ainda não é suficiente. Nesse caso, aceite o recall menor ou use binary quantization.
"""))

    # Add final summary
    c.append(md("""
## Resumo e Decisão

| Cenário | Recomendação | Motivo |
|---------|-------------|--------|
| Desenvolvimento local | `float32` | Simplicidade, sem surpresas |
| Produção geral | `int8` (Scalar) + rescore | 4x menos memória, ~99% recall |
| Grande escala (>10M docs) | Product Quantization | Compressão extrema necessária |
| Edge/mobile | `binary` | Memória mínima aceita recall ~70-80% |
| OpenAI/Anthropic API | `float32` ou `int8` | Custo de API domina, não memória |

**Takeaway principal:** Scalar Quantization (int8) com rescore é o sweet spot para a maioria dos sistemas RAG em produção. Você ganha 4x em memória e 30-50% em velocidade, com menos de 1% de perda de qualidade.

**Próximos passos:**
- [04 — Métricas de Distância](04_distance_metrics.html): qual distância usar com qual tipo de dado?
- [01 — Qdrant Intro](../02_vector_databases/01_qdrant_intro.html): como configurar tudo isso no Qdrant
"""))

    nb["cells"] = c
    save(nb, path)
    print(f"Documented: {path}")

# ---------------------------------------------------------------------------
# 04 — Distance Metrics
# ---------------------------------------------------------------------------

def doc_04_distance_metrics():
    path = "01_embeddings/04_distance_metrics.ipynb"
    nb = load(path)
    c = nb["cells"]

    c[0]["source"] = """
# 04 — Métricas de Distância: Cosine, Dot Product e Euclidiana

## Por que a métrica de distância importa?

Você tem dois embeddings e quer saber se os textos são parecidos. Como você calcula essa "parecença"? A escolha da fórmula matemática — chamada de **métrica de distância** — afeta diretamente a qualidade dos resultados de busca.

Parece um detalhe técnico, mas não é. Escolher a métrica errada pode fazer seu sistema de busca retornar resultados irrelevantes mesmo com embeddings de alta qualidade.

As três métricas mais usadas em busca vetorial são:

| Métrica | O que mede | Range |
|---------|-----------|-------|
| **Cosine Similarity** | Ângulo entre vetores | -1 a 1 (1 = idêntico) |
| **Dot Product** | Projeção de um vetor no outro | -∞ a +∞ |
| **Euclidean Distance** | Distância em linha reta | 0 a +∞ (0 = idêntico) |

Este notebook explica **quando usar cada uma** e por que para RAG com texto a escolha é quase sempre Cosine.
""".strip()

    for i, cell in enumerate(c):
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and "Cosine" in src and "1." in src:
            c[i]["source"] = """
## 4.1 Similaridade de Cosseno

A similaridade de cosseno mede o **ângulo** entre dois vetores, ignorando completamente o comprimento (norma) deles.

A fórmula é: `cos(θ) = (A · B) / (|A| × |B|)`

**Intuição geométrica:** imagine dois vetores saindo da origem. Se eles apontam para a mesma direção (ângulo = 0°), cos(0°) = 1. Se são perpendiculares (ângulo = 90°), cos(90°) = 0. Se opostos (180°), cos(180°) = -1.

**Por que ignorar o comprimento?** Em embeddings de texto, frases mais longas tendem a gerar vetores com norma maior — não porque são "mais importantes", mas porque têm mais tokens. Se usarmos distância euclidiana, frases longas pareceriam sempre "mais distantes" mesmo sendo semanticamente idênticas a frases curtas.

O cosseno elimina esse viés: só a *direção* importa.
""".strip()

        elif cell["cell_type"] == "markdown" and "Dot Product" in src or ("produto" in src.lower() and "escalar" in src.lower()):
            c[i]["source"] = """
## 4.2 Dot Product (Produto Escalar)

O dot product é calculado como: `A · B = Σ(aᵢ × bᵢ)`

**Relação com cosine:** quando os vetores estão normalizados (norma = 1), dot product e cosine similarity são **idênticos**. A maioria dos modelos modernos (incluindo o all-MiniLM) retorna vetores já normalizados, então na prática você pode usar qualquer um.

**Quando o dot product difere do cosine:** quando os vetores NÃO estão normalizados. Nesse caso, o dot product é afetado pela magnitude — vetores "maiores" (frases mais longas) terão scores maiores mesmo que o ângulo seja igual.

**Vantagem do dot product:** é computacionalmente mais barato (não precisa da divisão pelas normas). Para bilhões de operações, essa diferença importa.

**Quando usar:** sempre que seus vetores estiverem pré-normalizados (norma = 1). É o padrão do Qdrant com `COSINE` ou `DOT`.
""".strip()

        elif cell["cell_type"] == "markdown" and "Euclidean" in src or "Euclidiana" in src:
            c[i]["source"] = """
## 4.3 Distância Euclidiana (L2)

A distância euclidiana é a "distância em linha reta" entre dois pontos:

`d(A, B) = √(Σ(aᵢ - bᵢ)²)`

**Quando faz sentido:** quando a *posição absoluta* no espaço importa, não só a direção. Isso é verdade para dados físicos (coordenadas GPS, pixels de imagem) mas raramente para texto.

**O problema com texto:** dois documentos idênticos mas um escrito em caps lock ("gato" vs "GATO") podem ter vetores com normas diferentes e distância euclidiana grande — mesmo sendo semanticamente idênticos.

**Quando usar euclidiana em embeddings:** é mais comum em visão computacional (embeddings de imagem) do que em NLP. Para busca de texto puro, quase sempre prefira cosine.

**Nota Qdrant:** ao configurar `distance=Distance.EUCLID`, o Qdrant usa distância L2. Internamente ela é calculada eficientemente com otimizações SIMD.
""".strip()

    c = insert_after(c, "Distance.EUCLID", md("""
### Qual métrica escolher? Guia prático

**Para RAG com texto:** use **Cosine** (ou Dot Product se seus vetores estiverem normalizados — o resultado é idêntico). Esse é o padrão seguro para 95% dos casos.

**Para embeddings de imagem:** Euclidiana ou Dot Product, dependendo do modelo.

**Para rankings e recomendação** onde magnitude indica "popularidade" ou "relevância global": Dot Product não normalizado, pois modelos como DPR são treinados para que vetores mais relevantes tenham norma maior.

**Regra de ouro:** use a mesma métrica que o modelo foi treinado para usar. Consulte o card do modelo no HuggingFace — ele especifica qual métrica usar.
"""))

    c.append(md("""
## Resumo

| Métrica | Fórmula | Quando usar |
|---------|---------|-------------|
| **Cosine** | cos(θ) = A·B / (|A||B|) | Texto, embeddings normalizados — padrão para RAG |
| **Dot Product** | A · B = Σaᵢbᵢ | Texto normalizado (igual ao cosine) ou ranking com magnitude |
| **Euclidiana** | √Σ(aᵢ-bᵢ)² | Imagens, dados físicos, quando posição absoluta importa |

**Para o seu sistema RAG:** use `Distance.COSINE` no Qdrant e você estará correto na esmagadora maioria dos casos.

**Próximos passos:**
- [05 — Comparação de Modelos](05_models_comparison.html): qual modelo de embedding escolher?
- [01 — Qdrant Intro](../02_vector_databases/01_qdrant_intro.html): como criar coleções com cada métrica
"""))

    nb["cells"] = c
    save(nb, path)
    print(f"Documented: {path}")

# ---------------------------------------------------------------------------
# 05 — Models Comparison
# ---------------------------------------------------------------------------

def doc_05_models_comparison():
    path = "01_embeddings/05_models_comparison.ipynb"
    nb = load(path)
    c = nb["cells"]

    c[0]["source"] = """
# 05 — Comparação de Modelos de Embedding

## Como escolher o modelo certo?

Existem centenas de modelos de embedding disponíveis no HuggingFace. Escolher mal pode significar:
- Sistema lento demais para produção
- Qualidade de busca insatisfatória
- Custo de infraestrutura maior que o necessário

Este notebook compara os principais modelos em **três dimensões**:
1. **Velocidade** — quantos textos por segundo consegue processar?
2. **Qualidade** — qual a taxa de acerto na busca semântica?
3. **Tamanho** — quanto de memória/disco consome?

Não existe modelo "melhor" em tudo. O objetivo é entender os trade-offs para tomar a decisão certa para seu caso de uso.

**Modelos que vamos comparar:**
| Modelo | Dims | Parâmetros | Tipo |
|--------|------|------------|------|
| all-MiniLM-L6-v2 | 384 | 22M | Local, leve |
| all-mpnet-base-v2 | 768 | 110M | Local, qualidade |
| nomic-embed-text | 768 | 137M | Local via Ollama |
""".strip()

    for i, cell in enumerate(c):
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and "Velocidade" in src and "5." in src:
            c[i]["source"] = """
## 5.1 Benchmark de Velocidade

Velocidade importa em dois momentos:

**Indexação** (batch, offline): você processa milhares ou milhões de documentos uma vez. Aqui, velocidade reduz o tempo de setup — de horas para minutos.

**Query** (real-time, online): cada consulta do usuário precisa ser embedada em tempo real. Se isso leva 500ms, sua aplicação parece lenta. O objetivo é < 50ms por query.

O benchmark abaixo mede textos por segundo em batch de 32, em CPU. Com GPU, os números são 10-50x maiores mas a proporção entre modelos se mantém.

> **O que observar:** a diferença de velocidade entre MiniLM e MPNet. É proporcional ao tamanho do modelo?
""".strip()

        elif cell["cell_type"] == "markdown" and "Qualidade" in src and "5." in src:
            c[i]["source"] = """
## 5.2 Qualidade de Retrieval

Velocidade sem qualidade não serve. Agora vamos medir o que realmente importa: **o modelo encontra os documentos certos?**

As métricas:
- **Hit@1**: a query encontrou o documento mais relevante como 1º resultado?
- **MRR (Mean Reciprocal Rank)**: em que posição o documento relevante apareceu, em média? (1.0 = sempre 1º, 0.5 = sempre 2º)

Usamos um conjunto de 20 queries com ground truth manual — sabemos qual documento é o correto para cada query. Medimos quantas vezes cada modelo acerta.

> **O que observar:** o MPNet é significativamente melhor que o MiniLM, ou a diferença é marginal?
""".strip()

        elif cell["cell_type"] == "markdown" and "Ollama" in src:
            c[i]["source"] = """
## 5.3 Modelos via Ollama

O Ollama permite rodar modelos de embedding localmente sem depender de APIs externas. Isso é importante para:
- **Privacidade**: dados não saem da sua máquina
- **Custo**: sem cobrança por token
- **Offline**: funciona sem internet

Os modelos mais populares via Ollama para embedding:
- **nomic-embed-text**: 768d, excelente qualidade, similar ao MPNet
- **mxbai-embed-large**: 1024d, alta qualidade, maior

A desvantagem: latência um pouco maior que modelos carregados diretamente com sentence-transformers (overhead do servidor Ollama).
""".strip()

    c = insert_after(c, "decision", md("""
### Como tomar a decisão?

| Situação | Modelo | Motivo |
|----------|--------|--------|
| Protótipo rápido | MiniLM-L6-v2 | Leve, roda em qualquer máquina |
| Produção geral | MPNet-base-v2 | Melhor qualidade, ainda gerenciável |
| Privacidade/local | nomic-embed (Ollama) | Sem API, qualidade competitiva |
| Máxima qualidade | RoBERTa-large | Alto custo computacional, justificado para domínios críticos |
| Sem infra | OpenAI ada-002 / text-3 | Pague por token, sem GPU necessária |

**Regra dos 80/20:** para a maioria dos projetos RAG, `all-mpnet-base-v2` é suficiente. Comece com MiniLM para prototipar (mais rápido), migre para MPNet quando precisar de melhor qualidade, e só considere modelos maiores se você medir e confirmar que a qualidade ainda não é suficiente.

**Próximos passos:**
- [01 — Qdrant Intro](../02_vector_databases/01_qdrant_intro.html): indexando embeddings num banco vetorial real
"""))

    nb["cells"] = c
    save(nb, path)
    print(f"Documented: {path}")

# ---------------------------------------------------------------------------
# 02_vector_databases/01_qdrant_intro.ipynb
# ---------------------------------------------------------------------------

def doc_qdrant_intro():
    path = "02_vector_databases/01_qdrant_intro.ipynb"
    nb = load(path)
    c = nb["cells"]

    c[0]["source"] = """
# 01 — Introdução ao Qdrant

## Por que um banco de dados vetorial?

Você já sabe como gerar embeddings. Mas onde você armazena 1 milhão deles e como faz buscas eficientes?

A resposta ingênua seria: numpy array + busca linear. Problema: busca linear em 1M vetores de 768d leva segundos. Para um sistema RAG em produção, você precisa de resultados em milissegundos.

**Bancos de dados vetoriais** como o Qdrant resolvem isso com estruturas de índice especializadas (HNSW) que permitem busca aproximada em tempo sub-linear — tipicamente O(log N) em vez de O(N).

Mas o Qdrant faz mais que só armazenar vetores:
- **Payloads**: armazena metadados JSON junto com cada vetor (título, data, categoria, URL...)
- **Filtros**: combina busca vetorial com filtros nos metadados em uma única query
- **Múltiplas coleções**: diferentes espaços vetoriais para diferentes tipos de dado
- **Escalabilidade**: distribuído, suporta bilhões de vetores

Este notebook cobre as operações fundamentais que você vai usar em todo projeto RAG.
""".strip()

    for i, cell in enumerate(c):
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and "Criar" in src and "Coleção" in src:
            c[i]["source"] = """
## 1.1 Criando uma Coleção

Uma **coleção** no Qdrant é análoga a uma tabela num banco relacional — é onde você armazena um conjunto de vetores com características comuns (mesma dimensão, mesma métrica de distância).

Parâmetros fundamentais ao criar uma coleção:
- **`size`**: dimensão dos vetores (deve ser igual à saída do seu modelo de embedding)
- **`distance`**: métrica de distância (use `COSINE` para texto na maioria dos casos)

**Por que a distância é configurada na coleção e não na query?** Porque o Qdrant pode pré-computar e otimizar o índice HNSW para uma métrica específica. Mudar depois exigiria reindexar tudo.
""".strip()

        elif cell["cell_type"] == "markdown" and "Inserir" in src or ("Upsert" in src and "Point" in src):
            c[i]["source"] = """
## 1.2 Inserindo Documentos (Upsert)

Cada documento no Qdrant é chamado de **Point** e tem três componentes:
- **`id`**: identificador único (int ou UUID)
- **`vector`**: o embedding do documento (lista de floats)
- **`payload`**: metadados em JSON livre — qualquer coisa que você queira armazenar e filtrar depois

O **payload** é uma das features mais poderosas do Qdrant. Diferente de bancos vetoriais simples, você pode armazenar informações ricas junto com cada vetor e depois filtrar por elas sem sair do banco.

`upsert` (update + insert): se o ID já existir, atualiza. Se não existir, cria. Isso facilita reindexação incremental.
""".strip()

        elif cell["cell_type"] == "markdown" and "Busca" in src and "Semântica" in src:
            c[i]["source"] = """
## 1.3 Busca Semântica

A operação mais importante: dado um vetor de query, encontrar os K vetores mais similares no índice.

O Qdrant usa o algoritmo **HNSW** internamente para fazer essa busca em O(log N) — exploraremos os detalhes no próximo notebook. Por agora, o importante é entender os parâmetros:

- **`query_vector`**: o embedding da query do usuário
- **`limit`**: quantos resultados retornar (top-K)
- **`score_threshold`**: filtro mínimo de similaridade (opcional — útil para excluir resultados muito distantes)

O resultado inclui o **score** de cada ponto — a similaridade cosine com a query. Um score de 0.9 indica alta relevância; abaixo de 0.3, o resultado provavelmente não é relevante.
""".strip()

        elif cell["cell_type"] == "markdown" and "Filtr" in src and ("Combinando" in src or "Avançad" in src or "payload" in src.lower()):
            c[i]["source"] = """
## 1.4 Filtros por Payload: O Diferencial do Qdrant

Esta é a feature que separa o Qdrant de uma busca linear simples com numpy.

Você pode combinar busca vetorial com filtros nos metadados em **uma única operação eficiente**. Por exemplo:
- "Encontre artigos *semanticamente similares à minha query* E que sejam da *categoria tecnologia* E do *ano 2024*"

O Qdrant executa isso de forma otimizada — não é busca vetorial seguida de filtro (que seria lento). O índice HNSW é construído considerando os filtros.

**Estrutura de filtros:**
- `must`: AND — todos os filtros devem ser verdadeiros
- `should`: OR — pelo menos um deve ser verdadeiro
- `must_not`: NOT — nenhum desses deve ser verdadeiro

Isso permite queries complexas: "artigos de tecnologia OU ciência, mas não de política, com score > 0.7".
""".strip()

    c = insert_after(c, "must_not", md("""
### Por que filtros no banco vetorial são tão importantes para RAG?

Imagine um sistema RAG para uma empresa com documentos de vários departamentos. Quando um funcionário do RH faz uma pergunta, você não quer que o sistema retorne documentos confidenciais de finanças — mesmo que sejam semanticamente similares.

Com filtros de payload, você pode garantir: "busque apenas documentos onde `department == 'HR'` ou `visibility == 'public'`". Isso é essencial para sistemas multi-tenant ou com controle de acesso.

Outra aplicação comum: filtrar por data. "Mostre apenas documentos dos últimos 6 meses" — você adiciona `created_at` ao payload e filtra na query.
"""))

    c.append(md("""
## Resumo

| Operação | Método Qdrant | Uso |
|----------|--------------|-----|
| Criar índice | `create_collection` | Uma vez, na inicialização |
| Indexar docs | `upsert` | Indexação inicial e incremental |
| Buscar | `search` | Cada query do usuário |
| Buscar + filtrar | `search` + `query_filter` | RAG multi-tenant, filtros temporais |
| Ler por ID | `retrieve` | Debug, auditoria |
| Deletar | `delete` | Remover docs desatualizados |
| Atualizar metadata | `set_payload` | Sem precisar re-embedar |

**O que vem a seguir:**
- [02 — HNSW Indexing](02_hnsw_indexing.html): como o Qdrant faz buscas tão rápidas?
- [03 — Quantization](03_quantization.html): como comprimir o índice para economizar memória
"""))

    nb["cells"] = c
    save(nb, path)
    print(f"Documented: {path}")

# ---------------------------------------------------------------------------
# 02_vector_databases/02_hnsw_indexing.ipynb
# ---------------------------------------------------------------------------

def doc_hnsw():
    path = "02_vector_databases/02_hnsw_indexing.ipynb"
    nb = load(path)
    c = nb["cells"]

    c[0]["source"] = """
# 02 — HNSW: O Algoritmo por Trás da Busca Vetorial Rápida

## O problema que o HNSW resolve

Busca de nearest neighbor em alta dimensão é computacionalmente custosa.

A abordagem ingênua — calcular a distância de uma query para todos os N vetores — é O(N). Com 1 milhão de vetores de 768 dimensões, isso significa 768 milhões de multiplicações por query. Em CPU, isso leva segundos.

Para um sistema RAG em produção recebendo centenas de queries por segundo, isso é inviável.

**HNSW (Hierarchical Navigable Small World)** resolve isso com uma estrutura de grafo hierárquica que permite busca *aproximada* em O(log N) — encontra os K vizinhos mais próximos com ~95-99% de acurácia, mas em milissegundos.

## Intuição do algoritmo

Imagine uma rede de metrô em camadas:
- **Camada superior (poucas estações)**: conexões longas de "salto rápido" — você chega perto do destino rapidamente
- **Camadas intermediárias**: refinamento progressivo
- **Camada inferior (todas as estações)**: busca local fina para encontrar os vizinhos exatos

Na busca, você começa no topo (saltos grandes, cobertura ampla) e desce progressivamente até a camada onde estão os vizinhos mais próximos do ponto de busca.

Esse design imita o conceito de "small world networks" — como as 6 conexões que separam qualquer pessoa de qualquer outra no mundo.
""".strip()

    for i, cell in enumerate(c):
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and "Parâmetros" in src or ("m=" in src and "ef" in src):
            c[i]["source"] = """
## 2.1 Parâmetros do HNSW

O HNSW tem três parâmetros principais. Entender o que cada um faz é essencial para tunar o índice para o seu caso de uso:

### `m` — Número de conexões por nó
Cada ponto no grafo mantém `m` conexões com seus vizinhos. Mais conexões = grafo mais denso = busca mais precisa, mas mais memória e tempo de indexação.

- `m=4`: índice leve, busca rápida, recall menor (~90%)
- `m=16`: padrão do Qdrant, bom equilíbrio
- `m=32`: alta qualidade, 2x mais memória que m=16

### `ef_construct` — Qualidade da construção
Durante a indexação, o algoritmo considera `ef_construct` candidatos ao adicionar cada ponto. Maior valor = grafo de melhor qualidade = melhor recall, mas indexação mais lenta.

- `ef_construct=100`: padrão, bom para maioria dos casos
- `ef_construct=200+`: necessário apenas para datasets muito grandes ou alta precisão

### `ef` (search ef) — Candidatos na busca
Durante a query, considera `ef` candidatos. Aumentar melhora o recall sem exigir reindexação — você pode ajustar isso por query.

- `ef=128`: padrão, ~95% recall
- `ef=256+`: para quando você precisa de recall máximo e aceita latência maior
""".strip()

    c = insert_after(c, "ef_construct", md("""
### Como os parâmetros afetam memória?

O grafo HNSW ocupa memória adicional além dos vetores em si. Cada conexão é armazenada como um ponteiro.

Memória aproximada do grafo: `N × m × 2 × 8 bytes` (links em ambas as direções)

Para 1M vetores com m=16: `1M × 16 × 2 × 8 = ~256MB` de overhead de grafo. Não é o custo dominante (os vetores em si custam muito mais), mas vale considerar para datasets extremamente grandes.

**Regra prática:** use os padrões (`m=16`, `ef_construct=100`) para datasets até 10M de vetores. Para datasets maiores ou qualidade crítica, considere aumentar `m` para 32 ou `ef_construct` para 200.
"""))

    c.append(md("""
## Resumo: Quando tunar o HNSW?

Na maioria dos projetos, você **não precisa tunar o HNSW**. Os padrões do Qdrant funcionam bem para datasets de até alguns milhões de vetores.

Você deve considerar tuning quando:
- Dataset > 10M vetores: aumente `m` para melhor recall
- Latência crítica < 5ms: reduza `ef` na query (aceite recall menor)
- Precisão máxima necessária: aumente `ef_construct` e `ef`

| Situação | `m` | `ef_construct` | `ef` query |
|----------|-----|----------------|------------|
| Default / protótipo | 16 | 100 | 128 |
| Alta qualidade | 32 | 200 | 256 |
| Velocidade máxima | 8 | 100 | 64 |
| Dataset enorme | 16 | 100 | 128 + quantização |

**Próximos passos:**
- [03 — Quantization](03_quantization.html): como comprimir o índice para economizar memória sem perder recall
"""))

    nb["cells"] = c
    save(nb, path)
    print(f"Documented: {path}")

# ---------------------------------------------------------------------------
# 03_rag_fundamentals/01_naive_rag.ipynb
# ---------------------------------------------------------------------------

def doc_naive_rag():
    path = "03_rag_fundamentals/01_naive_rag.ipynb"
    nb = load(path)
    c = nb["cells"]

    c[0]["source"] = """
# 01 — Naive RAG: Pipeline Completo do Zero

## O que é RAG?

**RAG (Retrieval-Augmented Generation)** é uma técnica para fazer LLMs responderem perguntas sobre documentos que eles nunca viram durante o treinamento.

O problema que RAG resolve: um LLM como o Llama treinado até 2023 não sabe nada sobre sua documentação interna, dados atualizados ou qualquer informação privada. Fine-tuning seria caro e precisaria ser refeito a cada atualização.

RAG resolve isso em tempo de inferência:
1. **Indexação** (offline): converta seus documentos em embeddings e armazene num banco vetorial
2. **Retrieval** (online): quando o usuário pergunta algo, encontre os documentos mais relevantes
3. **Generation** (online): passe esses documentos como contexto para o LLM e peça que responda

O LLM não precisa "saber" a resposta — ele só precisa lê-la no contexto e reformulá-la.

## O que é "Naive RAG"?

"Naive" significa a versão mais simples possível, sem otimizações:
- Chunking fixo por tamanho
- Retrieval por similaridade cosine simples
- Prompt direto com os chunks recuperados

É o ponto de partida. Funciona surpreendentemente bem para casos simples. Os notebooks seguintes mostram como melhorá-lo quando não for suficiente.

**O que você vai construir:** um sistema RAG completo em ~100 linhas de código, do zero, sem frameworks.
""".strip()

    for i, cell in enumerate(c):
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and "INDEXING" in src and "Load" in src:
            c[i]["source"] = """
## Fase 1: INDEXAÇÃO (Offline)

A indexação acontece uma vez (ou periodicamente, quando seus documentos mudam). É o processo de preparar sua base de conhecimento para busca rápida.

### 1.1 Carregando Documentos

O primeiro passo é ler seus documentos do disco. Em produção, isso pode vir de:
- Arquivos markdown/PDF no filesystem
- Banco de dados (PostgreSQL, MongoDB)
- APIs externas (Notion, Confluence, Google Drive)
- Web crawlers

Para este exemplo, usamos arquivos markdown locais. O importante é que no final temos uma lista de textos para indexar.
""".strip()

        elif cell["cell_type"] == "markdown" and "Chunk" in src:
            c[i]["source"] = """
### 1.2 Chunking: Dividindo Documentos em Pedaços

LLMs têm um limite de tokens no contexto. Além disso, documentos longeiros inteiros como contexto são ineficientes — a parte relevante é geralmente pequena.

**Chunking** é dividir documentos em pedaços menores. A estratégia mais simples: tamanho fixo com overlap.

Por que **overlap**? Se você cortar um documento em partes de 500 caracteres exatos, pode cortar uma frase no meio:
- Chunk 1: "...o algoritmo HNSW usa uma estrutura hierárquica de"
- Chunk 2: "grafos para permitir busca eficiente..."

Com overlap de 50 caracteres, o início do Chunk 2 repete o fim do Chunk 1, garantindo que a ideia completa apareça em pelo menos um chunk.

**Tamanhos típicos em produção:** 256-512 tokens para documentação técnica, 128-256 para FAQ, 512-1024 para artigos longos.
""".strip()

        elif cell["cell_type"] == "markdown" and "Embed" in src and "Indexing" in src.upper():
            c[i]["source"] = """
### 1.3 Gerando Embeddings em Batch

Agora convertemos cada chunk em um vetor. Para eficiência, processamos em batches (grupos).

Por que batch? Carregar o modelo na GPU/CPU tem overhead. Processar 32 textos de uma vez é quase tão rápido quanto processar 1 — o overhead de GPU/CPU é amortizado.

**Estimativa de tempo:** ~500 textos/segundo em CPU com MiniLM. Para 100K documentos com chunks de 500 chars (≈ 200K chunks), isso leva ~6 minutos. Com GPU, < 30 segundos.

Este é o passo mais lento da indexação. Feito uma vez, o índice fica disponível para queries instantâneas.
""".strip()

        elif cell["cell_type"] == "markdown" and "QUERYING" in src:
            c[i]["source"] = """
## Fase 2: QUERYING (Online, Tempo Real)

O usuário fez uma pergunta. Agora precisamos: encontrar os documentos mais relevantes e gerar uma resposta.

### 2.1 Retrieval

O retrieval tem dois passos:
1. **Embed a query**: a pergunta do usuário vira um vetor (usando o MESMO modelo da indexação)
2. **Busca vetorial**: encontrar os K chunks com vetores mais similares

**Por que usar o mesmo modelo?** Se você indexou com MiniLM-L6-v2, precisa fazer a query com MiniLM-L6-v2. Modelos diferentes geram espaços vetoriais diferentes — a similaridade não seria comparável.

**Latência típica:** embedding da query < 10ms, busca vetorial no Qdrant < 5ms para 100K vetores.
""".strip()

        elif cell["cell_type"] == "markdown" and "Generat" in src:
            c[i]["source"] = """
### 2.2 Generation

Agora que temos os chunks mais relevantes, precisamos de um LLM para sintetizar uma resposta.

O LLM recebe:
- **Contexto**: os chunks recuperados (a "resposta bruta")
- **Query**: a pergunta original do usuário
- **Instrução**: como formatar a resposta

O LLM não busca informação — ele reformula o que está no contexto. Se o contexto não contém a resposta, um LLM bem instruído deve dizer "não encontrei essa informação nos documentos fornecidos".

**Latência:** a geração do LLM domina o tempo total (2-10 segundos com Ollama em CPU). Embedding + busca juntos levam < 50ms.
""".strip()

    c = insert_after(c, "resposta", md("""
### Análise do pipeline completo

Agora você tem um sistema RAG funcional. Vamos analisar o que funciona bem e o que não funciona.

**Funciona bem:**
- Perguntas diretas sobre informações presentes nos documentos
- Documentos bem estruturados com parágrafos coesos
- Queries similares linguisticamente ao texto dos documentos

**Falha nesses casos:**
- Queries ambíguas ("como funciona?") sem contexto suficiente
- Perguntas que requerem combinar informação de múltiplos documentos distantes
- Perguntas com negação ("o que NÃO é...") — embeddings são ruins para negação
- Informação que está "nas entrelinhas", não explícita no texto

Esses problemas motivam as técnicas de Advanced RAG que veremos adiante: reescrita de query, re-ranking, compressão de contexto.
"""))

    c.append(md("""
## Resumo do Naive RAG

**O que construímos:** pipeline completo de RAG em ~100 linhas sem framework.

**Latências típicas:**
| Fase | Tempo | Nota |
|------|-------|------|
| Indexação (100K docs) | ~10 min | Uma vez só, offline |
| Embedding da query | ~10ms | Por request |
| Busca vetorial | ~5ms | Por request |
| Geração LLM | 2-8s | Domina o tempo total |

**Quando Naive RAG é suficiente:**
- Documentação técnica bem estruturada
- Base < 100K documentos
- Queries relativamente simples e diretas
- Protótipo ou MVP

**Quando você precisa de mais:**
- Qualidade de resposta insatisfatória após testes com usuários reais
- Queries complexas com múltiplos passos de raciocínio
- Documentos muito longos ou mal estruturados
- Necessidade de citar fontes com precisão

**Próximos passos:**
- [02 — Chunking Strategies](02_chunking_strategies.html): estratégias melhores de divisão de documentos
- [03 — Retrieval Strategies](03_retrieval_strategies.html): hybrid search, re-ranking, MMR
"""))

    nb["cells"] = c
    save(nb, path)
    print(f"Documented: {path}")

# ---------------------------------------------------------------------------
# 03_rag_fundamentals/02_chunking_strategies.ipynb
# ---------------------------------------------------------------------------

def doc_chunking():
    path = "03_rag_fundamentals/02_chunking_strategies.ipynb"
    nb = load(path)
    c = nb["cells"]

    c[0]["source"] = """
# 02 — Estratégias de Chunking

## Por que chunking é um dos problemas mais importantes do RAG?

Na prática, chunking mal feito é responsável por boa parte das falhas de sistemas RAG. Um LLM excelente com chunks ruins gera respostas ruins. Chunks bons com um LLM mediano ainda funcionam razoavelmente.

**O desafio central:** você precisa dividir documentos em pedaços que:
1. Sejam pequenos o suficiente para caber no contexto do LLM
2. Sejam grandes o suficiente para conter uma ideia completa
3. Não quebrem no meio de conceitos importantes
4. Sejam similares em tamanho (para evitar que alguns chunks dominem os resultados)

Não existe uma estratégia perfeita para todos os casos. A escolha certa depende do tipo de documento.

**Estratégias que vamos comparar:**
| Estratégia | Complexidade | Quando usar |
|-----------|-------------|-------------|
| Fixed-size | Simples | Textos uniformes, prototipagem |
| Recursive split | Média | Documentação geral — recomendado como padrão |
| Semantic chunking | Alta | Documentos longos com mudanças de tópico |
""".strip()

    for i, cell in enumerate(c):
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and "Fixed" in src or ("fixo" in src.lower() and "tamanho" in src.lower()):
            c[i]["source"] = """
## 2.1 Fixed-Size Chunking

A estratégia mais simples: divida o texto a cada N caracteres, com overlap de M caracteres.

**Vantagem:** previsível, fácil de implementar, sem dependências.

**Desvantagem:** ignora completamente a estrutura do texto. Pode cortar no meio de:
- Uma frase: "O algoritmo HNSW usa uma estrutura / hierárquica para..."
- Um parágrafo: informação logicamente conectada fica em chunks separados
- Um código: função cortada ao meio não tem sentido isolada

**Quando usar:** protótipos, textos muito uniformes (transcrições de fala, logs), quando velocidade de implementação é prioridade.

**Parâmetros típicos:** chunk_size=500 chars, overlap=50 chars (10% do tamanho).
""".strip()

        elif cell["cell_type"] == "markdown" and "Recursive" in src or "recursiv" in src.lower():
            c[i]["source"] = """
## 2.2 Recursive Character Split

Uma evolução do fixed-size: em vez de cortar a cada N caracteres, tenta respeitar a hierarquia natural do texto.

**Como funciona:** define uma lista de separadores em ordem de preferência:
1. `\n\n` (parágrafo) — divide aqui se possível
2. `\n` (linha) — se o chunk ainda for grande, divide por linha
3. `. ` (frase) — se ainda for grande, divide por frase
4. ` ` (palavra) — último recurso
5. `""` (caractere) — último, último recurso

O algoritmo aplica esses separadores recursivamente até que todos os chunks estejam abaixo do tamanho máximo.

**Resultado:** chunks que respeitam a estrutura natural do documento, sem quebrar parágrafos ou frases no meio.

**Este é o padrão recomendado para a maioria dos casos.** LangChain e LlamaIndex usam isso como default.
""".strip()

        elif cell["cell_type"] == "markdown" and "Semântic" in src or "Semantic" in src:
            c[i]["source"] = """
## 2.3 Semantic Chunking

A estratégia mais sofisticada: divide onde o *significado* muda, não onde o texto tem quebras.

**Como funciona:**
1. Divide o texto em sentenças
2. Computa o embedding de cada sentença
3. Calcula a similaridade entre sentenças consecutivas
4. Quando a similaridade cai abruptamente (mudança de tópico), cria um novo chunk

**Exemplo:** num artigo que discute Python e depois machine learning, a fronteira semântica aparecerá quando o texto muda de "sintaxe de listas" para "gradient descent" — mesmo que não haja um parágrafo explícito separando.

**Custo:** caro. Precisa embedar todas as sentenças antes de chunkar. Para 1000 documentos, isso multiplica o tempo de indexação por 3-5x.

**Quando vale a pena:** documentos longos e heterogêneos (livros, relatórios extensos, transcrições de reuniões longas) onde mudanças de tópico são frequentes e importantes.
""".strip()

    c = insert_after(c, "recall", md("""
### O que os números nos dizem?

A comparação de recall revela algo importante: **a diferença entre fixed-size e recursive split é maior do que a maioria das pessoas espera**.

Fixed-size frequentemente corta no meio de informações cruciais. Quando o retrieval não encontra o chunk certo, o LLM não tem como responder corretamente — não importa quão bom seja o modelo.

**Insight chave:** invista tempo escolhendo a estratégia de chunking certa antes de otimizar o modelo de embedding ou o LLM. A qualidade do chunking tem um impacto desproporcional no resultado final.

**Recomendação prática:**
- Comece com recursive split (chunk_size=512, overlap=64)
- Meça o recall com um conjunto de queries de teste
- Se ainda insatisfatório, experimente semantic chunking
- Ajuste chunk_size baseado no tamanho médio dos documentos e no contexto do LLM
"""))

    c.append(md("""
## Resumo

| Estratégia | Recall típico | Custo | Recomendação |
|-----------|--------------|-------|--------------|
| Fixed-size | Baixo-médio | O(1) | Só para protótipos |
| Recursive split | Médio-alto | O(N) | **Padrão recomendado** |
| Semantic | Alto | O(N×M) | Documentos longos e heterogêneos |

**Parâmetros a experimentar:**
- `chunk_size`: 256 (respostas curtas e precisas) a 1024 (contexto rico)
- `overlap`: 10-20% do chunk_size
- Para código: chunke por função/classe, não por tamanho

**Próximos passos:**
- [03 — Retrieval Strategies](03_retrieval_strategies.html): uma vez que você tem bons chunks, como encontrá-los eficientemente?
"""))

    nb["cells"] = c
    save(nb, path)
    print(f"Documented: {path}")

# ---------------------------------------------------------------------------
# 03_rag_fundamentals/03_retrieval_strategies.ipynb
# ---------------------------------------------------------------------------

def doc_retrieval():
    path = "03_rag_fundamentals/03_retrieval_strategies.ipynb"
    nb = load(path)
    c = nb["cells"]

    c[0]["source"] = """
# 03 — Estratégias de Retrieval

## Por que retrieval importa tanto?

Uma resposta de RAG só pode ser tão boa quanto os documentos recuperados. Se o retrieval falha em encontrar o chunk relevante, o LLM não tem como responder — não importa quão inteligente seja o modelo.

**Falhas comuns do retrieval simples (dense only):**
- Query usa termos técnicos específicos que o modelo de embedding não captura bem
- Documentos usam sinônimos ou abreviações que o embedding aproxima mal
- Queries muito curtas ("erro 404") têm embeddings pouco informativos
- Queries com palavras raras ou nomes próprios específicos

**A solução:** combinar diferentes tipos de retrieval para cobrir os pontos cegos de cada um.

**Estratégias que vamos explorar:**
| Estratégia | Captura | Falha em |
|-----------|---------|----------|
| Dense (embedding) | Semântica, sinônimos | Keywords exatas, raros |
| Sparse (BM25) | Keywords exatas | Sinônimos, contexto |
| Hybrid (RRF) | Ambos | Quase nada — é o padrão de produção |
| MMR | Diversidade | Quando você quer documentos similares |
""".strip()

    for i, cell in enumerate(c):
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and "Dense" in src and "3." in src:
            c[i]["source"] = """
## 3.1 Dense Retrieval (Embedding-based)

O retrieval denso é o que você já conhece: embedding da query → similaridade cosine → top-K.

**Ponto forte:** entende *semântica*. "cachorro" e "cão" são similares. "como resolver problema X" encontra documentos que falam de "soluções para X" mesmo sem usar exatamente essas palavras.

**Ponto fraco:** para queries muito específicas com termos técnicos exatos (números de erro, códigos de produto, nomes próprios incomuns), o embedding pode não capturar bem a especificidade. O modelo foi treinado em linguagem geral — termos muito de nicho podem ter representações imprecisas.

**Também falha com negação:** "o que NÃO é machine learning?" — o embedding de "NÃO machine learning" fica próximo de embeddings de machine learning, porque o modelo entende o conceito, não a negação.
""".strip()

        elif cell["cell_type"] == "markdown" and "Sparse" in src or "BM25" in src:
            c[i]["source"] = """
## 3.2 Sparse Retrieval (BM25)

BM25 é uma versão melhorada do TF-IDF — o algoritmo clássico de busca textual que o Google usava antes das redes neurais.

**Como funciona:** pondera palavras por:
- **TF (Term Frequency)**: quantas vezes a palavra aparece no documento
- **IDF (Inverse Document Frequency)**: palavras raras recebem mais peso que palavras comuns
- **Normalização por tamanho**: documentos longos não ganham vantagem injusta

**Ponto forte:** encontra documentos que contêm exatamente as palavras da query. Para queries como "CUDA out of memory error", BM25 vai direto nos documentos que contêm essas palavras específicas.

**Ponto fraco:** completamente cego a semântica. "cachorro" e "cão" são palavras totalmente diferentes para o BM25. Um documento sobre "caninos domésticos" não aparece para uma query sobre "cachorros".

**Na prática:** BM25 é imbatível para buscas keyword-based (código, nomenclaturas técnicas, nomes próprios). Dense é melhor para linguagem natural e semântica.
""".strip()

        elif cell["cell_type"] == "markdown" and "Hybrid" in src or "RRF" in src:
            c[i]["source"] = """
## 3.3 Hybrid Search com RRF

Hybrid search combina os rankings de dense e sparse para capturar o melhor dos dois mundos.

**RRF (Reciprocal Rank Fusion)** é o algoritmo de fusão mais popular:

```
score_rrf(doc) = 1/(k + rank_dense) + 1/(k + rank_sparse)
```

Onde `k=60` é uma constante empírica. Um documento que aparece em 1º no dense e 1º no sparse tem score máximo. Um documento que aparece apenas em um dos rankings ainda contribui com algum score.

**Por que não simplesmente somar os scores?** Scores de sistemas diferentes têm escalas incompatíveis (cosine: 0-1, BM25: 0-∞). RRF usa apenas os *ranks* (posições), não os valores absolutos — isso é escala-invariante.

**O parâmetro `alpha`** controla o peso relativo:
- `alpha=1.0`: só dense
- `alpha=0.0`: só sparse
- `alpha=0.5`: equilíbrio — funciona bem como padrão
""".strip()

        elif cell["cell_type"] == "markdown" and "MMR" in src:
            c[i]["source"] = """
## 3.4 MMR (Max Marginal Relevance)

Um problema sutil do retrieval padrão: os top-K resultados podem ser muito similares entre si.

Se você busca "como funciona gradient descent?" e os 5 chunks mais relevantes são parágrafos consecutivos do mesmo artigo, você está desperdiçando espaço de contexto com informação redundante.

**MMR** resolve isso balanceando **relevância** (similar à query) com **diversidade** (dissimilar dos documentos já selecionados):

```
MMR = argmax[λ × sim(doc, query) - (1-λ) × max(sim(doc, selected))]
```

- `λ = 1.0`: só relevância (igual ao retrieval padrão)
- `λ = 0.0`: só diversidade (documentos o mais diferentes possíveis)
- `λ = 0.5`: equilíbrio — bom para respostas que precisam cobrir múltiplos ângulos

**Quando usar MMR:**
- Q&A sobre documentação extensa (evita repetir a mesma informação de formas diferentes)
- Sumarização de tópicos (você quer perspectivas diversas)
- Sistemas de recomendação (variedade é desejável)
""".strip()

    c = insert_after(c, "alpha", md("""
### Qual estratégia para qual situação?

| Tipo de Query | Melhor Estratégia | Por quê |
|--------------|------------------|---------|
| "Como funciona o backpropagation?" | Dense | Semântica, sinônimos |
| "CUDA error code 11" | Sparse / Hybrid | Keyword exata, código específico |
| "Quais são as vantagens do transformer?" | MMR + Hybrid | Múltiplos ângulos, evitar repetição |
| Query curta ("erro 404") | Hybrid | Dense fraca, BM25 complementa |
| Pergunta em português sobre doc em inglês | Dense | Modelos multilíngues capturam semântica cross-lingual |

**Recomendação padrão:** use Hybrid com `alpha=0.5` como default. Ajuste baseado nos seus dados e queries mais comuns.
"""))

    c.append(md("""
## Resumo

| Estratégia | Recall semântico | Recall keyword | Diversidade | Complexidade |
|-----------|-----------------|---------------|-------------|-------------|
| Dense | ★★★★★ | ★★☆☆☆ | ★★☆☆☆ | Baixa |
| Sparse (BM25) | ★★☆☆☆ | ★★★★★ | ★★☆☆☆ | Baixa |
| **Hybrid RRF** | **★★★★★** | **★★★★★** | **★★☆☆☆** | **Média** |
| MMR | ★★★★☆ | ★★☆☆☆ | ★★★★★ | Média |

**Hybrid com RRF é o padrão de produção moderno.** Sistemas como Elasticsearch, Qdrant e Weaviate têm suporte nativo a hybrid search exatamente por isso.

**Próximos passos:**
- [04 — Generation Prompts](04_generation_prompts.html): como instruir o LLM para gerar respostas precisas e evitar alucinações
"""))

    nb["cells"] = c
    save(nb, path)
    print(f"Documented: {path}")

# ---------------------------------------------------------------------------
# 03_rag_fundamentals/04_generation_prompts.ipynb
# ---------------------------------------------------------------------------

def doc_prompts():
    path = "03_rag_fundamentals/04_generation_prompts.ipynb"
    nb = load(path)
    c = nb["cells"]

    c[0]["source"] = """
# 04 — Prompt Engineering para RAG

## O prompt é o último quilômetro

Você indexou bem, fez retrieval eficiente, recuperou os documentos certos. Agora vem o último passo: pedir ao LLM que gere uma resposta.

O prompt determina:
- **Fidelidade**: o LLM usa apenas o contexto fornecido ou "inventa" informações?
- **Formato**: a resposta é curta e direta ou longa e detalhada?
- **Atribuição**: o LLM cita as fontes?
- **Honestidade**: quando a resposta não está no contexto, o LLM admite ou alucina?

**Alucinação** é o maior risco de sistemas RAG. Um LLM bem instruído diz "não encontrei essa informação nos documentos". Um LLM mal instruído inventa uma resposta convincente mas falsa.

Este notebook mostra a evolução do prompt: do básico (propenso a alucinação) ao production-grade (grounded, com citações, com fallback honesto).
""".strip()

    for i, cell in enumerate(c):
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and "Basic" in src or ("básico" in src.lower() and "prompt" in src.lower()):
            c[i]["source"] = """
## 4.1 Prompt Básico

O prompt mais simples possível: forneça o contexto e faça a pergunta.

Funciona para casos simples mas tem um problema grave: o LLM não recebe instruções explícitas sobre o que fazer quando a resposta não está no contexto. A tendência natural do LLM é completar o texto de forma coerente — o que significa *inventar* informações plausíveis.

Este comportamento (alucinação) pode parecer correto à primeira vista porque o LLM é convincente. É o maior risco em sistemas RAG de produção.

O exemplo abaixo mostra como um prompt básico se comporta:
""".strip()

        elif cell["cell_type"] == "markdown" and "Strict" in src or ("ground" in src.lower() and "citaç" not in src.lower()):
            c[i]["source"] = """
## 4.2 Prompt com Grounding Explícito

A solução para alucinação: instrua explicitamente o LLM a responder APENAS com base no contexto fornecido.

A adição de frases como "Responda SOMENTE com base nas informações fornecidas" e "Se a resposta não estiver no contexto, diga 'Não encontrei essa informação'" reduz drasticamente a taxa de alucinação.

**Por que funciona?** LLMs são treinados para seguir instruções. Quando você explicita o comportamento esperado, o modelo ajusta seu output. A instrução deve ser clara, direta e específica — instruções vagas ("use o contexto") são menos efetivas que instruções precisas ("responda SOMENTE com as informações do contexto, sem adicionar nada que não esteja lá").
""".strip()

        elif cell["cell_type"] == "markdown" and "Citaç" in src or "Citation" in src:
            c[i]["source"] = """
## 4.3 Prompt com Citações

Para sistemas onde a rastreabilidade é importante (suporte técnico, sistemas jurídicos, saúde), pedir ao LLM que cite as fontes muda tudo.

**Por que citações são importantes:**
1. **Verificação**: o usuário pode confirmar a informação na fonte original
2. **Confiança**: respostas com fontes parecem (e geralmente são) mais confiáveis
3. **Debug**: quando o sistema erra, você sabe exatamente qual chunk causou o problema
4. **Compliance**: em alguns domínios, rastreabilidade é requisito legal

O truque é numerar os documentos de contexto e pedir que o LLM use `[1]`, `[2]` etc. Os LLMs modernos seguem esse padrão de forma bastante confiável quando instruídos.
""".strip()

        elif cell["cell_type"] == "markdown" and "Hallucin" in src or "Alucinação" in src:
            c[i]["source"] = """
## 4.4 Teste de Alucinação

O teste mais importante: o que acontece quando você pergunta algo que NÃO está nos documentos?

Um sistema RAG bem construído deve responder "não sei" de forma honesta. Um sistema mal construído vai inventar uma resposta plausível — e isso é perigoso.

Vamos testar deliberadamente com uma pergunta cuja resposta não está nos documentos indexados, para ver como cada versão do prompt se comporta:
""".strip()

    c = insert_after(c, "hallucin", md("""
### O que o teste de alucinação revela?

A diferença entre um prompt básico e um com grounding explícito é dramática. O prompt básico frequentemente gera respostas inventadas que parecem corretas. O prompt com instrução de grounding força o LLM a admitir quando não sabe.

**Insight crítico:** testar com perguntas "fora do escopo" é tão importante quanto testar com perguntas que o sistema deveria saber responder. Sistemas RAG em produção sempre recebem queries para as quais não têm resposta.

**Como testar sistematicamente:** crie um conjunto de queries com resposta conhecida (para medir acurácia) E um conjunto de queries cujas respostas não estão nos documentos (para medir taxa de alucinação). Ambos os conjuntos são necessários para avaliar o sistema de forma completa.
"""))

    c.append(md("""
## Resumo: Evolução do Prompt

| Versão | Alucinação | Rastreabilidade | Complexidade |
|--------|-----------|-----------------|-------------|
| Básico | Alta | Nenhuma | Mínima |
| Com grounding | Baixa | Nenhuma | Baixa |
| Com citações | Baixa | Alta | Média |
| Estruturado | Baixíssima | Alta | Média-alta |

**Template de produção recomendado:**
```
Você é um assistente especializado. Responda APENAS com base nos documentos abaixo.
Se a resposta não estiver nos documentos, diga: "Não encontrei essa informação."

DOCUMENTOS:
[1] {doc1}
[2] {doc2}
...

PERGUNTA: {query}

RESPOSTA (cite as fontes como [1], [2] etc.):
```

**Próximos passos:**
- [01 — Naive RAG Architecture](../04_rag_architectures/01_naive_rag_arch.html): como essas peças se combinam numa arquitetura de produção?
"""))

    nb["cells"] = c
    save(nb, path)
    print(f"Documented: {path}")

# ---------------------------------------------------------------------------
# 00_quickstart.ipynb
# ---------------------------------------------------------------------------

def doc_quickstart():
    path = "00_quickstart.ipynb"
    nb = load(path)
    c = nb["cells"]

    c[0]["source"] = """
# RAG & Embeddings Playground — Quickstart

## O que é este playground?

Este é um ambiente de aprendizado hands-on sobre **embeddings**, **bancos de dados vetoriais** e **RAG (Retrieval-Augmented Generation)** — as tecnologias fundamentais por trás dos sistemas de IA com conhecimento próprio.

**Stack local:**
- **Ollama**: LLM local (Llama 3.2) para geração de texto — sem API key, sem custo por token
- **Qdrant**: banco de dados vetorial para busca semântica eficiente
- **sentence-transformers**: geração de embeddings de alta qualidade localmente

Este quickstart valida que tudo está funcionando e mostra um RAG completo do zero em menos de 5 minutos.

## O que você vai ver:

1. **Validação do stack**: Qdrant, Ollama e sentence-transformers funcionando
2. **Embeddings na prática**: frases similares ficam próximas no espaço vetorial
3. **Busca semântica no Qdrant**: indexar e buscar vetores em tempo real
4. **Geração com Ollama**: LLM local respondendo perguntas
5. **RAG completo**: query → retrieval → generation em ação

Se algum passo falhar, verifique se o Docker está rodando com `docker compose up -d`.
""".strip()

    for i, cell in enumerate(c):
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and ("1." in src or "Verif" in src) and "stack" in src.lower():
            c[i]["source"] = """
## 1. Verificando o Stack

Antes de qualquer coisa, vamos confirmar que todos os serviços estão disponíveis.

**Qdrant** (porta 6333): o banco de dados vetorial onde você vai indexar seus embeddings.

**Ollama** (porta 11434): servidor de LLM local. Roda modelos como Llama 3.2 sem precisar de GPU de ponta.

**sentence-transformers**: biblioteca Python para gerar embeddings de alta qualidade. O modelo `all-MiniLM-L6-v2` tem 22M parâmetros e gera vetores de 384 dimensões.

Se algum desses falhar, o resto do notebook não vai funcionar. Verifique com `docker compose ps` no terminal.
""".strip()

        elif cell["cell_type"] == "markdown" and "RAG" in src and "completo" in src.lower():
            c[i]["source"] = """
## 5. RAG Completo End-to-End

Agora que cada componente está validado individualmente, vamos combiná-los no pipeline completo:

```
Query do usuário
      ↓
Embedding (sentence-transformers) — 10ms
      ↓
Busca vetorial (Qdrant) — 5ms
      ↓
Top-K chunks relevantes
      ↓
Prompt com contexto → LLM (Ollama) — 2-8s
      ↓
Resposta gerada
```

O LLM recebe como contexto os documentos recuperados pelo Qdrant e gera uma resposta fundamentada neles. Se o documento relevante não for recuperado, a qualidade da resposta cai — isso é por que retrieval de qualidade é tão crítico.
""".strip()

    c = insert_after(c, "resposta", md("""
### Parabéns — o stack está funcionando!

Você acabou de ver um sistema RAG completo funcionando localmente:
- Embedding gerado em ~10ms
- Busca vetorial em ~5ms
- Resposta do LLM em ~2-8s (depende do hardware)

**O que explorar a seguir:**

Os notebooks estão organizados do fundamental ao avançado:

1. **Embeddings** (pasta `01_embeddings/`): o que são, como funcionam, como escolher o modelo certo
2. **Vector Databases** (pasta `02_vector_databases/`): Qdrant em profundidade — HNSW, quantização, filtros
3. **RAG Fundamentals** (pasta `03_rag_fundamentals/`): chunking, retrieval strategies, prompt engineering
4. **RAG Architectures** (pasta `04_rag_architectures/`): naive → advanced → agentic RAG

Comece por [01 — O que são Embeddings?](01_embeddings/01_what_are_embeddings.html) se você é novo no assunto.
"""))

    nb["cells"] = c
    save(nb, path)
    print(f"Documented: {path}")

# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    doc_quickstart()
    doc_01_what_are_embeddings()
    doc_03_float_types()
    doc_04_distance_metrics()
    doc_05_models_comparison()
    doc_qdrant_intro()
    doc_hnsw()
    doc_naive_rag()
    doc_chunking()
    doc_retrieval()
    doc_prompts()
    print("\nAll notebooks documented!")
