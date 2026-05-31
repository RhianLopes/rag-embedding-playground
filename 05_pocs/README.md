# Módulo 05 — PoCs (Proof of Concepts)

> Projetos completos que integram todos os conceitos aprendidos.

## PoCs Disponíveis

### [Semantic Search](semantic_search/)
Sistema de busca semântica sobre documentos locais.

**Stack**: sentence-transformers + Qdrant + rich (CLI)
**Aprenda**: Pipeline completo de ingestion e busca, filtragem por metadata

```bash
# Indexar documentos
python 05_pocs/semantic_search/ingest.py --docs data/sample_docs/

# Buscar
python 05_pocs/semantic_search/search.py "como funciona attention?"
```

---

### [Q&A Chatbot](qa_chatbot/)
Chatbot que responde perguntas sobre documentos customizados.

**Stack**: sentence-transformers + Qdrant + Ollama (llama3.2) + LangChain
**Aprenda**: RAG end-to-end, prompt engineering, streaming de respostas

```bash
# Indexar e conversar
python 05_pocs/qa_chatbot/pipeline.py
```

---

### [RAG Evaluation com RAGAs](evaluation_ragas/)
Avaliação sistemática de qualidade de um sistema RAG.

**Stack**: RAGAs + Qdrant + Ollama
**Aprenda**: Métricas de avaliação, como identificar e corrigir falhas no RAG

---

## Executar todos os demos

```bash
# Subir infraestrutura
docker compose up -d

# Instalar dependências
uv sync

# Notebook interativo (recomendado)
uv run jupyter lab
# Abre: 05_pocs/semantic_search/demo.ipynb
```

## Arquitetura dos PoCs

```
05_pocs/
├── semantic_search/
│   ├── ingest.py      ← Carrega, chunka, embeda e indexa documentos
│   ├── search.py      ← CLI de busca semântica
│   └── demo.ipynb     ← Demo interativo
│
├── qa_chatbot/
│   ├── pipeline.py    ← Pipeline RAG completo com Ollama
│   └── demo.ipynb     ← Demo interativo com chat
│
└── evaluation_ragas/
    ├── README.md      ← Como avaliar seu RAG
    └── evaluate.ipynb ← Notebook de avaliação com RAGAs
```
