#!/bin/bash
# Pull recommended Ollama models for this playground
# Usage: bash infrastructure/ollama/pull_models.sh
# Or:    docker exec ollama bash /path/to/pull_models.sh

set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

echo "🦙 Pulling Ollama models..."
echo "Host: $OLLAMA_HOST"
echo ""

# ── LLMs ──────────────────────────────────────────────────────────────────────
echo "📥 Pulling llama3.2 (default chat model, ~2GB)..."
ollama pull llama3.2

echo "📥 Pulling mistral (alternative, ~4GB)..."
ollama pull mistral

# ── Embedding models via Ollama ───────────────────────────────────────────────
echo "📥 Pulling nomic-embed-text (768d embedding model)..."
ollama pull nomic-embed-text

echo "📥 Pulling mxbai-embed-large (1024d embedding model)..."
ollama pull mxbai-embed-large

echo ""
echo "✅ All models pulled successfully!"
echo ""
echo "Available models:"
ollama list
