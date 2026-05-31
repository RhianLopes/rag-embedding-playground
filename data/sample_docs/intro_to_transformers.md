# Introduction to Transformers

## What are Transformers?

Transformers are a type of neural network architecture introduced in the seminal 2017 paper
"Attention is All You Need" by Vaswani et al. from Google Brain. They have become the dominant
architecture in natural language processing (NLP) and are the foundation of large language
models (LLMs) like GPT, BERT, and Claude.

## The Core Innovation: Self-Attention

The key innovation in Transformers is the **self-attention mechanism**, which allows the model
to weigh the importance of each token in the input when processing any other token. Unlike
recurrent neural networks (RNNs) that process sequences step by step, Transformers process
all tokens in parallel.

### How Self-Attention Works

For each token, the model computes three vectors:
- **Query (Q)**: What information am I looking for?
- **Key (K)**: What information do I contain?
- **Value (V)**: What information will I pass on?

The attention score between two tokens is computed as:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

Where `d_k` is the dimension of the key vectors, used for scaling.

## Multi-Head Attention

Instead of performing a single attention operation, Transformers use **multi-head attention**,
which runs multiple attention operations in parallel. Each "head" can focus on different aspects
of the relationships between tokens.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
where head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)
```

## Positional Encoding

Since Transformers don't process sequences recurrently, they need a way to encode the position
of each token. The original paper uses sinusoidal positional encodings:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Modern models often use learned positional embeddings or rotary position embeddings (RoPE).

## The Encoder-Decoder Architecture

The original Transformer had two main components:

### Encoder
- Processes the input sequence
- Each layer contains: Multi-Head Self-Attention → Add & Norm → Feed-Forward → Add & Norm
- BERT is an encoder-only model

### Decoder
- Generates the output sequence auto-regressively
- Uses masked self-attention (can't attend to future tokens)
- Also has cross-attention to attend to encoder outputs
- GPT is a decoder-only model

## Scaling Laws

Research has shown that Transformer performance scales predictably with:
1. **Model size** (number of parameters)
2. **Dataset size** (number of training tokens)
3. **Compute budget** (FLOPs used for training)

The Chinchilla scaling laws (Hoffmann et al., 2022) suggest that for optimal training,
model size and dataset size should scale proportionally.

## Applications

Transformers have been applied to:
- **Text**: Language modeling, translation, summarization, question answering
- **Images**: Vision Transformers (ViT), DALL-E, Stable Diffusion
- **Audio**: Whisper (speech recognition), MusicGen
- **Multimodal**: GPT-4V, Gemini, Claude

## Key Models Timeline

| Year | Model | Type | Parameters |
|------|-------|------|------------|
| 2017 | Transformer | Encoder-Decoder | ~65M |
| 2018 | BERT-base | Encoder | 110M |
| 2019 | GPT-2 | Decoder | 1.5B |
| 2020 | GPT-3 | Decoder | 175B |
| 2022 | ChatGPT (GPT-3.5) | Decoder | ~175B |
| 2023 | GPT-4 | Decoder | ~1.8T (est.) |
| 2023 | Llama 2 | Decoder | 7B-70B |
| 2024 | Llama 3 | Decoder | 8B-405B |
