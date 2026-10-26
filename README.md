---
title: VeriRAG
emoji: 🔎
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# VeriRAG

**Hallucination-aware retrieval-augmented generation with claim-level citation verification.**

A normal RAG system retrieves documents and lets an LLM write an answer with citations. It never checks whether those citations actually back what the LLM wrote. VeriRAG splits every answer into atomic claims and checks each claim against its cited source with a dedicated fact-checking model. Each claim gets one of four labels:

| Label | Meaning |
|---|---|
| 🟢 `supported` | The cited source supports the claim |
| 🔵 `miscited` | Another retrieved source supports it, but the citation is wrong or missing |
| 🟠 `partial` | Only weak support |
| 🔴 `unsupported` | No retrieved source supports it, so it is likely a hallucination |

## Pipeline

```
query
  ├─ BM25 (bm25s) ──────────┐
  └─ BGE-M3 + FAISS ────────┴─ RRF fusion → BGE reranker → top-5 sources
                                                              │
                         LLM (Qwen3 via Ollama / Groq) ◄──────┘
                              │  answer with [n] citations
                              ▼
                 claim decomposition (LLM atomic claims | per-sentence)
                              │
                              ▼
             verifier (MiniCheck | DeBERTa-v3 NLI) scores claim × source
                              │
                              ▼
          labels + grounded-rate + citation-accuracy
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# local LLM
ollama pull qwen3:4b

python scripts/build_index.py              # indexes everything in data/raw
python scripts/ask.py "When was Python 3.0 released?"
streamlit run app/streamlit_app.py
pytest
```

Put your own `.txt`, `.md`, or `.pdf` files in `data/raw/` and rebuild the index.

## Evaluation

```bash
# Retrieval ablation: BM25 vs dense vs hybrid vs hybrid+rerank
python scripts/eval_retrieval.py data/eval/retrieval_sample.jsonl --k 5

# Verifier benchmark on LLM-AggreFact (balanced accuracy)
python scripts/eval_verifier.py --verifier nli --limit 200
python scripts/eval_verifier.py --verifier minicheck --limit 200
```

## Deploy (Hugging Face Spaces, free)

1. Create a **Docker** Space and push this repo to it.
2. Add `GROQ_API_KEY` as a Space secret (free key from console.groq.com).
3. To skip the first-load wait, build `data/index/` on Colab and commit it.

## Stack (₹0)

bm25s · FAISS · BGE-M3 · bge-reranker-v2-m3 · Qwen3-4B (Ollama) / Groq free tier · MiniCheck · DeBERTa-v3 NLI · Streamlit · HF Spaces
