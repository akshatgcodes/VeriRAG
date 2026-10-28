# Changelog

All notable changes to **VeriRAG** are documented here.

---

## [Unreleased]

### ✨ Added
- SciFact retrieval ablation (`scripts/prepare_scifact.py`) with doc-level dedupe and nDCG@10, plus a Results section in the README
- GPL-3.0 `LICENSE` file

### 🔧 Changed
- Default Groq model is now `qwen/qwen3.8-27b`, since `llama-3.1-8b-instant` was retired

### 🐛 Fixed
- Ollama calls timing out when Qwen3's thinking mode looped under greedy decoding
- A stray `</think>` tag left at the start of answers

---

## [1.0.0] — 2026-10-26

### ✨ Added
- Hybrid retrieval: BM25 and BGE-M3 dense search fused with Reciprocal Rank Fusion, then reranked by `bge-reranker-v2-m3`
- Cited answers from Ollama (local) or the Groq free tier (hosted demo)
- Claim decomposition into atomic claims, with a per-sentence fallback
- Claim verification with MiniCheck or a DeBERTa-v3 NLI baseline, labelled `supported`, `miscited`, `partial` or `unsupported`
- Streamlit UI with colour-coded sentences, grounded-rate and citation-accuracy metrics, a claims table and sources
- CLI (`scripts/ask.py`) and evaluation scripts for retrieval, the verifier (LLM-AggreFact) and the full pipeline
- Docker image for Hugging Face Spaces

### 🐛 Fixed
- An "I don't know" answer is no longer counted as a hallucination
- macOS segfault when faiss was loaded before torch
- MiniCheck failing to load without `accelerate`, which now falls back to the NLI verifier instead of crashing

---

[1.0.0]: https://github.com/akshatgcodes/VeriRAG/releases/tag/v1.0.0

---

<div align="center">

<h3>Akshat Gupta</h3>

<a href="https://akshatgcodes.in"><img src="https://img.shields.io/badge/Website-akshatgcodes.in-111111?style=flat-square&logo=googlechrome&logoColor=white" alt="Website"></a>
<a href="https://linkedin.com/in/akshatgcodes"><img src="https://img.shields.io/badge/LinkedIn-akshatgcodes-0A66C2?style=flat-square" alt="LinkedIn"></a>
<a href="https://github.com/akshatgcodes"><img src="https://img.shields.io/badge/GitHub-akshatgcodes-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub"></a>
<a href="mailto:akshatgcodes@gmail.com"><img src="https://img.shields.io/badge/Email-akshatgcodes%40gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white" alt="Email"></a>

</div>
