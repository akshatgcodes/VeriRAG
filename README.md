<p align="center">
  <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24"
       fill="none" stroke="#16a34a" stroke-width="1.5"
       stroke-linecap="round" stroke-linejoin="round">
    <path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z" />
    <path d="m9 12 2 2 4-4" />
  </svg>
</p>

<h1 align="center">VeriRAG</h1>

<p align="center">
  <strong>Hallucination-aware retrieval-augmented generation with claim-level citation verification.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Ollama-000000?style=flat-square&logo=ollama&logoColor=white" alt="Ollama">
  <img src="https://img.shields.io/badge/Groq-F55036?style=flat-square" alt="Groq">
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/license-GPL--3.0-16a34a?style=flat-square" alt="License">
</p>

---

## What is VeriRAG?

A normal RAG system retrieves documents, lets an LLM write an answer with citations, and trusts it. It never checks whether the cited source actually says what the LLM claims. VeriRAG breaks every answer into atomic claims and has a dedicated fact-checking model verify each claim against every retrieved source. You can then see which sentences are grounded, which ones cite the wrong source, and which ones the model made up. The whole stack runs on free, open-source models.

---

## What you get

- **Claim-level verdicts.** Each claim is labelled `supported`, `miscited`, `partial` or `unsupported`, so a hallucination is visible right in the answer.
- **Citation repair.** If a claim is true but cites the wrong source (or none), VeriRAG names the source that actually supports it.
- **Hybrid retrieval.** BM25 and BGE-M3 dense search are fused with RRF, then reranked by a cross-encoder.
- **Two swappable verifiers.** MiniCheck (built for grounding checks) and a DeBERTa-v3 NLI baseline, with a benchmark script to compare them.
- **Eval harness.** Retrieval ablations (Recall@k, MRR, nDCG), verifier balanced accuracy on LLM-AggreFact, and an end-to-end hallucination report.

---

## Stack

| Layer | Tech |
|---|---|
| Retrieval | bm25s · FAISS · BGE-M3 · bge-reranker-v2-m3 |
| Generation | Qwen3-4B via Ollama (local) · Groq free tier (hosted demo) |
| Verification | MiniCheck (Flan-T5-Large) · DeBERTa-v3-large NLI |
| Frontend | Streamlit |
| Eval | LLM-AggreFact · custom Python scripts |
| Infra | Docker · Hugging Face Spaces |

---

## Results

### Retrieval

[BEIR SciFact](https://github.com/beir-cellar/beir) test set: 300 queries over 5,183 scientific abstracts. Each system's chunk ranking is collapsed to documents before scoring.

| System | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|
| BM25 | 0.774 | 0.626 | 0.656 |
| Dense (BGE-M3) | 0.790 | 0.616 | 0.651 |
| Hybrid (RRF) | 0.834 | 0.665 | 0.699 |
| Hybrid + rerank | **0.847** | **0.722** | **0.747** |

RRF beats both single retrievers without any weight tuning, and the cross-encoder adds another 0.05 nDCG@10. Reproduce with `scripts/prepare_scifact.py`, then `build_index.py` and `eval_retrieval.py --k 10`.

### Verifier

Balanced accuracy on [LLM-AggreFact](https://huggingface.co/datasets/lytang/LLM-AggreFact): 11 grounding benchmarks, with 100 shuffled test examples from each (seed 0) and a 0.5 threshold.

| Benchmark | DeBERTa-v3 NLI | MiniCheck (Flan-T5-L) |
|---|---|---|
| AggreFact-CNN | 0.462 | **0.666** |
| AggreFact-XSum | 0.616 | **0.682** |
| ClaimVerify | 0.621 | **0.670** |
| ExpertQA | **0.607** | 0.531 |
| FactCheck-GPT | 0.701 | **0.747** |
| Lfqa | 0.841 | **0.868** |
| RAGTruth | 0.750 | **0.882** |
| Reveal | 0.772 | **0.859** |
| TofuEval-MediaS | 0.582 | **0.738** |
| TofuEval-MeetB | 0.618 | **0.756** |
| Wice | 0.704 | **0.724** |
| **Average** | 0.661 | **0.738** |

MiniCheck wins on 10 of 11 benchmarks and by 7.7 points on average, which is why it is the default verifier. Its largest lead is on RAGTruth (+13 points), the benchmark closest to VeriRAG's own setting. The NLI baseline is weakest on long news articles (CNN), even with the sliding window. With 100 examples per benchmark, single-benchmark numbers can move by a few points. Reproduce with `eval_verifier.py --verifier {nli,minicheck} --limit 100`.

### End to end

100 SciFact test claims, each asked as "Is it true that …?", run through the full pipeline over the SciFact corpus. Answers come from `qwen/qwen3.8-27b` on Groq, claims are split by the same model, and MiniCheck verifies them.

| | |
|---|---|
| Questions answered | 79 (21 abstained with "I don't know") |
| Claims checked | 551 |
| Supported by the cited source | 79.3% |
| Miscited (true, but a different source supports it) | 4.2% |
| Partially supported | 4.4% |
| Unsupported | 12.2% |
| **Mean grounded rate** (supported or miscited) | **85.1%** |
| **Mean citation accuracy** (supported by the cited source) | **81.0%** |

So even a 27B model answering only from retrieved abstracts makes about one claim in eight that no source backs, and about one in twenty-five cites the wrong source. Read the unsupported rate as an upper bound: it also counts the model's own arithmetic ("493 per million ≈ 1 in 2,028") and statements about the sources themselves ("the sources do not say …"), which no abstract states word for word. Reproduce with `scripts/eval_e2e.py data/bench/scifact/questions_e2e_100.jsonl` (set `LLM_PROVIDER=groq` and `VERIRAG_DATA_DIR=data/bench/scifact`).

---

## Engineering Decisions

**Why a separate verifier instead of asking the LLM to check itself?**
An LLM grading its own answer shares the blind spots that produced the error. A small classifier trained for entailment is deterministic, cheap on CPU and can be benchmarked on its own.

**Why score each claim against every retrieved source, not just the cited one?**
It separates two different failures. A claim no source supports is a hallucination. A claim another source supports is a citation error. Checking only the cited source would call both "unsupported".

**Why RRF to merge BM25 and dense results?**
BM25 scores and cosine similarities live on different scales, so adding them needs tuning. Reciprocal Rank Fusion only uses ranks, so it needs no weights and still rewards chunks both retrievers agree on.

**Why slide a sentence window over chunks for the NLI verifier?**
NLI models are trained on short premises and truncate at 512 tokens. Scoring 3-sentence windows and taking the maximum keeps the premise in the range the model was trained on.

**What would you do differently in v2?**
Fine-tune the verifier on RAGTruth, export the models to ONNX/int8 for faster CPU inference, and rebuild the index automatically when documents change.

---

## Docs

| Document | Description |
|---|---|
| [PRD](docs/PRD.md) | Product requirements: goals, user stories, non-goals |
| [Architecture](docs/ARCHITECTURE.md) | System design, data flow, component breakdown |
| [Decisions](docs/DECISIONS.md) | Every major technical decision and why |
| [Setup](docs/SETUP.md) | Local dev setup and env vars |
| [Deploy](docs/DEPLOY.md) | Production deployment and rollback |
| [Contributing](CONTRIBUTING.md) | How to propose changes |
| [Changelog](CHANGELOG.md) | What changed in each release |
| [Security](SECURITY.md) | How to report a vulnerability |

---

<div align="center">

<h3>Akshat Gupta</h3>

<a href="https://akshatgcodes.in"><img src="https://img.shields.io/badge/Website-akshatgcodes.in-111111?style=flat-square&logo=googlechrome&logoColor=white" alt="Website"></a>
<a href="https://linkedin.com/in/akshatgcodes"><img src="https://img.shields.io/badge/LinkedIn-akshatgcodes-0A66C2?style=flat-square" alt="LinkedIn"></a>
<a href="https://github.com/akshatgcodes"><img src="https://img.shields.io/badge/GitHub-akshatgcodes-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub"></a>
<a href="mailto:akshatgcodes@gmail.com"><img src="https://img.shields.io/badge/Email-akshatgcodes%40gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white" alt="Email"></a>

</div>
