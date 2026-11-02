# VeriRAG — Architecture

---

## 1. Stack

| Layer | Tech |
|---|---|
| Language | Python 3.11 (local, CI) · 3.12 (Docker image) |
| Keyword search | bm25s (English stopwords) |
| Dense search | BGE-M3 (`BAAI/bge-m3`) via sentence-transformers, FAISS `IndexFlatIP` on normalised vectors |
| Fusion | Reciprocal Rank Fusion (k = 60) |
| Reranker | `BAAI/bge-reranker-v2-m3` cross-encoder |
| LLM | Ollama (`qwen3:4b`) or Groq (`qwen/qwen3.8-27b`), plain HTTP via `requests` |
| Verifier | MiniCheck `flan-t5-large` (primary), `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (baseline) |
| UI | Streamlit |
| Hosting | Docker image for Hugging Face Spaces (CPU) |

---

## 2. Components

```
verirag/
  config.py      Settings dataclass; every value overridable by env var or .env
  chunking.py    sentence splitting and sentence-aligned chunking
  ingest.py      reads data/raw, builds BM25 + FAISS indexes into data/index
  fusion.py      Reciprocal Rank Fusion
  retrieval.py   HybridRetriever: BM25 + dense -> RRF -> cross-encoder rerank
  llm.py         OllamaLLM / GroqLLM behind one chat() interface
  prompts.py     answer and decomposition prompts, abstention phrase
  claims.py      citation parsing, per-sentence and LLM claim decomposition
  verify.py      NLIVerifier, MiniCheckVerifier, label_claim()
  pipeline.py    VeriRAG: retrieve -> generate -> decompose -> verify
app/streamlit_app.py   web UI
scripts/               build_index, ask (CLI), eval_retrieval, eval_verifier, eval_e2e
tests/                 unit tests and a headless UI test (no model downloads)
```

### Ingestion (`ingest.py`, `chunking.py`)

Reads every `.txt`, `.md` and `.pdf` file under `data/raw/`. It splits the text into sentences and packs whole sentences into chunks of about 200 words, carrying the trailing ~40 words into the next chunk. It writes `chunks.jsonl`, a bm25s index and a FAISS index. It never runs during a query. The UI only builds the index if it is missing.

### Retriever (`retrieval.py`)

It takes the top 30 from BM25 and the top 30 from dense search, fuses them with RRF, and keeps the top 20. These are then scored by the cross-encoder and the best 5 are returned. Each `Hit` keeps both the RRF score and the rerank score. The reranker loads lazily, so the retrieval eval can measure BM25 and dense search without it.

### LLM providers (`llm.py`)

Both providers expose `chat(messages) -> str` at temperature 0. Ollama is called over its local REST API and Groq over its OpenAI-compatible endpoint. `<think>` blocks that Qwen3-style models emit are stripped.

### Claim extraction (`claims.py`)

Citations written after the full stop (`... 1991. [1]`) are first moved in front of it, so they stay with their sentence. Then the answer is split into sentences and the `[n]` markers are parsed. Citation numbers that point to no source are dropped. In `llm` mode, one extra LLM call splits the sentences into atomic claims, and each claim inherits its sentence's citations. In `sentence` mode, each sentence is one claim.

### Verifier (`verify.py`)

Both verifiers return P(supported) for a list of claims against one premise. The NLI verifier slides a 3-sentence window over the chunk and keeps the maximum entailment probability. MiniCheck handles long documents itself. `label_claim()` turns the score from the cited sources and the best score across all sources into a label.

### Pipeline (`pipeline.py`)

It wires the stages together, times each one, and returns a `Result` with the answer, sentences, sources, verified claims, `grounded_rate`, `citation_accuracy` and an `abstained` flag.

---

## 3. Data Flow

```
question
   │
   ├─► BM25 top-30 ─────┐
   └─► BGE-M3 top-30 ───┴─► RRF ─► top-20 ─► cross-encoder ─► top-5 sources
                                                                   │
        LLM: answer only from sources, cite [n] after every sentence ◄┘
                │
                ▼
        split sentences, parse + validate citations, drop abstention
                │
                ▼
        atomic claims (LLM JSON, fallback: one per sentence)
                │
                ▼
        verifier scores every claim × every source
                │
                ▼
        label = f(best cited score, best overall score)
                │
                ▼
        UI: coloured sentences · metrics · claims table · sources
```

1. The user submits a question in the Streamlit page (or `scripts/ask.py`).
2. The retriever returns 5 sources, each with its RRF and rerank scores.
3. The LLM writes a cited answer from the numbered sources, or abstains.
4. The answer is split into claims that carry their citations.
5. Each claim is scored against all 5 sources, labelled, and rendered. The sentence colour is the worst label among its claims.

---

## 4. Index layout

There is no database. Everything the query path reads is in `data/index/`:

- `chunks.jsonl`: one `{id, doc_id, source, text}` per chunk. `id` is the row number used by both indexes.
- `bm25/`: the bm25s sparse matrix and vocabulary.
- `dense.faiss`: a flat inner-product index over L2-normalised BGE-M3 vectors (inner product = cosine).

---

## 5. AI / LLM Design

### Input

The LLM sees only the top-5 chunks as `[n] (from <file>) <text>` and the question. Decomposition sees only the numbered answer sentences, with their citations removed.

### System prompt strategy

The answer prompt restricts the model to the given sources, requires a citation after every sentence, caps the answer at 5 sentences, and specifies an exact abstention phrase. The decomposition prompt asks for one fact per claim, replaces pronouns with their entities, and forbids adding, inferring or dropping information.

### Response schema

```jsonc
// decomposition output
[
  { "s": 1, "claim": "Python was created by Guido van Rossum." },
  { "s": 1, "claim": "Python was first released in 1991." }
]
```

### Validation

The JSON array is cut out of the reply (from the first `[` to the last `]`), parsed, and checked. A sentence index must be in range and a claim must be non-empty. Any parse error, or zero valid claims, falls back to one claim per sentence. The pipeline never fails on bad JSON.

### Failure handling

If the LLM can't be reached, the UI shows "Is `ollama serve` running?". If an HTTP error comes back, the UI shows it. If `GROQ_API_KEY` is missing, the UI shows that message instead of a traceback. If MiniCheck can't be imported or loaded, a warning is logged and the NLI verifier is used.

---

## 6. Security

- **Secrets:** `GROQ_API_KEY` is read from env vars or `.env`. It is gitignored and excluded by `.dockerignore`. In production it is a Hugging Face Space secret. The key is only used in server-side requests and never reaches the browser.
- **Rendering LLM output:** Answer sentences are passed through `html.escape` before being inserted into the coloured HTML, so a document or model reply can't inject markup into the page.
- **Untrusted documents:** Retrieved text goes into the prompt as data. Verification is done by a separate classifier against the source text, not by the LLM, so a prompt injection that steers the LLM can't also change a claim's label. VeriRAG checks *grounding*, not truth. A false statement written in an indexed document will verify as `supported`, so the index should only contain trusted documents.

---

## 7. Error Handling & Reliability

| Failure | Behaviour |
|---|---|
| Ollama not running | `requests.ConnectionError` is caught and the UI shows how to start it |
| Groq key missing | `ValueError` at load time is shown as a UI error |
| Decomposition returns bad JSON | Falls back to per-sentence claims |
| MiniCheck unavailable | Warning, then the NLI verifier is used |
| Index missing on first run | The UI builds it before loading models |
| LLM cites a source number that doesn't exist | The citation is dropped and the claim is treated as uncited |
| faiss loaded before torch on macOS | Segfault from duplicate OpenMP runtimes, prevented by importing torch first in `retrieval.py` and `ingest.py` |

---

## 8. Deployment

1. The `Dockerfile` builds a Python 3.12 slim image with CPU-only PyTorch and runs Streamlit on port 7860 as uid 1000, as Hugging Face Spaces expects.
2. `LLM_PROVIDER=groq` is set in the image. `GROQ_API_KEY` comes from a Space secret.
3. Models download from the Hugging Face Hub on first load. The index is built on first query if it isn't committed.

See [DEPLOY.md](DEPLOY.md).

---

## 9. Explicit Scope Cuts

- **No LangChain or LlamaIndex.** Every stage is a short, readable module, and the prompts and parsing are fully visible.
- **No incremental indexing.** Rebuild with `scripts/build_index.py` when documents change.
- **No automatic answer repair.** Claims are flagged, not rewritten.

---

<div align="center">

<h3>Akshat Gupta</h3>

<a href="https://akshatgcodes.in"><img src="https://img.shields.io/badge/Website-akshatgcodes.in-111111?style=flat-square&logo=googlechrome&logoColor=white" alt="Website"></a>
<a href="https://linkedin.com/in/akshatgcodes"><img src="https://img.shields.io/badge/LinkedIn-akshatgcodes-0A66C2?style=flat-square" alt="LinkedIn"></a>
<a href="https://github.com/akshatgcodes"><img src="https://img.shields.io/badge/GitHub-akshatgcodes-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub"></a>
<a href="mailto:akshatgcodes@gmail.com"><img src="https://img.shields.io/badge/Email-akshatgcodes%40gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white" alt="Email"></a>

</div>
