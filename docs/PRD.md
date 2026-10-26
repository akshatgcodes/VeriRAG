# VeriRAG — Product Requirements Document

**Status:** Draft
**Owner:** Akshat Gupta
**One-liner:** Hallucination-aware retrieval-augmented generation with claim-level citation verification.

---

## 1. Problem

RAG systems attach citations to their answers, and readers take a citation as proof. But the LLM writes the citation, and nothing checks it. A model can cite a real source for a sentence that the source never says, or mix a true fact with an invented one inside the same sentence. In high-stakes domains such as law, medicine or finance, an answer that looks sourced but isn't is worse than no answer at all, because it is trusted.

---

## 2. Goals (v1 / MVP)

1. Answer questions over a local document collection with numbered citations.
2. Split every answer into atomic claims and verify each claim against the retrieved sources.
3. Label every claim `supported`, `miscited`, `partial` or `unsupported`, and show the label inline in the answer.
4. Treat "I don't know based on the provided sources" as a correct answer, not as a hallucination.
5. Ship evaluation scripts for retrieval, for the verifier alone, and for the full pipeline.
6. Run entirely on free, open-source models, with no paid API required.

---

## 3. Non-Goals (explicit scope cuts)

- **Document upload in the UI.** Documents are added to `data/raw/` and indexed by a script. Upload means handling untrusted files and re-indexing on a shared CPU Space.
- **Fixing hallucinations automatically.** v1 flags and explains problems. Regenerating or deleting claims changes the answer the user asked for and needs its own evaluation.
- **Multi-user accounts and history.** The demo is single-session. Nothing is stored between queries.
- **GPU serving.** The hosted demo targets free CPU hardware. Speed is improved by smaller models and a hosted LLM, not by paid GPUs.

---

## 4. Users

**Primary:** Developers and researchers building RAG systems who want to measure how often their pipeline hallucinates or miscites.

**Secondary:** Recruiters and interviewers evaluating this as a portfolio piece. The labels have to be readable without knowing the internals.

---

## 5. User Stories

1. *As a user,* I ask a question and get a short answer with numbered citations so that I can check the sources myself.
2. *As a user,* I see each sentence underlined in a colour that matches its verdict so that I can tell at a glance which parts are safe to trust.
3. *As a user,* I see which source actually supports a miscited claim so that I can fix the citation.
4. *As a developer,* I switch between the MiniCheck and NLI verifiers with one env var so that I can compare them.
5. *As a developer,* I run one script to benchmark retrieval (BM25, dense, hybrid, hybrid + rerank) so that I can justify each stage.
6. *As a developer,* I run the full pipeline over a question set and get the share of supported, miscited, partial and unsupported claims.

---

## 6. Functional Requirements

### 6.1 Ingestion

- Read `.txt`, `.md` and `.pdf` files from `data/raw/`.
- Split documents into sentence-aligned chunks of about 200 words, with about 40 words of overlap.
- Build a BM25 index and a FAISS dense index, both saved under `data/index/`.

### 6.2 Retrieval

- Run BM25 and dense search, merge the results with Reciprocal Rank Fusion, and rerank the top 20 candidates with a cross-encoder.
- Pass the top 5 chunks to the LLM as numbered sources.

### 6.3 Generation

- The LLM answers only from the sources, cites after every sentence, and abstains with a fixed phrase when the sources don't contain the answer.
- The LLM provider is selected by config: Ollama locally, Groq for the hosted demo.

### 6.4 Verification

- Parse citations, including ones written after the full stop, and drop citation numbers that point to no source.
- Decompose the answer into atomic claims with the LLM. If the output can't be parsed, fall back to one claim per sentence.
- Score every claim against every source, then label it using the supported (0.7) and partial (0.4) thresholds.
- Report the grounded rate (supported + miscited) and citation accuracy (supported only).

### 6.5 Interface

- Streamlit page with the colour-coded answer, metrics, a claims table, the sources and per-stage timings.
- Clear error messages when the LLM can't be reached or the API key is missing.

---

## 7. Non-Functional Requirements

- **Cost:** ₹0 to run and host. All models are open source. The hosted LLM uses the Groq free tier.
- **Security:** API keys are read from env vars only. `.env` is gitignored. LLM output is HTML-escaped before it is rendered.
- **Reliability:** The pipeline doesn't depend on the LLM producing valid JSON. Bad decomposition output falls back to per-sentence claims. A failed MiniCheck load falls back to the NLI verifier.
- **Portability:** Runs on CPU on macOS and Linux. The Docker image installs CPU-only PyTorch.
- **Testability:** Core logic is covered by unit tests that need no model downloads, so CI stays fast.

---

## 8. Success Metrics

| Metric | Target |
|---|---|
| Verifier quality | Balanced accuracy on LLM-AggreFact reported for both verifiers |
| Retrieval quality | Hybrid + rerank matches or beats BM25-only and dense-only on Recall@5 for the eval set |
| Hallucination visibility | Every unsupported claim in the end-to-end report is shown in red in the UI |
| Live demo | Public Hugging Face Space answers a question end to end |

---

## 9. Risks & Open Questions

- **CPU latency.** Three transformer models run per query on free CPU hardware. Mitigated by reranking only 20 candidates, sending only 5 sources to the LLM, and using a hosted LLM for the demo.
- **Verifier errors.** Verifiers can mark a correct paraphrase as unsupported or miss a subtle contradiction. Mitigated by benchmarking both verifiers and showing raw scores next to labels.
- **Decomposition drift.** The LLM could add facts while splitting claims. The prompt forbids it, and per-sentence mode is available as a baseline.
- **Thresholds.** The 0.7 and 0.4 thresholds are defaults, not tuned values. **Open question:** tune them per verifier on LLM-AggreFact.

---

## 10. v2 Candidates

- Fine-tune the verifier on RAGTruth. It needs labelled data and GPU time on Colab.
- Regenerate or remove unsupported claims. This needs its own evaluation so it doesn't hide errors.
- Document upload with automatic re-indexing. It needs file validation and incremental indexing.
- ONNX/int8 export for faster CPU inference. It is an optimisation to make once the pipeline is stable.

---

<div align="center">

<h3>Akshat Gupta</h3>

<a href="https://akshatgcodes.in"><img src="https://img.shields.io/badge/Website-akshatgcodes.in-111111?style=flat-square&logo=googlechrome&logoColor=white" alt="Website"></a>
<a href="https://linkedin.com/in/akshatgcodes"><img src="https://img.shields.io/badge/LinkedIn-akshatgcodes-0A66C2?style=flat-square" alt="LinkedIn"></a>
<a href="https://github.com/akshatgcodes"><img src="https://img.shields.io/badge/GitHub-akshatgcodes-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub"></a>
<a href="mailto:akshatgcodes@gmail.com"><img src="https://img.shields.io/badge/Email-akshatgcodes%40gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white" alt="Email"></a>

</div>
