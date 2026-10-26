# Engineering Decisions — VeriRAG

---

## Decision 1 — A dedicated verifier model, not LLM self-checking

**Context:** Each claim needs a support check. One option was to ask the same LLM "is this claim supported?". The other was to use a separate model trained for entailment or fact-checking.

**Decision:** A separate verifier. MiniCheck is the primary one, with DeBERTa-v3 NLI as the baseline.

**Reason:** An LLM checking its own answer tends to repeat the reasoning that produced the error. A small classifier is deterministic and fast enough for CPU, and it can be benchmarked on its own (`scripts/eval_verifier.py` on LLM-AggreFact). MiniCheck is trained specifically to check LLM output against grounding documents, so it fits this job better than general NLI.

**Tradeoff:** It is one more model to download and hold in memory. A classifier also can't explain *why* a claim fails, it only gives a score.

---

## Decision 2 — Score every claim against every retrieved source

**Context:** The simplest check scores a claim only against the source it cites.

**Decision:** Score each claim against all top-5 sources. Keep both the best score among the cited sources and the best score overall.

**Reason:** This gives the `miscited` label. A claim that no source supports is a hallucination. A claim that a different source supports is a citation error, and the fix is different. It also handles claims with no citation at all.

**Tradeoff:** Verification costs `claims × 5` scores instead of `claims × cited sources`. On CPU it is the slowest stage.

---

## Decision 3 — Hybrid retrieval fused with RRF, then a reranker

**Context:** BM25 finds exact terms (names, years, versions) but misses paraphrases. Dense search finds paraphrases but can miss rare exact terms.

**Decision:** Run both, merge them with Reciprocal Rank Fusion, and rerank the top 20 with `bge-reranker-v2-m3`.

**Reason:** BM25 scores and cosine similarities are on different scales, so a weighted sum needs tuning. RRF uses only ranks, needs no weights, and rewards chunks that both retrievers rank highly. The cross-encoder reads the query and chunk together, which is more accurate than either retriever, but too slow to run on the whole corpus. That's why it only sees 20 candidates. `scripts/eval_retrieval.py` measures each stage separately so the choice can be checked.

**Tradeoff:** Three retrieval models instead of one, and the reranker adds latency on CPU.

---

## Decision 4 — Sliding sentence windows for the NLI verifier

**Context:** NLI models are trained on short, sentence-level premises and truncate input at 512 tokens. A retrieved chunk is about 200 words.

**Decision:** Split the chunk into sentences, score the claim against each 3-sentence window, and take the maximum entailment probability.

**Reason:** It keeps the premise close to what the model was trained on, and a fact anywhere in the chunk can still be found. In a local smoke test, the NLI verifier scored a supported claim at 1.00 and an invented one at 0.00 against the same chunk.

**Tradeoff:** It runs several forward passes per claim–chunk pair, and a fact spread across more than 3 sentences can be missed.

---

## Decision 5 — Ollama locally, Groq for the hosted demo, behind one interface

**Context:** The project must cost ₹0. Free Hugging Face Spaces run on CPU, and a 4B-parameter model generating on CPU next to three other transformer models would make each answer slow.

**Decision:** A two-provider `LLM` interface. It uses Ollama (`qwen3:4b`) for local development and Groq's free tier for the hosted demo, chosen with `LLM_PROVIDER`.

**Reason:** Development stays offline and free. The demo gets fast generation without paid hardware. Both are plain HTTP calls, so there is no SDK to install.

**Tradeoff:** The demo depends on a third-party free tier and its rate limits. The two providers run different models, so answers differ between local and hosted runs.

---

## Decision 6 — Abstention is not a claim

**Context:** For questions the sources don't cover, the prompt tells the model to say "I don't know based on the provided sources." Treated as a claim, this sentence has no support and would be labelled `unsupported`.

**Decision:** Sentences containing the abstention phrase are removed before decomposition, and the result is flagged `abstained`.

**Reason:** Abstaining is the correct behaviour. Counting it as a hallucination would penalise the model for being honest and make the metrics misleading.

**Tradeoff:** It depends on the model using the exact phrase. A differently worded refusal is still checked as a claim.

---

## What I'd do differently in v2

- **Fine-tune the verifier on RAGTruth.** Off-the-shelf verifiers aren't trained on this exact RAG answer style.
- **Tune the labelling thresholds per verifier.** The current 0.7 and 0.4 are defaults, not values fitted on data.
- **Export the models to ONNX/int8.** Verification dominates CPU latency.

---

## Explicit non-decisions (deferred to v2)

| Feature | Why deferred |
|---|---|
| Automatic answer repair | Rewriting claims needs its own evaluation, so that errors aren't hidden |
| Document upload in the UI | Needs untrusted-file handling and incremental indexing on a shared CPU Space |
| An orchestration framework (LangChain etc.) | Six short modules are easier to read and debug than framework abstractions at this size |

---

<div align="center">

<h3>Akshat Gupta</h3>

<a href="https://akshatgcodes.in"><img src="https://img.shields.io/badge/Website-akshatgcodes.in-111111?style=flat-square&logo=googlechrome&logoColor=white" alt="Website"></a>
<a href="https://linkedin.com/in/akshatgcodes"><img src="https://img.shields.io/badge/LinkedIn-akshatgcodes-0A66C2?style=flat-square" alt="LinkedIn"></a>
<a href="https://github.com/akshatgcodes"><img src="https://img.shields.io/badge/GitHub-akshatgcodes-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub"></a>
<a href="mailto:akshatgcodes@gmail.com"><img src="https://img.shields.io/badge/Email-akshatgcodes%40gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white" alt="Email"></a>

</div>
