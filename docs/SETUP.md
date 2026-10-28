# Local Setup — VeriRAG

> There is no live demo yet. This guide is for running VeriRAG locally.

---

## Prerequisites

- Python 3.11 (pinned in `.python-version`)
- [Ollama](https://ollama.com) for a local LLM, or a free [Groq](https://console.groq.com) API key
- About 8 GB of free disk space for model weights. BGE-M3 and the reranker are about 2 GB each, and the verifiers come on top of that.
- 16 GB of RAM is recommended. All models run on CPU.

---

## 1. Clone and install

```bash
git clone https://github.com/akshatgcodes/VeriRAG
cd VeriRAG
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs CPU-only PyTorch, the full model stack and MiniCheck. For a light install that is enough for the tests, use `pip install -e ".[dev]"`.

---

## 2. Environment variables

Copy `.env.example` to `.env` and fill in what you need:

```bash
cp .env.example .env
```

| Variable | Where to get it |
|---|---|
| `LLM_PROVIDER` | `ollama` (default) or `groq` |
| `OLLAMA_MODEL` | Any pulled Ollama model. The default is `qwen3:4b` |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys (only for `groq`) |
| `GROQ_MODEL` | Default `qwen/qwen3.8-27b` |
| `VERIRAG_VERIFIER` | `minicheck` (default) or `nli` |
| `VERIRAG_DECOMPOSE` | `llm` (atomic claims, default) or `sentence` |

Every other setting in `verirag/config.py` (chunk size, top-k, thresholds, model names) can be overridden with its `VERIRAG_*` variable.

---

## 3. Build the index

Put `.txt`, `.md` or `.pdf` files in `data/raw/` (two sample documents are included), then run:

```bash
python scripts/build_index.py
```

The first run downloads BGE-M3. For a large corpus, run this on Colab or Kaggle and copy `data/index/` back.

---

## 4. Run locally

```bash
ollama pull qwen3:4b                                      # once
python scripts/ask.py "When was Python 3.0 released?"     # CLI
streamlit run app/streamlit_app.py                        # web UI
pytest                                                    # tests
```

The UI runs at `http://localhost:8501`.

### Evaluation

```bash
python scripts/eval_retrieval.py data/eval/retrieval_sample.jsonl --k 5
python scripts/eval_verifier.py --verifier nli --limit 200
python scripts/eval_verifier.py --verifier minicheck --limit 200
python scripts/eval_e2e.py data/eval/questions_sample.jsonl
```

`eval_verifier.py` uses LLM-AggreFact. You may need to accept its terms on Hugging Face and run `huggingface-cli login`.

---

## 5. Deploy to production

See [DEPLOY.md](DEPLOY.md).

---

## Known local-only limitations

- The first run of each model downloads its weights from the Hugging Face Hub, which can take several minutes.
- On CPU, verifier speed varies a lot. In a local smoke test on an Apple Silicon laptop (3 claims against 2 sources), MiniCheck took about 2 seconds and the windowed NLI verifier took about 18–22 seconds.
- MiniCheck's weights (about 3 GB) are cached under `data/models/`. If a download is interrupted, delete that folder and run again. Otherwise the app falls back to the NLI verifier with a warning.
- On macOS, `torch` must be imported before `faiss`, or the process can segfault from duplicate OpenMP runtimes. The code already does this. Keep the order if you add new entry points.

---

<div align="center">

<h3>Akshat Gupta</h3>

<a href="https://akshatgcodes.in"><img src="https://img.shields.io/badge/Website-akshatgcodes.in-111111?style=flat-square&logo=googlechrome&logoColor=white" alt="Website"></a>
<a href="https://linkedin.com/in/akshatgcodes"><img src="https://img.shields.io/badge/LinkedIn-akshatgcodes-0A66C2?style=flat-square" alt="LinkedIn"></a>
<a href="https://github.com/akshatgcodes"><img src="https://img.shields.io/badge/GitHub-akshatgcodes-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub"></a>
<a href="mailto:akshatgcodes@gmail.com"><img src="https://img.shields.io/badge/Email-akshatgcodes%40gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white" alt="Email"></a>

</div>
