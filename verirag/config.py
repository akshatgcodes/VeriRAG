"""Central settings. Every value can be overridden with an environment variable."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _env(key: str, default):
    value = os.getenv(key)
    if value is None:
        return default
    return type(default)(value) if not isinstance(default, Path) else Path(value)


@dataclass
class Settings:
    # Paths
    data_dir: Path = _env("VERIRAG_DATA_DIR", ROOT / "data")

    # Retrieval
    embed_model: str = _env("VERIRAG_EMBED_MODEL", "BAAI/bge-m3")
    rerank_model: str = _env("VERIRAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    chunk_words: int = _env("VERIRAG_CHUNK_WORDS", 200)
    chunk_overlap_words: int = _env("VERIRAG_CHUNK_OVERLAP", 40)
    k_bm25: int = _env("VERIRAG_K_BM25", 30)
    k_dense: int = _env("VERIRAG_K_DENSE", 30)
    k_rerank: int = _env("VERIRAG_K_RERANK", 20)  # candidates sent to the reranker
    top_k: int = _env("VERIRAG_TOP_K", 5)  # sources given to the LLM
    rrf_k: int = _env("VERIRAG_RRF_K", 60)
    use_reranker: bool = _env("VERIRAG_USE_RERANKER", "1") == "1"

    # Generation: "ollama" for local dev, "groq" for the hosted demo
    llm_provider: str = _env("LLM_PROVIDER", "ollama")
    ollama_url: str = _env("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = _env("OLLAMA_MODEL", "qwen3:4b")
    groq_model: str = _env("GROQ_MODEL", "qwen/qwen3.8-27b")
    groq_api_key: str = _env("GROQ_API_KEY", "")

    # Verification: "minicheck" (primary) or "nli" (DeBERTa baseline)
    verifier: str = _env("VERIRAG_VERIFIER", "minicheck")
    minicheck_model: str = _env("VERIRAG_MINICHECK_MODEL", "flan-t5-large")
    nli_model: str = _env(
        "VERIRAG_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    )
    decompose: str = _env("VERIRAG_DECOMPOSE", "llm")  # "llm" or "sentence"
    supported_threshold: float = _env("VERIRAG_SUPPORTED_T", 0.7)
    partial_threshold: float = _env("VERIRAG_PARTIAL_T", 0.4)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"
