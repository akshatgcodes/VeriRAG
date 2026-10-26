"""Load raw documents, chunk them, and build the BM25 + FAISS indexes."""

import json
from pathlib import Path

from .chunking import Chunk, chunk_text
from .config import Settings

SUPPORTED = {".txt", ".md", ".pdf"}


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_chunks(raw_dir: Path, max_words: int, overlap_words: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED:
            continue
        doc_id = path.relative_to(raw_dir).as_posix()
        for text in chunk_text(read_file(path), max_words, overlap_words):
            chunks.append(Chunk(id=len(chunks), doc_id=doc_id, source=path.name, text=text))
    return chunks


def build_index(settings: Settings | None = None) -> int:
    import bm25s
    import faiss
    from sentence_transformers import SentenceTransformer

    settings = settings or Settings()
    chunks = load_chunks(settings.raw_dir, settings.chunk_words, settings.chunk_overlap_words)
    if not chunks:
        raise RuntimeError(f"No documents found in {settings.raw_dir}")

    out = settings.index_dir
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")

    texts = [c.text for c in chunks]

    bm25 = bm25s.BM25()
    bm25.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)
    bm25.save(str(out / "bm25"))

    embedder = SentenceTransformer(settings.embed_model)
    embedder.max_seq_length = 512
    vectors = embedder.encode(
        texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(out / "dense.faiss"))

    return len(chunks)


def index_exists(settings: Settings) -> bool:
    d = settings.index_dir
    return (d / "chunks.jsonl").exists() and (d / "dense.faiss").exists() and (d / "bm25").exists()
