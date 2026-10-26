"""Hybrid retrieval: BM25 + dense (BGE-M3/FAISS), fused with RRF, then cross-encoder rerank."""

import json
from dataclasses import dataclass

from .chunking import Chunk
from .config import Settings
from .fusion import rrf


@dataclass
class Hit:
    chunk: Chunk
    rrf_score: float
    rerank_score: float | None = None


class HybridRetriever:
    def __init__(self, settings: Settings | None = None):
        import bm25s
        import faiss
        from sentence_transformers import SentenceTransformer

        self.s = settings or Settings()
        d = self.s.index_dir
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [Chunk(**json.loads(line)) for line in f]
        self.bm25 = bm25s.BM25.load(str(d / "bm25"))
        self.index = faiss.read_index(str(d / "dense.faiss"))
        self.embedder = SentenceTransformer(self.s.embed_model)
        self.embedder.max_seq_length = 512
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.s.rerank_model, max_length=512)
        return self._reranker

    def bm25_search(self, query: str, k: int) -> list[int]:
        import bm25s

        k = min(k, len(self.chunks))
        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        results, scores = self.bm25.retrieve(tokens, k=k, show_progress=False)
        return [int(i) for i, s in zip(results[0], scores[0]) if s > 0]

    def dense_search(self, query: str, k: int) -> list[int]:
        k = min(k, len(self.chunks))
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        _, ids = self.index.search(q, k)
        return [int(i) for i in ids[0] if i >= 0]

    def search(self, query: str, top_k: int | None = None, rerank: bool | None = None) -> list[Hit]:
        top_k = top_k or self.s.top_k
        rerank = self.s.use_reranker if rerank is None else rerank

        fused = rrf(
            [self.bm25_search(query, self.s.k_bm25), self.dense_search(query, self.s.k_dense)],
            k=self.s.rrf_k,
        )
        hits = [Hit(self.chunks[i], score) for i, score in fused[: self.s.k_rerank]]

        if rerank and hits:
            scores = self.reranker.predict([(query, h.chunk.text) for h in hits])
            for h, sc in zip(hits, scores):
                h.rerank_score = float(sc)
            hits.sort(key=lambda h: h.rerank_score, reverse=True)

        return hits[:top_k]
