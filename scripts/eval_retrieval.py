"""Retrieval ablation: BM25 vs dense vs hybrid (RRF) vs hybrid + rerank.

Input: a JSONL file, one object per line:
    {"query": "...", "relevant_doc_ids": ["python.md"]}
doc ids are file paths relative to data/raw.

    python scripts/eval_retrieval.py data/eval/retrieval_sample.jsonl --k 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verirag import Settings  # noqa: E402
from verirag.fusion import rrf  # noqa: E402
from verirag.retrieval import HybridRetriever  # noqa: E402


def metrics(ranked_doc_ids: list[str], relevant: set[str], k: int) -> tuple[float, float]:
    top = ranked_doc_ids[:k]
    recall = len(relevant & set(top)) / len(relevant)
    mrr = next((1.0 / (i + 1) for i, d in enumerate(top) if d in relevant), 0.0)
    return recall, mrr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("qrels")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    s = Settings()
    r = HybridRetriever(s)
    rows = [json.loads(line) for line in open(args.qrels, encoding="utf-8") if line.strip()]

    def doc_ids(indices):
        return [r.chunks[i].doc_id for i in indices]

    systems = {
        "bm25": lambda q: doc_ids(r.bm25_search(q, s.k_bm25)),
        "dense": lambda q: doc_ids(r.dense_search(q, s.k_dense)),
        "hybrid_rrf": lambda q: doc_ids(i for i, _ in rrf(
            [r.bm25_search(q, s.k_bm25), r.dense_search(q, s.k_dense)], k=s.rrf_k)),
        "hybrid_rerank": lambda q: [h.chunk.doc_id for h in r.search(q, top_k=args.k, rerank=True)],
    }

    print(f"{'system':<15} Recall@{args.k}  MRR@{args.k}")
    for name, fn in systems.items():
        rec, mrr = zip(*(metrics(fn(row["query"]), set(row["relevant_doc_ids"]), args.k)
                         for row in rows))
        print(f"{name:<15} {sum(rec) / len(rec):.3f}      {sum(mrr) / len(mrr):.3f}")
