"""Retrieval ablation: BM25 vs dense vs hybrid (RRF) vs hybrid + rerank.

Input: a JSONL file, one object per line:
    {"query": "...", "relevant_doc_ids": ["python.md"]}
doc ids are file paths relative to data/raw. Rankings are over chunks, so each system's
list is collapsed to unique documents (first chunk wins) before scoring.

    python scripts/eval_retrieval.py data/eval/retrieval_sample.jsonl --k 5
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verirag import Settings  # noqa: E402
from verirag.fusion import rrf  # noqa: E402
from verirag.retrieval import HybridRetriever  # noqa: E402


def unique(doc_ids) -> list[str]:
    return list(dict.fromkeys(doc_ids))


def metrics(ranked_doc_ids: list[str], relevant: set[str], k: int) -> tuple[float, float, float]:
    """Recall@k, MRR@k and binary nDCG@k over a ranked list of unique doc ids."""
    top = ranked_doc_ids[:k]
    recall = len(relevant & set(top)) / len(relevant)
    mrr = next((1.0 / (i + 1) for i, d in enumerate(top) if d in relevant), 0.0)
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(top) if d in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return recall, mrr, dcg / idcg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("qrels")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=None, help="optional JSON file for the results")
    args = ap.parse_args()

    s = Settings()
    r = HybridRetriever(s)
    rows = [json.loads(line) for line in open(args.qrels, encoding="utf-8") if line.strip()]

    def doc_ids(indices):
        return unique(r.chunks[i].doc_id for i in indices)

    systems = {
        "bm25": lambda q: doc_ids(r.bm25_search(q, s.k_bm25)),
        "dense": lambda q: doc_ids(r.dense_search(q, s.k_dense)),
        "hybrid_rrf": lambda q: doc_ids(i for i, _ in rrf(
            [r.bm25_search(q, s.k_bm25), r.dense_search(q, s.k_dense)], k=s.rrf_k)),
        "hybrid_rerank": lambda q: unique(
            h.chunk.doc_id for h in r.search(q, top_k=s.k_rerank, rerank=True)),
    }

    k = args.k
    results = {}
    print(f"{'system':<15} Recall@{k}  MRR@{k}  nDCG@{k}")
    for name, fn in systems.items():
        rec, mrr, ndcg = (sum(col) / len(rows) for col in zip(
            *(metrics(fn(row["query"]), set(row["relevant_doc_ids"]), k) for row in rows)))
        results[name] = {f"recall@{k}": rec, f"mrr@{k}": mrr, f"ndcg@{k}": ndcg}
        print(f"{name:<15} {rec:.3f}      {mrr:.3f}   {ndcg:.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"queries": len(rows), **results}, indent=2))
