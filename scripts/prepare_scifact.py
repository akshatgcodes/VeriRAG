"""Download BEIR SciFact and lay it out as a VeriRAG data dir for the retrieval ablation.

Writes data/bench/scifact/raw/<doc_id>.txt (one abstract per file) and
data/bench/scifact/qrels_test.jsonl in the format eval_retrieval.py expects.

    python scripts/prepare_scifact.py
    VERIRAG_DATA_DIR=data/bench/scifact python scripts/build_index.py
    VERIRAG_DATA_DIR=data/bench/scifact python scripts/eval_retrieval.py \
        data/bench/scifact/qrels_test.jsonl --k 10
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

if __name__ == "__main__":
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bench/scifact")
    args = ap.parse_args()

    out = Path(args.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    for row in corpus:
        text = f"{row['title']}. {row['text']}" if row["title"] else row["text"]
        (raw / f"{row['_id']}.txt").write_text(text, encoding="utf-8")

    queries = {q["_id"]: q["text"] for q in load_dataset("BeIR/scifact", "queries", split="queries")}
    relevant: dict[str, list[str]] = defaultdict(list)
    for row in load_dataset("BeIR/scifact-qrels", split="test"):
        if row["score"] > 0:
            relevant[str(row["query-id"])].append(f"{row['corpus-id']}.txt")

    with open(out / "qrels_test.jsonl", "w", encoding="utf-8") as f:
        for qid, docs in relevant.items():
            f.write(json.dumps({"query": queries[qid], "relevant_doc_ids": docs}) + "\n")

    print(f"{len(corpus)} docs in {raw}, {len(relevant)} test queries in {out / 'qrels_test.jsonl'}")
