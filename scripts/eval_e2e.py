"""End-to-end run over a question set: how often does the LLM hallucinate or miscite?

Input: JSONL with one {"query": "..."} per line.

    python scripts/eval_e2e.py data/eval/questions_sample.jsonl --decompose llm
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verirag import Settings, VeriRAG  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("--decompose", choices=["llm", "sentence"], default=None)
    ap.add_argument("--out", default="data/eval/e2e_results.jsonl")
    ap.add_argument("--resume", action="store_true", help="keep answers already in --out, run the rest")
    args = ap.parse_args()

    pipe = VeriRAG(Settings())
    rows = [json.loads(line) for line in open(args.questions, encoding="utf-8") if line.strip()]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if args.resume and out.exists():
        records = [json.loads(line) for line in open(out, encoding="utf-8") if line.strip()]
        print(f"resuming, {len(records)} questions already answered", flush=True)
    done = {r["query"] for r in records}

    # Append one line per question so a crash (e.g. a network drop) loses nothing.
    with open(out, "a" if args.resume else "w", encoding="utf-8") as f:
        for row in rows:
            if row["query"] in done:
                continue
            res = pipe.answer(row["query"], decompose=args.decompose)
            rec = {
                "query": res.query,
                "answer": res.answer,
                "claims": [c.__dict__ for c in res.claims],
                "timings": res.timings,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            records.append(rec)
            print(f"{res.grounded_rate:>5.0%}  {row['query']}", flush=True)

    labels: Counter = Counter()
    latency, grounded, cit_acc = [], [], []
    for rec in records:
        claim_labels = [c["label"] for c in rec["claims"]]
        labels.update(claim_labels)
        latency.append(sum(rec["timings"].values()))
        if claim_labels:
            grounded.append(sum(lab in ("supported", "miscited") for lab in claim_labels) / len(claim_labels))
            cit_acc.append(claim_labels.count("supported") / len(claim_labels))

    total = sum(labels.values()) or 1
    print(f"\nllm={pipe.llm.name}  verifier={pipe.verifier.name}  questions={len(rows)}  claims={total}")
    for label in ("supported", "miscited", "partial", "unsupported"):
        print(f"  {label:<12} {labels[label] / total:.1%}")
    print(f"  mean grounded rate     {sum(grounded) / max(len(grounded), 1):.1%}")
    print(f"  mean citation accuracy {sum(cit_acc) / max(len(cit_acc), 1):.1%}")
    print(f"  mean latency           {sum(latency) / len(latency):.1f}s")
