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
    args = ap.parse_args()

    pipe = VeriRAG(Settings())
    rows = [json.loads(line) for line in open(args.questions, encoding="utf-8") if line.strip()]

    labels: Counter = Counter()
    latency, grounded, cit_acc = [], [], []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            res = pipe.answer(row["query"], decompose=args.decompose)
            labels.update(c.label for c in res.claims)
            latency.append(sum(res.timings.values()))
            if res.claims:
                grounded.append(res.grounded_rate)
                cit_acc.append(res.citation_accuracy)
            f.write(json.dumps({
                "query": res.query,
                "answer": res.answer,
                "claims": [c.__dict__ for c in res.claims],
                "timings": res.timings,
            }, ensure_ascii=False) + "\n")
            print(f"{res.grounded_rate:>5.0%}  {row['query']}")

    total = sum(labels.values()) or 1
    print(f"\nllm={pipe.llm.name}  verifier={pipe.verifier.name}  questions={len(rows)}  claims={total}")
    for label in ("supported", "miscited", "partial", "unsupported"):
        print(f"  {label:<12} {labels[label] / total:.1%}")
    print(f"  mean grounded rate     {sum(grounded) / max(len(grounded), 1):.1%}")
    print(f"  mean citation accuracy {sum(cit_acc) / max(len(cit_acc), 1):.1%}")
    print(f"  mean latency           {sum(latency) / len(latency):.1f}s")
