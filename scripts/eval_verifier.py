"""Benchmark a verifier on LLM-AggreFact (balanced accuracy per sub-dataset).

    python scripts/eval_verifier.py --verifier nli --limit 200
    python scripts/eval_verifier.py --verifier minicheck --limit 200

The dataset may require accepting its terms on Hugging Face and `huggingface-cli login`.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verirag import Settings  # noqa: E402
from verirag.verify import get_verifier  # noqa: E402


def balanced_accuracy(labels: list[int], preds: list[int]) -> float:
    tpr_tnr = []
    for cls in (0, 1):
        idx = [i for i, y in enumerate(labels) if y == cls]
        if idx:
            tpr_tnr.append(sum(preds[i] == cls for i in idx) / len(idx))
    return sum(tpr_tnr) / len(tpr_tnr)


if __name__ == "__main__":
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", choices=["nli", "minicheck"], default="nli")
    ap.add_argument("--limit", type=int, default=200, help="examples per sub-dataset")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="data/eval/verifier_results.json")
    args = ap.parse_args()

    settings = Settings()
    settings.verifier = args.verifier
    verifier = get_verifier(settings)

    ds = load_dataset("lytang/LLM-AggreFact", split="test")
    groups: dict[str, list] = defaultdict(list)
    for row in ds:
        if len(groups[row["dataset"]]) < args.limit:
            groups[row["dataset"]].append(row)

    results = {}
    for name, rows in sorted(groups.items()):
        probs = [verifier.score(r["doc"], [r["claim"]])[0] for r in rows]
        preds = [int(p >= args.threshold) for p in probs]
        results[name] = balanced_accuracy([r["label"] for r in rows], preds)
        print(f"{name:<20} BAcc={results[name]:.3f}  (n={len(rows)})")
    results["average"] = sum(results.values()) / len(results)
    print(f"{'average':<20} BAcc={results['average']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"verifier": verifier.name, **results}, indent=2))
