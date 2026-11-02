"""Benchmark a verifier on LLM-AggreFact (balanced accuracy per sub-dataset).

    python scripts/eval_verifier.py --verifier nli --limit 200
    python scripts/eval_verifier.py --verifier minicheck --limit 200

The dataset may require accepting its terms on Hugging Face and `huggingface-cli login`.
"""

import argparse
import json
import sys
import time
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="skip sub-datasets already in --out")
    ap.add_argument("--out", default="data/eval/verifier_results.json")
    args = ap.parse_args()

    settings = Settings()
    settings.verifier = args.verifier
    verifier = get_verifier(settings)

    # The test split is ordered (e.g. the first 100 CNN rows are all positive), so
    # shuffle before taking the first --limit rows of each sub-dataset.
    ds = load_dataset("lytang/LLM-AggreFact", split="test").shuffle(seed=args.seed)
    groups: dict[str, list] = defaultdict(list)
    for row in ds:
        if len(groups[row["dataset"]]) < args.limit:
            groups[row["dataset"]].append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save(results: dict) -> None:
        out.write_text(json.dumps({"verifier": verifier.name, "limit": args.limit, "seed": args.seed, **results}, indent=2))

    # Runs take hours on CPU, so print progress and save after every sub-dataset.
    results = {}
    if args.resume and out.exists():
        prev = json.loads(out.read_text())
        if (prev.get("verifier"), prev.get("limit"), prev.get("seed")) == (verifier.name, args.limit, args.seed):
            results = {k: v for k, v in prev.items() if k in groups}
            print(f"resuming, {len(results)} sub-datasets already done", flush=True)
    for name, rows in sorted(groups.items()):
        if name in results:
            continue
        t = time.time()
        probs = []
        for i, r in enumerate(rows, 1):
            probs.append(verifier.score(r["doc"], [r["claim"]])[0])
            if i % 25 == 0:
                print(f"  {name} {i}/{len(rows)}  {(time.time() - t) / i:.1f}s/example", flush=True)
        preds = [int(p >= args.threshold) for p in probs]
        results[name] = balanced_accuracy([r["label"] for r in rows], preds)
        print(f"{name:<20} BAcc={results[name]:.3f}  (n={len(rows)})", flush=True)
        save(results)
    results["average"] = sum(results.values()) / len(results)
    print(f"{'average':<20} BAcc={results['average']:.3f}")
    save(results)
