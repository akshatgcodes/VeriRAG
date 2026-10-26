"""Ask a question from the command line: python scripts/ask.py "When was Python 3 released?" """

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verirag import Settings, VeriRAG  # noqa: E402

MARK = {"supported": "OK ", "miscited": "CIT", "partial": "~  ", "unsupported": "XX "}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--decompose", choices=["llm", "sentence"], default=None)
    args = ap.parse_args()

    result = VeriRAG(Settings()).answer(args.question, decompose=args.decompose)

    print("\nANSWER\n" + result.answer + "\n\nCLAIMS")
    for c in result.claims:
        print(f"  [{MARK[c.label]}] {c.text}  (cites {c.citations or '-'}, "
              f"cited={c.cited_score:.2f}, best=[{c.best_source}] {c.best_score:.2f})")
    print(f"\ngrounded={result.grounded_rate:.0%}  citation_acc={result.citation_accuracy:.0%}")
    print("timings: " + ", ".join(f"{k}={v:.1f}s" for k, v in result.timings.items()))
