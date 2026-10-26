"""Chunk everything in data/raw and build the BM25 + FAISS indexes in data/index.

Run on Colab/Kaggle for big corpora, then copy data/index into the repo or Space.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verirag import Settings  # noqa: E402
from verirag.ingest import build_index  # noqa: E402

if __name__ == "__main__":
    settings = Settings()
    n = build_index(settings)
    print(f"Indexed {n} chunks into {settings.index_dir}")
