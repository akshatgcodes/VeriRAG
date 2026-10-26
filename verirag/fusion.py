"""Reciprocal Rank Fusion for merging ranked lists from different retrievers."""


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Each ranking is a list of doc indices, best first. Returns (index, score), best first."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
