"""Claim verifiers. Each returns P(claim is supported by premise) in [0, 1]."""

import warnings
from typing import Protocol

from .chunking import split_sentences
from .config import Settings

SUPPORTED, PARTIAL, MISCITED, UNSUPPORTED = "supported", "partial", "miscited", "unsupported"


class Verifier(Protocol):
    name: str

    def score(self, premise: str, claims: list[str]) -> list[float]: ...


class NLIVerifier:
    """DeBERTa NLI baseline. NLI models are trained on short premises, so we slide a
    window of sentences over the chunk and take the max entailment probability."""

    def __init__(self, model_name: str, window: int = 3, batch_size: int = 16):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.entail_idx = next(
            int(i) for i, label in self.model.config.id2label.items()
            if label.lower().startswith("entail")
        )
        self.window, self.batch_size = window, batch_size
        self.name = f"nli/{model_name.split('/')[-1]}"

    def _windows(self, premise: str) -> list[str]:
        sents = split_sentences(premise)
        if len(sents) <= self.window:
            return [premise]
        return [" ".join(sents[i : i + self.window]) for i in range(len(sents) - self.window + 1)]

    def score(self, premise: str, claims: list[str]) -> list[float]:
        import torch

        windows = self._windows(premise)
        pairs = [(w, c) for c in claims for w in windows]
        probs: list[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                enc = self.tok(
                    [p for p, _ in batch], [h for _, h in batch],
                    truncation=True, max_length=512, padding=True, return_tensors="pt",
                )
                logits = self.model(**enc).logits
                probs.extend(logits.softmax(-1)[:, self.entail_idx].tolist())
        n = len(windows)
        return [max(probs[i * n : (i + 1) * n]) for i in range(len(claims))]


class MiniCheckVerifier:
    """MiniCheck (Tang et al., 2024): a small model trained specifically to check
    LLM output against grounding documents. Handles long documents internally."""

    def __init__(self, model_name: str, cache_dir: str):
        from minicheck.minicheck import MiniCheck

        self.scorer = MiniCheck(model_name=model_name, cache_dir=cache_dir)
        self.name = f"minicheck/{model_name}"

    def score(self, premise: str, claims: list[str]) -> list[float]:
        _, probs, _, _ = self.scorer.score(docs=[premise] * len(claims), claims=claims)
        return [float(p) for p in probs]


def get_verifier(settings: Settings) -> Verifier:
    if settings.verifier == "minicheck":
        try:
            return MiniCheckVerifier(
                settings.minicheck_model, str(settings.data_dir / "models" / "minicheck")
            )
        except (ImportError, ValueError, OSError) as e:
            warnings.warn(f"MiniCheck unavailable ({e}); falling back to the NLI verifier")
    return NLIVerifier(settings.nli_model)


def label_claim(cited_score: float, best_score: float, supported_t: float, partial_t: float) -> str:
    """cited_score: best support among the sources the claim cites (0 if it cites none).
    best_score: best support among all retrieved sources."""
    if cited_score >= supported_t:
        return SUPPORTED
    if best_score >= supported_t:
        return MISCITED  # true per the context, but cited the wrong source (or none)
    if max(cited_score, best_score) >= partial_t:
        return PARTIAL
    return UNSUPPORTED
