"""Turn a cited answer into claims: parse citations, then split by sentence or by LLM."""

import json
import re
from dataclasses import dataclass, field

from .chunking import split_sentences
from .prompts import DECOMPOSE

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_CITE_RUN = r"(?:\[\d+(?:\s*,\s*\d+)*\]\s*)+"


@dataclass
class Sentence:
    text: str
    citations: list[int] = field(default_factory=list)


@dataclass
class Claim:
    text: str
    citations: list[int]
    sentence_idx: int


def parse_citations(sentence: str) -> tuple[str, list[int]]:
    cites: list[int] = []
    for m in _CITE.finditer(sentence):
        cites.extend(int(x) for x in m.group(1).split(","))
    clean = _CITE.sub("", sentence)
    clean = re.sub(r"\s+([.,!?;:])", r"\1", clean)
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    return clean, sorted(set(cites))


def split_answer(answer: str, n_sources: int | None = None) -> list[Sentence]:
    # Move citations written after the period ("... 1991. [1]") in front of it,
    # so they stay attached to the sentence they support.
    answer = re.sub(
        rf"([.!?])\s*({_CITE_RUN})",
        lambda m: f" {m.group(2).strip()}{m.group(1)} ",
        answer,
    )
    sentences = []
    for raw in split_sentences(answer):
        text, cites = parse_citations(raw)
        if n_sources is not None:
            cites = [c for c in cites if 1 <= c <= n_sources]
        if text:
            sentences.append(Sentence(text, cites))
    return sentences


def sentence_claims(sentences: list[Sentence]) -> list[Claim]:
    return [Claim(s.text, s.citations, i) for i, s in enumerate(sentences)]


def _parse_json_array(text: str) -> list:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("no JSON array in output")
    return json.loads(text[start : end + 1])


def llm_claims(sentences: list[Sentence], llm) -> list[Claim]:
    """Atomic claims via the LLM; falls back to one claim per sentence if output is unusable."""
    if not sentences:
        return []
    numbered = "\n".join(f"{i + 1}. {s.text}" for i, s in enumerate(sentences))
    try:
        raw = llm.chat([{"role": "user", "content": DECOMPOSE.format(sentences=numbered)}])
        claims = []
        for item in _parse_json_array(raw):
            idx = int(item["s"]) - 1
            text = str(item["claim"]).strip()
            if 0 <= idx < len(sentences) and text:
                claims.append(Claim(text, sentences[idx].citations, idx))
        if claims:
            return claims
    except (ValueError, KeyError, TypeError):
        pass
    return sentence_claims(sentences)
