"""Sentence splitting and sentence-aware chunking."""

import re
from dataclasses import dataclass

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class Chunk:
    id: int
    doc_id: str
    source: str
    text: str


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_BOUNDARY.split(text) if s.strip()]


def chunk_text(text: str, max_words: int = 200, overlap_words: int = 40) -> list[str]:
    """Pack whole sentences into chunks of ~max_words, carrying ~overlap_words forward."""
    chunks, current, count = [], [], 0
    for sent in split_sentences(text):
        n = len(sent.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            # keep trailing sentences as overlap
            carry, carried = [], 0
            for s in reversed(current):
                carried += len(s.split())
                if carried > overlap_words:
                    break
                carry.insert(0, s)
            current, count = carry, sum(len(s.split()) for s in carry)
        current.append(sent)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks
