"""End-to-end: retrieve -> generate cited answer -> decompose into claims -> verify each claim."""

import time
from dataclasses import dataclass, field

from .claims import Claim, Sentence, llm_claims, sentence_claims, split_answer
from .config import Settings
from .llm import LLM, get_llm
from .prompts import ABSTAIN, ANSWER_SYSTEM, ANSWER_USER
from .retrieval import Hit, HybridRetriever
from .verify import MISCITED, SUPPORTED, Verifier, get_verifier, label_claim


@dataclass
class VerifiedClaim:
    text: str
    citations: list[int]
    sentence_idx: int
    label: str
    cited_score: float
    best_score: float
    best_source: int  # 1-based source number with the strongest support


@dataclass
class Result:
    query: str
    answer: str
    sentences: list[Sentence]
    sources: list[Hit]
    claims: list[VerifiedClaim]
    timings: dict[str, float] = field(default_factory=dict)
    abstained: bool = False  # model said the sources don't contain the answer

    @property
    def grounded_rate(self) -> float:
        """Share of claims supported by some retrieved source (hallucination = 1 - this)."""
        if not self.claims:
            return 0.0
        return sum(c.label in (SUPPORTED, MISCITED) for c in self.claims) / len(self.claims)

    @property
    def citation_accuracy(self) -> float:
        """Share of claims supported by the source they actually cite."""
        if not self.claims:
            return 0.0
        return sum(c.label == SUPPORTED for c in self.claims) / len(self.claims)


def format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] (from {h.chunk.source}) {h.chunk.text}" for i, h in enumerate(hits, 1))


class VeriRAG:
    def __init__(
        self,
        settings: Settings | None = None,
        retriever: HybridRetriever | None = None,
        llm: LLM | None = None,
        verifier: Verifier | None = None,
    ):
        self.s = settings or Settings()
        self.retriever = retriever or HybridRetriever(self.s)
        self.llm = llm or get_llm(self.s)
        self.verifier = verifier or get_verifier(self.s)

    def verify(self, claims: list[Claim], sources: list[Hit]) -> list[VerifiedClaim]:
        if not claims or not sources:
            return []
        texts = [c.text for c in claims]
        # scores[j][i] = support of claim i by source j
        scores = [self.verifier.score(h.chunk.text, texts) for h in sources]

        out = []
        for i, c in enumerate(claims):
            per_source = [scores[j][i] for j in range(len(sources))]
            best_j = max(range(len(sources)), key=lambda j: per_source[j])
            cited = max((per_source[n - 1] for n in c.citations), default=0.0)
            out.append(
                VerifiedClaim(
                    text=c.text,
                    citations=c.citations,
                    sentence_idx=c.sentence_idx,
                    label=label_claim(
                        cited, per_source[best_j],
                        self.s.supported_threshold, self.s.partial_threshold,
                    ),
                    cited_score=cited,
                    best_score=per_source[best_j],
                    best_source=best_j + 1,
                )
            )
        return out

    def answer(self, query: str, decompose: str | None = None) -> Result:
        decompose = decompose or self.s.decompose
        t: dict[str, float] = {}

        t0 = time.perf_counter()
        sources = self.retriever.search(query)
        t["retrieval"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        answer = self.llm.chat([
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": ANSWER_USER.format(
                sources=format_sources(sources), question=query)},
        ])
        t["generation"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        sentences = split_answer(answer, n_sources=len(sources))
        # An abstention is the correct behaviour, not a claim to fact-check.
        abstain_key = ABSTAIN.lower().rstrip(".")
        keep = [i for i, s in enumerate(sentences) if abstain_key not in s.text.lower()]
        to_check = [sentences[i] for i in keep]
        abstained = bool(sentences) and not keep
        if decompose == "llm":
            claims = llm_claims(to_check, self.llm)
        else:
            claims = sentence_claims(to_check)
        # claims index into to_check; map back to positions in `sentences` for rendering
        for c in claims:
            c.sentence_idx = keep[c.sentence_idx]
        t["decomposition"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        verified = self.verify(claims, sources)
        t["verification"] = time.perf_counter() - t0

        return Result(query, answer, sentences, sources, verified, t, abstained)
