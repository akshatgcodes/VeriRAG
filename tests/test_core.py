from verirag.chunking import chunk_text, split_sentences
from verirag.claims import llm_claims, parse_citations, split_answer
from verirag.fusion import rrf
from verirag.verify import MISCITED, PARTIAL, SUPPORTED, UNSUPPORTED, label_claim


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    assert fused[0][0] == 1  # high in both lists
    assert {i for i, _ in fused} == {1, 2, 3, 4}


def test_split_sentences():
    assert split_sentences("Hello world. It is 1991! Is it?") == ["Hello world.", "It is 1991!", "Is it?"]


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(50))
    chunks = chunk_text(text, max_words=25, overlap_words=10)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 30 for c in chunks)
    assert chunks[0].split(".")[-2] in chunks[1]  # last sentence carried over


def test_parse_citations():
    assert parse_citations("Python was released in 1991 [1][3].") == ("Python was released in 1991.", [1, 3])
    assert parse_citations("Both [2, 4] agree.") == ("Both agree.", [2, 4])


def test_split_answer_moves_trailing_citations_and_drops_bad_ones():
    sents = split_answer("Python came out in 1991. [1] It was named after Monty Python [2][9].", n_sources=3)
    assert [s.text for s in sents] == ["Python came out in 1991.", "It was named after Monty Python."]
    assert [s.citations for s in sents] == [[1], [2]]


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages, temperature=0.0):
        return self.reply


def test_llm_claims_inherit_citations_and_fall_back():
    sents = split_answer("Guido created Python in 1991 [1].")
    claims = llm_claims(sents, FakeLLM('[{"s": 1, "claim": "Guido created Python."},'
                                       ' {"s": 1, "claim": "Python was created in 1991."}]'))
    assert [c.citations for c in claims] == [[1], [1]]
    assert len(llm_claims(sents, FakeLLM("sorry, no JSON"))) == 1


class FakeRetriever:
    def search(self, query):
        from verirag.chunking import Chunk
        from verirag.retrieval import Hit

        return [Hit(Chunk(0, "a.md", "a.md", "Python was released in 1991."), 1.0)]


class FakeVerifier:
    name = "fake"

    def score(self, premise, claims):
        return [0.9 if "1991" in c else 0.1 for c in claims]


def test_pipeline_abstention_and_labels():
    from verirag.config import Settings
    from verirag.pipeline import VeriRAG

    pipe = VeriRAG(Settings(), FakeRetriever(), FakeLLM("I don't know based on the provided sources."),
                   FakeVerifier())
    res = pipe.answer("population of Paris?", decompose="sentence")
    assert res.abstained and res.claims == []

    pipe.llm = FakeLLM("Python was released in 1991 [1]. It was written in Rust [1].")
    res = pipe.answer("When?", decompose="sentence")
    assert [c.label for c in res.claims] == [SUPPORTED, UNSUPPORTED]
    assert res.grounded_rate == 0.5 and not res.abstained


def test_label_claim():
    assert label_claim(0.9, 0.9, 0.7, 0.4) == SUPPORTED
    assert label_claim(0.1, 0.8, 0.7, 0.4) == MISCITED
    assert label_claim(0.5, 0.5, 0.7, 0.4) == PARTIAL
    assert label_claim(0.1, 0.2, 0.7, 0.4) == UNSUPPORTED
