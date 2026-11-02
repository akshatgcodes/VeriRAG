from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from verirag.chunking import Chunk  # noqa: E402
from verirag.retrieval import Hit  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


class FakeRetriever:
    def search(self, query):
        return [Hit(Chunk(0, "python.md", "python.md", "Python 3.0 was released in December 2008."), 1.0, 0.9)]


class FakeLLM:
    name = "fake"

    def chat(self, messages, temperature=0.0):
        return "Python 3.0 was released in December 2008 [1]. It was written in Rust [1]."


class FakeVerifier:
    name = "fake"

    def score(self, premise, claims):
        return [0.95 if "2008" in c else 0.05 for c in claims]


def test_app_renders_verified_answer(monkeypatch):
    import verirag.pipeline as pipeline

    original = pipeline.VeriRAG.__init__

    def fake_init(self, settings=None, **_):
        original(self, settings, retriever=FakeRetriever(), llm=FakeLLM(), verifier=FakeVerifier())

    monkeypatch.setattr(pipeline.VeriRAG, "__init__", fake_init)

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    at.text_input[0].input("When was Python 3.0 released?").run()
    assert not at.exception and not at.error
    assert [m.value for m in at.metric][:2] == ["50%", "50%"]
    assert any("#dc2626" in md.value for md in at.markdown)  # hallucination underlined red


def test_example_picker_and_clear_button(monkeypatch):
    import verirag.pipeline as pipeline

    original = pipeline.VeriRAG.__init__

    def fake_init(self, settings=None, **_):
        original(self, settings, retriever=FakeRetriever(), llm=FakeLLM(), verifier=FakeVerifier())

    monkeypatch.setattr(pipeline.VeriRAG, "__init__", fake_init)

    at = AppTest.from_file(APP, default_timeout=60).run()
    example = at.selectbox(key="example").options[0]
    at.selectbox(key="example").select(example).run()
    assert not at.exception
    assert at.text_input(key="query").value == example
    assert at.selectbox(key="example").value is None  # reset so it can be picked again
    assert at.metric  # the picked question was answered

    at.button[0].click().run()
    assert at.text_input(key="query").value == ""
    assert not at.metric
