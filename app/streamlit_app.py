import html
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verirag import Settings  # noqa: E402
from verirag.ingest import build_index, index_exists  # noqa: E402

COLORS = {
    "supported": ("#16a34a", "Supported by the cited source"),
    "miscited": ("#2563eb", "True per the sources, but the citation is wrong or missing"),
    "partial": ("#d97706", "Only partially supported"),
    "unsupported": ("#dc2626", "Not supported by any retrieved source: likely hallucination"),
}
SEVERITY = ["supported", "miscited", "partial", "unsupported"]

st.set_page_config(page_title="VeriRAG", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading models (first run downloads them)...")
def load_pipeline():
    from verirag import VeriRAG

    settings = Settings()
    if not index_exists(settings):
        build_index(settings)
    return VeriRAG(settings)


def render_answer(result) -> str:
    """Color each sentence by the worst label among its claims."""
    worst: dict[int, str] = {}
    for c in result.claims:
        prev = worst.get(c.sentence_idx, "supported")
        worst[c.sentence_idx] = max(prev, c.label, key=SEVERITY.index)
    parts = []
    for i, s in enumerate(result.sentences):
        label = worst.get(i)
        cites = "".join(f"<sup>[{n}]</sup>" for n in s.citations)
        text = html.escape(s.text) + cites
        if label:
            color, tip = COLORS[label]
            text = (f'<span title="{tip}" style="border-bottom:3px solid {color};'
                    f'padding-bottom:1px">{text}</span>')
        parts.append(text)
    return '<p style="font-size:1.1rem;line-height:1.9">' + " ".join(parts) + "</p>"


st.title("🔎 VeriRAG")
st.caption("Hallucination-aware RAG: every claim in the answer is checked against its cited source.")

with st.sidebar:
    st.header("Settings")
    decompose = st.radio("Claim splitting", ["llm", "sentence"],
                         format_func={"llm": "Atomic claims (LLM)", "sentence": "Per sentence"}.get)
    st.markdown("**Legend**")
    for label, (color, tip) in COLORS.items():
        st.markdown(f'<span style="color:{color}">■</span> **{label}**: {tip}',
                    unsafe_allow_html=True)

query = st.text_input("Ask a question about the indexed documents",
                      placeholder="When was Python 3.0 released?")

if query:
    pipe = load_pipeline()
    with st.spinner("Retrieving, answering, verifying..."):
        result = pipe.answer(query, decompose=decompose)

    st.markdown(render_answer(result), unsafe_allow_html=True)
    if result.abstained:
        st.info("The model declined to answer because the sources don't cover this question.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Grounded claims", f"{result.grounded_rate:.0%}")
    c2.metric("Citation accuracy", f"{result.citation_accuracy:.0%}")
    c3.metric("Total time", f"{sum(result.timings.values()):.1f}s")

    st.subheader("Claims")
    st.dataframe(
        [{
            "claim": c.text,
            "label": c.label,
            "cites": ", ".join(map(str, c.citations)) or "none",
            "cited support": round(c.cited_score, 2),
            "best source": c.best_source,
            "best support": round(c.best_score, 2),
        } for c in result.claims],
        width="stretch", hide_index=True,
    )

    st.subheader("Sources")
    for i, h in enumerate(result.sources, 1):
        with st.expander(f"[{i}] {h.chunk.source}"):
            st.write(h.chunk.text)

    with st.expander("Timings and model info"):
        st.json({**{k: round(v, 2) for k, v in result.timings.items()},
                 "llm": pipe.llm.name, "verifier": pipe.verifier.name})
