ABSTAIN = "I don't know based on the provided sources."

ANSWER_SYSTEM = f"""You answer questions using ONLY the numbered sources provided.
Rules:
- After every sentence, cite the source(s) that support it, like [1] or [2][3].
- Do not use outside knowledge. If the sources do not contain the answer, reply exactly:
  "{ABSTAIN}"
- Be concise: at most 5 sentences."""

ANSWER_USER = """Sources:
{sources}

Question: {question}"""

DECOMPOSE = """Break each numbered sentence into atomic, self-contained factual claims.
- Each claim states exactly one fact.
- Replace pronouns with the entities they refer to, so each claim is understandable alone.
- Do not add, infer, or drop information.
- Skip sentences with no factual content.

Return ONLY a JSON array, no prose:
[{{"s": <sentence number>, "claim": "<claim>"}}]

Sentences:
{sentences}"""
