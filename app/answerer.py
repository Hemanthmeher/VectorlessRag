from . import config, llm

SYSTEM = (
    "You answer questions strictly from the provided sources. Rules:\n"
    "- Use only the sources. If they do not contain the answer, say the documents do not cover it.\n"
    "- Be concise and structured; use short bullets for lists.\n"
    "- Cite every claim as [filename, p.N] using the [p.N] markers in the sources. "
    "For several pages write [filename, p.3, p.5]. Never invent page numbers."
)


def stream_answer(question, context, history):
    msgs = [{"role": "system", "content": SYSTEM}]
    for m in history[-4:]:
        msgs.append({"role": m["role"], "content": m["content"][:1200]})
    msgs.append({"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"})
    yield from llm.stream(msgs, config.SMART_MODEL)
