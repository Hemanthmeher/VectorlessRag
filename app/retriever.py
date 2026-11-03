"""Two-stage reasoning retrieval: pick documents, then pick tree nodes (structured JSON, exact ID match)."""
from . import config, llm, pdf_utils, store


def condense_question(question, history):
    if not history:
        return question
    turns = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history[-4:])
    try:
        out = llm.chat([
            {"role": "system", "content": "Rewrite the last question as a standalone question using the conversation. Return only the question."},
            {"role": "user", "content": f"{turns}\n\nlast question: {question}"}],
            config.FAST_MODEL, max_tokens=120).strip()
        return out or question
    except Exception:
        return question


def select_documents(question, docs, limit=3):
    if len(docs) <= 2:
        return docs
    listing = "\n".join(f"[{d['id']}] {d['filename']}: {d['summary'][:300]}" for d in docs)
    try:
        data = llm.chat_json([
            {"role": "system", "content": f'Pick up to {limit} documents most likely to answer the question. '
                                          'Return JSON: {"doc_ids": ["<id>", ...]}'},
            {"role": "user", "content": f"Question: {question}\n\nDocuments:\n{listing}"}],
            config.FAST_MODEL, max_tokens=200)
        ids = {str(i) for i in data.get("doc_ids", [])}
        picked = [d for d in docs if d["id"] in ids]
        return picked or docs[:limit]
    except Exception:
        return docs[:limit]


def _outline(label, tree, index, parent_of):
    lines = []

    def walk(nodes, depth, parent):
        for n in nodes:
            key = f"{label}:{n['id']}"
            index[key] = n
            parent_of[key] = parent
            lines.append(f"{'  ' * depth}[{key}] {n['title']} (pp.{n['start']}-{n['end']}): {n['summary'][:160]}")
            walk(n["children"], depth + 1, key)

    walk(tree, 0, None)
    return lines


def select_nodes(question, docs):
    """Returns (reasoning, [ {doc, node, text_range} ])."""
    index, parent_of, doc_of, blocks = {}, {}, {}, []
    for i, d in enumerate(docs, start=1):
        label = f"D{i}"
        tree = store.load_tree(d["id"])
        lines = _outline(label, tree, index, parent_of)
        for k in [k for k in index if k.startswith(label + ":")]:
            doc_of[k] = d
        blocks.append(f"## {label} = {d['filename']}\n{d['summary']}\n" + "\n".join(lines))
    outline = "\n\n".join(blocks)[:config.OUTLINE_CHAR_BUDGET * max(1, len(docs))]

    data = llm.chat_json([
        {"role": "system", "content":
            "You navigate a document outline like a table of contents. Choose the most specific sections "
            f"(max {config.MAX_NODES_SELECTED}) whose text would contain the answer. Prefer leaf sections over "
            'parents. Return JSON: {"reasoning": "<2-3 sentences>", "node_ids": ["D1:5", ...]}. '
            "Use only IDs that appear in the outline. Return an empty list if nothing is relevant."},
        {"role": "user", "content": f"Question: {question}\n\nOutline:\n{outline}"}],
        config.SMART_MODEL, max_tokens=900)

    chosen = []
    for raw in data.get("node_ids", []):
        key = str(raw).strip().upper()
        if key in index and key not in chosen:
            chosen.append(key)
    # drop a parent when one of its descendants is also chosen
    drop = set()
    for key in chosen:
        p = parent_of.get(key)
        while p:
            drop.add(p)
            p = parent_of.get(p)
    chosen = [k for k in chosen if k not in drop][:config.MAX_NODES_SELECTED]
    return str(data.get("reasoning", "")), [{"doc": doc_of[k], "node": index[k]} for k in chosen]


def build_context(selected):
    """Slice page text for the selected nodes within a character budget."""
    parts, used, cache = [], 0, {}
    for s in selected:
        remaining = config.CONTEXT_CHAR_BUDGET - used
        if remaining < 500:
            break
        d, n = s["doc"], s["node"]
        pages = cache.setdefault(d["id"], store.load_pages(d["id"]))
        text = pdf_utils.slice_pages(pages, n["start"], n["end"])[:min(config.MAX_NODE_CHARS, remaining)]
        parts.append(f"### SOURCE: {d['filename']} | {n['title']} | pages {n['start']}-{n['end']}\n{text}")
        used += len(text)
    return "\n\n".join(parts)
