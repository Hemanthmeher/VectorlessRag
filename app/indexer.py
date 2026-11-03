"""Build a hierarchical page-range tree for a PDF. Node text is sliced from pages, not LLM-copied."""
import re

from . import config, llm, pdf_utils, store


def _progress(doc_id, stage, pct):
    store.update_doc(doc_id, stage=stage, progress=pct)


def build_tree(headings, n_pages):
    """headings: [(level, title, page)] -> nested nodes with start/end page ranges."""
    heads = sorted(headings, key=lambda h: h[2])          # stable: keeps reading order within a page
    nodes, counter = [], [0]

    def new(level, title, start, end):
        counter[0] += 1
        return {"id": counter[0], "title": title, "level": level,
                "start": start, "end": end, "summary": "", "children": []}

    if heads and heads[0][2] > 1:
        nodes.append(new(1, "Front matter", 1, heads[0][2] - 1))
    for i, (lvl, title, pg) in enumerate(heads):
        end = n_pages
        for l2, _, p2 in heads[i + 1:]:
            if l2 <= lvl:
                end = pg if p2 <= pg else p2 - 1
                break
        nodes.append(new(lvl, title[:120], pg, max(pg, end)))

    roots, stack = [], []
    for n in nodes:
        while stack and stack[-1]["level"] >= n["level"]:
            stack.pop()
        (stack[-1]["children"] if stack else roots).append(n)
        stack.append(n)

    def split_long(n):
        if n["children"]:
            for c in n["children"]:
                split_long(c)
        elif n["end"] - n["start"] + 1 > config.MAX_LEAF_PAGES:
            for a in range(n["start"], n["end"] + 1, config.WINDOW_PAGES):
                b = min(a + config.WINDOW_PAGES - 1, n["end"])
                n["children"].append(new(n["level"] + 1, f"{n['title']} (pp. {a}-{b})", a, b))

    for r in roots:
        split_long(r)
    return roots


def _leaves(nodes):
    for n in nodes:
        if n["children"]:
            yield from _leaves(n["children"])
        else:
            yield n


def _fill_parent_summaries(nodes):
    for n in nodes:
        if n["children"]:
            _fill_parent_summaries(n["children"])
            n["summary"] = "Covers: " + "; ".join(c["title"] for c in n["children"][:8])


def _summarize_leaves(doc_id, leaves, pages):
    batches = [leaves[i:i + config.SUMMARY_BATCH] for i in range(0, len(leaves), config.SUMMARY_BATCH)]
    for bi, batch in enumerate(batches):
        payload = "\n\n".join(
            f"### ID {n['id']} | {n['title']}\n{pdf_utils.slice_pages(pages, n['start'], n['end'], False)[:1200]}"
            for n in batch)
        try:
            data = llm.chat_json([
                {"role": "system", "content":
                    'Summarise each section in one factual sentence (max 25 words) that says what '
                    'questions it can answer. Return JSON: {"summaries": {"<ID>": "<sentence>"}}'},
                {"role": "user", "content": payload}], config.FAST_MODEL, max_tokens=600)
            sm = data.get("summaries", {})
            for n in batch:
                n["summary"] = str(sm.get(str(n["id"]), "")).strip()
        except Exception:
            pass
        for n in batch:   # fallback so retrieval always has something to read
            if not n["summary"]:
                n["summary"] = re.sub(r"\s+", " ", pdf_utils.slice_pages(pages, n["start"], n["end"], False))[:180]
        _progress(doc_id, f"Summarising sections ({bi + 1}/{len(batches)})", 25 + int(65 * (bi + 1) / len(batches)))


def _doc_summary(roots, pages):
    outline = "\n".join(f"- {r['title']}: {r['summary']}" for r in roots)[:3000]
    try:
        return llm.chat([
            {"role": "system", "content": "Describe this document in two sentences: what it is and what topics it covers."},
            {"role": "user", "content": outline}], config.FAST_MODEL, max_tokens=150).strip()
    except Exception:
        return re.sub(r"\s+", " ", pages[0])[:300]


def index_document(doc_id):
    try:
        path = config.PDF_DIR / f"{doc_id}.pdf"
        _progress(doc_id, "Reading pages", 5)
        pages = pdf_utils.extract_pages(path)
        if len(pages) > config.MAX_PAGES:
            raise ValueError(f"Document has {len(pages)} pages; the limit is {config.MAX_PAGES}.")
        if sum(len(p) for p in pages) < 200:
            raise ValueError("No extractable text found. Scanned PDFs need OCR, which is not supported yet.")
        store.save_json(config.PAGES_DIR / f"{doc_id}.json", pages)
        store.update_doc(doc_id, num_pages=len(pages))

        _progress(doc_id, "Detecting structure", 15)
        heads = pdf_utils.toc_headings(path, len(pages))
        if len(heads) < 3:
            heads = pdf_utils.font_headings(path, len(pages))
        if len(heads) < 3:
            heads = pdf_utils.window_headings(len(pages), config.WINDOW_PAGES)
        roots = build_tree(heads, len(pages))

        _summarize_leaves(doc_id, list(_leaves(roots)), pages)
        _fill_parent_summaries(roots)
        _progress(doc_id, "Writing document summary", 94)
        summary = _doc_summary(roots, pages)

        store.save_json(config.TREES_DIR / f"{doc_id}.json", roots)
        store.update_doc(doc_id, status="ready", stage="Ready", progress=100, summary=summary)
    except Exception as e:
        store.update_doc(doc_id, status="failed", stage="Failed", error=str(e)[:300])
