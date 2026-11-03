"""PDF reading and structure detection. No LLM calls here."""
import re
from collections import Counter

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24.3
except ImportError:  # older versions
    import fitz


def extract_pages(path):
    with fitz.open(path) as doc:
        return [re.sub(r"[ \t]+\n", "\n", p.get_text("text")).strip() for p in doc]


def slice_pages(pages, start, end, markers=True):
    """Text of pages start..end (1-based, inclusive). Sliced from the PDF, never LLM-copied."""
    out = []
    for p in range(max(1, start), min(end, len(pages)) + 1):
        out.append(f"[p.{p}]\n{pages[p - 1]}" if markers else pages[p - 1])
    return "\n".join(out)


def toc_headings(path, n_pages):
    """Headings from the PDF's own bookmarks: [(level, title, page)]."""
    with fitz.open(path) as doc:
        toc = doc.get_toc()
    return [(min(l, 3), t.strip(), p) for l, t, p in toc if t.strip() and 1 <= p <= n_pages]


def font_headings(path, n_pages, max_headings=150):
    """Heuristic: lines set noticeably larger (or bold and short) than body text."""
    sizes, lines = Counter(), []
    with fitz.open(path) as doc:
        for pno, page in enumerate(doc, start=1):
            for b in page.get_text("dict")["blocks"]:
                for l in b.get("lines", []):
                    spans = [s for s in l["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    text = " ".join(s["text"].strip() for s in spans)
                    size = round(max(s["size"] for s in spans), 1)
                    bold = all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans)
                    sizes[size] += len(text)
                    lines.append((pno, text, size, bold))
    if not sizes:
        return []
    body = sizes.most_common(1)[0][0]
    freq = Counter(t for _, t, _, _ in lines)
    cands = []
    for pno, text, size, bold in lines:
        if not 3 <= len(text) <= 90 or re.fullmatch(r"[\d\W]+", text):
            continue
        if freq[text] > max(3, 0.3 * n_pages):          # running headers/footers
            continue
        big = size >= body * 1.15
        bold_label = bold and size >= body and len(text) <= 60 and not text.endswith((".", ","))
        if big or bold_label:
            cands.append((pno, text, size))
    top = sorted({s for _, _, s in cands}, reverse=True)[:3]
    level_of = {s: i + 1 for i, s in enumerate(top)}
    return [(level_of[s], t, p) for p, t, s in cands if s in level_of][:max_headings]


def window_headings(n_pages, w):
    return [(1, f"Pages {a}-{min(a + w - 1, n_pages)}", a) for a in range(1, n_pages + 1, w)]
