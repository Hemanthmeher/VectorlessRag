"""PDF reading and structure detection. No LLM calls here."""
import logging
import os
import re
import threading
from collections import Counter

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24.3
except ImportError:  # older versions
    import fitz


log = logging.getLogger(__name__)

# PARSER=docling uses IBM Docling (layout model, OCR, tables); anything else uses PyMuPDF only.
PARSER = os.getenv("PARSER", "pymupdf").lower()

_converter = None
_docling_lock = threading.Lock()
_docling_cache = {}          # holds the most recent converted document, so pages + headings convert once


def _docling_doc(path):
    """Convert a PDF with Docling (slow the first time; cached for the next call on the same path)."""
    global _converter
    key = str(path)
    with _docling_lock:
        if key not in _docling_cache:
            if _converter is None:
                from docling.document_converter import DocumentConverter   # heavy import, load lazily
                _converter = DocumentConverter()
            _docling_cache.clear()
            _docling_cache[key] = _converter.convert(key).document
        return _docling_cache[key]


def _docling_pages(path):
    doc = _docling_doc(path)
    n = max(doc.pages) if doc.pages else 0
    pages = []
    for i in range(1, n + 1):
        md = doc.export_to_markdown(page_no=i).replace("<!-- image -->", "").strip()
        pages.append(md)
    return pages


def _infer_levels(heads):
    """PDF heading levels from Docling are often all 1. If so, derive depth from numbering (1. / 1.2 / 1.2.3)."""
    if len({l for l, _, _ in heads}) > 1:
        return heads
    depths = []
    for _, t, _ in heads:
        m = re.match(r"^(\d+(?:\.\d+)*)[.)]?\s", t)
        depths.append(min(m.group(1).count(".") + 1, 3) if m else None)
    if sum(d is not None for d in depths) < 0.6 * len(heads):
        return heads                                   # not a numbered document
    out, last = [], 1
    for (l, t, p), d in zip(heads, depths):
        last = d if d is not None else last
        out.append((last, t, p))
    return out


def _docling_headings(path, n_pages):
    from docling_core.types.doc import SectionHeaderItem
    heads = []
    for item, _ in _docling_doc(path).iterate_items():
        if isinstance(item, SectionHeaderItem) and item.text.strip() and item.prov:
            page = item.prov[0].page_no
            if 1 <= page <= n_pages:
                heads.append((min(max(item.level, 1), 3), item.text.strip()[:120], page))
    return _infer_levels(heads)


def extract_pages(path):
    if PARSER == "docling":
        try:
            pages = _docling_pages(path)
            if pages:
                return pages
        except Exception as e:                          # not installed, model download failed, etc.
            log.warning("Docling failed (%s); falling back to PyMuPDF", e)
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
    """Layout-based headings: Docling when enabled, otherwise (or if it finds <3) a font-size heuristic."""
    if PARSER == "docling":
        try:
            heads = _docling_headings(path, n_pages)
            if len(heads) >= 3:
                return heads[:max_headings]
        except Exception as e:
            log.warning("Docling heading detection failed (%s); using font heuristic", e)
    return _font_headings_pymupdf(path, n_pages, max_headings)


def _font_headings_pymupdf(path, n_pages, max_headings=150):
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
