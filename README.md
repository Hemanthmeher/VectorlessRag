# Vectorless RAG

Document Q&A without embeddings or a vector database. Each PDF becomes a hierarchical outline
(chapter, section, subsection, with page ranges and summaries). An LLM reads the outline like a
table of contents, picks the sections that hold the answer, and the answer is written from the
exact text of those pages, with page citations.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY (free at console.groq.com)
uvicorn app.main:app --reload
```
Open http://localhost:8000. Docker: `docker build -t vectorless-rag . && docker run -p 8000:8000 --env-file .env -v rag-data:/srv/data vectorless-rag`

## How it works

1. **Indexing** (background job): PyMuPDF extracts text per page. Structure comes from PDF bookmarks,
   else font-size heading detection, else fixed page windows. Long leaves are split into 3-page windows.
   A small model writes a one-line summary per leaf (batched, 4 per call). Node text is sliced
   from the stored pages, never copied by the LLM, so it cannot be truncated or hallucinated.
2. **Cache**: indexes are keyed by the file's SHA-256; re-uploading the same PDF is instant.
3. **Retrieval**: (a) with 3+ documents, pick relevant documents from their summaries; (b) show the model
   the outline of those documents and get back JSON node IDs. IDs are matched exactly and validated,
   parents are dropped when a child is chosen, and text is capped by a character budget.
4. **Answer**: streamed over SSE, grounded in the selected pages, cited as `[file.pdf, p.12]`.
   Follow-up questions are rewritten into standalone questions before retrieval.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | /api/documents | upload PDF (returns immediately, indexes in background) |
| GET | /api/documents | list with status and progress |
| GET | /api/documents/{id}/tree | outline tree |
| DELETE | /api/documents/{id} | remove document and index |
| POST | /api/chat | SSE stream: status, retrieval, token, done, error |

## Limits and next steps

- Scanned PDFs (no text layer) are rejected; add OCR (e.g. Tesseract) to support them.
- Groq free-tier rate limits vary by model and change over time. Calls retry with backoff; if you hit
  429s often, lower `CONTEXT_CHAR_BUDGET` or switch `SMART_MODEL` in `.env`.
- Auth, per-user document isolation, and a task queue (Celery/RQ) are the next steps for multi-user deployment.

## Resume bullets

- Built a vectorless, reasoning-based RAG system (FastAPI, Groq LLMs, PyMuPDF) that indexes PDFs into hierarchical page-range trees and retrieves via LLM tree navigation instead of embeddings.
- Designed a multi-document, two-stage retrieval pipeline (document selection, then section selection) with structured JSON output and validated node IDs, with page-level citations in answers.
- Implemented background indexing with progress polling, SHA-256 index caching, rate-limit-aware retries with exponential backoff, and SSE token streaming to a single-page frontend.
