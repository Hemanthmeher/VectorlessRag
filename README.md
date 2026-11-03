# Vectorless RAG Engine
*Document Q&A without embeddings or a vector database.*

## Overview
Traditional RAG relies on vector databases and chunk embeddings, which often break document context and introduce retrieval noise. **Vectorless RAG** takes a structural approach:
1. **Hierarchical Indexing:** Each uploaded PDF is parsed into a logical tree outline (Chapters, Sections, Subsections) with exact page ranges.
2. **Reasoning Retrieval:** Instead of vector similarity search, an LLM acts as a navigator, reading the document tree outline like a Table of Contents to select the precise nodes containing the answer.
3. **Exact Slicing:** Text is sliced directly from the stored PDF pages based on page ranges—the LLM never rewrites or copies the source text, eliminating hallucinations and truncation.
4. **Grounded Answers:** Responses are streamed in real-time via Server-Sent Events (SSE) and include explicit page citations (e.g., `[file.pdf, p.12]`).

---

## Architecture & How It Works

[ Upload PDF ] ──> [ FastAPI Backend ] ──> [ PDF Parser (PyMuPDF) ]
│
[ Hierarchical Tree ]
│
[ User Query ] ──> [ 2-Stage Reasoning Router ] ──> [ Exact Page Slicing ] ──> [ Grounded Streaming Answer ]
1. **Indexing Pipeline:** 
   - Extracts page-wise text using PyMuPDF.
   - Detects document structure via bookmarks or font-size heuristics (with a 3-page window fallback).
   - Generates concise node summaries in batches using Groq LLMs to save tokens.
   - Caches indexes using SHA-256 file hashes for instant re-uploads.
2. **Retrieval Pipeline:**
   - **Stage 1 (Document Selection):** Selects relevant documents from metadata summaries (for multi-document queries).
   - **Stage 2 (Node Selection):** Feeds the document tree outline to the LLM, which returns structured JSON node IDs. Validates IDs strictly and filters out redundant parent/child nodes.
3. **Answering & Streaming:**
   - Assembles text within a strict character budget.
   - Streams responses over **SSE (Server-Sent Events)** with status updates, reasoning traces, and citation chips.

---

## Tech Stack
- **Backend:** FastAPI, Python, SQLite (metadata & state storage)
- **PDF Processing:** PyMuPDF (`fitz`)
- **LLM & Orchestration:** Groq API (`llama-3.1-8b-instant`, `openai/gpt-oss-120b`), custom JSON parsing & retry backoff engine
- **Frontend:** Vanilla HTML5, CSS (Modern Glassmorphism / Dark Minimalist styling), JavaScript (SSE streaming client)
- **Deployment:** Docker & Docker Compose

---

## Quick Start

### 1. Local Setup
```bash
# Clone the repository
git clone [https://github.com/Hemanthmeher/VectorlessRag.git](https://github.com/Hemanthmeher/VectorlessRag.git)
cd VectorlessRag

# Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Open .env and add your GROQ_API_KEY (free at console.groq.com)

# Run the FastAPI server
uvicorn app.main:app --reload
Open http://localhost:8000 in your browser.2. Docker SetupBashdocker build -t vectorless-rag .
