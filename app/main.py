import hashlib
import json
import uuid

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import answerer, config, indexer, retriever, store

app = FastAPI(title="Vectorless RAG")
store.init()


def public(d):
    return {k: d[k] for k in ("id", "filename", "status", "stage", "progress", "error", "num_pages", "summary")}


@app.get("/")
def home():
    return FileResponse(config.ROOT / "frontend" / "index.html")


@app.post("/api/documents")
async def upload(background: BackgroundTasks, file: UploadFile = File(...)):
    name = file.filename or "document.pdf"
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "This file is not a valid PDF.")
    sha = hashlib.sha256(data).hexdigest()
    existing = store.find_by_hash(sha)
    if existing and existing["status"] != "failed":
        return public(existing)                       # same PDF: reuse the cached index
    if existing:
        store.delete_doc(existing["id"])
    doc_id = uuid.uuid4().hex[:12]
    (config.PDF_DIR / f"{doc_id}.pdf").write_bytes(data)
    store.create_doc(doc_id, name, sha)
    background.add_task(indexer.index_document, doc_id)
    return public(store.get_doc(doc_id))


@app.get("/api/documents")
def list_documents():
    return [public(d) for d in store.list_docs()]


@app.get("/api/documents/{doc_id}/tree")
def tree(doc_id: str):
    d = store.get_doc(doc_id)
    if not d or d["status"] != "ready":
        raise HTTPException(404, "Document is not ready.")
    return store.load_tree(doc_id)


@app.delete("/api/documents/{doc_id}")
def delete(doc_id: str):
    if not store.get_doc(doc_id):
        raise HTTPException(404, "Document not found.")
    store.delete_doc(doc_id)
    return {"deleted": doc_id}


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] = []
    history: list[dict] = []


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/chat")
def chat(req: ChatRequest):
    def run():
        try:
            ready = [d for d in store.list_docs() if d["status"] == "ready"]
            if req.document_ids:
                ready = [d for d in ready if d["id"] in req.document_ids]
            if not ready:
                yield sse("error", {"message": "Select at least one indexed document first."})
                return
            history = [m for m in req.history if m.get("role") in ("user", "assistant") and m.get("content")]

            yield sse("status", {"message": "Understanding the question"})
            question = retriever.condense_question(req.question, history)

            yield sse("status", {"message": "Choosing documents"})
            docs = retriever.select_documents(question, ready)

            yield sse("status", {"message": "Reading the outline to find sections"})
            reasoning, selected = retriever.select_nodes(question, docs)
            yield sse("retrieval", {"reasoning": reasoning, "nodes": [
                {"doc": s["doc"]["filename"], "title": s["node"]["title"],
                 "start": s["node"]["start"], "end": s["node"]["end"]} for s in selected]})
            if not selected:
                yield sse("token", {"text": "I couldn't find a section in the selected documents that covers this."})
                yield sse("done", {})
                return

            yield sse("status", {"message": "Writing the answer"})
            context = retriever.build_context(selected)
            for tok in answerer.stream_answer(question, context, history):
                yield sse("token", {"text": tok})
            yield sse("done", {})
        except Exception as e:
            yield sse("error", {"message": str(e)[:300]})

    return StreamingResponse(run(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
