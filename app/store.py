import json
import sqlite3
import time
from contextlib import closing

from . import config

FIELDS = {"status", "stage", "progress", "error", "num_pages", "summary"}


def _run(sql, args=(), one=False, many=False):
    with closing(sqlite3.connect(config.DB_PATH, timeout=30)) as c:
        c.row_factory = sqlite3.Row
        cur = c.execute(sql, args)
        c.commit()
        if one:
            r = cur.fetchone()
            return dict(r) if r else None
        if many:
            return [dict(r) for r in cur.fetchall()]


def init():
    for d in (config.PDF_DIR, config.PAGES_DIR, config.TREES_DIR):
        d.mkdir(parents=True, exist_ok=True)
    _run("""CREATE TABLE IF NOT EXISTS documents(
        id TEXT PRIMARY KEY, filename TEXT, sha256 TEXT,
        status TEXT, stage TEXT DEFAULT '', progress INTEGER DEFAULT 0,
        error TEXT DEFAULT '', num_pages INTEGER DEFAULT 0,
        summary TEXT DEFAULT '', created_at REAL)""")


def create_doc(doc_id, filename, sha):
    _run("INSERT INTO documents(id,filename,sha256,status,stage,created_at) VALUES(?,?,?,?,?,?)",
         (doc_id, filename, sha, "indexing", "Queued", time.time()))


def update_doc(doc_id, **kw):
    kw = {k: v for k, v in kw.items() if k in FIELDS}
    if not kw:
        return
    sets = ", ".join(f"{k}=?" for k in kw)
    _run(f"UPDATE documents SET {sets} WHERE id=?", (*kw.values(), doc_id))


def get_doc(doc_id):
    return _run("SELECT * FROM documents WHERE id=?", (doc_id,), one=True)


def find_by_hash(sha):
    return _run("SELECT * FROM documents WHERE sha256=? ORDER BY created_at DESC", (sha,), one=True)


def list_docs():
    return _run("SELECT * FROM documents ORDER BY created_at DESC", many=True)


def delete_doc(doc_id):
    _run("DELETE FROM documents WHERE id=?", (doc_id,))
    for p in (config.PDF_DIR / f"{doc_id}.pdf",
              config.PAGES_DIR / f"{doc_id}.json",
              config.TREES_DIR / f"{doc_id}.json"):
        p.unlink(missing_ok=True)


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def load_pages(doc_id):
    return json.loads((config.PAGES_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))


def load_tree(doc_id):
    return json.loads((config.TREES_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))
