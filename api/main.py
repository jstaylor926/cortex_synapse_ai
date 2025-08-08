from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import select, func, text
from sqlalchemy.orm import Session

from settings import Settings, get_settings
from db import get_session, Base, engine
from models import Document, Chunk, Embedding
from llm.provider import get_llm

app = FastAPI(title="Dev Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

class ImportIn(BaseModel):
    project_id: Optional[int] = None
    title: str
    text: str

@app.post("/import")
def import_text(payload: ImportIn, sess: Session = Depends(get_session), settings: Settings = Depends(get_settings)):
    # simple 1-doc import: create Document, Chunk(s), Embedding(s)
    doc = Document(title=payload.title, project_id=payload.project_id)
    sess.add(doc)
    sess.flush()

    # naive chunking (improve later)
    chunks = []
    CHUNK = 800
    OVER = 120
    txt = payload.text
    i = 0
    while i < len(txt):
        end = min(len(txt), i + CHUNK)
        chunk_text = txt[i:end]
        chunks.append(Chunk(document_id=doc.id, content=chunk_text))
        i = end - OVER

    sess.add_all(chunks)
    sess.flush()

    # embeddings via provider
    llm = get_llm(settings)
    vectors = llm.embed([c.content for c in chunks])
    embs = []
    for c, v in zip(chunks, vectors):
        embs.append(Embedding(chunk_id=c.id, vector=v))
    sess.add_all(embs)
    sess.commit()
    return {"document_id": doc.id, "chunks": len(chunks)}

class SearchIn(BaseModel):
    query: str
    k: int = 8

@app.post("/search/hybrid")
def search_hybrid(payload: SearchIn, sess: Session = Depends(get_session), settings: Settings = Depends(get_settings)):
    # keyword score using tsvector, vector score via cosine distance
    # 1) text search candidates
    q = payload.query
    sql = text("""
        SELECT c.id AS chunk_id, c.content,
               ts_rank(c.tsv, plainto_tsquery('english', :q)) AS kw_score
        FROM chunks c
        WHERE c.tsv @@ plainto_tsquery('english', :q)
        ORDER BY kw_score DESC
        LIMIT 32
    """)
    kw_rows = sess.execute(sql, {"q": q}).mappings().all()

    # 2) vector search
    llm = get_llm(settings)
    qvec = llm.embed([q])[0]
    vsql = text("""
        SELECT e.chunk_id, c.content, 1 - (e.vector <=> :qvec) AS vec_score
        FROM embeddings e
        JOIN chunks c ON c.id = e.chunk_id
        ORDER BY e.vector <=> :qvec
        LIMIT 32
    """)
    v_rows = sess.execute(vsql, {"qvec": qvec}).mappings().all()

    # fuse by max-normalized sum
    from collections import defaultdict
    import math
    by_id = defaultdict(lambda: {"kw":0.0,"vec":0.0,"content":""})
    max_kw = max([r["kw_score"] for r in kw_rows], default=1e-9)
    max_vec = max([r["vec_score"] for r in v_rows], default=1e-9)
    for r in kw_rows:
        by_id[r["chunk_id"]]["kw"] = r["kw_score"]/max_kw if max_kw>0 else 0.0
        by_id[r["chunk_id"]]["content"] = r["content"]
    for r in v_rows:
        by_id[r["chunk_id"]]["vec"] = r["vec_score"]/max_vec if max_vec>0 else 0.0
        by_id[r["chunk_id"]]["content"] = r["content"]
    ranked = sorted(by_id.items(), key=lambda kv: (kv[1]["kw"]+kv[1]["vec"]), reverse=True)
    top = [{"chunk_id": cid, "score": s["kw"]+s["vec"], "content": s["content"]} for cid, s in ranked[:payload.k]]
    return {"results": top}

class RagIn(BaseModel):
    question: str
    k: int = 6
    max_tokens: int = 512
    temperature: float = 0.2

@app.post("/rag/answer")
def rag_answer(payload: RagIn, sess: Session = Depends(get_session), settings: Settings = Depends(get_settings)):
    # retrieve
    search = search_hybrid(SearchIn(query=payload.question, k=payload.k), sess, settings)
    contexts = [r["content"] for r in search["results"]]
    citations = [r["chunk_id"] for r in search["results"]]

    # prompt
    system = "You are a local dev/study assistant. Prefer the user's notes/snippets. Always cite with [chunk:<id>]. If insufficient context, say so."
    ctx = "\n\n---\n\n".join([f"[chunk:{cid}]\n{txt}" for cid, txt in zip(citations, contexts)])
    user = f"Question: {payload.question}\n\nUse the context above. If you don't know, say so."

    llm = get_llm(settings)
    answer = llm.generate(prompt=f"{ctx}\n\n{user}", system=system, temperature=payload.temperature, max_tokens=payload.max_tokens)
    return {"answer": answer, "citations": citations}
