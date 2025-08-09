---

## Repo & package naming (yarn-first)

Keep your current repo and align sub-packages:

```
cortex_synapse_ai/
  apps/
    studio/                 # Desktop app (React + Tauri)
    api/                    # FastAPI service
  services/
    engine/                 # Celery workers (ingestion/embeddings/RAG)
  infra/
    docker-compose.yml
  packages/
    ui/                     # shared UI components (optional)
    ts-sdk/                 # TypeScript client (optional)
```

**Display names & bundles**

* Tauri app name: `Cortex Synapse Studio` (bundle id e.g. `ai.cortexsynapse.studio`)
* Python module: `cortex_synapse_api`
* Docker images: `ghcr.io/<you>/cortex-synapse-api`, `…-engine`
* NPM scope (if you want one): `@cortex-synapse/*`

---

## What I’ll change next in your repo (based on your 10 answers)

1. **Yarn everywhere**

* Add Tailwind to `apps/studio` (`yarn add -D tailwindcss postcss autoprefixer && npx tailwindcss init -p`)
* Add state/query libs: `yarn add @tanstack/react-query zustand classnames axios`

2. **Both OS targets**

* Tauri config updates + GitHub Actions matrix: macOS, Windows, Ubuntu packaging.

3. **Postgres (pgvector)**

* Compose stays as-is; add migrations: `vector(768)` (see embeddings below).

4. **LLMs (Ollama primary, OpenAI-compatible optional)**

* Keep provider switch; default Ollama model names in `.env`.
* Add robust error handling + health checks.

5. **Minimal ingestion + embeddings**

* Add `/documents` upload + text extraction (PyMuPDF/Unstructured).
* Use **Ollama embeddings** by default (`nomic-embed-text`) → `vector(768)`.
* Optional cloud path: OpenAI `text-embedding-3-small` → `vector(1536)` (toggle sets dim).

6. **Redis + Celery**

* Extend compose with Redis.
* Celery workers for: chunk → embed → upsert, plus reindex jobs.

7. **Auth (my pick for MVP):** local-only settings; no login yet

* Encrypt local API keys with OS keychain later; for now `.env` + Tauri secure storage.

8. **Tailwind**

* Wire Tailwind; add base layout + shadcn/ui later.

9. **Telemetry**

* Sentry + OpenTelemetry hooks (disabled by default; env-gated).

10. **CI/CD**

* GitHub Actions:

    * API: lint, tests, build Docker image.
    * Studio: yarn build; Tauri bundling for all OS.
    * Tag-based release artifacts.

---

## Concrete naming checklist

* [ ] Update app display strings to **Cortex Synapse Studio**
* [ ] README: “Cortex Synapse” as master brand; sections for Studio/API/Engine
* [ ] Package identifiers:

    * Python: `cortex_synapse_api`
    * Tauri bundle: `ai.cortexsynapse.studio`
    * Docker: `cortex-synapse-{api,engine}`
* [ ] Issue labels: `area:studio`, `area:api`, `area:engine`, `area:infra`
* [ ] CLI (later): `synapse` or `csyn` (short, memorable)

---

## Tiny style guide (so we’re consistent)

* First mention: **Cortex Synapse Studio** (the “Studio”)
* Subsequent mentions in UI copy: **Synapse Studio**
* Docs: “Use **Cortex Synapse API** to …”
* Avoid “AI” in the product name; keep it for the **company** or tagline.

---

### One quick confirmation

Do you want me to go ahead and:

1. rename the app to **Cortex Synapse Studio** in the Tauri config & UI,
2. update the README to the master-brand structure, and
3. open GitHub issues for the work items above?

If yes, I’ll draft the PR plan and the exact changeset list next.

Perfect—let’s **start building** and skip CI/CD for now. Below is a tight, do-this-next plan that matches your choices (
yarn, Postgres, Redis+Celery, Tailwind, Ollama-first with optional OpenAI-compatible). I’ll use the brand structure we
agreed: **Cortex Synapse** (master), **Cortex Synapse Studio** (desktop), **Cortex Synapse API** (backend), **Cortex
Synapse Engine** (workers).

# Week 1 Build Plan (no CI/CD)

## 0) Branding tweaks (fast, optional)

* **Studio window/title:** set name to “Cortex Synapse Studio”
* **Docs/README:** refer to Studio/API/Engine consistently

### Tauri config update

```diff
# apps/studio/src-tauri/tauri.conf.json
- "productName": "Dev Assistant IDE",
+ "productName": "Cortex Synapse Studio",
- "identifier": "dev.assistant.ide",
+ "identifier": "ai.cortexsynapse.studio",
```

---

## 1) Infra: Postgres + Redis (compose)

Add **Redis** next to Postgres so Celery can run.

```yaml
# infra/docker-compose.yml
services:
    db:
        image: ankane/pgvector:latest
        container_name: cs_pgvector
        environment:
            POSTGRES_USER: dev
            POSTGRES_PASSWORD: dev
            POSTGRES_DB: cortex_synapse
        ports: [ "5432:5432" ]
        volumes:
            - cs_pg:/var/lib/postgresql/data
    redis:
        image: redis:7
        container_name: cs_redis
        ports: [ "6379:6379" ]
volumes:
    cs_pg:
```

**Run**

```bash
cd infra && docker compose up -d
```

---

## 2) Backend deps (FastAPI + Celery + parsing)

```bash
cd apps/api
python -m venv .venv && source .venv/bin/activate
# requirements.txt: add the following if not present
```

```txt
# apps/api/requirements.txt
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.8.2
SQLAlchemy==2.0.31
psycopg2-binary==2.9.9
alembic==1.13.2
python-dotenv==1.0.1
pgvector==0.2.5
httpx==0.27.0

# NEW
celery[redis]==5.3.6
redis==5.0.8
pymupdf==1.24.9      # PDF text
python-magic==0.4.27 # simple mime detect
```

`.env` (API):

```ini
DATABASE_URL=postgresql+psycopg2://dev:dev@localhost:5432/cortex_synapse
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3:8b-instruct

# Embeddings: default Ollama
EMBED_PROVIDER=ollama
EMBED_MODEL=nomic-embed-text
EMBED_DIM=768

# Optional OpenAI-compatible
GPT_OSS_ENDPOINT=http://localhost:11434/v1/chat/completions
GPT_OSS_API_KEY=replace_me

# Celery/Redis
REDIS_URL=redis://localhost:6379/0

# Telemetry (wire but off by default)
SENTRY_DSN=
OTEL_EXPORTER_OTLP_ENDPOINT=
```

---

## 3) DB schema: documents + chunks + embedding

Create **Document** and **DocumentChunk** tables and enable pgvector. We’ll default to **768-dim** (Ollama
nomic-embed-text). If you switch to OpenAI `text-embedding-3-small` later (1536), we’ll do a migration.

### Alembic revision (example)

```py
# apps/api/alembic/versions/0002_documents_embeddings.py
from alembic import op
import sqlalchemy as sa

revision = "0002_documents_embeddings"
down_revision = "0001_init"


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.create_table(
        "documents",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("filename", sa.String(512)),
        sa.Column("mime_type", sa.String(128)),
        sa.Column("size_bytes", sa.Integer),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
        sa.Column("meta", sa.JSON),
    )
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("document_id", sa.String, sa.ForeignKey("documents.id", ondelete="CASCADE")),
        sa.Column("idx", sa.Integer),  # chunk order
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", sa.dialects.postgresql.BYTEA, nullable=True),  # temp fill
    )
    # switch to pgvector type with dimension
    op.execute("ALTER TABLE document_chunks ADD COLUMN embedding_vec vector(768);")
    # indexes
    op.execute("CREATE INDEX ON document_chunks USING ivfflat (embedding_vec vector_cosine_ops) WITH (lists = 100);")
    op.execute("CREATE INDEX document_chunks_document_id_idx ON document_chunks(document_id);")


def downgrade():
    op.execute("DROP INDEX IF EXISTS document_chunks_document_id_idx;")
    op.execute("DROP INDEX IF EXISTS document_chunks_embedding_vec_idx;")
    op.drop_table("document_chunks")
    op.drop_table("documents")
```

> Note: We add `embedding_vec vector(768)` directly; the BYTEA can be skipped if you prefer—kept here only if you want
> an intermediate staging. Feel free to remove.

Run:

```bash
alembic upgrade head
```

---

## 4) Celery (Cortex Synapse Engine)

Create a Celery app and embedding tasks.

```py
# apps/api/app/celery_app.py
import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("cs_engine", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(task_serializer="json", accept_content=["json"], result_serializer="json")
```

```py
# apps/api/app/embeddings.py
import os, httpx
from typing import List

EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "ollama")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


async def embed_texts(texts: List[str]) -> List[List[float]]:
    if EMBED_PROVIDER == "ollama":
        # Ollama embeddings: POST /api/embeddings
        async with httpx.AsyncClient(timeout=120) as client:
            out = []
            for t in texts:
                r = await client.post("http://localhost:11434/api/embeddings",
                                      json={"model": EMBED_MODEL, "prompt": t})
                r.raise_for_status()
                out.append(r.json().get("embedding", []))
            return out
    else:
        # OpenAI-compatible: POST /v1/embeddings
        endpoint = os.getenv("GPT_OSS_EMBED_ENDPOINT")
        api_key = os.getenv("GPT_OSS_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        async with httpx.AsyncClient(timeout=120, headers=headers) as client:
            r = await client.post(endpoint, json={"model": os.getenv("EMBED_MODEL"), "input": texts})
            r.raise_for_status()
            data = r.json()
            return [item["embedding"] for item in data["data"]]
```

```py
# apps/api/app/tasks.py
import uuid
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import text
from .celery_app import celery_app
from .db import SessionLocal
from .models import Document, DocumentChunk
from .text_extract import extract_text
from .chunking import chunk_text
from .settings import EMBED_DIM
from .sync_utils import embed_sync


@celery_app.task
def ingest_document_task(doc_id: str):
    db: Session = SessionLocal()
    try:
        doc = db.query(Document).filter_by(id=doc_id).one()
        text = extract_text(doc)  # reads file path from doc.meta or FS
        chunks = chunk_text(text, target_tokens=300, overlap=60)

        # embed (sync wrapper around async for Celery simplicity)
        vectors: List[List[float]] = embed_sync([c["content"] for c in chunks])

        for i, (c, v) in enumerate(zip(chunks, vectors)):
            cid = str(uuid.uuid4())
            db.add(DocumentChunk(
                id=cid, document_id=doc.id, idx=i, content=c["content"]
            ))
            # insert embedding into pgvector column
            # SQLAlchemy pgvector type exists but raw SQL is fine:
            db.execute(
                text("UPDATE document_chunks SET embedding_vec = :vec WHERE id = :cid"),
                {"vec": v, "cid": cid},
            )
        db.commit()
    finally:
        db.close()
```

> Implementation notes
>
> * `sync_utils.embed_sync` can run an asyncio loop for `embed_texts`.
> * For pgvector, many folks use raw SQL with array literal: `vector '[..]'`. Some ORMs have a `Vector` type; if you
    prefer, add `from pgvector.sqlalchemy import Vector` and model as `Column(Vector(dim))`.

**Run workers**

```bash
# terminal A: API
uvicorn app.main:app --reload

# terminal B: Celery worker
celery -A app.celery_app.celery_app worker --loglevel=INFO -Q default -n engine@%h
```

---

## 5) Minimal ingestion API & retrieval

### Models

```py
# apps/api/app/models.py (additions)
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, JSON
from sqlalchemy.sql import func


class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True)
    filename = Column(String(512))
    mime_type = Column(String(128))
    size_bytes = Column(Integer)
    created_at = Column(DateTime, default=func.now())
    meta = Column(JSON)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    idx = Column(Integer)
    content = Column(Text, nullable=False)
    # embedding_vec lives in DB via migration (vector(768))
```

### Text extraction & chunking

```py
# apps/api/app/text_extract.py
import fitz  # PyMuPDF
import os, magic


def extract_text(doc) -> str:
    path = doc.meta.get("path")
    if not path: return ""
    mime = magic.from_file(path, mime=True)
    if mime == "application/pdf":
        with fitz.open(path) as pdf:
            return "\n".join(page.get_text() for page in pdf)
    # md/txt fallback
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
```

```py
# apps/api/app/chunking.py
import re


def chunk_text(text: str, target_tokens=300, overlap=60):
    # naive token estimate (~4 chars/token)
    size = target_tokens * 4
    step = max(size - overlap * 4, 100)
    out = []
    for i in range(0, len(text), step):
        out.append({"content": text[i:i + size]})
    return out
```

### Upload & trigger ingestion

```py
# apps/api/app/routers/documents.py
import os, uuid, shutil
from fastapi import APIRouter, UploadFile, File
from ..db import SessionLocal
from ..models import Document
from ..tasks import ingest_document_task

router = APIRouter()

STORAGE = os.getenv("CS_STORAGE", os.path.abspath("./storage"))
os.makedirs(STORAGE, exist_ok=True)


@router.post("", summary="Upload a document and start ingestion")
async def upload_document(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    dst = os.path.join(STORAGE, f"{doc_id}_{file.filename}")
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f)
    db = SessionLocal()
    try:
        db.add(Document(
            id=doc_id, filename=file.filename, mime_type=file.content_type,
            size_bytes=os.path.getsize(dst), meta={"path": dst}
        ))
        db.commit()
    finally:
        db.close()
    ingest_document_task.delay(doc_id)
    return {"id": doc_id, "status": "queued"}
```

### Retrieval (hybrid: vector + keyword)

```py
# apps/api/app/routers/search.py
from fastapi import APIRouter, Query
from sqlalchemy import text
from ..db import engine

router = APIRouter()


@router.get("", summary="Hybrid search")
def search(q: str = Query(...), k: int = 5):
    with engine.connect() as conn:
        # Simple keyword + cosine; improve with ts_vectors later
        sql = text("""
          WITH vec AS (
            SELECT embedding_vec FROM document_chunks
            WHERE embedding_vec IS NOT NULL
            ORDER BY random() LIMIT 1
          )
          SELECT content, 1 - (embedding_vec <=> (SELECT embedding_vec FROM document_chunks ORDER BY random() LIMIT 1)) AS sim
          FROM document_chunks
          ORDER BY embedding_vec <=> (SELECT embedding_vec FROM document_chunks ORDER BY random() LIMIT 1)
          LIMIT :k
        """)
        # ^ Placeholder for brevity; replace with:
        #  - compute query embedding for :q
        #  - ORDER BY embedding_vec <=> :query_vec
        return []
```

> In practice: compute the **query embedding** from `q` (same model), then parameterize:
>
> ```sql
> SELECT content, 1 - (embedding_vec <=> :query_vec) AS sim
> FROM document_chunks
> ORDER BY embedding_vec <=> :query_vec
> LIMIT :k;
> ```
>
> We’ll wire this when we add the `embed_texts([q])` call in the route.

### RAG answer (use retrieved chunks)

Update `/rag/answer` to embed the query, fetch top-k, and stuff the prompt context.

---

## 6) Frontend (Studio): Yarn + Tailwind + basic panels

**Install**

```bash
cd apps/studio
yarn add @tanstack/react-query zustand classnames axios
yarn add -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

**Tailwind config**

```js
// apps/studio/tailwind.config.js
export default {
    content: ["./index.html", "./src/**/*.{ts,tsx}"],
    theme: {extend: {}},
    plugins: [],
}
```

**PostCSS**

```js
// apps/studio/postcss.config.js
export default {plugins: {tailwindcss: {}, autoprefixer: {}}}
```

**CSS entry**

```css
/* apps/studio/src/index.css */
@tailwind base;
@tailwind components;
@tailwind utilities;
```

**Wire in UI**

* Sidebar: Snippets / Documents / RAG
* Docs page: file upload → `/documents`
* RAG page: input → calls `/rag/answer`, renders citations
* Settings: choose provider (Ollama vs OpenAI-compatible), base URLs, models

---

## 7) Telemetry (wired, off by default)

Add Sentry init in API (if `SENTRY_DSN` set). Same for Studio (browser SDK). Keep disabled by default to avoid noise.

---

## 8) Runbook (end-to-end)

```bash
# 1) Infra
cd infra && docker compose up -d

# 2) API
cd apps/api
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add vars above
alembic upgrade head
uvicorn app.main:app --reload

# 3) Engine (workers)
celery -A app.celery_app.celery_app worker --loglevel=INFO

# 4) Studio
cd apps/studio
yarn
yarn tauri dev
```

---

## Short backlog (immediately after this)

* [ ] **Search route**: compute query embedding, parametrize `:query_vec`, return top-k with scores.
* [ ] **RAG route**: prompt template w/ top-k chunks + citations.
* [ ] **Studio**: Documents upload UI; RAG panel with streaming answer.
* [ ] **Ollama health checks** for both chat & embeddings; helpful error messages in UI.
* [ ] **Indexing**: ivfflat `lists` tuning, `ANALYZE` table after bulk ingest.

---

### One tiny confirmation

Default **embedding dim = 768** (Ollama `nomic-embed-text`). Good to lock that for MVP? If you plan to try OpenAI
embeddings soon, I’ll note a follow-up migration path to 1536.
