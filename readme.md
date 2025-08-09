# Dev Assistant (Local-first Study/IDE/Assistant)

Local-first desktop-friendly app with:

- **FastAPI** backend
- **Postgres + pgvector** for hybrid search and RAG
- **Ollama** for local LLM + embeddings
- **React (Vite) + Monaco** for notebooks and snippets

## Quick start

1) Bring up infra (Postgres + Ollama):

```bash
cd infra
cp .env.example .env
docker compose up -d
```

2) Install Python deps and run migrations:

```bash
cd ../app-api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
fastapi dev main.py --port 8000
```

*Note: `fastapi dev` requires installing FastAPI with standard extras (`pip install "fastapi[standard]"`). This
project’s `requirements.txt` already includes that extra. If you prefer using Uvicorn directly, you can still
run: `uvicorn main:app --reload --port 8000`.*

```

3) (Optional) Pull Ollama models:

```bash
curl -s http://localhost:11434/api/pull -d '{"model":"llama3"}'
curl -s http://localhost:11434/api/pull -d '{"model":"nomic-embed-text"}'
```

4) Start web app (using yarn):

```bash
cd ../app-web
yarn install
yarn dev
```

## Env

- **API** reads `.env` (or env vars) for DB and LLM settings.
- **DB** connection defaults to `postgresql+psycopg://dev:dev@localhost:5432/dev_assistant`

## Endpoints (subset)

- `GET /health`
- `POST /import` (stub) – accept text and index
- `POST /search/hybrid` – BM25 (tsvector) + vector cosine
- `POST /rag/answer` – retrieve top chunks and ask LLM with citations

## Notes

- First run creates pgvector extension + tsvector column for BM25-ish scoring.
- This is a minimal scaffold; extend models and React UI per roadmap.

Got it — you want a **system design** for a **smart study assistant + note-taking application** that uses **GPT-OSS** (
open-source GPT-style models) for **local development**.
We’ll build this in a way that supports **offline/LLM-powered features**, integrates **structured notes**, and enables *
*smart study capabilities** like summarization, Q\&A, flashcard generation, and schedule creation — all without
requiring a proprietary API like OpenAI’s.

---

## **1. Goals and Core Requirements**

**Functional**

* **Note Taking**: Create, edit, tag, search notes (Markdown & rich text).
* **AI Assistance** (using GPT-OSS locally):

  * Summarize notes/textbooks.
  * Answer questions based on selected notes or topics.
  * Generate flashcards & quizzes.
  * Suggest study schedules.
* **File Import**: PDF, DOCX, plain text → chunk + embed for retrieval.
* **Study Dashboard**: Upcoming study items, flashcard review, AI suggestions.
* **Offline-First**: Entire pipeline runs locally using OSS LLMs.

**Non-Functional**

* Modular architecture to swap LLM models.
* Scalable to larger local datasets without cloud dependency.
* Easy developer setup for local testing.

---

## **2. High-Level Architecture**

```
 ┌─────────────────────────────────────────────┐
 │                 Frontend (React TS)         │
 │─────────────────────────────────────────────│
 │ - Note Editor (Markdown/Rich Text)          │
 │ - Study Dashboard                           │
 │ - Flashcard Viewer                          │
 │ - File Upload UI                            │
 │ - Chat/Q&A Interface                        │
 └─────────────────────────────────────────────┘
                 │  REST/WebSocket API
                 ▼
 ┌─────────────────────────────────────────────┐
 │                Backend (FastAPI)            │
 │─────────────────────────────────────────────│
 │ Notes Service      ── CRUD + Tags            │
 │ File Processing    ── PDF/DOCX → Text        │
 │ Embedding Service  ── Sentence Transformers  │
 │ Vector Store       ── Chroma / Weaviate      │
 │ LLM Service        ── GPT-OSS via Ollama /   │
 │                      GPT4All / LM Studio     │
 │ RAG Pipeline       ── Query + Retrieval      │
 │ Study Logic        ── Schedule, Flashcards   │
 └─────────────────────────────────────────────┘
                 │
                 ▼
 ┌─────────────────────────────────────────────┐
 │           Local Storage Layer                │
 │─────────────────────────────────────────────│
 │ PostgreSQL  ── Structured notes, metadata    │
 │ Vector DB   ── Embeddings for RAG            │
 │ Filesystem  ── Uploaded files/raw text       │
 └─────────────────────────────────────────────┘
```

---

## **3. Component Breakdown**

### **Frontend (React + TypeScript)**

* **Note Editor**:

  * Markdown with formatting toolbar.
  * Side panel for tags/topics.
* **Study Dashboard**:

  * “Study Today” list from backend.
  * AI suggestions for what to review.
* **File Upload**:

  * Drag/drop → sends to backend for processing.
* **Flashcard View**:

  * Flip, quiz mode, spaced repetition.
* **Chat/Q\&A**:

  * Messages stream from GPT-OSS model locally.

---

### **Backend (FastAPI)**

**1. Notes Service**

* CRUD for notes with tags, topics, linked resources.
* Endpoints:

  ```
  GET /notes
  POST /notes
  PUT /notes/{id}
  DELETE /notes/{id}
  ```

**2. File Processing Service**

* Use `pdfplumber` or `PyMuPDF` for PDFs.
* Use `python-docx` for DOCX.
* Output cleaned text + chunk for embeddings.

**3. Embedding + Vector Store**

* Use `sentence-transformers` locally for embeddings (e.g., `all-MiniLM-L6-v2`).
* Store in ChromaDB or Weaviate (Docker local instance).
* Metadata: `note_id`, `tags`, `topics`.

**4. LLM Service (GPT-OSS)**

* Run locally via:

  * **Ollama** → `ollama run mistral` or `llama2`.
  * **GPT4All** → Python API.
* Configurable model selection.
* Prompt templates for:

  * Summarization.
  * Q\&A.
  * Flashcard generation.

**5. RAG Pipeline**

* Input → Embed → Vector search → Context injection into GPT prompt.
* Example:

  ```python
  def rag_query(question):
      results = vector_db.similarity_search(question, k=5)
      context = "\n".join([r.page_content for r in results])
      prompt = f"Use the following context to answer:\n{context}\n\nQ: {question}\nA:"
      return gpt_oss.generate(prompt)
  ```

**6. Study Logic**

* Generate spaced repetition schedule from flashcards.
* Recommend daily tasks via rules:

  * Due flashcards.
  * Unread summaries.
  * Upcoming exam topics.

---

### **Storage Layer**

* **PostgreSQL** → Note metadata, study schedule, flashcards.
* **ChromaDB** → Embedding store for semantic search.
* **Local FS** → Uploaded raw files and processed text.

---

## **4. Example Local Development Stack**

| Layer           | Tech                         |
|-----------------|------------------------------|
| Frontend        | React + Vite + TypeScript    |
| Backend         | FastAPI + Uvicorn            |
| LLM             | Ollama / GPT4All / LM Studio |
| Embeddings      | sentence-transformers        |
| Vector Store    | ChromaDB (Docker)            |
| DB              | PostgreSQL (Docker)          |
| File Processing | pdfplumber, python-docx      |

---

## **5. Example Local Dev Flow**

1. **Setup**:

   ```bash
   # Install Ollama
   brew install ollama
   ollama pull mistral
   # Start Docker services
   docker-compose up -d
   ```
2. **Backend**:

   ```bash
   pip install fastapi uvicorn sentence-transformers chromadb pdfplumber python-docx
   uvicorn app.main:app --reload
   ```
3. **Frontend**:

   ```bash
   npm create vite@latest study-assistant -- --template react-ts
   npm install axios react-router-dom
   npm run dev
   ```
4. **Load Notes**:

* Upload PDF/DOCX.
* Process + embed.

5. **Ask Questions**:

* Query runs through RAG pipeline.
* GPT-OSS answers using local embeddings.

---

## **6. AI Prompting Strategy**

For **summarization**:

```
Summarize the following text into key bullet points for study:
{context}
```

For **flashcard generation**:

```
From the following text, create Q&A flashcards:
{context}
Format:
Q: ...
A: ...
```

For **study schedule suggestion**:

```
You are a study planner. Based on the upcoming exam date {date} and topics {topics}, generate a daily study plan.
```

---

## **7. Deployment (Local Dev)**

* **All services** run in Docker.
* Volume mounts for persistent storage of:

  * PostgreSQL DB.
  * ChromaDB index.
  * Uploaded files.

Example `docker-compose.yml` snippet:

```yaml
services:
  db:
    image: postgres:15
    environment:
      POSTGRES_USER: study
      POSTGRES_PASSWORD: study
      POSTGRES_DB: studydb
    ports:
      - "5432:5432"
    volumes:
      - ./data/db:/var/lib/postgresql/data
  chromadb:
    image: ghcr.io/chroma-core/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - ./data/chroma:/data
```

---

If you like, I can now **draw a detailed architecture diagram** showing **frontend-backend-LLM-RAG connections**, plus *
*data flow** for:

* Note creation.
* File import.
* Question answering.

That way you’ll have both **logical** and **physical** views of the system.

# Cortex Synapse

> **Where code, notes, and insight connect.**

**Cortex Synapse** is a local-first IDE and study environment that blends Monaco editing, notebooks, and an AI copilot.
It ingests your documents, builds embeddings in Postgres + pgvector, and serves grounded answers—without leaving your
editor.

## Packages

- **Cortex Synapse Studio** — Desktop app (Tauri + React): `app-web/` (or `apps/studio/` in future layout)
- **Cortex Synapse API** — FastAPI backend: `app-api/`
- **Cortex Synapse Engine** — Workers (Celery + Redis): `services/engine/` (new path introduced by this PR)
- **Cortex Synapse Infra** — Docker Compose: `infra/`

## Quickstart

```bash
# Infra
cd infra && docker compose up -d

# API
cd app-api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
alembic upgrade head
uvicorn app.main:app --reload

# Engine (Celery)
# in another terminal from app-api or services/engine (if later split)
celery -A app.celery_app.celery_app worker --loglevel=INFO

# Studio
cd app-web
yarn && yarn tauri dev
```

## Configuration

- Env prefix guidance: `CS_` (additive, non-breaking). See component READMEs.
- Default LLM: **Ollama** (`llama3:8b-instruct`, `nomic-embed-text`)
- Optional: OpenAI-compatible endpoints (aka "gpt-oss")

## Brand Architecture

- **Master brand:** **Cortex Synapse**
  - **Cortex Synapse Studio** (desktop IDE)
  - **Cortex Synapse API** (backend)
  - **Cortex Synapse Engine** (workers/RAG)
  - **Cortex Synapse Infra** (compose, provisioning)

## License

MIT (placeholder)

---

© 2025 Cortex Synapse contributors.

# Cortex Synapse Engine

Celery workers for document ingestion, chunking, embeddings, and indexing.

## Run

```bash
# from app-api/ (if the Celery app lives there)
celery -A app.celery_app.celery_app worker --loglevel=INFO -Q default -n engine@%h
```

## Tasks

- `ingest_document_task(doc_id)` — extract, chunk, embed, upsert

## Notes

- Default embeddings via Ollama (`nomic-embed-text`, dim=768)
- Switch to OpenAI‑compatible by setting EMBED vars

# Cortex Synapse

> **Where code, notes, and insight connect.**

**Cortex Synapse** is a local-first IDE and study environment that blends Monaco editing, notebooks, and an AI copilot.
It ingests your documents, builds embeddings in Postgres + pgvector, and serves grounded answers—without leaving your
editor.

## Packages

- **Cortex Synapse Studio** — Desktop app (Tauri + React): `app-web/` (or `apps/studio/` in future layout)
- **Cortex Synapse API** — FastAPI backend: `app-api/`
- **Cortex Synapse Engine** — Workers (Celery + Redis): `services/engine/` (new path introduced by this PR)
- **Cortex Synapse Infra** — Docker Compose: `infra/`

## Quickstart

```bash
# Infra
cd infra && docker compose up -d

# API
cd app-api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
alembic upgrade head
uvicorn app.main:app --reload

# Engine (Celery)
# in another terminal from app-api or services/engine (if later split)
celery -A app.celery_app.celery_app worker --loglevel=INFO

# Studio
cd app-web
yarn && yarn tauri dev
```

## Configuration

- Env prefix guidance: `CS_` (additive, non-breaking). See component READMEs.
- Default LLM: **Ollama** (`llama3:8b-instruct`, `nomic-embed-text`)
- Optional: OpenAI-compatible endpoints (aka "gpt-oss")

## Brand Architecture

- **Master brand:** **Cortex Synapse**
  - **Cortex Synapse Studio** (desktop IDE)
  - **Cortex Synapse API** (backend)
  - **Cortex Synapse Engine** (workers/RAG)
  - **Cortex Synapse Infra** (compose, provisioning)

## License

MIT (placeholder)

---

© 2025 Cortex Synapse contributors.

# Cortex Synapse Studio

Desktop IDE where code, notes, and AI study tools meet.

## Stack

- Tauri 2, React 18, Vite
- Monaco Editor
- TailwindCSS
- State: Zustand
- Data: TanStack Query

## Scripts

```bash
yarn dev        # vite dev server
yarn tauri dev  # desktop dev
yarn build      # build web assets
yarn tauri build
```

## Config

- `CS_API_URL` (default `http://127.0.0.1:8000`)
- `CS_OLLAMA_URL` (default `http://localhost:11434`)

## Pages

- Snippets, Documents (upload), RAG (chat), Settings
