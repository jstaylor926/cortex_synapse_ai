You are an expert software developer assisting with the Cortex Synapse AI project.

**Project Overview:**

The project is a local-first Dev Assistant that combines a notebook/IDE with a personal knowledge base and an AI assistant. It uses a FastAPI backend, a React frontend, and a PostgreSQL database with pgvector for RAG.

**Key Technologies:**

* **Backend**: FastAPI, SQLAlchemy, Alembic, Pydantic
* **Frontend**: React, TypeScript, Vite, Monaco Editor
* **Database**: PostgreSQL with pgvector for vector search
* **AI/ML**: Ollama for local LLMs, sentence-transformers for embeddings

**File Structure:**

* `api/`: FastAPI backend code
    * `main.py`: API endpoints
    * `models.py`: SQLAlchemy models
    * `db.py`: Database session management
    * `llm/`: LLM provider implementations
* `web/`: React frontend code
    * `src/main.tsx`: Main application entry point
    * `src/App.tsx`: Root App component
* `infra/`: Docker compose for Postgres and Ollama
* `db_migrations/`: Alembic database migrations

When providing code examples, please adhere to the existing coding style and best practices found in the project. For Python, follow PEP 8. For TypeScript/React, use functional components with hooks.


# MVP (4–6 weeks)

**Must-haves**

* Monaco editor + notebooks (markdown + Python cells; TS cells optional)
* Snippets library (CRUD, tags, quick-insert, copy)
* Import PDFs/DOCX → chunk → embed
* Hybrid search (keyword + vector)
* RAG Q\&A panel with **citations to your sources**
* Local LLM via Ollama (configurable), offline-by-default
* Basic actions: RunCell, CaptureSnippet, IndexDocument, AskRag
* Minimal Tasks (todo/doing/done), linked anywhere
* Export (markdown/json) and simple backup

**Quality-of-life**

* Global hotkey “Capture Snippet”
* Per-notebook virtualenv (Python) and cached outputs
* Settings page (model selection, privacy toggles)

# v1 (productization)

* **Refactor actions** (propose diffs), **test generation**, docstring tools
* **Flashcards + spaced repetition** (“Study Today” view)
* **Pack publishing** (versioned bundles you can share)
* **Sync** (optional) via encrypted store (S3/minio/Tailscale)—user holds keys
* **Auth** (local single-user → passkey for cloud)
* **Telemetry (opt-in, local-first)** success/failure metrics only
* **Plugin API** (register new Actions & Tools with minimal boilerplate)

# Data & Retrieval Plan

* **Chunk sizes** \~800 tokens, 100 overlap; code chunks include language, path, and title for weighting
* **Hybrid retrieval**: BM25 (SQL/SQLite FTS) + vector cosine; score fusion
* **Rerank (optional)**: cross-encoder for top 50 → top 10
* **Citations**: source title + chunk index; click to open exact section

# Security & Privacy (from day one)

* Local-by-default; **no network calls unless explicitly enabled**
* Configurable **denylist** of outbound domains; key vault in OS keychain
* Content hashing for duplicate detection
* For SaaS later: row-level security, per-tenant keys, encrypted at rest

# Packaging & Distribution

* **Tauri app** for macOS/Win/Linux
* Auto-update channel (optional), offline installers
* CLI: `dev-assistant export --all` (tar of DB dump + markdown + attachments)

# Monetization & Open-Source Strategy

* **Core OSS** (local app, basic RAG, notes/snippets)
* **Pro** (one-time or subscription): refactor/test-gen, advanced search, packs marketplace, multi-device sync
* **Team plan**: shared spaces, role gates, review workflows
* License: permissive for core; commercial add-ons under separate EULA

# Multi-Tenant SaaS Path (when ready)

* Keep FastAPI; add **Postgres RLS** (tenant\_id), or schema-per-tenant
* Object storage for attachments (S3-compatible) + CDN
* Background workers (Redis) + task queue
* Observability (OpenTelemetry) and rate limiting


Ensure you use modern technologies and focus on local development and implementation.

Got it — you want a **system design** for a **smart study assistant + note-taking application** that uses **GPT-OSS** (open-source GPT-style models) for **local development**.
We’ll build this in a way that supports **offline/LLM-powered features**, integrates **structured notes**, and enables **smart study capabilities** like summarization, Q\&A, flashcard generation, and schedule creation — all without requiring a proprietary API like OpenAI’s.

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
| --------------- | ---------------------------- |
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

If you like, I can now **draw a detailed architecture diagram** showing **frontend-backend-LLM-RAG connections**, plus **data flow** for:

* Note creation.
* File import.
* Question answering.

That way you’ll have both **logical** and **physical** views of the system.
