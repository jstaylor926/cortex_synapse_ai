# Implementation Roadmap (8–10 weeks)

## Sprint 1 — Skeleton & Persistence

* Repo scaffold (web, api, db, infra), Docker compose (Postgres + pgvector)
* Alembic baseline: projects, notebooks, cells, snippets, documents, tasks, chunks, embeddings, jobs
* FastAPI CRUD for Snippets/Notebooks; Monaco editor shell

**Exit**: Create snippets/notebooks, edit markdown, list/filter

## Sprint 2 — Execution & Jobs

* Python cell execution (isolated venv per notebook), output caching
* Job runner (embeddings/import/long runs) + WS progress

**Exit**: Run code cells, see streamed output, jobs tracked

## Sprint 3 — Embeddings & Hybrid Search

* Chunker + embeddings pipeline; BM25 + pgvector with score fusion
* Global search UI and citations drawer

**Exit**: Ask a question → get cited answers grounded in your notes

## Sprint 4 — Local LLM & RAG Polishing

* Ollama integration (model picker), prompt templates, failure fallbacks
* “Capture Snippet” hotkey (Tauri tray)

**Exit**: Smooth offline Q\&A, easy snippet capture anywhere

## Sprint 5 — Assistant Actions (Dev QoL)

* Refactor suggestion (diff view), docstring generator, TODO extraction
* Basic flashcard generation from selection

**Exit**: Practical assistant that improves your code & notes

## Sprint 6 — Packs & Export

* Packs (assemble/validate/publish), versioning
* Full export/backup CLI

**Exit**: Shareable bundles; reliable backup story

## Sprint 7+ — Pro Readiness

* Spaced repetition dashboard, plugin API, opt-in telemetry
* Optional encrypted sync + multi-device

# Minimal API Sketch (FastAPI)

* `POST /snippets`, `GET /snippets?query=&tags=`
* `POST /notebooks`, `POST /notebooks/{id}/cells` (`run=true`)
* `POST /import` (pdf/docx) → jobs + embeddings
* `POST /search/hybrid`
* `POST /rag/answer`
* `POST /actions/refactor`, `/actions/generate-tests`, `/actions/summarize`

# Folder Layout

```
/dev-assistant
  /app-web (React+Vite)
  /app-api (FastAPI)
  /db (alembic)
  /infra (docker-compose, env)
  /scripts
  README.md
```

# Prompts (starter)

* **System**: “You are a local dev/study assistant. Prefer the user’s own notes/snippets. Always cite sources. If
  context is insufficient, say so and suggest what to add.”
* **Tools**: retrieval, diff, tests, flashcards (callable actions)

# Success Criteria (MVP)

* Query over your notes returns a **grounded** answer with **clickable citations**
* Insert/refactor snippets from assistant with a single click
* Import a PDF, generate flashcards, and review them in the app
* All of the above **works offline**

---