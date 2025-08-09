# Cortex Synapse — Brand System & Docs Pack (v0.1)

> Working brand + doc templates to keep naming, messaging, and repos consistent while we build. Edit freely.

---

## 1) Brand Architecture

* **Master brand:** **Cortex Synapse**
  **Tagline (optional):** *Where code, notes, and insight connect.*

* **Products / Components**

    * **Cortex Synapse Studio** — Desktop IDE (Tauri + React). Short: **“Synapse Studio”**.
    * **Cortex Synapse API** — HTTP backend (FastAPI). Short: **“Synapse API”**.
    * **Cortex Synapse Engine** — Workers for ingestion/embeddings/RAG (Celery + Redis). Short: **“Synapse Engine”**.
    * **Cortex Synapse Infra** — Local infra (Docker Compose, provisioning). Short: **“Synapse Infra”**.
    * *(Future)* **Cortex Synapse Cloud**, **SDK**, **CLI**.

**Why this works**

* Clarity across docs and UI
* Scales to new surfaces (Cloud/SDK/CLI) without renaming
* Short names read cleanly in UI after first mention

---

## 2) Naming Conventions & Identifiers

### Display Names & Abbreviations

* First mention in docs: **Cortex Synapse Studio** (the “Studio”) → later as **Synapse Studio**.
* Similar for **Cortex Synapse API** → **Synapse API**; **Cortex Synapse Engine** → **Synapse Engine**.

### Package / Binary / Bundle IDs

* **Tauri bundle identifier:** `ai.cortexsynapse.studio`
* **Docker images:**

    * `ghcr.io/<your-gh-user>/cortex-synapse-api`
    * `ghcr.io/<your-gh-user>/cortex-synapse-engine`
* **Python package/module (API):** `cortex_synapse_api`
* **NPM scope (optional):** `@cortex-synapse/*`
* **CLI name (reserved):** `synapse` (fallback: `csyn`)

### Env Var Prefix & Config

* Use **`CS_`** prefix for cross-cutting config; keep component-specific ones too.
* Examples:

    * `CS_ENV=dev|staging|prod`
    * `CS_API_URL=http://127.0.0.1:8000`
    * `CS_OLLAMA_URL=http://localhost:11434`
    * `CS_EMBED_MODEL=nomic-embed-text`

### Filesystem & Code Style

* **Folders:** kebab-case for web (`/apps/studio/src/pages/rag-panel/`), snake\_case for Python.
* **Branch names:** `feat/…`, `fix/…`, `docs/…`, `chore/…`.
* **Issue labels:** `area:studio`, `area:api`, `area:engine`, `area:infra`, `type:bug`, `type:feat`.

---

## 3) Messaging Cheatsheet

### One-liner (master)

**Cortex Synapse** unifies code, notes, and AI assistants—so you learn faster and build without context switching.

### Product blurbs (short)

* **Synapse Studio:** A focused desktop IDE where your code, notes, and AI study tools live side-by-side.
* **Synapse API:** A fast, typed HTTP backend that powers snippets, notebooks, and retrieval-augmented answers.
* **Synapse Engine:** Background workers for document ingestion, embeddings, and hybrid search at low latency.
* **Synapse Infra:** Local-first Postgres/pgvector + Redis in Docker to get you productive in minutes.

### 30‑sec pitch (master)

Cortex Synapse is a local‑first IDE that blends Monaco editing, notebooks, and an AI study copilot. It ingests your
PDFs/notes, builds embeddings in Postgres + pgvector, and serves grounded answers—without leaving your editor. Start
with Ollama locally, add OpenAI‑compatible providers as needed, and scale to Celery workers when your knowledge base
grows.

---

## 4) Visual Identity (starter — iterate later)

> Placeholder choices to get moving; we can refine once UI is stable.

* **Primary color:** Indigo 600 (#4F46E5)
* **Accent:** Cyan 500 (#06B6D4)
* **Neutral:** Zinc/Slate scale (Tailwind)
* **Typography:**

    * UI: Inter (system fallbacks)
    * Mono: JetBrains Mono
* **Iconography:** Lucide (outline, consistent stroke)

**Logo/Wordmark Brief (v0)**

* Wordmark: “Cortex Synapse” in a geometric sans (e.g., Inter, semi‑bold)
* Mark: minimal neuron/synapse node connected to a bracket `{}` motif
* Dark/light variants; 24px and 48px app icon cuts

**Asset folders**

```
/docs/brand/
  logo/
    cortex-synapse-wordmark.svg
    cortex-synapse-mark.svg
  colors.md
  typography.md
```

---

## 5) Repo Layout & Names (proposed)

```
repo-root/
  apps/
    studio/   # Cortex Synapse Studio (React + Tauri)
    api/      # Cortex Synapse API (FastAPI)
  services/
    engine/   # Cortex Synapse Engine (Celery workers)
  infra/
    docker-compose.yml  # Postgres + pgvector + Redis
  docs/
    brand/    # identity assets
  .editorconfig
  .gitignore
  README.md
```

---

## 6) Top‑Level README Template

````md
# Cortex Synapse

> Where code, notes, and insight connect.

## Packages

- **Cortex Synapse Studio** — Desktop app (Tauri + React): `apps/studio`
- **Cortex Synapse API** — FastAPI backend: `apps/api`
- **Cortex Synapse Engine** — Workers (Celery + Redis): `services/engine`
- **Cortex Synapse Infra** — Docker Compose: `infra/`

## Quickstart

```bash
# Infra
cd infra && docker compose up -d

# API
cd apps/api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
alembic upgrade head
uvicorn app.main:app --reload

# Engine (Celery)
celery -A app.celery_app.celery_app worker --loglevel=INFO

# Studio
cd apps/studio
yarn && yarn tauri dev
````

## Configuration

* Env prefix: `CS_` (see component READMEs)
* Default LLM: **Ollama** (`llama3:8b-instruct`, `nomic-embed-text`)
* Optional: OpenAI‑compatible endpoints

## License

MIT (placeholder)

````

---

## 7) Component READMEs (templates)

### 7.1 Studio (apps/studio/README.md)
```md
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
````

## Config

* `CS_API_URL` (default `http://127.0.0.1:8000`)
* `CS_OLLAMA_URL` (default `http://localhost:11434`)

## Pages

* Snippets, Documents (upload), RAG (chat), Settings

````

### 7.2 API (apps/api/README.md)
```md
# Cortex Synapse API

FastAPI service for snippets, notebooks, documents, and RAG.

## Endpoints (MVP)
- `GET /snippets` — list
- `POST /snippets` — create
- `POST /documents` — upload + queue ingestion
- `POST /rag/answer` — answer a query (uses top‑k chunks)

## Env
```ini
DATABASE_URL=postgresql+psycopg2://dev:dev@localhost:5432/cortex_synapse
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3:8b-instruct
EMBED_PROVIDER=ollama
EMBED_MODEL=nomic-embed-text
EMBED_DIM=768
REDIS_URL=redis://localhost:6379/0
````

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

````

### 7.3 Engine (services/engine/README.md)
```md
# Cortex Synapse Engine

Celery workers for document ingestion, chunking, embeddings, and indexing.

## Run
```bash
celery -A app.celery_app.celery_app worker --loglevel=INFO -Q default -n engine@%h
````

## Tasks

* `ingest_document_task(doc_id)` — extract, chunk, embed, upsert

## Notes

* Default embeddings via Ollama (`nomic-embed-text`, dim=768)
* Switch to OpenAI‑compatible by setting EMBED vars

````

### 7.4 Infra (infra/README.md)
```md
# Cortex Synapse Infra

Local infrastructure via Docker Compose.

## Services
- **db:** Postgres + pgvector
- **redis:** Redis 7

## Commands
```bash
cd infra && docker compose up -d
docker compose ps
docker compose logs -f db redis
````

````

---

## 8) CONTRIBUTING.md (template)
```md
# Contributing to Cortex Synapse

Thanks for taking time to contribute!

## Setup
- Yarn for web
- Python 3.11+ for API/Engine
- Docker for Postgres/Redis

## Branching
- `feat/…`, `fix/…`, `docs/…`, `chore/…`

## PRs
- Link to issue, describe changes, add screenshots for UI

## Code Style
- TypeScript strict mode
- Ruff/black (if added) for Python

## DCO/CLA
- TBD
````

---

## 9) SECURITY.md (template)

```md
# Security Policy

- Report vulnerabilities to security@cortexsynapse.dev (placeholder)
- Avoid filing public issues for sensitive reports
- We aim to acknowledge within 72 hours
```

---

## 10) CHANGELOG.md (keep a simple convention)

```md
# Changelog

All notable changes to this repo will be documented here.

## [Unreleased]

- …

## [0.1.0] - YYYY-MM-DD

### Added

- Initial brand system and docs templates
```

---

## 11) Brand Usage Do’s & Don’ts (quick)

* **Do**: Use full name on first mention (Cortex Synapse Studio), then short name.
* **Do**: Use `CS_` prefix for env vars.
* **Do**: Keep product descriptors capitalized (Studio, API, Engine).
* **Don’t**: Mix “AI” into the product names; reserve for company/tagline.
* **Don’t**: Use internal abbreviations (CSS) in user-facing copy.

---

## 12) Next Steps (Docs & Branding)

1. Approve/adjust names + identifiers
2. Generate the component READMEs in the repo
3. Add `/docs/brand` with wordmark + palette tokens
4. Update Tauri window title to **Cortex Synapse Studio**
5. Update `.env.example` keys to `CS_` where appropriate

> If you want, I can turn these templates into actual files in your repo or a PR-ready patchset.

# Cortex Synapse — Brand System & Docs Pack (v0.1)

> Working brand + doc templates to keep naming, messaging, and repos consistent while we build. Edit freely.

---

## 1) Brand Architecture

* **Master brand:** **Cortex Synapse**
  **Tagline (optional):** *Where code, notes, and insight connect.*

* **Products / Components**

    * **Cortex Synapse Studio** — Desktop IDE (Tauri + React). Short: **“Synapse Studio”**.
    * **Cortex Synapse API** — HTTP backend (FastAPI). Short: **“Synapse API”**.
    * **Cortex Synapse Engine** — Workers for ingestion/embeddings/RAG (Celery + Redis). Short: **“Synapse Engine”**.
    * **Cortex Synapse Infra** — Local infra (Docker Compose, provisioning). Short: **“Synapse Infra”**.
    * *(Future)* **Cortex Synapse Cloud**, **SDK**, **CLI**.

**Why this works**

* Clarity across docs and UI
* Scales to new surfaces (Cloud/SDK/CLI) without renaming
* Short names read cleanly in UI after first mention

---

## 2) Naming Conventions & Identifiers

### Display Names & Abbreviations

* First mention in docs: **Cortex Synapse Studio** (the “Studio”) → later as **Synapse Studio**.
* Similar for **Cortex Synapse API** → **Synapse API**; **Cortex Synapse Engine** → **Synapse Engine**.

### Package / Binary / Bundle IDs

* **Tauri bundle identifier:** `ai.cortexsynapse.studio`
* **Docker images:**

    * `ghcr.io/<your-gh-user>/cortex-synapse-api`
    * `ghcr.io/<your-gh-user>/cortex-synapse-engine`
* **Python package/module (API):** `cortex_synapse_api`
* **NPM scope (optional):** `@cortex-synapse/*`
* **CLI name (reserved):** `synapse` (fallback: `csyn`)

### Env Var Prefix & Config

* Use **`CS_`** prefix for cross-cutting config; keep component-specific ones too.
* Examples:

    * `CS_ENV=dev|staging|prod`
    * `CS_API_URL=http://127.0.0.1:8000`
    * `CS_OLLAMA_URL=http://localhost:11434`
    * `CS_EMBED_MODEL=nomic-embed-text`

### Filesystem & Code Style

* **Folders:** kebab-case for web (`/apps/studio/src/pages/rag-panel/`), snake\_case for Python.
* **Branch names:** `feat/…`, `fix/…`, `docs/…`, `chore/…`.
* **Issue labels:** `area:studio`, `area:api`, `area:engine`, `area:infra`, `type:bug`, `type:feat`.

---

## 3) Messaging Cheatsheet

### One-liner (master)

**Cortex Synapse** unifies code, notes, and AI assistants—so you learn faster and build without context switching.

### Product blurbs (short)

* **Synapse Studio:** A focused desktop IDE where your code, notes, and AI study tools live side-by-side.
* **Synapse API:** A fast, typed HTTP backend that powers snippets, notebooks, and retrieval-augmented answers.
* **Synapse Engine:** Background workers for document ingestion, embeddings, and hybrid search at low latency.
* **Synapse Infra:** Local-first Postgres/pgvector + Redis in Docker to get you productive in minutes.

### 30‑sec pitch (master)

Cortex Synapse is a local‑first IDE that blends Monaco editing, notebooks, and an AI study copilot. It ingests your
PDFs/notes, builds embeddings in Postgres + pgvector, and serves grounded answers—without leaving your editor. Start
with Ollama locally, add OpenAI‑compatible providers as needed, and scale to Celery workers when your knowledge base
grows.

---

## 4) Visual Identity (starter — iterate later)

> Placeholder choices to get moving; we can refine once UI is stable.

* **Primary color:** Indigo 600 (#4F46E5)
* **Accent:** Cyan 500 (#06B6D4)
* **Neutral:** Zinc/Slate scale (Tailwind)
* **Typography:**

    * UI: Inter (system fallbacks)
    * Mono: JetBrains Mono
* **Iconography:** Lucide (outline, consistent stroke)

**Logo/Wordmark Brief (v0)**

* Wordmark: “Cortex Synapse” in a geometric sans (e.g., Inter, semi‑bold)
* Mark: minimal neuron/synapse node connected to a bracket `{}` motif
* Dark/light variants; 24px and 48px app icon cuts

**Asset folders**

```
/docs/brand/
  logo/
    cortex-synapse-wordmark.svg
    cortex-synapse-mark.svg
  colors.md
  typography.md
```

---

## 5) Repo Layout & Names (proposed)

```
repo-root/
  apps/
    studio/   # Cortex Synapse Studio (React + Tauri)
    api/      # Cortex Synapse API (FastAPI)
  services/
    engine/   # Cortex Synapse Engine (Celery workers)
  infra/
    docker-compose.yml  # Postgres + pgvector + Redis
  docs/
    brand/    # identity assets
  .editorconfig
  .gitignore
  README.md
```

---

## 6) Top‑Level README Template

````md
# Cortex Synapse

> Where code, notes, and insight connect.

## Packages

- **Cortex Synapse Studio** — Desktop app (Tauri + React): `apps/studio`
- **Cortex Synapse API** — FastAPI backend: `apps/api`
- **Cortex Synapse Engine** — Workers (Celery + Redis): `services/engine`
- **Cortex Synapse Infra** — Docker Compose: `infra/`

## Quickstart

```bash
# Infra
cd infra && docker compose up -d

# API
cd apps/api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
alembic upgrade head
uvicorn app.main:app --reload

# Engine (Celery)
celery -A app.celery_app.celery_app worker --loglevel=INFO

# Studio
cd apps/studio
yarn && yarn tauri dev
````

## Configuration

* Env prefix: `CS_` (see component READMEs)
* Default LLM: **Ollama** (`llama3:8b-instruct`, `nomic-embed-text`)
* Optional: OpenAI‑compatible endpoints

## License

MIT (placeholder)

````

---

## 7) Component READMEs (templates)

### 7.1 Studio (apps/studio/README.md)
```md
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
````

## Config

* `CS_API_URL` (default `http://127.0.0.1:8000`)
* `CS_OLLAMA_URL` (default `http://localhost:11434`)

## Pages

* Snippets, Documents (upload), RAG (chat), Settings

````

### 7.2 API (apps/api/README.md)
```md
# Cortex Synapse API

FastAPI service for snippets, notebooks, documents, and RAG.

## Endpoints (MVP)
- `GET /snippets` — list
- `POST /snippets` — create
- `POST /documents` — upload + queue ingestion
- `POST /rag/answer` — answer a query (uses top‑k chunks)

## Env
```ini
DATABASE_URL=postgresql+psycopg2://dev:dev@localhost:5432/cortex_synapse
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3:8b-instruct
EMBED_PROVIDER=ollama
EMBED_MODEL=nomic-embed-text
EMBED_DIM=768
REDIS_URL=redis://localhost:6379/0
````

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

````

### 7.3 Engine (services/engine/README.md)
```md
# Cortex Synapse Engine

Celery workers for document ingestion, chunking, embeddings, and indexing.

## Run
```bash
celery -A app.celery_app.celery_app worker --loglevel=INFO -Q default -n engine@%h
````

## Tasks

* `ingest_document_task(doc_id)` — extract, chunk, embed, upsert

## Notes

* Default embeddings via Ollama (`nomic-embed-text`, dim=768)
* Switch to OpenAI‑compatible by setting EMBED vars

````

### 7.4 Infra (infra/README.md)
```md
# Cortex Synapse Infra

Local infrastructure via Docker Compose.

## Services
- **db:** Postgres + pgvector
- **redis:** Redis 7

## Commands
```bash
cd infra && docker compose up -d
docker compose ps
docker compose logs -f db redis
````

````

---

## 8) CONTRIBUTING.md (template)
```md
# Contributing to Cortex Synapse

Thanks for taking time to contribute!

## Setup
- Yarn for web
- Python 3.11+ for API/Engine
- Docker for Postgres/Redis

## Branching
- `feat/…`, `fix/…`, `docs/…`, `chore/…`

## PRs
- Link to issue, describe changes, add screenshots for UI

## Code Style
- TypeScript strict mode
- Ruff/black (if added) for Python

## DCO/CLA
- TBD
````

---

## 9) SECURITY.md (template)

```md
# Security Policy

- Report vulnerabilities to security@cortexsynapse.dev (placeholder)
- Avoid filing public issues for sensitive reports
- We aim to acknowledge within 72 hours
```

---

## 10) CHANGELOG.md (keep a simple convention)

```md
# Changelog

All notable changes to this repo will be documented here.

## [Unreleased]

- …

## [0.1.0] - YYYY-MM-DD

### Added

- Initial brand system and docs templates
```

---

## 11) Brand Usage Do’s & Don’ts (quick)

* **Do**: Use full name on first mention (Cortex Synapse Studio), then short name.
* **Do**: Use `CS_` prefix for env vars.
* **Do**: Keep product descriptors capitalized (Studio, API, Engine).
* **Don’t**: Mix “AI” into the product names; reserve for company/tagline.
* **Don’t**: Use internal abbreviations (CSS) in user-facing copy.

---

## 12) Next Steps (Docs & Branding)

1. Approve/adjust names + identifiers
2. Generate the component READMEs in the repo
3. Add `/docs/brand` with wordmark + palette tokens
4. Update Tauri window title to **Cortex Synapse Studio**
5. Update `.env.example` keys to `CS_` where appropriate

> If you want, I can turn these templates into actual files in your repo or a PR-ready patchset.
