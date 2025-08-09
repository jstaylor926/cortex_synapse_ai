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
