# Corext Synapse

A privacy-first desktop app that blends:

* **Notebook + IDE** (markdown + runnable code cells + Monaco editor)
* **Personal code/notes knowledge base** (snippets, docs, PDFs)
* **AI assistant** with **RAG** grounded in your own notes/snippets
* **Workflow backbone** using DAIA-style **objects, actions, gates** (so it scales cleanly to teams/SaaS later)

# Architecture (local-first → SaaS-ready)

* **UI**: React + TypeScript (Vite), Monaco, shadcn/ui
* **Desktop**: **Tauri** (lightweight, global hotkey, native fs), browser fallback for dev
* **API**: FastAPI (Python) + SQLAlchemy + Alembic
* **DB**: PostgreSQL + **pgvector** (or SQLite + FAISS for simple local mode)
* **Embeddings/LLM**: Local by default (**Ollama**), pluggable cloud providers later
* **Vector/RAG**: Hybrid retrieval (BM25 + vector), optional reranker
* **Background jobs**: RQ/Arq (embeddings, imports, long runs)
* **FS**: `~/.dev-assistant` (attachments, caches)

# DAIA-Style Domain Model (generic)

**Objects**

* **Project** – top-level container (courses, repos, study tracks)
* **Notebook** – ordered **Cell**s (markdown | code)
* **Snippet** – reusable code + explanation + tags
* **Document** – longform notes (markdown) or imported PDF/DOCX
* **Task** – lightweight TODOs linked to any object
* **Pack** – publishable bundle (snippets + docs + templates)
* **Chunk** / **Embedding** – retrieval units for RAG
* **Job** – background operations (embed, import, run)

**Actions (Action Types)**

* `CreateNotebook`, `AddCell`, `RunCell`, `ExportNotebook`
* `CaptureSnippet`, `RefactorSnippet`, `GenerateTests`, `Docstringify`
* `ImportDocument`, `IndexDocument`, `ReindexProject`
* `AskRag`, `Summarize`, `GenerateFlashcards`
* `CreatePack`, `PromotePack`, `PublishPack`

**Workflows & Gates (simple but extensible)**

* **Notebook**: `Draft → Ready → Published`
* **Snippet**: `Draft → Reviewed → Approved`
* **Pack**: `Assemble → Validate (lint/tests) → Publish`
* Gates are just **preconditions + actions** (e.g., “Pass linting & tests before Publish”).

# What’s inside

* **infra/**

  * `docker-compose.yml` → Postgres (pgvector) + Ollama
  * `.env.example` → DB + LLM settings
* **app-api/** (FastAPI)

  * `main.py` → `/health`, `/import`, `/search/hybrid`, `/rag/answer`
  * `db.py`, `models.py`, `settings.py`
  * `llm/` → provider shim (`base.py`, `ollama.py`, `provider.py`)
  * `requirements.txt`
  * `alembic.ini`, `db_migrations/` → **pgvector** extension, tsvector triggers, IVFFlat index
* **app-web/** (Vite React + **Monaco**)

  * Minimal notebook editor + RAG panel (import sample, ask, citations in text)
* **README.md** with quickstart

# Quickstart (copy/paste)

```bash
# 1) Infra
cd infra
cp .env.example .env
docker compose up -d

# 2) API
cd ../app-api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn main:app --reload --port 8000
```

Pull models for Ollama (in a new terminal):

```bash
curl -s http://localhost:11434/api/pull -d '{"model":"llama3"}'
curl -s http://localhost:11434/api/pull -d '{"model":"nomic-embed-text"}'
```

Then run the web app:

```bash
cd ../app-web
pnpm i || npm i
pnpm dev || npm run dev
```

Open `http://localhost:5173` → click **Import Sample**, ask a question in **RAG**, watch it answer using your local
models.

# Notes & Next steps

* Embedding dimension is set to **768** for `nomic-embed-text`; if you change embedding models, update the migration (
  vector size) accordingly.
* Hybrid search = **BM25-ish (tsvector)** + **vector cosine** with simple score fusion.
* This is a minimal spine. Next sprints we can add:

  * **Snippets/Notebooks** as first-class DB objects with actions
  * **Job runner** for async imports/embeds
  * **Diff/refactor** actions, docstring/test-gen
  * **Flashcards + spaced repetition**

Want me to extend this with a background worker + real PDF/DOCX import pipeline next?

Awesome—here’s a tidy, weekend-sized project that lets you kick the tires on both **Hugging Face** and **OpenAI’s new
gpt-oss** models, with clean upgrade paths if you want to go deeper later.

# 5) Optional stretch goals

* **UI**: Wrap the API with a 1-file Flask/HTMX or Streamlit app.
* **Quantization**: Try 4-bit to fit tighter memory budgets (see model card for quant tips). ([Hugging Face][2])
* **Compare models**: Keep your baseline (e.g., Llama 3.x) and print latency/quality diffs.
* **Fine-tune**: Do a tiny LoRA on a small instruction set later.

# 6) Run the mini server (5 min)

```python
# app.py
from flask import Flask, request, jsonify
from rag import answer

app = Flask(__name__)


@app.post("/chat")
def chat():
  q = request.json.get("message", "")
  return jsonify({"answer": answer(q)})


if __name__ == "__main__":
  app.run(port=7860, debug=True)
```

```bash
python app.py
# POST {"message": "..."} to http://localhost:7860/chat
```

# Useful references (skim these once)

* OpenAI blog & model card for **gpt-oss** (what it is, licenses, perf, and formats). ([OpenAI][3])
* Hugging Face model page for downloads/quantization notes. ([Hugging Face][2])
* OpenAI cookbook snippet for **Transformers + harmony** formatting. ([OpenAI Cookbook][4])
* Hardware/availability backgrounders & coverage if you’re curious. ([IT Pro][1])

---

If you want, I can turn this into a ready-to-run repo (with a `requirements.txt`, Makefile, and a tiny web UI).

[1]: https://www.itpro.com/technology/artificial-intelligence/openai-open-weight-ai-models?utm_source=chatgpt.com "Everything you need to know about OpenAI's new open weight AI models, including price, performance, and where you can access them"

[2]: https://huggingface.co/openai/gpt-oss-120b?utm_source=chatgpt.com "openai/gpt-oss-120b"

[3]: https://openai.com/index/introducing-gpt-oss/?utm_source=chatgpt.com "Introducing gpt-oss"

[4]: https://cookbook.openai.com/articles/gpt-oss/run-transformers?utm_source=chatgpt.com "How to run gpt-oss with Transformers"

[5]: https://openai.com/index/gpt-oss-model-card/?utm_source=chatgpt.com "gpt-oss-120b & gpt-oss-20b Model Card"
