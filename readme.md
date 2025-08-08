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
uvicorn main:app --reload --port 8000
```

3) (Optional) Pull Ollama models:

```bash
curl -s http://localhost:11434/api/pull -d '{"model":"llama3"}'
curl -s http://localhost:11434/api/pull -d '{"model":"nomic-embed-text"}'
```

4) Start web app:

```bash
cd ../app-web
pnpm i || npm i
pnpm dev || npm run dev
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
