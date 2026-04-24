# Decision Intelligence Assistant (DIA)

A full-stack AI application that analyses customer support tickets using Retrieval-Augmented Generation (RAG), an ML classifier, and an LLM — all running in parallel and compared side-by-side in real time.

Submit a support ticket query and DIA returns four simultaneous outputs: a RAG answer grounded in historical tickets with brand citations, a general non-RAG answer, an ML priority prediction (urgent/normal), and an LLM zero-shot priority prediction. An Observability tab logs every request with metrics and a trace drawer for full detail.

---

## Architecture

```
Browser (React + Vite)
        │
        ▼
FastAPI Backend (port 8000)
  ├── /api/query          → RAG retrieval + Gemini generation (parallel)
  ├── /api/priority/ml    → scikit-learn classifier
  ├── /api/priority/llm   → Gemini 2.5 Flash (Ollama fallback)
  ├── /api/ingest         → CSV → embed → Qdrant upsert
  ├── /api/observability  → JSONL request log + metrics
  └── /api/health
        │
        ├── Qdrant (port 6333)   — vector store for embedded tickets
        └── Ollama (port 11434)  — nomic-embed-text embeddings + LLM fallback
```

**LLM priority:** Gemini 2.5 Flash (primary) → Ollama local model (fallback)  
**Embeddings:** always via Ollama `nomic-embed-text`  
**RAG store:** Qdrant with cosine similarity, configurable score threshold  
**ML model:** scikit-learn classifier trained on telecom support tickets  

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose v2
- [uv](https://github.com/astral-sh/uv) (Python package manager, for local scripts)
- A Google Gemini API key (free tier works) — optional but recommended

---

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd project3_decision_intelligence_assistant

# 2. Create .env from the example
cp .env.example .env

# 3. Add your Gemini API key (optional — Ollama fallback works without it)
#    Edit .env and set:
#    GOOGLE_API_KEY=your-key-here

# 4. Pull the Ollama models (one-time, ~2 GB)
docker volume create ollama_data
docker compose run --rm ollama ollama pull nomic-embed-text
docker compose run --rm ollama ollama pull gemma4:31b-cloud

# 5. Start the full stack
docker compose up -d
```

The app is ready when `docker compose ps` shows all services healthy.

---

## How to Run

```bash
# Start everything
docker compose up -d

# Stop everything
docker compose down

# Rebuild after backend code changes
docker compose build backend && docker compose up -d backend

# Stream backend logs
docker logs -f dia-backend

# Stream ingest-specific logs only
docker logs -f dia-backend 2>&1 | grep --line-buffered -iE "batch|embed|upsert|ingest|error"
```

| Service  | URL                          |
|----------|------------------------------|
| Frontend | http://localhost:3000        |
| Backend  | http://localhost:8000        |
| API docs | http://localhost:8000/docs   |
| Qdrant   | http://localhost:6333/dashboard |

---

## Ingest Knowledge Base

Before querying, embed the support ticket dataset into Qdrant:

```bash
# Check current ingest state
python backend/app/rag/ingest_all.py --status

# Run full ingest with live progress
python backend/app/rag/ingest_all.py

# Install deps first (one-time)
cd backend && uv sync && cd ..
```

The script drives `POST /api/ingest` in a loop, showing per-batch progress, cursor position, embeddings count, and Qdrant point count. It handles Ctrl-C gracefully and resumes from where it left off on re-run.

Source data: `data/knowledge/thread_chunks.csv` — pre-processed telecom support ticket threads.

---

## Environment Variables

All variables live in `.env` (see `.env.example` for a template with placeholder values).

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | No | — | Gemini API key. If blank, falls back to Ollama for LLM calls |
| `OLLAMA_BASE_URL` | Yes | `http://ollama:11434` | Ollama service URL (use `http://localhost:11434` for local dev) |
| `OLLAMA_LLM_MODEL` | Yes | `gemma4:31b-cloud` | Ollama model for LLM fallback |
| `OLLAMA_EMBED_MODEL` | Yes | `nomic-embed-text` | Embedding model (used for all RAG operations) |
| `OLLAMA_TIMEOUT_SECONDS` | No | `30` | Ollama request timeout |
| `QDRANT_HOST` | Yes | `qdrant` | Qdrant hostname |
| `QDRANT_PORT` | No | `6333` | Qdrant HTTP port |
| `QDRANT_COLLECTION` | No | `support_tickets` | Qdrant collection name |
| `QDRANT_TOP_K` | No | `3` | Number of tickets to retrieve per query |
| `RAG_SCORE_THRESHOLD` | No | `0.6` | Minimum cosine similarity to proceed with LLM generation |
| `LOG_LEVEL` | No | `INFO` | Python log level (`DEBUG`, `INFO`, `WARNING`) |

---

## Project Structure

```
.
├── backend/                  FastAPI service
│   ├── app/
│   │   ├── routers/          API endpoints (one file per resource group)
│   │   ├── schemas/          Pydantic request/response models
│   │   ├── services/         Business logic (LLM, ML, obs logger)
│   │   ├── rag/              Embedder, retriever, loader, chunker, store, prompts
│   │   ├── utils/            Prompt injection guard, ingest runner
│   │   └── core/             Settings, shared constants
│   ├── pyproject.toml
│   └── Dockerfile
├── frontend/                 React + Vite SPA
│   ├── src/
│   │   ├── components/       QueryInput, AnswerPanel, ComparisonTable, ObservabilityPage
│   │   ├── api/              Typed API client
│   │   └── types/            TypeScript interfaces mirroring backend schemas
│   └── Dockerfile
├── data/
│   └── knowledge/            thread_chunks.csv (read-only raw data)
├── models/                   Trained ML model artefacts (.joblib)
├── notebooks/                Exploration notebooks (not production code)
├── scripts/                  (empty — reserved for future shell scripts)
├── docker-compose.yml
├── .env.example
└── create_presentation.py    Generates DIA_presentation.pptx
```

---

## Key Features

**RAG Answers** — retrieves the most similar historical support tickets from Qdrant (cosine similarity), passes them to Gemini with brand citations. If no ticket scores above the threshold, returns a clear "no data" response rather than hallucinating.

**Non-RAG Answers** — parallel general-knowledge answer from Gemini (max 4 sentences). Runs concurrently with RAG so total latency equals the slower of the two, not the sum.

**ML Priority Predictor** — scikit-learn classifier, sub-millisecond latency, zero cost per call. Engineered features: ticket length, keyword signals, punctuation patterns.

**LLM Priority Predictor** — Gemini zero-shot, reads tone and implicit urgency. Gemini 2.5 Flash is primary; Ollama is the automatic fallback. Cost tracked per call from usage metadata.

**Observability** — every query is recorded to `observations.jsonl`. The Observability tab shows metric cards (total queries, avg latency, total cost, urgent rate), a logs grid, and a slide-in trace drawer with full RAG/non-RAG/ML/LLM detail per request.

**Security** — prompt injection prevention via `sanitize_user_input()` and `<user_input>` delimiters in all templates. Pydantic validation at every API boundary. LRU-cached model clients (loaded once per process).

---

## Deployment

Both services have Dockerfiles and are orchestrated by `docker-compose.yml`. To deploy:

1. Point `VITE_API_BASE_URL` in `docker-compose.yml` to your public backend URL before building the frontend image.
2. Set `GOOGLE_API_KEY` in your host environment or secrets manager.
3. Run `docker compose up -d` on the target host.

The frontend is served by nginx on port 3000. The backend exposes port 8000. Both communicate with Qdrant and Ollama by Docker service name over the `dia-net` bridge network.
