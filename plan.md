# Decision Intelligence Assistant — Implementation Plan

## Goal

Build a full-stack AI knowledge assistant that:
- Answers customer support queries using RAG (retrieval-augmented generation)
- Predicts ticket priority using both a trained ML classifier and an LLM
- Shows a four-way comparison: RAG answer, non-RAG answer, ML prediction, LLM zero-shot prediction
- Runs entirely with `docker compose up --build`

---

## Technology Choices

| Component | Choice | Reason |
|-----------|--------|--------|
| LLM (primary) | `gemma4:31b-cloud` via Ollama | Already running locally; free, no API key |
| LLM (fallback) | `gemini-2.5-flash` via Gemini API | Activates when Ollama is unreachable or times out |
| Embeddings | `nomic-embed-text` via Ollama | Consistent embedding endpoint, no extra service |
| Vector DB | Qdrant | Already in docker-compose; grpc + REST; persistent |
| Backend | FastAPI + Python | Required by brief |
| Frontend | React + Vite + TypeScript | Required by brief |
| ML baseline | scikit-learn | Standard, interpretable, fast |
| Dep manager | `uv` | Required by CLAUDE.md |

**LLM client**: Ollama Python SDK (`ollama`) primary, `google-generativeai` SDK fallback.  
`GOOGLE_API_KEY` is **optional** — if unset, the fallback is disabled and Ollama errors surface directly.

---

## Architecture

```
User Browser
    │
    ▼
[React frontend :3000]
    │  HTTP → http://backend:8000
    ▼
[FastAPI backend :8000]
    ├── /api/query        → RAG + non-RAG answers (parallel)
    ├── /api/priority/ml  → ML classifier prediction
    ├── /api/priority/llm → LLM zero-shot prediction
    └── /api/health       → health check
    │
    ├──→ [Qdrant :6333]        vector search
    └──→ [Ollama :11434]       LLM generation + embeddings
```

All services share a Docker network `dia-net`. Qdrant and Ollama data live in named volumes.

---

## Dataset

**Customer Support on Twitter** (Kaggle: `thoughtvector/customer-support-on-twitter`)

Each tweet = one support ticket. Filter to inbound customer messages only (not brand replies).

### Priority Labeling Function (Weak Supervision)

Label a ticket as `urgent` if **any** of the following:
1. Contains urgency keywords: `refund`, `broken`, `cancel`, `outage`, `down`, `not working`, `urgent`, `asap`, `help`, `fix`
2. Has 2+ exclamation marks
3. Has ALL-CAPS ratio > 0.3

Otherwise `normal`. This rule will be documented in the notebook. The ML model will partly learn this rule — that is expected and acknowledged.

---

## Project Structure

```
project-root/
├── plan.md                          ← this file
├── CLAUDE.md
├── docker-compose.yml               ← extended with backend, frontend, ollama-init
├── .env.example
├── .gitignore
├── .dockerignore
├── .pre-commit-config.yaml
├── README.md
│
├── notebooks/
│   └── notebook.ipynb               ← EDA, labeling, features, ML training, analysis
│
├── data/
│   ├── raw/                         ← read-only, downloaded dataset
│   └── processed/                   ← cleaned CSV, feature CSV, chunked tickets
│
├── models/
│   └── priority_classifier_v1.joblib
│
├── logs/                            ← JSONL query logs
│
├── backend/
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── app/
│   │   ├── main.py                  ← FastAPI app init, router registration
│   │   ├── core/
│   │   │   └── settings.py          ← env vars, Ollama URL, Qdrant URL, model names
│   │   ├── routers/
│   │   │   ├── query.py             ← /api/query (RAG + non-RAG)
│   │   │   ├── priority.py          ← /api/priority/ml + /api/priority/llm
│   │   │   └── health.py            ← /api/health
│   │   ├── schemas/
│   │   │   ├── query.py             ← QueryRequest, QueryResponse
│   │   │   └── priority.py          ← PriorityRequest, PriorityResponse
│   │   ├── services/
│   │   │   ├── retriever.py         ← Qdrant search, embed query, return top-k tickets
│   │   │   ├── llm_client.py        ← LLM abstraction: tries Ollama first, falls back to Gemini
│   │   │   ├── generator.py         ← RAG answer + non-RAG answer via llm_client
│   │   │   ├── ml_predictor.py      ← load joblib model, extract features, predict
│   │   │   ├── llm_predictor.py     ← zero-shot priority prediction via llm_client
│   │   │   └── query_logger.py      ← write JSONL log per query (records which provider was used)
│   │   └── utils/
│   │       └── feature_extractor.py ← text length, keyword flags, caps ratio, etc.
│   ├── scripts/
│   │   └── ingest.py                ← embed tickets, upsert to Qdrant (run once)
│   └── tests/
│       ├── test_health.py
│       ├── test_retriever.py
│       ├── test_ml_predictor.py
│       └── test_feature_extractor.py
│
└── frontend/
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    ├── Dockerfile
    ├── nginx.conf
    ├── .dockerignore
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── api/
        │   └── client.ts            ← typed fetch wrappers for backend endpoints
        ├── components/
        │   ├── QueryInput.tsx        ← text box + submit
        │   ├── AnswerPanel.tsx       ← RAG answer + non-RAG answer side by side
        │   ├── SourcePanel.tsx       ← retrieved tickets + similarity scores
        │   └── ComparisonTable.tsx  ← ML vs LLM: label, confidence, latency, cost
        └── types/
            └── index.ts             ← TypeScript interfaces matching backend schemas
```

---

## Build Sequence

### Phase 0 — Scaffold (do first, no logic yet)

| Step | Task | Verify |
|------|------|--------|
| 0.1 | Create `.gitignore`, `.dockerignore`, `.env.example` at root | files exist |
| 0.2 | Create `.pre-commit-config.yaml` (black, isort, flake8, mypy, gitleaks) | `pre-commit run --all-files` passes |
| 0.3 | Create placeholder dirs: `data/raw/`, `data/processed/`, `models/`, `logs/`, `notebooks/` | `find . -type d` shows them |
| 0.4 | Scaffold `backend/` layout (pyproject.toml, Dockerfile, app skeleton) | `uv sync` succeeds |
| 0.5 | Scaffold `frontend/` layout (package.json, vite, tsconfig, Dockerfile, nginx.conf) | `npm install` succeeds |
| 0.6 | Extend `docker-compose.yml` — add backend, frontend, ollama-init services | `docker compose config` validates |

### Phase 1 — Data & Notebook

| Step | Task | Verify |
|------|------|--------|
| 1.1 | Download dataset to `data/raw/` | CSV file present |
| 1.2 | EDA cell: shape, nulls, sample rows, class distribution | notebook runs to end |
| 1.3 | Implement and apply labeling function | `urgent`/`normal` column in processed CSV |
| 1.4 | Feature engineering: 7+ features documented with justification | feature CSV in `data/processed/` |
| 1.5 | Train 3 ML models (LogReg, RandomForest, GradientBoosting) with 5-fold CV | metrics table with MAE, F1, AUC per model |
| 1.6 | Select best model, save as `models/priority_classifier_v1.joblib` | file exists, loads cleanly |
| 1.7 | Generate embeddings for all tickets, upsert to Qdrant via `scripts/ingest.py` | Qdrant collection has N points |

### Phase 2 — Backend Services

| Step | Task | Verify |
|------|------|--------|
| 2.1 | `core/settings.py` — load all env vars, fail-fast guards; `GOOGLE_API_KEY` optional | import succeeds |
| 2.2 | `services/llm_client.py` — `generate(prompt)` tries Ollama; on `ConnectionError`/timeout falls back to Gemini 2.5 Flash; logs which provider was used | fallback triggers when Ollama is stopped |
| 2.3 | `services/retriever.py` — embed query via Ollama, search Qdrant top-5 | returns list of tickets with scores |
| 2.4 | `services/generator.py` — RAG prompt (system + context + query) + non-RAG prompt via `llm_client` | both return string answers |
| 2.5 | `services/ml_predictor.py` — load model with `lru_cache`, extract features, predict | returns label + confidence + latency_ms |
| 2.6 | `services/llm_predictor.py` — zero-shot priority prompt via `llm_client` | returns `urgent`/`normal` + latency_ms + provider used |
| 2.7 | `services/query_logger.py` — JSONL log with all fields including `llm_provider` field | log file grows on each call |
| 2.8 | `routers/query.py` — `POST /api/query` runs retrieval + generation in parallel | 200 with full response |
| 2.9 | `routers/priority.py` — `POST /api/priority/ml` and `POST /api/priority/llm` | 200 each |
| 2.10 | `routers/health.py` — `GET /api/health` checks Qdrant + Ollama reachability; reports Gemini fallback availability | 200 `{"status": "ok"}` |
| 2.11 | Write tests (80% coverage target); include test for fallback path with mocked Ollama failure | `uv run pytest` passes |

### Phase 3 — Frontend

| Step | Task | Verify |
|------|------|--------|
| 3.1 | `api/client.ts` — typed fetch wrappers matching backend schemas | TypeScript compiles |
| 3.2 | `QueryInput.tsx` — text box + submit, shows loading state | renders and submits |
| 3.3 | `AnswerPanel.tsx` — RAG and non-RAG answers side by side | displays both answers |
| 3.4 | `SourcePanel.tsx` — list retrieved tickets with similarity score badge | shows top-k tickets |
| 3.5 | `ComparisonTable.tsx` — 2×3 table: ML row + LLM row × accuracy/latency/cost | values populated |
| 3.6 | Wire everything in `App.tsx` | full flow works end to end in browser |

### Phase 4 — Docker & Integration

| Step | Task | Verify |
|------|------|--------|
| 4.1 | Backend `Dockerfile` — multi-stage, non-root user | `docker build` succeeds |
| 4.2 | Frontend `Dockerfile` — build stage + nginx serve stage | `docker build` succeeds |
| 4.3 | `docker-compose.yml` — all 5 services: ollama, qdrant, ollama-init, backend, frontend | `docker compose config` passes |
| 4.4 | `docker compose up --build` — full stack starts | frontend loads in browser |
| 4.5 | Run ingest inside container, verify Qdrant collection populated | query returns real tickets |
| 4.6 | End-to-end smoke test — submit a query, all 4 outputs appear | no errors in browser or logs |

### Phase 5 — Polish

| Step | Task | Verify |
|------|------|--------|
| 5.1 | Write `README.md` (all required sections per CLAUDE.md §21) | a stranger can run it |
| 5.2 | Pre-review checklist (CLAUDE.md §24) — all boxes ticked | checklist complete |

---

## Key Design Decisions

**Primary LLM: `gemma4:31b-cloud` via Ollama**  
Running locally means zero API cost, no rate limits, and no outbound calls. Already pulled. The tradeoff: if Ollama is down or the container hasn't started yet, all LLM calls fail — hence the fallback.

**Fallback LLM: `gemini-2.5-flash` via Gemini API**  
`services/llm_client.py` is the single place that knows about both providers. It catches `ollama.ResponseError`, `ConnectionError`, and `TimeoutError`; logs a `WARNING` with the error; then retries the same prompt against Gemini. If `GOOGLE_API_KEY` is not set, the fallback is skipped and the original error is re-raised. No other service knows about this — `generator.py` and `llm_predictor.py` just call `llm_client.generate()`. The log entry records `"llm_provider": "ollama"` or `"llm_provider": "gemini-fallback"` so you can track how often the fallback fires.

**Why Qdrant over Chroma?**  
Qdrant runs as a separate container with a REST + gRPC API, which matches the multi-service architecture requirement exactly. It also provides similarity scores natively and has a web UI on port 6333.

**Embedding model: `nomic-embed-text` via Ollama**  
Same endpoint as the LLM — no extra service. 768-dim, ~274 MB, purpose-built for retrieval. Not currently pulled; the `ollama-init` Docker service pulls it on first `docker compose up`. Chosen over `mxbai-embed-large` (670 MB, marginal uplift on short tweet-length text) and `all-minilm` (too small for semantic nuance).

**ML labeling function (weak supervision)**  
The labeling rule is documented here and in the notebook. The model will reproduce it with high accuracy — that is not a bug, it is the expected outcome of weak supervision. The honest answer to "is it learning urgency?" is: partly. It is learning our proxy signal for urgency.

**Cost accounting**  
Ollama calls: `$0.00` (self-hosted). Gemini fallback calls: `$0.00075 / 1M input tokens` for Flash 2.5 (free tier covers 1,500 req/day). The log records which provider was used, so cost is always attributable. ML model: `$0.00`.

---

## Environment Variables (`.env.example`)

```
# Ollama (primary LLM + embeddings)
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_LLM_MODEL=gemma4:31b-cloud
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_TIMEOUT_SECONDS=30

# Gemini (fallback LLM — optional; leave blank to disable fallback)
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_LLM_MODEL=gemini-2.5-flash

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=support_tickets
QDRANT_TOP_K=5

# Backend
LOG_LEVEL=INFO
LOG_DIR=/app/logs

# Frontend (build-time)
VITE_API_BASE_URL=http://localhost:8000
```

All variables are loaded **once** in `backend/app/core/settings.py` using a Pydantic `BaseSettings` model with `lru_cache`. No `os.getenv()` call appears anywhere else in the codebase — every service and router imports from `core.settings`. This is the single source of truth for all configuration.

```python
# core/settings.py — sketch
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ollama_base_url: str
    ollama_llm_model: str
    ollama_embed_model: str
    ollama_timeout_seconds: int = 30
    google_api_key: str | None = None   # optional — fallback disabled if None
    gemini_llm_model: str = "gemini-2.5-flash"
    qdrant_host: str
    qdrant_port: int = 6333
    qdrant_collection: str
    qdrant_top_k: int = 5
    log_level: str = "INFO"
    log_dir: str = "/app/logs"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

---

## Four-Way Comparison — What Gets Measured

For every query the frontend shows:

| Output | Source | Accuracy | Latency | Cost |
|--------|--------|----------|---------|------|
| RAG answer | Ollama (gemma4) + Qdrant context | qualitative | wall-clock ms | $0.00 |
| Non-RAG answer | Ollama (gemma4) alone | qualitative | wall-clock ms | $0.00 |
| ML priority | scikit-learn classifier | F1 on test set | ~2ms | $0.00 |
| LLM priority | Ollama zero-shot | measured on test set | ~800ms+ | $0.00 |

The comparison section answers: **at 10,000 tickets/hour, which priority predictor would you deploy?**

---

## Logging Schema (JSONL, one object per query)

```json
{
  "timestamp": "2026-04-21T10:00:00Z",
  "query": "my account is broken",
  "retrieved_tickets": [{"text": "...", "score": 0.87}, ...],
  "rag_answer": "...",
  "non_rag_answer": "...",
  "ml_prediction": {"label": "urgent", "confidence": 0.91, "latency_ms": 2},
  "llm_prediction": {"label": "urgent", "latency_ms": 1240},
  "errors": []
}
```
