"""Ingest router — embed and upsert pre-processed thread chunks into Qdrant.

POST /api/ingest        — process one batch of chunks, return immediately (202).
GET  /api/ingest/status — live state: index progress, cursor, batch count.

Workflow:
  1. Run notebooks/knowledge_preprocessing.ipynb → data/knowledge/thread_chunks.csv
  2. POST /api/ingest  (first call loads the CSV, subsequent calls advance cursor)
  3. GET  /api/ingest/status  (poll until status="ready" or "done")
  4. Test retrieval in the app, then POST again for the next batch.
"""

import logging
from functools import lru_cache

import ollama
from fastapi import APIRouter, BackgroundTasks, HTTPException
from qdrant_client import QdrantClient

from app.core.settings import get_settings
from app.rag import chunker, embedder, loader, store
from app.schemas.ingest import (
    IngestRequest,
    IngestResponse,
    IngestStatusResponse,
    ThreadChunk,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ingest", tags=["ingest"])

# Pipeline state — persists for the lifetime of the process.
_state: dict = {
    "status": "idle",       # idle | loading | running | ready | done | error
    "all_chunks": None,     # list[ThreadChunk] — loaded once from CSV
    "cursor": 0,            # index of next chunk to process
    "batches_completed": 0,
    "threads_embedded": 0,  # cumulative across all batches
    "qdrant_count": 0,
    "error": None,
}


@lru_cache
def _qdrant() -> QdrantClient:
    """Cached Qdrant client — one instance per process."""
    s = get_settings()
    return QdrantClient(host=s.qdrant_host, port=s.qdrant_port)


@lru_cache
def _ollama() -> ollama.Client:
    """Cached Ollama client — one instance per process."""
    return ollama.Client(host=get_settings().ollama_base_url)


def _run_batch(request: IngestRequest) -> None:
    """Background task: load CSV if needed, then embed + upsert one batch.

    Updates _state throughout so GET /api/ingest/status reflects live progress.
    """
    settings = get_settings()

    try:
        # ── Phase 1: load knowledge CSV (only on first call or after reset) ──
        if _state["all_chunks"] is None:
            _state["status"] = "loading"
            all_chunks: list[ThreadChunk] = loader.build_index(request.knowledge_csv_path)
            _state["all_chunks"] = all_chunks
            logger.info("Knowledge CSV loaded: %d chunks", len(all_chunks))

        all_chunks = _state["all_chunks"]
        cursor: int = _state["cursor"]

        # ── Phase 2: slice next batch ──────────────────────────────────────
        batch = loader.get_batch(all_chunks, cursor, request.batch_size)

        if not batch:
            _state["status"] = "done"
            logger.info("All %d chunks processed — ingest complete.", cursor)
            return

        _state["status"] = "running"
        batch_num = _state["batches_completed"] + 1
        logger.info(
            "Batch %d: chunks[%d:%d] (%d chunks)",
            batch_num, cursor, cursor + len(batch), len(batch),
        )

        # ── Phase 3: validate → embed → upsert ────────────────────────────
        chunks = chunker.build_chunks(batch)
        if not chunks:
            logger.warning("Batch %d: all chunks empty after validation — advancing cursor.", batch_num)
            _state["cursor"] += len(batch)
            _state["batches_completed"] += 1
            _state["status"] = "ready"
            return

        probe = _ollama().embed(model=settings.ollama_embed_model, input=chunks[0].text)
        dims = len(probe.embeddings[0])
        store.ensure_collection(_qdrant(), settings.qdrant_collection, dims)

        vectors = embedder.embed_chunks(chunks, _ollama(), settings.ollama_embed_model)
        count = store.upsert_chunks(_qdrant(), settings.qdrant_collection, chunks, vectors)

        # ── Phase 4: advance state ─────────────────────────────────────────
        _state["cursor"] += len(batch)
        _state["batches_completed"] += 1
        _state["threads_embedded"] += len(chunks)
        _state["qdrant_count"] = count
        _state["status"] = "ready" if _state["cursor"] < len(all_chunks) else "done"

        logger.info(
            "Batch %d done — %d embedded, %d total in Qdrant, cursor=%d/%d",
            batch_num, len(chunks), count, _state["cursor"], len(all_chunks),
        )

    except Exception as exc:
        logger.exception("Ingest batch failed: %s", exc)
        _state["status"] = "error"
        _state["error"] = str(exc)


@router.post("", response_model=IngestResponse, status_code=202)
def start_batch(request: IngestRequest, background_tasks: BackgroundTasks) -> IngestResponse:
    """Accept one ingest batch and start it in the background.

    Returns HTTP 202 immediately. Poll GET /api/ingest/status for progress.
    Call again (without reset=True) to continue with the next batch.

    Args:
        request: IngestRequest with batch_size, csv path, and optional reset flag.
        background_tasks: FastAPI BackgroundTasks injected by the framework.

    Returns:
        IngestResponse confirming batch acceptance.

    Raises:
        HTTPException 409: If a batch or CSV load is already in progress.
    """
    if _state["status"] in ("loading", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"Pipeline is busy (status={_state['status']}). "
                   "Poll GET /api/ingest/status and retry when ready.",
        )

    if request.reset:
        _state.update({
            "status": "idle",
            "all_chunks": None,
            "cursor": 0,
            "batches_completed": 0,
            "threads_embedded": 0,
            "qdrant_count": 0,
            "error": None,
        })
        logger.info("Ingest state reset.")

    total = len(_state["all_chunks"]) if _state["all_chunks"] is not None else -1
    cursor = _state["cursor"]
    batch_num = _state["batches_completed"] + 1
    roots_remaining = max(total - cursor - request.batch_size, 0) if total >= 0 else -1

    _state["error"] = None
    background_tasks.add_task(_run_batch, request)

    return IngestResponse(
        message=(
            f"Batch {batch_num} started (chunks {cursor}–{cursor + request.batch_size}). "
            "Poll GET /api/ingest/status for progress."
        ),
        batch_number=batch_num,
        roots_remaining=roots_remaining,
    )


@router.get("/status", response_model=IngestStatusResponse)
def get_status() -> IngestStatusResponse:
    """Return the live state of the ingest pipeline.

    Returns:
        IngestStatusResponse with status, cursor, batch count, and Qdrant point count.
    """
    settings = get_settings()
    qdrant_count = _state["qdrant_count"]

    if _state["status"] in ("idle", "ready", "done"):
        qdrant_count = store.get_collection_count(_qdrant(), settings.qdrant_collection)

    return IngestStatusResponse(
        status=_state["status"],
        total_roots=len(_state["all_chunks"]) if _state["all_chunks"] is not None else 0,
        cursor=_state["cursor"],
        batches_completed=_state["batches_completed"],
        threads_embedded=_state["threads_embedded"],
        qdrant_count=qdrant_count,
        error=_state["error"],
    )
