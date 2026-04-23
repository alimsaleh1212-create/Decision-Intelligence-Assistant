"""Ingest router — populate Qdrant with conversation-thread chunks.

POST /api/ingest        — process one batch of threads, return immediately (202).
GET  /api/ingest/status — live state of the pipeline (index progress + batch cursor).

Design:
  The adjacency index (tweet_by_id, children, all_roots) is built once on the first POST
  and cached in module-level _state. Every subsequent POST reuses the cached index and
  advances the cursor by batch_size roots. The caller can test retrieval after each batch,
  then trigger the next batch with another POST.
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
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ingest", tags=["ingest"])

# Pipeline state — persists for the lifetime of the process.
_state: dict = {
    "status": "idle",       # idle | indexing | running | ready | done | error
    # Index — built once from the CSV, reused for all batches
    "tweet_by_id": None,
    "children": None,
    "all_roots": None,      # list[int] of all root tweet_ids
    # Batch progress
    "cursor": 0,            # next root index to process
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
    """Background task: build index if needed, then process one batch.

    Updates _state throughout so GET /api/ingest/status reflects live progress.
    """
    settings = get_settings()

    try:
        # ── Phase 1: build index (only on first call or after reset) ──────────
        if _state["tweet_by_id"] is None:
            _state["status"] = "indexing"
            tweet_by_id, children, all_roots = loader.build_index(request.raw_csv_path)
            _state["tweet_by_id"] = tweet_by_id
            _state["children"] = children
            _state["all_roots"] = all_roots
            logger.info("Index ready: %d roots", len(all_roots))

        all_roots: list[int] = _state["all_roots"]
        cursor: int = _state["cursor"]

        # ── Phase 2: slice the next batch from the root list ──────────────────
        batch_roots = all_roots[cursor : cursor + request.batch_size]

        if not batch_roots:
            _state["status"] = "done"
            logger.info("All %d roots processed — ingest complete.", cursor)
            return

        _state["status"] = "running"
        batch_num = _state["batches_completed"] + 1
        logger.info(
            "Batch %d: processing roots[%d:%d] (%d threads)",
            batch_num,
            cursor,
            cursor + len(batch_roots),
            len(batch_roots),
        )

        # ── Phase 3: reconstruct → chunk → embed → upsert ────────────────────
        threads = loader.reconstruct_batch(
            batch_roots, _state["tweet_by_id"], _state["children"]
        )
        chunks = chunker.build_chunks(threads)

        if not chunks:
            logger.warning("Batch %d produced 0 chunks — advancing cursor.", batch_num)
            _state["cursor"] += len(batch_roots)
            _state["batches_completed"] += 1
            _state["status"] = "ready"
            return

        # Probe embedding dims once per process (collection setup)
        probe = _ollama().embed(model=settings.ollama_embed_model, input=chunks[0].text)
        dims = len(probe.embeddings[0])
        store.ensure_collection(_qdrant(), settings.qdrant_collection, dims)

        vectors = embedder.embed_chunks(chunks, _ollama(), settings.ollama_embed_model)
        count = store.upsert_chunks(_qdrant(), settings.qdrant_collection, chunks, vectors)

        # ── Phase 4: update state ─────────────────────────────────────────────
        _state["cursor"] += len(batch_roots)
        _state["batches_completed"] += 1
        _state["threads_embedded"] += len(chunks)
        _state["qdrant_count"] = count
        _state["status"] = "ready" if _state["cursor"] < len(all_roots) else "done"

        logger.info(
            "Batch %d done: %d threads embedded, %d total in Qdrant, cursor=%d/%d",
            batch_num,
            len(chunks),
            count,
            _state["cursor"],
            len(all_roots),
        )

    except Exception as exc:
        logger.exception("Ingest batch failed: %s", exc)
        _state["status"] = "error"
        _state["error"] = str(exc)


@router.post("", response_model=IngestResponse, status_code=202)
def start_batch(request: IngestRequest, background_tasks: BackgroundTasks) -> IngestResponse:
    """Accept one ingest batch and start it in the background.

    Returns immediately with HTTP 202. Poll GET /api/ingest/status for progress.
    Call again (without reset=True) to process the next batch after the first completes.

    Args:
        request: IngestRequest with batch_size, optional csv path, and reset flag.
        background_tasks: FastAPI BackgroundTasks injected by the framework.

    Returns:
        IngestResponse with batch number and remaining root count.

    Raises:
        HTTPException 409: If indexing or a batch is already running.
    """
    if _state["status"] in ("indexing", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"Pipeline is busy (status={_state['status']}). "
                   "Poll GET /api/ingest/status and retry when ready.",
        )

    if request.reset:
        _state.update({
            "status": "idle",
            "tweet_by_id": None,
            "children": None,
            "all_roots": None,
            "cursor": 0,
            "batches_completed": 0,
            "threads_embedded": 0,
            "qdrant_count": 0,
            "error": None,
        })
        logger.info("Ingest state reset — index will be rebuilt on next batch.")

    total_roots = len(_state["all_roots"]) if _state["all_roots"] is not None else -1
    cursor = _state["cursor"]
    batch_num = _state["batches_completed"] + 1
    roots_remaining = max(total_roots - cursor - request.batch_size, 0) if total_roots >= 0 else -1

    _state["error"] = None
    background_tasks.add_task(_run_batch, request)

    logger.info(
        "Batch %d accepted: batch_size=%d cursor=%d",
        batch_num,
        request.batch_size,
        cursor,
    )

    return IngestResponse(
        message=(
            f"Batch {batch_num} started (roots {cursor}–{cursor + request.batch_size}). "
            "Poll GET /api/ingest/status for progress."
        ),
        batch_number=batch_num,
        roots_remaining=roots_remaining,
    )


@router.get("/status", response_model=IngestStatusResponse)
def get_status() -> IngestStatusResponse:
    """Return the live state of the ingest pipeline.

    Returns:
        IngestStatusResponse with status, cursor, batch counter, and Qdrant count.
    """
    settings = get_settings()
    qdrant_count = _state["qdrant_count"]

    # Fetch live count from Qdrant when pipeline is idle/ready (not mid-run)
    if _state["status"] in ("idle", "ready", "done"):
        qdrant_count = store.get_collection_count(_qdrant(), settings.qdrant_collection)

    return IngestStatusResponse(
        status=_state["status"],
        total_roots=len(_state["all_roots"]) if _state["all_roots"] is not None else 0,
        cursor=_state["cursor"],
        batches_completed=_state["batches_completed"],
        threads_embedded=_state["threads_embedded"],
        qdrant_count=qdrant_count,
        error=_state["error"],
    )
