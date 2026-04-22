"""Ingest router — populate Qdrant with conversation-thread chunks.

POST /api/ingest  — accepts an IngestRequest, starts a background job, returns immediately.
GET  /api/ingest/status — returns the live state of the current or last-run job.
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

# Module-level job state — single ingest job at a time.
_job_state: dict = {
    "status": "idle",
    "threads_total": 0,
    "threads_embedded": 0,
    "qdrant_count": 0,
    "error": None,
}


@lru_cache
def _get_qdrant_client() -> QdrantClient:
    """Return a cached Qdrant client (one per process)."""
    settings = get_settings()
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


@lru_cache
def _get_ollama_client() -> ollama.Client:
    """Return a cached Ollama client (one per process)."""
    settings = get_settings()
    return ollama.Client(host=settings.ollama_base_url)


def _run_ingest(request: IngestRequest) -> None:
    """Background task: load → chunk → embed → upsert.

    Updates _job_state throughout so GET /api/ingest/status reflects live progress.
    """
    global _job_state
    settings = get_settings()
    qdrant = _get_qdrant_client()
    ollama_client = _get_ollama_client()

    try:
        _job_state.update({"status": "running", "error": None})

        # 1. Load threads from raw CSV
        threads = loader.load_threads(request.raw_csv_path, limit=request.limit)
        _job_state["threads_total"] = len(threads)

        if not threads:
            _job_state.update({"status": "done", "qdrant_count": 0})
            logger.warning("No threads found in %s", request.raw_csv_path)
            return

        # 2. Convert to ThreadChunk objects
        chunks = chunker.build_chunks(threads)

        # 3. Probe embedding dimensions for collection setup
        probe = ollama_client.embed(model=settings.ollama_embed_model, input=chunks[0].text)
        dims = len(probe.embeddings[0])
        store.ensure_collection(qdrant, settings.qdrant_collection, dims)

        # 4. Embed all chunks
        vectors = embedder.embed_chunks(chunks, ollama_client, settings.ollama_embed_model)
        _job_state["threads_embedded"] = len(chunks)

        # 5. Upsert into Qdrant
        count = store.upsert_chunks(qdrant, settings.qdrant_collection, chunks, vectors)
        _job_state.update({"status": "done", "qdrant_count": count})

        logger.info("Ingest job complete: %d threads → %d points.", len(chunks), count)

    except Exception as exc:
        logger.exception("Ingest job failed: %s", exc)
        _job_state.update({"status": "error", "error": str(exc)})


@router.post("", response_model=IngestResponse, status_code=202)
def start_ingest(request: IngestRequest, background_tasks: BackgroundTasks) -> IngestResponse:
    """Accept an ingest request and start the background job.

    Returns immediately with HTTP 202. Poll GET /api/ingest/status for progress.

    Args:
        request: IngestRequest containing limit and optional CSV path override.
        background_tasks: FastAPI BackgroundTasks injected by the framework.

    Returns:
        IngestResponse confirming the job was accepted.

    Raises:
        HTTPException 409: If an ingest job is already running.
    """
    if _job_state["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="An ingest job is already running. Check GET /api/ingest/status.",
        )

    settings = get_settings()
    # Resolve the actual thread count to report — loader counts after loading
    # so we report the limit as the expected upper bound.
    threads_requested = request.limit if request.limit > 0 else -1

    _job_state.update(
        {
            "status": "running",
            "threads_total": 0,
            "threads_embedded": 0,
            "qdrant_count": 0,
            "error": None,
        }
    )

    background_tasks.add_task(_run_ingest, request)

    logger.info(
        "Ingest job accepted: csv=%s limit=%d embed_model=%s",
        request.raw_csv_path,
        request.limit,
        settings.ollama_embed_model,
    )

    return IngestResponse(
        message="Ingest job started. Poll GET /api/ingest/status for progress.",
        threads_requested=threads_requested,
    )


@router.get("/status", response_model=IngestStatusResponse)
def get_ingest_status() -> IngestStatusResponse:
    """Return the live state of the current or last-run ingest job.

    Returns:
        IngestStatusResponse with status, progress counters, and any error.
    """
    settings = get_settings()
    qdrant_count = _job_state["qdrant_count"]

    # If idle, fetch the live count from Qdrant so the status is always fresh.
    if _job_state["status"] == "idle":
        qdrant_count = store.get_collection_count(
            _get_qdrant_client(), settings.qdrant_collection
        )

    return IngestStatusResponse(
        status=_job_state["status"],
        threads_total=_job_state["threads_total"],
        threads_embedded=_job_state["threads_embedded"],
        qdrant_count=qdrant_count,
        error=_job_state["error"],
    )
