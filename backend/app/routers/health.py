"""Health check endpoint.

Returns the reachability status of Ollama, Qdrant, and Gemini fallback.
"""

import logging

import httpx
from fastapi import APIRouter
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["health"])


class ServiceStatus(BaseModel):
    """Reachability status of a single downstream service."""

    reachable: bool
    detail: str


class HealthResponse(BaseModel):
    """Overall health of the application and its dependencies."""

    status: str
    ollama: ServiceStatus
    qdrant: ServiceStatus
    gemini_fallback_configured: bool


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check reachability of Ollama and Qdrant and report Gemini configuration.

    Returns:
        HealthResponse with per-service status and overall status string.
    """
    settings = get_settings()

    # Check Ollama
    ollama_status = await _check_ollama(settings.ollama_base_url)

    # Check Qdrant
    qdrant_status = _check_qdrant(settings.qdrant_host, settings.qdrant_port)

    overall = "ok" if ollama_status.reachable and qdrant_status.reachable else "degraded"

    logger.info(
        "Health check",
        extra={
            "status": overall,
            "ollama": ollama_status.reachable,
            "qdrant": qdrant_status.reachable,
        },
    )

    return HealthResponse(
        status=overall,
        ollama=ollama_status,
        qdrant=qdrant_status,
        gemini_fallback_configured=settings.gemini_fallback_enabled,
    )


async def _check_ollama(base_url: str) -> ServiceStatus:
    """Probe Ollama's /api/tags endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{base_url}/api/tags")
            resp.raise_for_status()
        return ServiceStatus(reachable=True, detail="ok")
    except Exception as exc:
        logger.warning("Ollama health check failed: %s", exc)
        return ServiceStatus(reachable=False, detail=str(exc))


def _check_qdrant(host: str, port: int) -> ServiceStatus:
    """Probe Qdrant by listing collections."""
    try:
        client = QdrantClient(host=host, port=port, timeout=5)
        client.get_collections()
        return ServiceStatus(reachable=True, detail="ok")
    except (UnexpectedResponse, Exception) as exc:
        logger.warning("Qdrant health check failed: %s", exc)
        return ServiceStatus(reachable=False, detail=str(exc))
