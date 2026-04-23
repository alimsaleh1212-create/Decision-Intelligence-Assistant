"""Observability API router — record, logs, and metrics endpoints."""

from fastapi import APIRouter

from app.schemas.observability import (
    LogsResponse,
    MetricsResponse,
    ObservationRecord,
    RecordRequest,
)
from app.services import obs_logger

router = APIRouter(prefix="/api/observability", tags=["observability"])


@router.post("/record", response_model=ObservationRecord, status_code=201)
def record(req: RecordRequest) -> ObservationRecord:
    """Append one observation record to the log."""
    return obs_logger.record_observation(req)


@router.get("/logs", response_model=LogsResponse)
def logs(limit: int = 100) -> LogsResponse:
    """Return the most recent observation records."""
    return obs_logger.get_logs(limit=limit)


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Return aggregate metrics across all stored observations."""
    return obs_logger.get_metrics()
