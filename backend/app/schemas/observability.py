"""Pydantic schemas for the observability API.

Request/response models for recording and retrieving observations.
"""

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class PredictorSnapshot(BaseModel):
    """Snapshot of one predictor's output for a single request."""

    label: str
    confidence: float | None
    latency_ms: float
    provider: str
    cost_usd: float


class RecordRequest(BaseModel):
    """Payload sent by the frontend after each query."""

    query: str
    brand: str | None = None
    rag_score_threshold: float | None = None
    rag_answer: str
    non_rag_answer: str
    retrieved_tickets_count: int
    ml: PredictorSnapshot
    llm: PredictorSnapshot


class ObservationRecord(BaseModel):
    """A single persisted observation with auto-generated id and timestamp."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    query: str
    brand: str | None = None
    rag_score_threshold: float | None = None
    rag_answer: str
    non_rag_answer: str
    retrieved_tickets_count: int
    ml: PredictorSnapshot
    llm: PredictorSnapshot


class LogsResponse(BaseModel):
    """Paginated list of observation records."""

    records: list[ObservationRecord]
    total: int


class MetricsResponse(BaseModel):
    """Aggregate metrics computed across all stored observations."""

    total_queries: int
    avg_llm_latency_ms: float
    avg_ml_latency_ms: float
    total_cost_usd: float
    urgent_rate: float
    ml_urgent_rate: float
