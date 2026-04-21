"""Pydantic schemas for /api/priority endpoints."""

from typing import Literal

from pydantic import BaseModel, Field


class PriorityRequest(BaseModel):
    """Incoming ticket text to classify."""

    text: str = Field(min_length=1, max_length=2000)


class PriorityResponse(BaseModel):
    """Priority prediction result."""

    label: Literal["urgent", "normal"]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    latency_ms: float
    provider: str
    cost_usd: float = 0.0
