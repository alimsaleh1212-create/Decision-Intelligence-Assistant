"""Priority prediction endpoints — ML classifier and LLM zero-shot.

Exposes two prediction routes so the frontend can compare the two
approaches on accuracy, latency, and cost.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.priority import PriorityRequest, PriorityResponse
from app.services import llm_predictor, ml_predictor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/priority", tags=["priority"])


@router.post("/ml", response_model=PriorityResponse)
async def predict_ml(request: PriorityRequest) -> PriorityResponse:
    """Predict ticket priority using the trained scikit-learn classifier.

    Args:
        request: Validated priority request payload.

    Returns:
        PriorityResponse with label, confidence, latency_ms, and cost.

    Raises:
        HTTPException 500: If the ML predictor raises an unexpected error.
    """
    try:
        return await ml_predictor.predict(request.text)
    except Exception as exc:
        logger.error("ML prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="ML prediction failed") from exc


@router.post("/llm", response_model=PriorityResponse)
async def predict_llm(request: PriorityRequest) -> PriorityResponse:
    """Predict ticket priority using LLM zero-shot classification.

    Args:
        request: Validated priority request payload.

    Returns:
        PriorityResponse with label, latency_ms, provider, and cost.

    Raises:
        HTTPException 500: If the LLM predictor raises an unexpected error.
    """
    try:
        return await llm_predictor.predict(request.text)
    except Exception as exc:
        logger.error("LLM priority prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="LLM prediction failed") from exc
