"""ML classifier priority predictor.

Loads the trained scikit-learn model once (lru_cache) and predicts
ticket priority from engineered features. Model artifact is mounted
into the container at runtime.
"""

import logging
import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from app.core.settings import get_settings
from app.schemas.priority import PriorityResponse
from app.utils.feature_extractor import extract_features

logger = logging.getLogger(__name__)


def _model_path() -> Path:
    """Resolve model path — settings first, then relative fallback for local dev."""
    configured = Path(get_settings().model_dir) / "priority_classifier_v1.joblib"
    if configured.exists():
        return configured
    # When running outside Docker, models/ sits two levels above backend/
    local = Path(__file__).parent.parent.parent.parent / "models" / "priority_classifier_v1.joblib"
    return local


class MLPredictorError(Exception):
    """Raised when model loading or prediction fails."""


@lru_cache(maxsize=1)
def _load_model() -> Pipeline:
    """Load the ML model from disk, cached for the process lifetime.

    Returns:
        Loaded scikit-learn Pipeline.

    Raises:
        MLPredictorError: If the model file does not exist or cannot be loaded.
    """
    path = _model_path()
    if not path.exists():
        raise MLPredictorError(f"Model file not found: {path}")
    model: Pipeline = joblib.load(path)
    logger.info("ML model loaded", extra={"path": str(path)})
    return model


async def predict(text: str) -> PriorityResponse:
    """Predict ticket priority using the trained classifier.

    Args:
        text: Raw ticket text.

    Returns:
        PriorityResponse with label, confidence, latency_ms, and provider.

    Raises:
        MLPredictorError: If model loading or inference fails.
    """
    model = _load_model()
    features = extract_features(text)
    feature_array = np.array(list(features.values())).reshape(1, -1)

    start = time.perf_counter()
    label_idx: int = model.predict(feature_array)[0]
    proba: np.ndarray = model.predict_proba(feature_array)[0]
    latency_ms = (time.perf_counter() - start) * 1000

    label = "urgent" if label_idx == 1 else "normal"
    confidence = float(proba[label_idx])

    logger.info(
        "ML prediction",
        extra={"label": label, "confidence": round(confidence, 3), "latency_ms": round(latency_ms, 2)},
    )
    return PriorityResponse(
        label=label,
        confidence=confidence,
        latency_ms=latency_ms,
        provider="sklearn",
        cost_usd=0.0,
    )
