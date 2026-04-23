"""Observability logger — append and read structured observation records.

Writes one JSON line per request to observations.jsonl in the shared
log directory. Computes aggregate metrics on read — no database needed.
"""

import logging
from pathlib import Path

from app.schemas.observability import (
    LogsResponse,
    MetricsResponse,
    ObservationRecord,
    RecordRequest,
)

logger = logging.getLogger(__name__)

_LOG_DIR = Path("/app/logs")
_OBS_FILE = _LOG_DIR / "observations.jsonl"


def record_observation(req: RecordRequest) -> ObservationRecord:
    """Create an ObservationRecord and append it to the JSONL log.

    Args:
        req: The record request payload from the frontend.

    Returns:
        The persisted ObservationRecord with id and timestamp set.
    """
    obs = ObservationRecord(**req.model_dump())
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with _OBS_FILE.open("a", encoding="utf-8") as f:
            f.write(obs.model_dump_json() + "\n")
    except OSError as exc:
        logger.warning("Could not write observation log: %s", exc)
    return obs


def _read_all() -> list[ObservationRecord]:
    """Parse every line in the JSONL file into an ObservationRecord.

    Returns:
        List of records in chronological order; skips malformed lines.
    """
    if not _OBS_FILE.exists():
        return []
    records: list[ObservationRecord] = []
    try:
        with _OBS_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(ObservationRecord.model_validate_json(line))
                except Exception as exc:
                    logger.warning("Skipping malformed observation line: %s", exc)
    except OSError as exc:
        logger.warning("Could not read observation log: %s", exc)
    return records


def get_logs(limit: int = 100) -> LogsResponse:
    """Return the most recent observations in reverse-chronological order.

    Args:
        limit: Maximum number of records to return.

    Returns:
        LogsResponse with records and total count.
    """
    all_records = _read_all()
    total = len(all_records)
    recent = list(reversed(all_records[-limit:]))
    return LogsResponse(records=recent, total=total)


def get_metrics() -> MetricsResponse:
    """Compute aggregate metrics from all stored observations.

    Returns:
        MetricsResponse with averages and totals.
    """
    records = _read_all()
    n = len(records)
    if n == 0:
        return MetricsResponse(
            total_queries=0,
            avg_llm_latency_ms=0.0,
            avg_ml_latency_ms=0.0,
            total_cost_usd=0.0,
            urgent_rate=0.0,
            ml_urgent_rate=0.0,
        )

    avg_llm_latency = sum(r.llm.latency_ms for r in records) / n
    avg_ml_latency = sum(r.ml.latency_ms for r in records) / n
    total_cost = sum(r.llm.cost_usd + r.ml.cost_usd for r in records)
    urgent_rate = sum(1 for r in records if r.llm.label == "urgent") / n
    ml_urgent_rate = sum(1 for r in records if r.ml.label == "urgent") / n

    return MetricsResponse(
        total_queries=n,
        avg_llm_latency_ms=round(avg_llm_latency, 1),
        avg_ml_latency_ms=round(avg_ml_latency, 1),
        total_cost_usd=round(total_cost, 6),
        urgent_rate=round(urgent_rate, 3),
        ml_urgent_rate=round(ml_urgent_rate, 3),
    )
