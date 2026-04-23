"""Structured JSONL query logger.

Appends one JSON object per query to a rotating log file.
Each entry records everything needed for offline evaluation.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from app.core.settings import get_settings
from app.schemas.query import RetrievedTicket

logger = logging.getLogger(__name__)


async def log_query(
    query: str,
    tickets: list[RetrievedTicket],
    rag_answer: str,
    non_rag_answer: str,
    errors: list[str] | None = None,
    llm_provider: str = "unknown",
) -> None:
    """Append a query record to the JSONL log file.

    Args:
        query: The user's raw query.
        tickets: Retrieved tickets with similarity scores.
        rag_answer: Answer generated with RAG context.
        non_rag_answer: Answer generated without context.
        errors: List of error strings encountered during processing.
        llm_provider: Which LLM provider was used (ollama / gemini-fallback).
    """
    settings = get_settings()
    log_dir = Path(settings.log_dir)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "llm_provider": llm_provider,
        "retrieved_tickets": [
            {"text": t.text, "score": t.score} for t in tickets
        ],
        "rag_answer": rag_answer,
        "non_rag_answer": non_rag_answer,
        "errors": errors or [],
    }

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "queries.jsonl"
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error("Failed to write query log: %s", exc)
