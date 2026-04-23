"""Answer generation service — RAG and non-RAG via llm_client.

Runs both generation calls concurrently using asyncio.gather so total
latency equals the slower of the two, not the sum.
"""

import asyncio
import logging

from app.core.settings import RAG_NO_DATA_RESPONSE, get_settings
from app.rag.prompts import (
    NON_RAG_SYSTEM_PROMPT,
    NON_RAG_TEMPLATE,
    RAG_SYSTEM_PROMPT,
    RAG_TEMPLATE,
)
from app.schemas.query import RetrievedTicket
from app.services.llm_client import LLMResult, generate
from app.utils.prompt_guard import sanitize_user_input

logger = logging.getLogger(__name__)


async def generate_both(
    query: str,
    tickets: list[RetrievedTicket],
    score_threshold: float | None = None,
) -> tuple[str, str]:
    """Generate RAG and non-RAG answers concurrently.

    Args:
        query: Raw user query from the API (sanitized inside this function).
        tickets: Retrieved context tickets (may be empty).
        score_threshold: Override the settings default score threshold for this request.

    Returns:
        Tuple of (rag_answer, non_rag_answer) strings.
    """
    threshold = score_threshold if score_threshold is not None else get_settings().rag_score_threshold

    rag_task = asyncio.get_event_loop().run_in_executor(
        None, _generate_rag, query, tickets, threshold
    )
    non_rag_task = asyncio.get_event_loop().run_in_executor(
        None, _generate_non_rag, query
    )

    rag_result, non_rag_result = await asyncio.gather(rag_task, non_rag_task)
    return rag_result.text, non_rag_result.text


def _generate_rag(query: str, tickets: list[RetrievedTicket], threshold: float) -> LLMResult:
    """Build RAG prompt and call the LLM.

    Args:
        query: User query — sanitized here before prompt interpolation.
        tickets: Retrieved context tickets.
        threshold: Minimum score to proceed with LLM generation.

    Returns:
        LLMResult from llm_client.
    """
    safe_query = sanitize_user_input(query)

    # Quality gate: skip LLM if the best retrieved chunk is below the threshold.
    best_score = max((t.score for t in tickets), default=0.0)
    if best_score < threshold:
        logger.info(
            "RAG skipped — best score below threshold",
            extra={"best_score": best_score, "threshold": threshold},
        )
        return LLMResult(text=RAG_NO_DATA_RESPONSE, provider="rag-no-match", latency_ms=0.0)

    context_parts = [
        f"Ticket {i + 1}: {t.text}" for i, t in enumerate(tickets)
    ]
    context = "\n".join(context_parts)
    prompt = RAG_TEMPLATE.format(context=context, query=safe_query)
    result = generate(prompt, system=RAG_SYSTEM_PROMPT)
    logger.info(
        "RAG generation complete",
        extra={"provider": result.provider, "latency_ms": round(result.latency_ms)},
    )
    return result


def _generate_non_rag(query: str) -> LLMResult:
    """Build non-RAG prompt and call the LLM.

    Args:
        query: User query — sanitized here before prompt interpolation.

    Returns:
        LLMResult from llm_client.
    """
    safe_query = sanitize_user_input(query)
    prompt = NON_RAG_TEMPLATE.format(query=safe_query)
    # Hard token cap enforces the 1-2 sentence rule regardless of model compliance.
    result = generate(prompt, system=NON_RAG_SYSTEM_PROMPT, max_tokens=200)
    logger.info(
        "Non-RAG generation complete",
        extra={"provider": result.provider, "latency_ms": round(result.latency_ms)},
    )
    return result
