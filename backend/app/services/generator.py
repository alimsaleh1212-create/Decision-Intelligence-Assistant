"""Answer generation service — RAG and non-RAG via llm_client.

Runs both generation calls concurrently using asyncio.gather so total
latency equals the slower of the two, not the sum.
"""

import asyncio
import logging

from app.core.settings import RAG_NO_DATA_RESPONSE, get_settings
from app.schemas.query import RetrievedTicket
from app.services.llm_client import LLMResult, generate
from app.utils.prompt_guard import sanitize_user_input

logger = logging.getLogger(__name__)

# RAG: model must stay strictly within the retrieved context.
_RAG_SYSTEM_PROMPT = (
    "You are a customer support assistant analysing historical support tickets. "
    "Answer the user's question using ONLY the information in the retrieved support tickets provided. "
    "If the retrieved tickets do not contain sufficient information to answer, respond with: "
    "'The retrieved support tickets do not contain enough information to answer this question.' "
    "Do not use any knowledge outside of the provided context. "
    "Do not fabricate details, invent resolutions, or extrapolate beyond what the tickets show."
)

# Non-RAG: model answers from general knowledge and must be transparent about it.
_NON_RAG_SYSTEM_PROMPT = (
    "You are a general-purpose customer support assistant. "
    "You are answering from your general training knowledge — you have no access to this customer's "
    "account, order history, or any retrieved tickets. "
    "Answer clearly based on general customer support best practices. "
    "If you cannot answer without account-specific information, say so explicitly. "
    "Do not fabricate specific details or account-level information."
)

# Retrieved context injected before the user query; model instructed not to escape it.
_RAG_TEMPLATE = """\
Answer the user's question using ONLY the retrieved support tickets below. \
Do not use any knowledge outside of this context.

<retrieved_context>
{context}
</retrieved_context>

<user_input>
{query}
</user_input>
"""

# No context: explicit acknowledgment embedded in the template framing.
_NON_RAG_TEMPLATE = """\
Answer based on your general knowledge of customer support. \
You have no retrieved tickets or account-specific information.

<user_input>
{query}
</user_input>
"""


async def generate_both(
    query: str,
    tickets: list[RetrievedTicket],
) -> tuple[str, str]:
    """Generate RAG and non-RAG answers concurrently.

    Args:
        query: Raw user query from the API (sanitized inside this function).
        tickets: Retrieved context tickets (may be empty).

    Returns:
        Tuple of (rag_answer, non_rag_answer) strings.
    """
    rag_task = asyncio.get_event_loop().run_in_executor(
        None, _generate_rag, query, tickets
    )
    non_rag_task = asyncio.get_event_loop().run_in_executor(
        None, _generate_non_rag, query
    )

    rag_result, non_rag_result = await asyncio.gather(rag_task, non_rag_task)
    return rag_result.text, non_rag_result.text


def _generate_rag(query: str, tickets: list[RetrievedTicket]) -> LLMResult:
    """Build RAG prompt and call the LLM.

    Args:
        query: User query — sanitized here before prompt interpolation.
        tickets: Retrieved context tickets.

    Returns:
        LLMResult from llm_client.
    """
    safe_query = sanitize_user_input(query)

    # Quality gate: skip LLM if the best retrieved chunk is below the threshold.
    # This avoids hallucination on irrelevant context and saves an LLM call.
    best_score = max((t.score for t in tickets), default=0.0)
    threshold = get_settings().rag_score_threshold
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
    prompt = _RAG_TEMPLATE.format(context=context, query=safe_query)
    result = generate(prompt, system=_RAG_SYSTEM_PROMPT)
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
    prompt = _NON_RAG_TEMPLATE.format(query=safe_query)
    result = generate(prompt, system=_NON_RAG_SYSTEM_PROMPT)
    logger.info(
        "Non-RAG generation complete",
        extra={"provider": result.provider, "latency_ms": round(result.latency_ms)},
    )
    return result
