"""Answer generation service — RAG and non-RAG via llm_client.

Runs both generation calls concurrently using asyncio.gather so total
latency equals the slower of the two, not the sum.
"""

import asyncio
import logging

from app.schemas.query import RetrievedTicket
from app.services.llm_client import LLMResult, generate

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful customer support assistant. "
    "Answer clearly and concisely based on the information provided. "
    "If you do not know the answer, say so. "
    "Never fabricate information."
)

_RAG_TEMPLATE = """\
Use the following retrieved support tickets as context to answer the user's question.

<retrieved_context>
{context}
</retrieved_context>

<user_input>
{query}
</user_input>
"""

_NON_RAG_TEMPLATE = """\
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
        query: Sanitized user query.
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
        query: User query (already sanitized at API boundary).
        tickets: Retrieved context tickets.

    Returns:
        LLMResult from llm_client.
    """
    context_parts = [
        f"Ticket {i + 1}: {t.text}" for i, t in enumerate(tickets)
    ]
    context = "\n".join(context_parts) if context_parts else "No relevant tickets found."
    prompt = _RAG_TEMPLATE.format(context=context, query=query)
    result = generate(prompt, system=_SYSTEM_PROMPT)
    logger.info(
        "RAG generation complete",
        extra={"provider": result.provider, "latency_ms": round(result.latency_ms)},
    )
    return result


def _generate_non_rag(query: str) -> LLMResult:
    """Build non-RAG prompt and call the LLM.

    Args:
        query: User query (already sanitized at API boundary).

    Returns:
        LLMResult from llm_client.
    """
    prompt = _NON_RAG_TEMPLATE.format(query=query)
    result = generate(prompt, system=_SYSTEM_PROMPT)
    logger.info(
        "Non-RAG generation complete",
        extra={"provider": result.provider, "latency_ms": round(result.latency_ms)},
    )
    return result
