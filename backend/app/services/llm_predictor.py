"""LLM zero-shot priority predictor.

Asks the LLM whether a ticket is urgent or normal without any training.
Uses llm_client so Ollama → Gemini fallback applies automatically.
"""

import logging
import re

from app.schemas.priority import PriorityResponse
from app.services.llm_client import LLMError, generate

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a customer support triage assistant. "
    "Classify the ticket as exactly one word: urgent or normal. "
    "Reply with only that single word and nothing else."
)

_PROMPT_TEMPLATE = """\
Classify the following support ticket as urgent or normal.

<user_input>
{text}
</user_input>

Reply with exactly one word: urgent or normal.
"""


async def predict(text: str) -> PriorityResponse:
    """Predict ticket priority using LLM zero-shot classification.

    Args:
        text: Raw ticket text.

    Returns:
        PriorityResponse with label, latency_ms, provider, and cost.

    Raises:
        LLMError: If both Ollama and Gemini fail.
    """
    prompt = _PROMPT_TEMPLATE.format(text=text)
    result = generate(prompt, system=_SYSTEM_PROMPT)

    raw = result.text.strip().lower()
    label = _parse_label(raw)

    logger.info(
        "LLM priority prediction",
        extra={
            "label": label,
            "provider": result.provider,
            "latency_ms": round(result.latency_ms),
            "raw_response": raw[:50],
        },
    )
    return PriorityResponse(
        label=label,
        confidence=None,
        latency_ms=result.latency_ms,
        provider=result.provider,
        cost_usd=result.cost_usd,
    )


def _parse_label(raw: str) -> str:
    """Extract urgent/normal from LLM output, defaulting to normal.

    Args:
        raw: Stripped, lowercased LLM response.

    Returns:
        "urgent" or "normal".
    """
    if re.search(r"\burgent\b", raw):
        return "urgent"
    return "normal"
