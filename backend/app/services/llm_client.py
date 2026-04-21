"""LLM client abstraction — Ollama primary, Gemini fallback.

All LLM generation goes through generate(). No other module calls
Ollama or Gemini directly. The caller never knows which provider responded.
"""

import logging
import time
from dataclasses import dataclass

import ollama

from app.core.settings import get_settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when both Ollama and Gemini (if configured) fail."""


@dataclass
class LLMResult:
    """Result from a single LLM call."""

    text: str
    provider: str
    latency_ms: float
    cost_usd: float = 0.0


def generate(prompt: str, system: str = "") -> LLMResult:
    """Generate text from the configured LLM with automatic fallback.

    Tries Ollama first. On ConnectionError or timeout falls back to
    Gemini 2.5 Flash if GOOGLE_API_KEY is configured. If neither
    succeeds, raises LLMError.

    Args:
        prompt: The user/task prompt.
        system: Optional system instruction prepended to the conversation.

    Returns:
        LLMResult with generated text, provider name, latency, and cost.

    Raises:
        LLMError: When all configured providers fail.
    """
    try:
        return _call_ollama(prompt, system)
    except (ollama.ResponseError, ConnectionError, TimeoutError, Exception) as exc:
        settings = get_settings()
        logger.warning(
            "Ollama call failed, checking fallback: %s",
            exc,
            extra={"provider": "ollama"},
        )
        if not settings.gemini_fallback_enabled:
            raise LLMError(f"Ollama failed and Gemini fallback is not configured: {exc}") from exc

    return _call_gemini(prompt, system)


def _call_ollama(prompt: str, system: str) -> LLMResult:
    """Call Ollama chat API.

    Args:
        prompt: User message content.
        system: System message content (empty string = no system message).

    Returns:
        LLMResult from Ollama.

    Raises:
        ollama.ResponseError: On model or server errors.
        ConnectionError: When Ollama is unreachable.
    """
    settings = get_settings()
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    start = time.perf_counter()
    client = ollama.Client(
        host=settings.ollama_base_url,
        timeout=settings.ollama_timeout_seconds,
    )
    response = client.chat(model=settings.ollama_llm_model, messages=messages)
    latency_ms = (time.perf_counter() - start) * 1000

    text = response.message.content or ""
    logger.info(
        "Ollama call complete",
        extra={"latency_ms": round(latency_ms), "chars": len(text)},
    )
    return LLMResult(text=text, provider="ollama", latency_ms=latency_ms)


def _call_gemini(prompt: str, system: str) -> LLMResult:
    """Call Gemini 2.5 Flash as fallback.

    Args:
        prompt: User message content.
        system: System instruction (Gemini supports this natively).

    Returns:
        LLMResult from Gemini.

    Raises:
        LLMError: If the Gemini call also fails.
    """
    import google.generativeai as genai  # imported lazily — only when fallback fires

    settings = get_settings()
    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel(
        model_name=settings.gemini_llm_model,
        system_instruction=system or None,
    )

    start = time.perf_counter()
    try:
        response = model.generate_content(prompt)
    except Exception as exc:
        logger.error("Gemini fallback also failed: %s", exc)
        raise LLMError(f"Both Ollama and Gemini failed. Gemini error: {exc}") from exc
    latency_ms = (time.perf_counter() - start) * 1000

    text = response.text or ""
    logger.warning(
        "Gemini fallback used",
        extra={"latency_ms": round(latency_ms), "chars": len(text)},
    )
    # Gemini Flash 2.5 free tier — $0.00 for ≤1500 req/day; we report $0 for simplicity
    return LLMResult(text=text, provider="gemini-fallback", latency_ms=latency_ms, cost_usd=0.0)
