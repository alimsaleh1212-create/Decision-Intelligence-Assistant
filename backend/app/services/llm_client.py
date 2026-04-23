"""LLM client abstraction — Ollama primary, Gemini fallback.

All LLM generation goes through generate(). No other module calls
Ollama or Gemini directly. The caller never knows which provider responded.
"""

import logging
import time
from dataclasses import dataclass
from functools import lru_cache

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


@lru_cache(maxsize=1)
def _get_ollama_client() -> ollama.Client:
    """Return a single cached Ollama client for the process lifetime.

    Settings are also cached via lru_cache, so host/timeout are stable.
    """
    settings = get_settings()
    return ollama.Client(
        host=settings.ollama_base_url,
        timeout=settings.ollama_timeout_seconds,
    )


@lru_cache(maxsize=8)
def _get_gemini_model(system: str) -> "genai.GenerativeModel":  # type: ignore[name-defined]
    """Return a cached Gemini model instance keyed by system instruction.

    In practice there are 3 distinct system prompts (RAG, non-RAG,
    priority predictor), so this cache never holds more than 3 entries.
    genai.configure() is idempotent; calling it once per unique key is safe.

    Args:
        system: System instruction string baked into the model at construction.

    Returns:
        Configured GenerativeModel instance, cached for the process lifetime.
    """
    import google.generativeai as genai  # lazy import — only when fallback fires

    settings = get_settings()
    genai.configure(api_key=settings.google_api_key)
    return genai.GenerativeModel(
        model_name=settings.gemini_llm_model,
        system_instruction=system or None,
    )


def generate(prompt: str, system: str = "") -> LLMResult:
    """Generate text from the configured LLM with automatic fallback.

    Tries Ollama first. On any failure falls back to Gemini 2.5 Flash
    if GOOGLE_API_KEY is configured. If neither succeeds, raises LLMError.

    Args:
        prompt: The user/task prompt (caller is responsible for sanitizing).
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
    """Call Ollama chat API using the cached client.

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
    client = _get_ollama_client()

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    start = time.perf_counter()
    response = client.chat(model=settings.ollama_llm_model, messages=messages)
    latency_ms = (time.perf_counter() - start) * 1000

    text = response.message.content or ""
    logger.info(
        "Ollama call complete",
        extra={"latency_ms": round(latency_ms), "chars": len(text)},
    )
    return LLMResult(text=text, provider="ollama", latency_ms=latency_ms)


def _call_gemini(prompt: str, system: str) -> LLMResult:
    """Call Gemini 2.5 Flash as fallback using the cached model instance.

    Args:
        prompt: User message content.
        system: System instruction (used as the cache key for model lookup).

    Returns:
        LLMResult from Gemini.

    Raises:
        LLMError: If the Gemini call also fails.
    """
    model = _get_gemini_model(system)

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
    return LLMResult(text=text, provider="gemini-fallback", latency_ms=latency_ms, cost_usd=0.0)
