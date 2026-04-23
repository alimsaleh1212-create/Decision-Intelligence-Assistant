"""LLM client abstraction — Gemini primary, Ollama fallback.

All LLM generation goes through generate(). No other module calls
Gemini or Ollama directly. The caller never knows which provider responded.
"""

import logging
import time
from dataclasses import dataclass
from functools import lru_cache

import ollama

from app.core.settings import get_settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when both Gemini and Ollama fail."""


@dataclass
class LLMResult:
    """Result from a single LLM call."""

    text: str
    provider: str
    latency_ms: float
    cost_usd: float = 0.0


@lru_cache(maxsize=1)
def _get_ollama_client() -> ollama.Client:
    """Return a single cached Ollama client for the process lifetime."""
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
    import google.generativeai as genai  # lazy import — only when Gemini is used

    settings = get_settings()
    genai.configure(api_key=settings.google_api_key)
    return genai.GenerativeModel(
        model_name=settings.gemini_llm_model,
        system_instruction=system or None,
    )


def generate(prompt: str, system: str = "", max_tokens: int | None = None) -> LLMResult:
    """Generate text — Gemini primary, Ollama fallback.

    Tries Gemini first if GOOGLE_API_KEY is configured. On any Gemini failure
    (or if no API key) falls back to Ollama. If neither succeeds, raises LLMError.

    Args:
        prompt: The user/task prompt (caller is responsible for sanitizing).
        system: Optional system instruction prepended to the conversation.
        max_tokens: Hard cap on output tokens. None = model default.

    Returns:
        LLMResult with generated text, provider name, latency, and cost.

    Raises:
        LLMError: When all configured providers fail.
    """
    settings = get_settings()

    if settings.gemini_configured:
        try:
            return _call_gemini(prompt, system, max_tokens)
        except LLMError as exc:
            logger.warning(
                "Gemini primary failed, falling back to Ollama: %s",
                exc,
                extra={"provider": "gemini"},
            )

    # Ollama: primary when Gemini not configured, fallback otherwise.
    try:
        return _call_ollama(prompt, system, max_tokens)
    except Exception as exc:
        raise LLMError(f"All LLM providers failed. Ollama error: {exc}") from exc


def _call_gemini(prompt: str, system: str, max_tokens: int | None) -> LLMResult:
    """Call Gemini as the primary LLM using the cached model instance.

    Args:
        prompt: User message content.
        system: System instruction (used as the cache key for model lookup).
        max_tokens: Hard cap on generated tokens (None = model default).

    Returns:
        LLMResult from Gemini.

    Raises:
        LLMError: If the Gemini call fails.
    """
    import google.generativeai as genai  # type: ignore[import]

    model = _get_gemini_model(system)
    generation_config = genai.GenerationConfig(max_output_tokens=max_tokens) if max_tokens else None

    start = time.perf_counter()
    try:
        response = model.generate_content(prompt, generation_config=generation_config)
    except Exception as exc:
        logger.error("Gemini call failed: %s", exc)
        raise LLMError(f"Gemini failed: {exc}") from exc
    latency_ms = (time.perf_counter() - start) * 1000

    text = response.text or ""
    logger.info(
        "Gemini call complete",
        extra={"latency_ms": round(latency_ms), "chars": len(text)},
    )
    return LLMResult(text=text, provider="gemini", latency_ms=latency_ms, cost_usd=0.0)


def _call_ollama(prompt: str, system: str, max_tokens: int | None) -> LLMResult:
    """Call Ollama as the fallback LLM using the cached client.

    Args:
        prompt: User message content.
        system: System message content (empty string = no system message).
        max_tokens: Hard cap on generated tokens (None = model default).

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

    options: dict = {}
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    start = time.perf_counter()
    response = client.chat(
        model=settings.ollama_llm_model,
        messages=messages,
        options=options if options else None,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    text = response.message.content or ""
    logger.warning(
        "Ollama fallback used",
        extra={"latency_ms": round(latency_ms), "chars": len(text)},
    )
    return LLMResult(text=text, provider="ollama-fallback", latency_ms=latency_ms)
