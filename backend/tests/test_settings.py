"""Tests for app.core.settings.

Validates environment parsing, fail-fast guards, and fallback flag.
"""

import pytest
from pydantic import ValidationError

from app.core.settings import Settings


def test_settings_defaults():
    """Settings loads with defaults when no env vars set."""
    s = Settings()
    assert s.ollama_base_url == "http://ollama:11434"
    assert s.qdrant_top_k == 5
    assert s.gemini_fallback_enabled is False


def test_settings_gemini_fallback_enabled_when_key_set():
    """gemini_fallback_enabled is True when GOOGLE_API_KEY is non-empty."""
    s = Settings(google_api_key="test-key-123")
    assert s.gemini_fallback_enabled is True


def test_settings_gemini_fallback_disabled_when_key_empty():
    """gemini_fallback_enabled is False when GOOGLE_API_KEY is empty string."""
    s = Settings(google_api_key="")
    assert s.gemini_fallback_enabled is False


def test_settings_top_k_validation_rejects_zero():
    """QDRANT_TOP_K < 1 raises ValidationError."""
    with pytest.raises(ValidationError):
        Settings(qdrant_top_k=0)


def test_settings_top_k_validation_rejects_negative():
    """Negative QDRANT_TOP_K raises ValidationError."""
    with pytest.raises(ValidationError):
        Settings(qdrant_top_k=-5)


def test_settings_ollama_url_rejected_when_empty():
    """Empty OLLAMA_BASE_URL raises ValidationError."""
    with pytest.raises(ValidationError):
        Settings(ollama_base_url="")
