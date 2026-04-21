"""Tests for app.services.llm_client.

Verifies Ollama-first logic and Gemini fallback behaviour.
"""

from unittest.mock import MagicMock, patch

import ollama
import pytest

from app.services.llm_client import LLMError, LLMResult, _call_ollama, generate


@patch("app.services.llm_client.get_settings")
@patch("app.services.llm_client.ollama.Client")
def test_generate_uses_ollama_when_available(mock_client_cls, mock_settings):
    """generate() returns Ollama result when Ollama is healthy."""
    mock_settings.return_value = MagicMock(
        ollama_base_url="http://ollama:11434",
        ollama_llm_model="gemma4:31b-cloud",
        ollama_timeout_seconds=30,
        gemini_fallback_enabled=False,
    )
    mock_msg = MagicMock()
    mock_msg.content = "Hello from Ollama"
    mock_response = MagicMock()
    mock_response.message = mock_msg
    mock_client_cls.return_value.chat.return_value = mock_response

    result = generate("test prompt")
    assert result.text == "Hello from Ollama"
    assert result.provider == "ollama"
    assert result.latency_ms >= 0


@patch("app.services.llm_client.get_settings")
@patch("app.services.llm_client.ollama.Client")
def test_generate_raises_llm_error_when_ollama_fails_no_fallback(mock_client_cls, mock_settings):
    """generate() raises LLMError when Ollama fails and no fallback configured."""
    mock_settings.return_value = MagicMock(
        ollama_base_url="http://ollama:11434",
        ollama_llm_model="gemma4:31b-cloud",
        ollama_timeout_seconds=30,
        gemini_fallback_enabled=False,
    )
    mock_client_cls.return_value.chat.side_effect = ConnectionError("refused")

    with pytest.raises(LLMError, match="not configured"):
        generate("test prompt")


@patch("app.services.llm_client._call_gemini")
@patch("app.services.llm_client.get_settings")
@patch("app.services.llm_client.ollama.Client")
def test_generate_falls_back_to_gemini_on_connection_error(
    mock_client_cls, mock_settings, mock_gemini
):
    """generate() calls Gemini when Ollama raises ConnectionError."""
    mock_settings.return_value = MagicMock(
        ollama_base_url="http://ollama:11434",
        ollama_llm_model="gemma4:31b-cloud",
        ollama_timeout_seconds=30,
        gemini_fallback_enabled=True,
    )
    mock_client_cls.return_value.chat.side_effect = ConnectionError("refused")
    mock_gemini.return_value = LLMResult(
        text="Gemini answer", provider="gemini-fallback", latency_ms=800.0
    )

    result = generate("test prompt")
    assert result.provider == "gemini-fallback"
    assert result.text == "Gemini answer"
    mock_gemini.assert_called_once()
