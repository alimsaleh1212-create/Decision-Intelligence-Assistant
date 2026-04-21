"""Tests for app.services.llm_predictor.

LLM calls are fully mocked — we test label parsing and response schema.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.services.llm_predictor import _parse_label, predict


def test_parse_label_urgent():
    """'urgent' in response maps to urgent."""
    assert _parse_label("urgent") == "urgent"


def test_parse_label_normal():
    """'normal' or unrecognised text maps to normal."""
    assert _parse_label("normal") == "normal"
    assert _parse_label("something else") == "normal"


def test_parse_label_urgent_embedded_in_sentence():
    """'urgent' anywhere in the response is detected."""
    assert _parse_label("this ticket is urgent please") == "urgent"


def test_parse_label_case_insensitive_already_lowered():
    """Input is already lowercased before _parse_label — verify it handles it."""
    assert _parse_label("urgent") == "urgent"


@pytest.mark.asyncio
@patch("app.services.llm_predictor.generate")
async def test_predict_returns_urgent(mock_generate):
    """predict() returns urgent label when LLM says urgent."""
    mock_result = MagicMock()
    mock_result.text = "urgent"
    mock_result.provider = "ollama"
    mock_result.latency_ms = 400.0
    mock_result.cost_usd = 0.0
    mock_generate.return_value = mock_result

    response = await predict("my account is broken ASAP!!!")
    assert response.label == "urgent"
    assert response.provider == "ollama"
    assert response.latency_ms == 400.0


@pytest.mark.asyncio
@patch("app.services.llm_predictor.generate")
async def test_predict_returns_normal(mock_generate):
    """predict() returns normal label when LLM says normal."""
    mock_result = MagicMock()
    mock_result.text = "normal"
    mock_result.provider = "ollama"
    mock_result.latency_ms = 350.0
    mock_result.cost_usd = 0.0
    mock_generate.return_value = mock_result

    response = await predict("thanks for the update")
    assert response.label == "normal"


@pytest.mark.asyncio
@patch("app.services.llm_predictor.generate")
async def test_predict_uses_gemini_fallback_provider(mock_generate):
    """predict() records gemini-fallback as provider when fallback fires."""
    mock_result = MagicMock()
    mock_result.text = "urgent"
    mock_result.provider = "gemini-fallback"
    mock_result.latency_ms = 900.0
    mock_result.cost_usd = 0.0
    mock_generate.return_value = mock_result

    response = await predict("system is down!")
    assert response.provider == "gemini-fallback"
