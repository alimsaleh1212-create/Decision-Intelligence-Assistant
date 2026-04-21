"""Tests for GET /api/health.

Qdrant and Ollama are mocked — we test routing, schema, and logic
without requiring live services.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers.health import ServiceStatus

client = TestClient(app)


def _mock_settings(**overrides: object) -> MagicMock:
    """Return a Settings-like mock with sensible defaults."""
    m = MagicMock()
    m.ollama_base_url = "http://ollama:11434"
    m.qdrant_host = "qdrant"
    m.qdrant_port = 6333
    m.gemini_fallback_enabled = False
    for k, v in overrides.items():
        setattr(m, k, v)
    return m


@patch("app.routers.health.get_settings")
@patch("app.routers.health._check_ollama", return_value=ServiceStatus(reachable=True, detail="ok"))
@patch("app.routers.health._check_qdrant", return_value=ServiceStatus(reachable=True, detail="ok"))
def test_health_ok(mock_qdrant, mock_ollama, mock_settings):
    """Health endpoint returns 200 and status=ok when both services are up."""
    mock_settings.return_value = _mock_settings()

    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["ollama"]["reachable"] is True
    assert body["qdrant"]["reachable"] is True


@patch("app.routers.health.get_settings")
@patch("app.routers.health._check_ollama", return_value=ServiceStatus(reachable=True, detail="ok"))
@patch("app.routers.health._check_qdrant", return_value=ServiceStatus(reachable=False, detail="connection refused"))
def test_health_degraded_when_qdrant_down(mock_qdrant, mock_ollama, mock_settings):
    """Health endpoint returns status=degraded when Qdrant is unreachable."""
    mock_settings.return_value = _mock_settings()

    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["qdrant"]["reachable"] is False


@patch("app.routers.health.get_settings")
@patch("app.routers.health._check_ollama", return_value=ServiceStatus(reachable=False, detail="timeout"))
@patch("app.routers.health._check_qdrant", return_value=ServiceStatus(reachable=True, detail="ok"))
def test_health_degraded_when_ollama_down(mock_qdrant, mock_ollama, mock_settings):
    """Health endpoint returns status=degraded when Ollama is unreachable."""
    mock_settings.return_value = _mock_settings()

    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"


@patch("app.routers.health.get_settings")
@patch("app.routers.health._check_ollama", return_value=ServiceStatus(reachable=True, detail="ok"))
@patch("app.routers.health._check_qdrant", return_value=ServiceStatus(reachable=True, detail="ok"))
def test_health_reports_gemini_configured(mock_qdrant, mock_ollama, mock_settings):
    """Health endpoint reports gemini_fallback_configured=True when key is set."""
    mock_settings.return_value = _mock_settings(gemini_fallback_enabled=True)

    resp = client.get("/api/health")
    assert resp.json()["gemini_fallback_configured"] is True


@patch("app.routers.health.get_settings")
@patch("app.routers.health._check_ollama", return_value=ServiceStatus(reachable=True, detail="ok"))
@patch("app.routers.health._check_qdrant", return_value=ServiceStatus(reachable=True, detail="ok"))
def test_health_reports_gemini_not_configured_by_default(mock_qdrant, mock_ollama, mock_settings):
    """Health endpoint reports gemini_fallback_configured=False when key is absent."""
    mock_settings.return_value = _mock_settings()

    resp = client.get("/api/health")
    assert resp.json()["gemini_fallback_configured"] is False
