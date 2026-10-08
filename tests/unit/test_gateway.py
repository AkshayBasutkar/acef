"""
Unit tests for the API Gateway (api/gateway.py).

Uses FastAPI TestClient to test:
  - POST /query (success and 422 validation error)
  - POST /ingest/code (success and 404 on nonexistent directory)
  - POST /ingest/document (success and 400 on unsupported format)
  - GET /health (service status)
"""

import io
from unittest.mock import MagicMock, patch
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient

from api.gateway import app, _services
from core.schemas import QueryResponse, QueryType


@pytest.fixture
def client():
    # Setup mock pipeline in _services
    mock_pipeline = MagicMock()
    mock_mem_rag = MagicMock()
    mock_driver = MagicMock()
    mock_chroma = MagicMock()
    mock_embed = MagicMock()

    _services["pipeline"] = mock_pipeline
    _services["mem_rag"] = mock_mem_rag
    _services["neo4j_driver"] = mock_driver
    _services["chroma_client"] = mock_chroma
    _services["embedding_model"] = mock_embed

    with TestClient(app) as test_client:
        yield test_client, mock_pipeline, mock_mem_rag


def test_query_success(client):
    test_client, mock_pipeline, _ = client

    mock_resp = QueryResponse(
        session_id=uuid4(),
        response="Test answer from mock",
        query_type=QueryType.SIMPLE,
        from_cache=False,
        sources=[],
        latency_ms=12.5,
    )
    mock_pipeline.handle_query.return_value = mock_resp

    response = test_client.post("/query", json={"query": "Hello world"})
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Test answer from mock"
    assert data["query_type"] == "simple"
    assert data["from_cache"] is False


def test_query_empty_string_validation_error(client):
    test_client, _, _ = client
    response = test_client.post("/query", json={"query": ""})
    assert response.status_code == 422


def test_query_missing_field_validation_error(client):
    test_client, _, _ = client
    response = test_client.post("/query", json={})
    assert response.status_code == 422


def test_ingest_code_outside_allowed_roots_returns_400(client):
    test_client, _, _ = client
    # Path escaping allowed roots
    response = test_client.post(
        "/ingest/code",
        json={"root_path": "../"},
    )
    assert response.status_code == 400
    assert "outside allowed ingest roots" in response.json()["detail"]


def test_ingest_code_nonexistent_path(client):
    test_client, _, _ = client
    response = test_client.post(
        "/ingest/code",
        json={"root_path": "./data/nonexistent_path_xyz"},
    )
    assert response.status_code == 404


@patch("api.gateway.ingest_codebase")
def test_ingest_code_success(mock_ingest, client, monkeypatch, tmp_path):
    test_client, _, _ = client
    mock_ingest.return_value = {"files": 3, "symbols": 4, "calls": 2, "imports": 2}

    # Allow tmp_path in settings for this test
    from config.settings import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "allowed_ingest_roots", ["./data", str(tmp_path)])

    response = test_client.post(
        "/ingest/code",
        json={"root_path": str(tmp_path)},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["summary"]["files"] == 3


@patch("api.gateway.ingest_document")
def test_ingest_document_success(mock_ingest_doc, client):
    test_client, _, mock_mem_rag = client
    mock_ingest_doc.return_value = 4

    file_content = b"# Sample Document\nPricing tiers and docs"
    response = test_client.post(
        "/ingest/document",
        files={"file": ("guide.md", io.BytesIO(file_content), "text/markdown")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "guide.md"
    assert data["chunks_stored"] == 4


def test_ingest_document_payload_too_large(client, monkeypatch):
    test_client, _, _ = client
    from config.settings import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "max_upload_bytes", 20)  # 20 bytes limit

    large_content = b"This is content that is definitely longer than twenty bytes"
    response = test_client.post(
        "/ingest/document",
        files={"file": ("large.txt", io.BytesIO(large_content), "text/plain")},
    )
    assert response.status_code == 413
    assert "exceeds maximum allowed upload size" in response.json()["detail"]


def test_ingest_document_unsupported_format(client):
    test_client, _, _ = client
    file_content = b"Binary executable content"
    response = test_client.post(
        "/ingest/document",
        files={"file": ("malicious.exe", io.BytesIO(file_content), "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_health_check(client):
    test_client, _, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "neo4j_connected" in data
    assert "chroma_ready" in data
    assert "llm_configured" in data


def test_query_llm_service_error_maps_to_503(client):
    from modules.llm_client import LLMServiceError
    test_client, mock_pipe, _ = client
    mock_pipe.handle_query.side_effect = LLMServiceError("Service unavailable", status_code=503)

    response = test_client.post("/query", json={"query": "test query", "session_id": str(uuid4())})
    assert response.status_code == 503
    assert response.json() == {"detail": "Service unavailable"}


def test_query_llm_rate_limit_maps_to_429(client):
    from modules.llm_client import LLMServiceError
    test_client, mock_pipe, _ = client
    mock_pipe.handle_query.side_effect = LLMServiceError("Rate limit exceeded", status_code=429)

    response = test_client.post("/query", json={"query": "test query", "session_id": str(uuid4())})
    assert response.status_code == 429
    assert response.json() == {"detail": "Rate limit exceeded"}

