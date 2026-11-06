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


def test_ingest_code_outside_allowed_roots_returns_400(client, monkeypatch):
    test_client, _, _ = client
    from config.settings import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "allowed_ingest_roots", ["./data"])
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
    assert mock_ingest_doc.call_args[1]["source_name"] == "guide.md"


@patch("api.gateway.ingest_document")
def test_ingest_document_zero_text_returns_400(mock_ingest_doc, client):
    test_client, _, _ = client
    mock_ingest_doc.side_effect = ValueError("No extractable text was found in this file. Image-only PDFs are not supported.")

    file_content = b"   "
    response = test_client.post(
        "/ingest/document",
        files={"file": ("empty.md", io.BytesIO(file_content), "text/markdown")},
    )
    assert response.status_code == 400
    assert "No extractable text was found in this file. Image-only PDFs are not supported." in response.json()["detail"]


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


@patch("api.gateway.ingest_document")
def test_ingest_documents_endpoint_multi_success(mock_ingest_doc, client):
    test_client, _, _ = client
    mock_ingest_doc.return_value = 3

    files = [
        ("files", ("doc1.md", io.BytesIO(b"# First Document\nContent one"), "text/markdown")),
        ("files", ("doc2.txt", io.BytesIO(b"Second document plain text"), "text/plain")),
    ]
    response = test_client.post("/ingest/documents", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_files"] == 2
    assert data["successful"] == 2
    assert data["failed"] == 0
    assert data["total_chunks_stored"] == 6
    assert len(data["results"]) == 2
    assert data["results"][0]["filename"] == "doc1.md"
    assert data["results"][1]["filename"] == "doc2.txt"


@patch("api.gateway.ingest_document")
def test_ingest_document_endpoint_multi_files(mock_ingest_doc, client):
    test_client, _, _ = client
    mock_ingest_doc.return_value = 2

    files = [
        ("file", ("file_a.md", io.BytesIO(b"# Doc A"), "text/markdown")),
        ("file", ("file_b.txt", io.BytesIO(b"Doc B content"), "text/plain")),
    ]
    response = test_client.post("/ingest/document", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_files"] == 2
    assert data["successful"] == 2
    assert data["total_chunks_stored"] == 4


@patch("api.gateway.ingest_document")
def test_ingest_documents_partial_success(mock_ingest_doc, client):
    test_client, _, _ = client
    mock_ingest_doc.return_value = 5

    files = [
        ("files", ("valid.md", io.BytesIO(b"# Valid doc"), "text/markdown")),
        ("files", ("invalid.exe", io.BytesIO(b"binary"), "application/octet-stream")),
    ]
    response = test_client.post("/ingest/documents", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "partial_success"
    assert data["total_files"] == 2
    assert data["successful"] == 1
    assert data["failed"] == 1
    assert data["total_chunks_stored"] == 5
    assert data["results"][0]["status"] == "success"
    assert data["results"][1]["status"] == "error"
    assert "Unsupported file type" in data["results"][1]["error"]


def test_ingest_documents_all_failed_returns_400(client):
    test_client, _, _ = client
    files = [
        ("files", ("bad1.exe", io.BytesIO(b"bin1"), "application/octet-stream")),
        ("files", ("bad2.bin", io.BytesIO(b"bin2"), "application/octet-stream")),
    ]
    response = test_client.post("/ingest/documents", files=files)
    assert response.status_code == 400
    assert "All 2 file(s) failed ingestion" in response.json()["detail"]


def test_ingest_documents_empty_files_returns_400(client):
    test_client, _, _ = client
    response = test_client.post("/ingest/documents", files=[])
    assert response.status_code == 400


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


def test_concurrent_queries_run_in_parallel_in_threadpool(client):
    """I-14: Five concurrent requests against a 0.5s pipeline stub must finish in < 1.5s total."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    test_client, mock_pipe, _ = client

    def slow_handle(req):
        time.sleep(0.5)
        return QueryResponse(
            session_id=req.session_id,
            response="Slow response",
            query_type=QueryType.SIMPLE,
            from_cache=False,
            sources=[],
            latency_ms=500.0,
        )

    mock_pipe.handle_query.side_effect = slow_handle

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                test_client.post,
                "/query",
                json={"query": f"Query {i}", "session_id": str(uuid4())},
            )
            for i in range(5)
        ]
        results = [f.result() for f in futures]

    total_time = time.time() - start_time
    for r in results:
        assert r.status_code == 200
    assert total_time < 1.5


def test_health_check_responds_promptly_during_slow_query(client):
    """I-14: /health answers in < 100ms while a slow query is in flight."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    test_client, mock_pipe, _ = client

    def slow_handle(req):
        time.sleep(0.5)
        return QueryResponse(
            session_id=req.session_id,
            response="Slow response",
            query_type=QueryType.SIMPLE,
            from_cache=False,
            sources=[],
            latency_ms=500.0,
        )

    mock_pipe.handle_query.side_effect = slow_handle

    with ThreadPoolExecutor(max_workers=2) as executor:
        query_future = executor.submit(
            test_client.post,
            "/query",
            json={"query": "slow", "session_id": str(uuid4())},
        )
        time.sleep(0.05)

        t_health_start = time.time()
        health_resp = test_client.get("/health")
        t_health = time.time() - t_health_start

        query_resp = query_future.result()

    assert query_resp.status_code == 200
    assert health_resp.status_code == 200
    assert t_health < 0.10


def test_clear_graph_success(client):
    """Test clearing graph via POST /graph/clear."""
    test_client, _, _ = client
    mock_driver = _services["neo4j_driver"]
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_count_result = MagicMock()
    mock_count_result.single.return_value = {"cnt": 42}
    mock_session.run.side_effect = [mock_count_result, None]

    response = test_client.post("/graph/clear")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["nodes_deleted"] == 42
    assert "42 nodes deleted" in data["message"]


def test_clear_graph_delete_method_success(client):
    """Test clearing graph via DELETE /graph/clear."""
    test_client, _, _ = client
    mock_driver = _services["neo4j_driver"]
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_count_result = MagicMock()
    mock_count_result.single.return_value = {"cnt": 15}
    mock_session.run.side_effect = [mock_count_result, None]

    response = test_client.delete("/graph/clear")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["nodes_deleted"] == 15


def test_clear_graph_no_driver_returns_503(client):
    """Test 503 when Neo4j driver is unavailable."""
    test_client, _, _ = client
    saved_driver = _services.get("neo4j_driver")
    try:
        _services["neo4j_driver"] = None
        with patch("api.gateway.init_services"):
            response = test_client.post("/graph/clear")
            assert response.status_code == 503
            assert "Neo4j driver not connected" in response.json()["detail"]
    finally:
        _services["neo4j_driver"] = saved_driver


def test_clear_cache_endpoint(client):
    """Test clearing semantic cache via /cache/clear."""
    test_client, mock_pipeline, _ = client
    mock_pipeline.cache = MagicMock()
    mock_pipeline.cache.clear.return_value = None

    response = test_client.post("/cache/clear")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_pipeline.cache.clear.assert_called_once()

    # Also test DELETE method
    del_resp = test_client.delete("/cache/clear")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "success"





