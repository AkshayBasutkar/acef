"""
Integration tests for Pipeline and API Gateway.

Tests end-to-end routing, cache hits, branch assertion, and API responses.
"""

from unittest.mock import MagicMock, patch
from uuid import uuid4
import pytest
from httpx import ASGITransport, AsyncClient

from api.gateway import app, _services
from core.pipeline import Pipeline
from core.schemas import QueryRequest, QueryResponse, QueryType, RetrievedChunk
from modules.cache import RAGCacheLite
from modules.llm_client import LLMClient


@pytest.fixture
def mock_clients():
    mock_chroma = MagicMock()
    mock_embed = MagicMock()
    mock_embed.encode.return_value = [0.0] * 384
    mock_driver = MagicMock()

    mock_cache = MagicMock()
    mock_graph_rag = MagicMock()
    mock_mem_rag = MagicMock()
    mock_llm = MagicMock()

    mock_cache.lookup.return_value = None
    mock_mem_rag.session_turn_count.return_value = 0
    mock_llm.generate.return_value = "Integration test LLM answer"

    pipe = Pipeline(
        cache=mock_cache,
        graph_rag=mock_graph_rag,
        mem_rag=mock_mem_rag,
        llm_client=mock_llm,
    )
    return pipe, mock_cache, mock_graph_rag, mock_mem_rag, mock_llm


# ── Pipeline Integration Tests ───────────────────────────────────────────────

def test_pipeline_integration_code_branch(mock_clients):
    pipe, _, mock_graph_rag, mock_mem_rag, mock_llm = mock_clients
    mock_graph_rag.retrieve.return_value = [
        RetrievedChunk(text="class AuthManager: ...", source="auth.py")
    ]

    req = QueryRequest(query="Where is the class AuthManager defined in python?")
    res = pipe.handle_query(req)

    assert res.query_type == QueryType.CODE
    assert res.from_cache is False
    assert "auth.py" in res.sources
    mock_graph_rag.retrieve.assert_called_once()
    mock_mem_rag.retrieve.assert_not_called()
    mock_llm.generate.assert_called_once()


def test_pipeline_integration_memory_branch(mock_clients):
    pipe, _, mock_graph_rag, mock_mem_rag, mock_llm = mock_clients
    mock_mem_rag.retrieve.return_value = [
        RetrievedChunk(text="Pro plan is $49", source="pricing.md")
    ]

    req = QueryRequest(query="What was mentioned earlier about pricing in the document?")
    res = pipe.handle_query(req)

    assert res.query_type == QueryType.MEMORY
    assert res.from_cache is False
    assert "pricing.md" in res.sources
    mock_mem_rag.retrieve.assert_called_once()
    mock_graph_rag.retrieve.assert_not_called()
    mock_mem_rag.append_turn.assert_called_once()


def test_pipeline_integration_simple_branch(mock_clients):
    pipe, _, mock_graph_rag, mock_mem_rag, mock_llm = mock_clients

    req = QueryRequest(query="What is the capital of Italy?")
    res = pipe.handle_query(req)

    assert res.query_type == QueryType.SIMPLE
    assert res.from_cache is False
    assert res.sources == []
    mock_graph_rag.retrieve.assert_not_called()
    mock_mem_rag.retrieve.assert_not_called()
    mock_llm.generate.assert_called_once()


# ── Gateway Integration Tests with AsyncClient ───────────────────────────────

@pytest.mark.asyncio
async def test_gateway_async_query_and_health():
    mock_pipeline = MagicMock()
    mock_resp = QueryResponse(
        query_id=uuid4(),
        session_id=uuid4(),
        response="Async integration response",
        query_type=QueryType.SIMPLE,
        from_cache=False,
        sources=[],
        latency_ms=15.0,
    )
    mock_pipeline.handle_query.return_value = mock_resp

    _services["pipeline"] = mock_pipeline
    _services["neo4j_driver"] = MagicMock()
    _services["chroma_client"] = MagicMock()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Test /health
        health_resp = await ac.get("/health")
        assert health_resp.status_code == 200

        # Test /query
        query_resp = await ac.post("/query", json={"query": "Test query"})
        assert query_resp.status_code == 200

        # Test 422 on empty query
        bad_resp = await ac.post("/query", json={"query": ""})
        assert bad_resp.status_code == 422
