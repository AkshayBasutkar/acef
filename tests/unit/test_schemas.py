"""
Unit tests for core/schemas.py — M1 acceptance check.

Verifies that QueryRequest, QueryResponse, QueryType, and RetrievedChunk
import cleanly and validate correctly.
"""

import pytest
from uuid import UUID

# Ensure the acef package root is importable
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.schemas import QueryType, QueryRequest, QueryResponse, RetrievedChunk, HealthResponse


class TestQueryType:
    def test_enum_values(self):
        assert QueryType.SIMPLE.value == "simple"
        assert QueryType.CODE.value == "code"
        assert QueryType.MEMORY.value == "memory"

    def test_enum_is_str(self):
        """QueryType members should be usable as strings."""
        assert isinstance(QueryType.SIMPLE, str)
        assert QueryType.CODE == "code"


class TestQueryRequest:
    def test_valid_request(self):
        req = QueryRequest(query="What is Python?")
        assert req.query == "What is Python?"
        assert isinstance(req.session_id, UUID)

    def test_empty_query_rejected(self):
        with pytest.raises(Exception):  # Pydantic ValidationError
            QueryRequest(query="")

    def test_custom_session_id(self):
        sid = UUID("12345678-1234-5678-1234-567812345678")
        req = QueryRequest(query="test", session_id=sid)
        assert req.session_id == sid


class TestQueryResponse:
    def test_full_response(self):
        sid = UUID("12345678-1234-5678-1234-567812345678")
        resp = QueryResponse(
            session_id=sid,
            response="Paris",
            query_type=QueryType.SIMPLE,
            from_cache=False,
            sources=[],
            latency_ms=42.5,
        )
        assert resp.response == "Paris"
        assert resp.query_type == QueryType.SIMPLE
        assert resp.from_cache is False
        assert isinstance(resp.query_id, UUID)

    def test_sources_default_empty(self):
        sid = UUID("12345678-1234-5678-1234-567812345678")
        resp = QueryResponse(
            session_id=sid,
            response="test",
            query_type=QueryType.CODE,
            from_cache=True,
            latency_ms=1.0,
        )
        assert resp.sources == []


class TestRetrievedChunk:
    def test_basic_chunk(self):
        chunk = RetrievedChunk(text="def foo(): pass", source="utils.py")
        assert chunk.text == "def foo(): pass"
        assert chunk.source == "utils.py"


class TestHealthResponse:
    def test_health(self):
        h = HealthResponse(status="ok", neo4j_connected=True, chroma_ready=True)
        assert h.status == "ok"
