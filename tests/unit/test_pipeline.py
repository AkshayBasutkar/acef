"""
Unit tests for Pipeline orchestration (core/pipeline.py).

Verifies the 5-step lifecycle, caching branch, classification branching,
graceful error handling, and memory write-back.
"""

from unittest.mock import MagicMock
from uuid import uuid4
import pytest

from core.pipeline import Pipeline
from core.schemas import QueryRequest, QueryResponse, QueryType, RetrievedChunk


@pytest.fixture
def mock_pipeline():
    mock_cache = MagicMock()
    mock_graph_rag = MagicMock()
    mock_mem_rag = MagicMock()
    mock_llm = MagicMock()

    # Defaults
    mock_cache.lookup.return_value = None
    mock_mem_rag.session_turn_count.return_value = 0
    mock_llm.generate.return_value = "Mock LLM answer"

    pipe = Pipeline(
        cache=mock_cache,
        graph_rag=mock_graph_rag,
        mem_rag=mock_mem_rag,
        llm_client=mock_llm,
    )
    return pipe, mock_cache, mock_graph_rag, mock_mem_rag, mock_llm


def test_cache_hit_short_circuits(mock_pipeline):
    pipe, mock_cache, mock_graph_rag, mock_mem_rag, mock_llm = mock_pipeline
    mock_cache.lookup.return_value = "Cached instant answer"

    req = QueryRequest(query="What is Paris?")
    res = pipe.handle_query(req)

    assert res.from_cache is True
    assert res.response == "Cached instant answer"
    assert res.sources == []
    mock_graph_rag.retrieve.assert_not_called()
    mock_mem_rag.retrieve.assert_not_called()
    mock_llm.generate.assert_not_called()
    # I-02: Cache hits are recorded in conversation memory
    mock_mem_rag.append_turn.assert_called_once_with(
        str(req.session_id), "What is Paris?", "Cached instant answer", query_type=QueryType.SIMPLE
    )


def test_simple_query_flow(mock_pipeline):
    pipe, mock_cache, mock_graph_rag, mock_mem_rag, mock_llm = mock_pipeline

    req = QueryRequest(query="What is 2 + 2?")
    res = pipe.handle_query(req)

    assert res.from_cache is False
    assert res.query_type == QueryType.SIMPLE
    assert res.response == "Mock LLM answer"
    mock_graph_rag.retrieve.assert_not_called()
    mock_mem_rag.retrieve.assert_not_called()
    mock_llm.generate.assert_called_once_with(
        query="What is 2 + 2?",
        query_type="simple",
        context_chunks=[],
    )
    mock_cache.store.assert_called_once()
    # I-02: Simple queries are recorded in conversation memory
    mock_mem_rag.append_turn.assert_called_once_with(
        str(req.session_id), "What is 2 + 2?", "Mock LLM answer", query_type=QueryType.SIMPLE
    )


def test_code_query_flow(mock_pipeline):
    pipe, mock_cache, mock_graph_rag, mock_mem_rag, mock_llm = mock_pipeline
    mock_chunk = RetrievedChunk(text="def login(): ...", source="auth.py")
    mock_graph_rag.retrieve.return_value = [mock_chunk]

    req = QueryRequest(query="Explain the login function in auth.py")
    res = pipe.handle_query(req)

    assert res.query_type == QueryType.CODE
    assert res.sources == ["auth.py"]
    mock_graph_rag.retrieve.assert_called_once_with("Explain the login function in auth.py")
    mock_mem_rag.retrieve.assert_not_called()
    mock_llm.generate.assert_called_once_with(
        query="Explain the login function in auth.py",
        query_type="code",
        context_chunks=["def login(): ..."],
    )
    # I-01: CODE answers depend on codebase state and are never cached
    mock_cache.store.assert_not_called()
    # I-02: Code queries are recorded in conversation memory
    mock_mem_rag.append_turn.assert_called_once_with(
        str(req.session_id), "Explain the login function in auth.py", "Mock LLM answer", query_type=QueryType.CODE
    )


def test_memory_query_flow(mock_pipeline):
    pipe, mock_cache, mock_graph_rag, mock_mem_rag, mock_llm = mock_pipeline
    mock_chunk = RetrievedChunk(text="Tier 1: Free", source="pricing.md")
    mock_mem_rag.retrieve.return_value = [mock_chunk]

    sess_id = uuid4()
    req = QueryRequest(query="What was mentioned in the document about pricing?", session_id=sess_id)
    res = pipe.handle_query(req)

    assert res.query_type == QueryType.MEMORY
    assert res.sources == ["pricing.md"]
    mock_mem_rag.retrieve.assert_called_once()
    mock_graph_rag.retrieve.assert_not_called()
    mock_llm.generate.assert_called_once_with(
        query=req.query,
        query_type="memory",
        context_chunks=["Tier 1: Free"],
    )
    # I-01: MEMORY answers depend on session/document state and are never cached
    mock_cache.store.assert_not_called()
    mock_mem_rag.append_turn.assert_called_once_with(
        str(sess_id), req.query, "Mock LLM answer", query_type=QueryType.MEMORY
    )


def test_empty_llm_response_not_cached_and_not_recorded(mock_pipeline):
    pipe, mock_cache, _, mock_mem_rag, mock_llm = mock_pipeline
    mock_llm.generate.return_value = "   "
    req = QueryRequest(query="What is 2 + 2?")
    res = pipe.handle_query(req)
    mock_cache.store.assert_not_called()
    mock_mem_rag.append_turn.assert_not_called()


def test_retrieval_exception_gracefully_degrades(mock_pipeline):
    pipe, mock_cache, mock_graph_rag, mock_mem_rag, mock_llm = mock_pipeline
    mock_graph_rag.retrieve.side_effect = RuntimeError("Neo4j database down")

    req = QueryRequest(query="Where is function execute_query defined?")
    res = pipe.handle_query(req)

    # Should not crash; proceeds with empty context
    assert res.query_type == QueryType.CODE
    assert res.sources == []
    mock_llm.generate.assert_called_once_with(
        query=req.query,
        query_type="code",
        context_chunks=[],
    )
