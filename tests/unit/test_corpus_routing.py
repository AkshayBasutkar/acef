"""
Unit tests for I-05: Corpus-aware routing signals and single embedding computation.

Acceptance criteria verified:
  1. With demo corpus ingested, demo queries route as documented:
     - "What does login call?" -> CODE
     - "What are the pricing tiers?" -> MEMORY
  2. With empty corpus, the same queries route to SIMPLE.
  3. Neo4j or Chroma failure never raises from classify — signals degrade to "none".
  4. The query embedding is computed exactly once per request.
"""

from unittest.mock import MagicMock
import pytest

from core.corpus_index import CorpusIndex
from core.pipeline import Pipeline
from core.query_analyzer import classify
from core.schemas import QueryRequest, QueryType, RetrievedChunk
from modules.cache import RAGCacheLite
from modules.graph_rag import GraphRAG
from modules.mem_rag import MemRAG


class CountingEmbedder:
    """Fake embedding model that counts calls to encode()."""

    def __init__(self):
        self.call_count = 0
        self.encoded_texts = []

    def encode(self, text):
        self.call_count += 1
        self.encoded_texts.append(text)
        return [0.1, 0.2, 0.3, 0.4]


class TestCorpusIndex:
    """Test CorpusIndex caching and Neo4j loading behavior."""

    def test_init_filters_and_lowercases(self):
        index = CorpusIndex(["Login", "abc", "hash_password", 123])
        assert "login" in index.symbols
        assert "hash_password" in index.symbols
        assert "abc" not in index.symbols  # len < 4 filtered out

    def test_refresh_from_neo4j_success(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = [
            {"name": "login"},
            {"name": "hash_password"},
            {"name": "fn"},  # length < 4
        ]

        index = CorpusIndex()
        count = index.refresh_from_neo4j(mock_driver)
        assert count == 2
        assert index.contains("login")
        assert index.contains("hash_password")
        assert not index.contains("fn")

    def test_refresh_from_neo4j_failure_never_raises(self):
        mock_driver = MagicMock()
        mock_driver.session.side_effect = RuntimeError("Neo4j connection refused")

        index = CorpusIndex(["initial_symbol"])
        count = index.refresh_from_neo4j(mock_driver)
        # Keeps previous symbols safely
        assert count == 1
        assert index.contains("initial_symbol")

    def test_refresh_from_neo4j_none_driver(self):
        index = CorpusIndex()
        assert index.refresh_from_neo4j(None) == 0


class TestDocumentSimilarityProbe:
    """Test MemRAG.best_document_similarity."""

    def test_empty_collection_returns_zero(self):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mem_rag = MemRAG.__new__(MemRAG)
        mem_rag._collection = mock_collection
        mem_rag._embed = CountingEmbedder()

        sim = mem_rag.best_document_similarity(query="What are the pricing tiers?")
        assert sim == 0.0

    def test_document_similarity_calculation(self):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "ids": [["doc1"]],
            "distances": [[0.2]],  # cosine distance 0.2 -> similarity 0.8
        }
        mem_rag = MemRAG.__new__(MemRAG)
        mem_rag._collection = mock_collection
        mem_rag._embed = CountingEmbedder()

        sim = mem_rag.best_document_similarity(query_vec=[0.1, 0.2])
        assert pytest.approx(sim, 0.01) == 0.8

    def test_chroma_exception_returns_zero_gracefully(self):
        mock_collection = MagicMock()
        mock_collection.count.side_effect = RuntimeError("Chroma unavailable")
        mem_rag = MemRAG.__new__(MemRAG)
        mem_rag._collection = mock_collection
        mem_rag._embed = CountingEmbedder()

        sim = mem_rag.best_document_similarity(query="test query")
        assert sim == 0.0


class TestCorpusAwareRouting:
    """Test routing with and without corpus awareness."""

    def test_demo_queries_with_corpus_route_as_documented(self):
        # Demo query 1: "What does login call?"
        # login is in known symbols -> 1 weak signal; 'call' is relationship word -> 1 weak signal -> 2 weak -> CODE
        q1 = "What does login call?"
        res1 = classify(q1, known_symbols=frozenset({"login", "hash_password"}))
        assert res1 == QueryType.CODE

        # Demo query 2: "What are the pricing tiers?"
        # High document similarity -> MEMORY
        q2 = "What are the pricing tiers?"
        res2 = classify(q2, doc_similarity_fn=lambda: 0.85)
        assert res2 == QueryType.MEMORY

    def test_demo_queries_with_empty_corpus_route_simple(self):
        # With empty corpus, "What does login call?" only has 'call' (1 weak signal < 2) -> SIMPLE
        q1 = "What does login call?"
        res1 = classify(q1, known_symbols=frozenset(), doc_similarity_fn=lambda: 0.0)
        assert res1 == QueryType.SIMPLE

        # "What are the pricing tiers?" has no keywords and 0 similarity -> SIMPLE
        q2 = "What are the pricing tiers?"
        res2 = classify(q2, known_symbols=frozenset(), doc_similarity_fn=lambda: 0.0)
        assert res2 == QueryType.SIMPLE

    def test_neo4j_or_chroma_down_never_raises_from_classify(self):
        def crashing_probe():
            raise RuntimeError("Database connection down")

        # Classify must catch and degrade to SIMPLE
        result = classify(
            "What are the pricing tiers?",
            known_symbols=frozenset(),
            doc_similarity_fn=crashing_probe,
        )
        assert result == QueryType.SIMPLE


class TestEmbedOnce:
    """Verify that the query embedding is computed strictly once per request."""

    def test_embedding_computed_once_per_pipeline_request(self):
        counting_embed = CountingEmbedder()

        mock_chroma = MagicMock()
        mock_cache_coll = MagicMock()
        mock_cache_coll.count.return_value = 0
        mock_mem_coll = MagicMock()
        mock_mem_coll.count.return_value = 1
        mock_mem_coll.query.return_value = {
            "ids": [["doc1"]],
            "documents": [["Pricing details"]],
            "metadatas": [[{"source": "pricing.md"}]],
            "distances": [[0.1]],
        }
        mock_mem_coll.get.return_value = {"ids": [], "metadatas": []}

        def get_or_create(name, **kwargs):
            return mock_cache_coll if name == "cache" else mock_mem_coll

        mock_chroma.get_or_create_collection.side_effect = get_or_create

        cache = RAGCacheLite(mock_chroma, counting_embed)
        mem_rag = MemRAG(mock_chroma, counting_embed)
        graph_rag = MagicMock(spec=GraphRAG)
        llm = MagicMock()
        llm.generate.return_value = "Here is the pricing information."

        corpus_index = CorpusIndex(["login"])
        pipeline = Pipeline(
            cache=cache,
            graph_rag=graph_rag,
            mem_rag=mem_rag,
            llm_client=llm,
            corpus_index=corpus_index,
        )

        import uuid
        query_text = "What are the pricing tiers?"
        response = pipeline.handle_query(
            QueryRequest(query=query_text, session_id=uuid.uuid4())
        )

        assert response.query_type == QueryType.MEMORY
        assert response.from_cache is False

        # Query text itself was encoded exactly once across cache check, doc probe, and memory retrieval
        query_encodes = [t for t in counting_embed.encoded_texts if t == query_text]
        assert len(query_encodes) == 1, (
            f"Expected query to be encoded exactly 1 time, but was encoded {len(query_encodes)} times"
        )
