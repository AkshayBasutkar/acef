"""
Live integration test suite — requires running Neo4j container and a valid LLM API key.

Deselected by default via pytest.ini (addopts = -m "not live").
Run with:
    pytest -m live
"""

import os
from pathlib import Path
import uuid
import pytest

from config.settings import get_settings
from core.corpus_index import CorpusIndex
from core.pipeline import Pipeline
from core.schemas import QueryRequest, QueryType
from ingestion.code_ingest import ingest_codebase
from ingestion.document_ingest import ingest_document
from modules.cache import RAGCacheLite
from modules.graph_rag import GraphRAG
from modules.llm_client import LLMClient
from modules.mem_rag import MemRAG


@pytest.mark.live
class TestLivePipelineEndToEnd:
    """Live verification against real Neo4j, Chroma, and LLM."""

    @pytest.fixture(autouse=True)
    def setup_services(self):
        settings = get_settings()
        active_key = settings.gemini_api_key if settings.llm_provider == "gemini" else settings.groq_api_key
        if not active_key or active_key.startswith("your_"):
            pytest.skip(f"Live API key not configured for provider '{settings.llm_provider}'")

        from neo4j import GraphDatabase
        try:
            driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            with driver.session() as s:
                s.run("RETURN 1").single()
        except Exception as e:
            pytest.skip(f"Neo4j offline at {settings.neo4j_uri}: {e}")

        import chromadb
        from sentence_transformers import SentenceTransformer

        persist_dir = Path(settings.chroma_persist_dir).resolve()
        persist_dir.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        embed_model = SentenceTransformer(settings.embedding_model)

        cache = RAGCacheLite(chroma_client, embed_model)
        cache.clear()
        mem_rag = MemRAG(chroma_client, embed_model)
        graph_rag = GraphRAG(driver)
        llm = LLMClient(settings)
        corpus_index = CorpusIndex()
        corpus_index.refresh_from_neo4j(driver)

        self.driver = driver
        self.pipeline = Pipeline(
            cache=cache,
            graph_rag=graph_rag,
            mem_rag=mem_rag,
            llm_client=llm,
            corpus_index=corpus_index,
        )
        self.settings = settings
        yield
        driver.close()

    def test_live_end_to_end_flow(self):
        # 1. Ingest demo codebase
        code_path = Path(self.settings.demo_codebase_path).resolve()
        assert code_path.exists(), f"Demo codebase path missing: {code_path}"
        code_summary = ingest_codebase(str(code_path), self.driver)
        assert code_summary["files_scanned"] >= 3
        assert code_summary["symbols_created"] >= 3
        assert code_summary["call_relationships"] >= 2
        assert code_summary["import_relationships"] >= 1
        self.pipeline.corpus_index.refresh_from_neo4j(self.driver)

        # 2. Ingest demo document
        doc_path = Path("data/demo_documents/pricing.md").resolve()
        assert doc_path.exists(), f"Demo document missing: {doc_path}"
        chunks_count = ingest_document(
            str(doc_path),
            self.pipeline.mem_rag.collection,
            self.pipeline.mem_rag._embed,
        )
        assert chunks_count >= 1

        session_a = uuid.uuid4()

        # 3. Code query: calls relationship
        res_code1 = self.pipeline.handle_query(
            QueryRequest(query="What does the login function call?", session_id=session_a)
        )
        assert res_code1.query_type == QueryType.CODE
        assert any("auth.py" in s for s in res_code1.sources)
        assert "hash_password" in res_code1.response.lower()

        # 4. Code query: function body reasoning
        res_code2 = self.pipeline.handle_query(
            QueryRequest(query="Why might login fail?", session_id=session_a)
        )
        assert res_code2.query_type == QueryType.CODE
        resp_lower = res_code2.response.lower()
        assert "password" in resp_lower or "user" in resp_lower or "hash" in resp_lower

        # 5. Document query: pricing tiers
        res_doc = self.pipeline.handle_query(
            QueryRequest(query="What are the pricing tiers?", session_id=session_a)
        )
        assert res_doc.query_type == QueryType.MEMORY
        assert any("pricing.md" in s for s in res_doc.sources)
        resp_doc_lower = res_doc.response.lower()
        assert "free" in resp_doc_lower or "pro" in resp_doc_lower or "enterprise" in resp_doc_lower

        # 6. Simple query: Cache miss then hit
        res_simple1 = self.pipeline.handle_query(
            QueryRequest(query="What is the capital of France?", session_id=session_a)
        )
        assert res_simple1.query_type == QueryType.SIMPLE
        assert res_simple1.from_cache is False

        res_simple2 = self.pipeline.handle_query(
            QueryRequest(query="What is the capital of France?", session_id=session_a)
        )
        assert res_simple2.query_type == QueryType.SIMPLE
        assert res_simple2.from_cache is True

        # 7. Session isolation
        session_b = uuid.uuid4()
        res_b = self.pipeline.handle_query(
            QueryRequest(query="What did we discuss earlier?", session_id=session_b)
        )
        assert "capital of france" not in res_b.response.lower()
