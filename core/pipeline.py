"""
Pipeline Orchestration — the central flow connecting cache, classifier, retrieval, and LLM.

Executes the 5 steps:
  1. Cache check (RAGCacheLite) -> return if hit
  2. Query classification (query_analyzer.classify) -> SIMPLE, CODE, MEMORY
  3. Context retrieval branch:
     - CODE -> GraphRAG (Neo4j)
     - MEMORY -> MemRAG (Chroma)
     - SIMPLE -> skip retrieval
  4. Call LLM (Gemini or Groq) with retrieved chunks
  5. Write-back:
     - Store in cache
     - If MEMORY, append turn to session history
"""

import logging
import time
from uuid import uuid4

from config.settings import get_settings
from core.query_analyzer import classify
from core.schemas import QueryRequest, QueryResponse, QueryType, RetrievedChunk
from modules.cache import RAGCacheLite
from modules.graph_rag import GraphRAG
from modules.llm_client import LLMClient
from modules.mem_rag import MemRAG

logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end query processing pipeline."""

    def __init__(
        self,
        cache: RAGCacheLite,
        graph_rag: GraphRAG,
        mem_rag: MemRAG,
        llm_client: LLMClient,
    ):
        """
        Parameters
        ----------
        cache : RAGCacheLite
            Semantic cache instance.
        graph_rag : GraphRAG
            Code graph retrieval instance.
        mem_rag : MemRAG
            Document & conversation memory retrieval instance.
        llm_client : LLMClient
            Unified Gemini/Groq LLM client.
        """
        self.cache = cache
        self.graph_rag = graph_rag
        self.mem_rag = mem_rag
        self.llm = llm_client
        self.settings = get_settings()

    def handle_query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a user query through the 5-step pipeline.

        Parameters
        ----------
        request : QueryRequest
            User query and session ID.

        Returns
        -------
        QueryResponse
            Structured response with classification, cache flag, sources, and latency.
        """
        start = time.time()

        # ── 1. Cache Check ───────────────────────────────────────────────────
        try:
            cached = self.cache.lookup(request.query)
            if cached is not None:
                return QueryResponse(
                    session_id=request.session_id,
                    response=cached,
                    query_type=QueryType.SIMPLE,
                    from_cache=True,
                    sources=[],
                    latency_ms=(time.time() - start) * 1000,
                )
        except Exception as e:
            logger.warning("Cache lookup failed (continuing to pipeline): %s", e)

        # ── 2. Classification ────────────────────────────────────────────────
        turn_count = self.mem_rag.session_turn_count(str(request.session_id))
        query_type = classify(request.query, turn_count)
        logger.info("Query classified as: %s (turn_count=%d)", query_type.value, turn_count)

        # ── 3. Context Retrieval Branch ──────────────────────────────────────
        chunks: list[RetrievedChunk] = []
        try:
            if query_type == QueryType.CODE:
                chunks = self.graph_rag.retrieve(request.query)
            elif query_type == QueryType.MEMORY:
                chunks = self.mem_rag.retrieve(
                    request.query,
                    str(request.session_id),
                    top_k=self.settings.memory_top_k,
                )
            # SIMPLE: chunks remains empty
        except Exception as e:
            # Important: log and proceed with empty context rather than failing
            logger.error("Context retrieval failed for %s query: %s", query_type.value, e, exc_info=True)
            chunks = []

        # ── 4. Call LLM ──────────────────────────────────────────────────────
        context_texts = [c.text for c in chunks]
        response_text = self.llm.generate(
            query=request.query,
            query_type=query_type.value,
            context_chunks=context_texts,
        )

        # ── 5. Write-back ────────────────────────────────────────────────────
        query_id = str(uuid4())
        try:
            self.cache.store(request.query, response_text, query_id)
        except Exception as e:
            logger.warning("Cache store failed: %s", e)

        if query_type == QueryType.MEMORY:
            try:
                self.mem_rag.append_turn(str(request.session_id), request.query, response_text)
            except Exception as e:
                logger.warning("Conversation append_turn failed: %s", e)

        latency_ms = (time.time() - start) * 1000
        sources = [c.source for c in chunks]

        return QueryResponse(
            session_id=request.session_id,
            response=response_text,
            query_type=query_type,
            from_cache=False,
            sources=sources,
            latency_ms=latency_ms,
        )
