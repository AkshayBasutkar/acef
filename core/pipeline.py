"""
Pipeline Orchestration — the central flow connecting cache, classifier, retrieval, and LLM.

Executes the 5 steps:
  1. Cache check (RAGCacheLite) -> return if hit
  2. Query classification (heuristic or LLM via core.classifier) -> SIMPLE, CODE, MEMORY
  3. Context retrieval branch:
     - CODE -> GraphRAG (Neo4j)
     - MEMORY -> MemRAG (Chroma)
     - SIMPLE -> skip retrieval
  4. Call LLM (Gemini or Groq) with retrieved chunks
  5. Write-back:
     - Store in cache
     - Append turn to session history

The active classifier backend is controlled by settings.classifier_mode:
  "heuristic" (default) | "llm" | "local"
"""

import logging
import time
from uuid import uuid4

from config.settings import get_settings
from core.classifier import ClassificationContext, get_classifier
from core.corpus_index import CorpusIndex
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
        corpus_index: CorpusIndex | None = None,
        classifier=None,
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
        corpus_index : CorpusIndex, optional
            Known symbols index for corpus-aware classification.
        classifier : HeuristicClassifier | LLMClassifier | LocalClassifier, optional
            Query classifier backend. If None, one is created from settings
            (classifier_mode). Inject explicitly in tests for determinism.
        """
        self.cache = cache
        self.graph_rag = graph_rag
        self.mem_rag = mem_rag
        self.llm = llm_client
        self.corpus_index = corpus_index if corpus_index is not None else CorpusIndex()
        self.settings = get_settings()
        self.classifier = classifier if classifier is not None else get_classifier(
            self.settings, llm_client
        )

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

        # Embed once per request to avoid redundant encoding across components
        query_vec = None
        try:
            if hasattr(self.mem_rag, "embed_query"):
                query_vec = self.mem_rag.embed_query(request.query)
        except Exception as e:
            logger.warning("Failed to precompute query embedding: %s", e)

        # ── 1. Cache Check ───────────────────────────────────────────────────
        try:
            cached = self.cache.lookup(request.query, query_vec=query_vec)
            if cached is not None:
                # I-02: Record cache hits in conversation memory so full session history is preserved
                try:
                    self.mem_rag.append_turn(
                        str(request.session_id),
                        request.query,
                        cached,
                        query_type=QueryType.SIMPLE,
                    )
                except Exception as e:
                    logger.warning("Conversation append_turn failed on cache hit: %s", e)

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
        last_turn = self.mem_rag.last_turn_type(str(request.session_id))

        doc_probe = None
        if hasattr(self.mem_rag, "best_document_similarity"):
            doc_probe = lambda: self.mem_rag.best_document_similarity(
                query=request.query, query_vec=query_vec
            )

        ctx = ClassificationContext(
            session_turn_count=turn_count,
            known_symbols=self.corpus_index.symbols,
            doc_similarity_fn=doc_probe,
            last_turn_type=last_turn,
        )
        query_type = self.classifier.classify(request.query, ctx)
        logger.info(
            "Query classified as: %s (mode=%s, turn_count=%d, last_turn=%s)",
            query_type.value,
            self.settings.classifier_mode,
            turn_count,
            last_turn,
        )

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
                    query_vec=query_vec,
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
        # I-01: Cache only SIMPLE answers; CODE and MEMORY answers depend on state
        if query_type == QueryType.SIMPLE and response_text and response_text.strip():
            try:
                self.cache.store(request.query, response_text, query_id)
            except Exception as e:
                logger.warning("Cache store failed: %s", e)

        # I-02: Record EVERY successful turn in conversation memory regardless of query type
        if response_text and response_text.strip():
            try:
                self.mem_rag.append_turn(
                    str(request.session_id),
                    request.query,
                    response_text,
                    query_type=query_type,
                )
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
