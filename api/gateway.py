"""
API Gateway — FastAPI application exposing ACEF endpoints.

Endpoints:
  - POST /query: Process query through the 5-step Pipeline
  - POST /ingest/code: Ingest Python codebase into Neo4j
  - POST /ingest/document: Upload and ingest .txt/.md/.pdf into Chroma
  - GET /health: Status check for Neo4j and ChromaDB
"""

import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

import chromadb
from config.settings import get_settings
from core.pipeline import Pipeline
from core.schemas import HealthResponse, QueryRequest, QueryResponse
from ingestion.code_ingest import ingest_codebase
from ingestion.document_ingest import ingest_document
from modules.cache import RAGCacheLite
from modules.graph_rag import GraphRAG
from modules.llm_client import LLMClient
from modules.mem_rag import MemRAG

logger = logging.getLogger(__name__)

# ── Global Container for Pipeline & Shared Clients ───────────────────────────
_services = {
    "neo4j_driver": None,
    "chroma_client": None,
    "embedding_model": None,
    "pipeline": None,
    "mem_rag": None,
}


def _get_embedding_model():
    """Lazy-load the SentenceTransformer model once."""
    if _services["embedding_model"] is None:
        from sentence_transformers import SentenceTransformer
        settings = get_settings()
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _services["embedding_model"] = SentenceTransformer(settings.embedding_model)
    return _services["embedding_model"]


def init_services():
    """Initialize shared drivers, stores, modules, and pipeline."""
    if _services["pipeline"] is not None:
        return _services

    settings = get_settings()

    # 1. Neo4j Driver
    if _services["neo4j_driver"] is None:
        try:
            _services["neo4j_driver"] = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
        except Exception as e:
            logger.warning("Could not initialize Neo4j driver at startup: %s", e)

    # 2. Chroma Client (persistent local directory)
    if _services["chroma_client"] is None:
        persist_dir = Path(settings.chroma_persist_dir).resolve()
        persist_dir.mkdir(parents=True, exist_ok=True)
        _services["chroma_client"] = chromadb.PersistentClient(path=str(persist_dir))

    # 3. Embedding Model
    embed_model = _get_embedding_model()

    # 4. Modules
    cache = RAGCacheLite(_services["chroma_client"], embed_model)
    mem_rag = MemRAG(_services["chroma_client"], embed_model)
    graph_rag = GraphRAG(_services["neo4j_driver"])
    llm_client = LLMClient(settings)

    # 5. Pipeline
    _services["mem_rag"] = mem_rag
    _services["pipeline"] = Pipeline(
        cache=cache,
        graph_rag=graph_rag,
        mem_rag=mem_rag,
        llm_client=llm_client,
    )
    return _services


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifespan context."""
    init_services()
    yield
    # Shutdown logic
    if _services["neo4j_driver"] is not None:
        _services["neo4j_driver"].close()


app = FastAPI(
    title="ACEF — Lean Build",
    description="Agentic Context Engineering Framework with GraphRAG, MemRAG, and Semantic Caching",
    version="1.0.0",
    lifespan=lifespan,
)


def get_pipeline() -> Pipeline:
    """Dependency helper returning the Pipeline singleton."""
    if _services["pipeline"] is None:
        init_services()
    return _services["pipeline"]


# ── Request Models ────────────────────────────────────────────────────────────

class IngestCodeRequest(BaseModel):
    """Payload for code ingestion."""
    root_path: str = Field(
        default="./data/demo_codebase",
        description="Path to Python codebase root",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, summary="Execute Query Pipeline")
async def handle_query(request: QueryRequest) -> QueryResponse:
    """
    Execute the 5-step query processing pipeline.
    Empty queries (<1 char) or oversized queries (>4096 chars) return 422 automatically.
    """
    pipeline = get_pipeline()
    return pipeline.handle_query(request)


@app.post("/ingest/code", summary="Ingest Python Codebase")
async def ingest_code(payload: Optional[IngestCodeRequest] = None, root_path: Optional[str] = None) -> dict:
    """
    Parse a Python codebase into Neo4j nodes and edges.
    Accepts root_path via JSON body or query parameter.
    """
    path_to_ingest = root_path
    if not path_to_ingest and payload:
        path_to_ingest = payload.root_path
    if not path_to_ingest:
        path_to_ingest = get_settings().demo_codebase_path

    resolved_path = Path(path_to_ingest).resolve()
    if not resolved_path.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Codebase path does not exist: {path_to_ingest}",
        )

    driver = _services.get("neo4j_driver")
    if driver is None:
        init_services()
        driver = _services.get("neo4j_driver")

    try:
        summary = ingest_codebase(str(resolved_path), driver)
        if _services.get("pipeline") and hasattr(_services["pipeline"], "cache"):
            _services["pipeline"].cache.clear()
        return {"status": "success", "summary": summary}
    except Exception as e:
        logger.error("Failed to ingest codebase: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest codebase: {str(e)}",
        )


@app.post("/ingest/document", summary="Ingest Document File")
async def ingest_doc(file: UploadFile = File(...)) -> dict:
    """
    Upload and ingest a .txt, .md, or .pdf document into Chroma's memory collection.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: .txt, .md, .pdf",
        )

    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        if _services["mem_rag"] is None:
            init_services()

        mem_rag = _services["mem_rag"]
        embed_model = _get_embedding_model()

        chunk_count = ingest_document(
            file_path=str(tmp_path),
            chroma_collection=mem_rag.collection,
            embedding_model=embed_model,
        )
        if _services.get("pipeline") and hasattr(_services["pipeline"], "cache"):
            _services["pipeline"].cache.clear()
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_stored": chunk_count,
        }
    except Exception as e:
        logger.error("Document ingestion failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.get("/health", response_model=HealthResponse, summary="System Health Check")
async def health_check() -> HealthResponse:
    """
    Verify Neo4j connectivity and ChromaDB readiness.
    """
    neo4j_ok = False
    driver = _services.get("neo4j_driver")
    if driver:
        try:
            with driver.session() as session:
                result = session.run("RETURN 1 AS ping").single()
                if result and result["ping"] == 1:
                    neo4j_ok = True
        except Exception:
            neo4j_ok = False

    chroma_ok = False
    chroma_client = _services.get("chroma_client")
    if chroma_client:
        try:
            # PersistentClient heartbeat check
            chroma_client.heartbeat()
            chroma_ok = True
        except Exception:
            chroma_ok = False

    overall_status = "ok" if (neo4j_ok and chroma_ok) else "degraded"

    return HealthResponse(
        status=overall_status,
        neo4j_connected=neo4j_ok,
        chroma_ready=chroma_ok,
    )
