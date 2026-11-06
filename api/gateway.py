"""
API Gateway — FastAPI application exposing ACEF endpoints.

Endpoints:
  - POST /query: Process query through the 5-step Pipeline
  - POST /ingest/code: Ingest Python codebase into Neo4j
  - POST /ingest/document: Upload and ingest .txt/.md/.pdf into Chroma
  - POST /ingest/documents: Batch upload and ingest multiple .txt/.md/.pdf documents into Chroma
  - GET /health: Status check for Neo4j and ChromaDB
"""

import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

import chromadb
from config.settings import get_settings
from core.corpus_index import CorpusIndex
from core.pipeline import Pipeline
from core.schemas import HealthResponse, QueryRequest, QueryResponse
from ingestion.code_ingest import ingest_codebase
from ingestion.document_ingest import ingest_document
from modules.cache import RAGCacheLite
from modules.graph_rag import GraphRAG
from modules.llm_client import LLMClient, LLMServiceError
from modules.mem_rag import MemRAG
from core.classifier import get_classifier

logger = logging.getLogger(__name__)

# ── Global Container for Pipeline & Shared Clients ───────────────────────────
_services = {
    "neo4j_driver": None,
    "chroma_client": None,
    "embedding_model": None,
    "pipeline": None,
    "mem_rag": None,
    "corpus_index": None,
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
    graph_rag = GraphRAG(
        _services["neo4j_driver"],
        max_hops=settings.graph_retrieval_max_hops,
        min_score=settings.graph_retrieval_min_score,
        root=str(Path(settings.demo_codebase_path).resolve()),
    )
    llm_client = LLMClient(settings)

    # 5. Corpus Index
    corpus_index = CorpusIndex()
    if _services["neo4j_driver"] is not None:
        corpus_index.refresh_from_neo4j(_services["neo4j_driver"])
    _services["corpus_index"] = corpus_index

    # 6. Classifier — mode controlled by CLASSIFIER_MODE in .env
    classifier = get_classifier(settings, llm_client)
    logger.info("Query classifier mode: %s", settings.classifier_mode)

    # 7. Pipeline
    _services["mem_rag"] = mem_rag
    _services["llm_client"] = llm_client
    _services["pipeline"] = Pipeline(
        cache=cache,
        graph_rag=graph_rag,
        mem_rag=mem_rag,
        llm_client=llm_client,
        corpus_index=corpus_index,
        classifier=classifier,
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


@app.exception_handler(LLMServiceError)
async def llm_service_error_handler(request: Request, exc: LLMServiceError):
    """Map LLM provider failures to clean JSON status responses without stack traces."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
    )


# ── Static UI Assets & Dashboard Route ────────────────────────────────────────
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", summary="ACEF Web Studio Dashboard", include_in_schema=False)
async def serve_dashboard():
    """Serve the interactive web UI dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "ACEF API Gateway is running. Interactive UI not found."}


@app.get("/graph/data", summary="Get Code Knowledge Graph Data")
async def get_graph_data():
    """Return nodes and edges from Neo4j for frontend graph visualization."""
    if _services.get("neo4j_driver") is None:
        init_services()
    driver = _services.get("neo4j_driver")
    if not driver:
        return {"nodes": [], "links": [], "error": "Neo4j driver not connected"}

    def _fetch():
        nodes_dict = {}
        links = []
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (n)
                    RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props
                """)
                for record in result:
                    nid = str(record["id"])
                    labels = record["labels"] or []
                    props = record["props"] or {}
                    node_type = labels[0] if labels else "Unknown"
                    name = props.get("name") or props.get("path") or nid
                    if node_type == "File":
                        name = Path(str(name)).name
                    nodes_dict[nid] = {
                        "id": nid,
                        "name": str(name),
                        "type": node_type,
                        "kind": props.get("kind", node_type.lower()),
                        "qualname": props.get("qualname", str(name)),
                        "parent_class": props.get("parent_class", ""),
                        "docstring": props.get("docstring", ""),
                        "file_path": props.get("file_path", props.get("path", "")),
                        "lineno": props.get("lineno", 0),
                        "end_lineno": props.get("end_lineno", props.get("lineno", 0)),
                        "source": props.get("source", ""),
                    }

                rel_result = session.run("""
                    MATCH (a)-[r]->(b)
                    RETURN elementId(a) AS source, elementId(b) AS target, type(r) AS type
                """)
                for record in rel_result:
                    src = str(record["source"])
                    tgt = str(record["target"])
                    if src in nodes_dict and tgt in nodes_dict:
                        links.append({
                            "source": src,
                            "target": tgt,
                            "type": record["type"],
                        })

            return {"nodes": list(nodes_dict.values()), "links": links}
        except Exception as e:
            logger.error("Failed to fetch graph data: %s", e)
            return {"nodes": [], "links": [], "error": str(e)}

    return await run_in_threadpool(_fetch)


@app.post("/graph/clear", summary="Clear Neo4j Knowledge Graph")
@app.delete("/graph/clear", summary="Clear Neo4j Knowledge Graph")
async def clear_graph():
    """
    Delete all nodes and relationships from the Neo4j knowledge graph.
    Refreshes the CorpusIndex and clears semantic cache.
    """
    if _services.get("neo4j_driver") is None:
        init_services()
    driver = _services.get("neo4j_driver")
    if not driver:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Neo4j driver not connected")

    def _execute_clear():
        try:
            with driver.session() as session:
                # Count nodes before deletion for response summary
                count_res = session.run("MATCH (n) RETURN count(n) AS cnt").single()
                nodes_count = count_res["cnt"] if count_res else 0

                # Purge all nodes and relationships
                session.run("MATCH (n) DETACH DELETE n")

            # Refresh corpus index if available
            if _services.get("corpus_index") is not None:
                _services["corpus_index"].refresh_from_neo4j(driver)
            if _services.get("pipeline"):
                if hasattr(_services["pipeline"], "corpus_index"):
                    _services["pipeline"].corpus_index.refresh_from_neo4j(driver)
                if hasattr(_services["pipeline"], "graph_rag"):
                    _services["pipeline"].graph_rag._root = None
                if hasattr(_services["pipeline"], "cache"):
                    _services["pipeline"].cache.clear()

            logger.info("Cleared entire Neo4j graph (%d nodes deleted).", nodes_count)
            return {
                "status": "success",
                "message": f"Successfully cleared graph ({nodes_count} nodes deleted)",
                "nodes_deleted": nodes_count,
            }
        except Exception as e:
            logger.error("Failed to clear Neo4j graph: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to clear graph: {str(e)}",
            )

    return await run_in_threadpool(_execute_clear)


@app.post("/cache/clear", summary="Clear Semantic Cache")
@app.delete("/cache/clear", summary="Clear Semantic Cache")
async def clear_cache_endpoint():
    """
    Clear all cached query-response pairs from RAGCacheLite.
    Returns confirmation of purged cache entries.
    """
    def _execute_cache_clear():
        cleared = False
        pipeline = _services.get("pipeline")
        if pipeline and hasattr(pipeline, "cache") and pipeline.cache is not None:
            pipeline.cache.clear()
            cleared = True
        elif _services.get("chroma_client") and _services.get("embedding_model"):
            cache = RAGCacheLite(_services["chroma_client"], _services["embedding_model"])
            cache.clear()
            cleared = True

        if cleared:
            logger.info("Cleared semantic cache.")
            return {"status": "success", "message": "Semantic cache cleared successfully"}

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear semantic cache: cache service not initialized",
        )

    return await run_in_threadpool(_execute_cache_clear)


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
    return await run_in_threadpool(pipeline.handle_query, request)


@app.post("/ingest/code", summary="Ingest Python Codebase")
async def ingest_code(payload: Optional[IngestCodeRequest] = None, root_path: Optional[str] = None) -> dict:
    """
    Parse a Python codebase into Neo4j nodes and edges.
    Accepts root_path via JSON body or query parameter.
    """
    settings = get_settings()
    path_to_ingest = root_path
    if not path_to_ingest and payload:
        path_to_ingest = payload.root_path
    if not path_to_ingest:
        path_to_ingest = settings.demo_codebase_path

    resolved_path = Path(path_to_ingest).resolve()

    # I-10: Path traversal and root allowlist validation
    if "*" not in settings.allowed_ingest_roots and settings.allowed_ingest_roots:
        allowed_roots = [Path(p).resolve() for p in settings.allowed_ingest_roots]
        is_allowed = any(
            resolved_path == r or r in resolved_path.parents for r in allowed_roots
        )
        if not is_allowed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Path '{path_to_ingest}' is outside allowed ingest roots: {settings.allowed_ingest_roots}",
            )

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
        def _execute_code_ingest():
            summary = ingest_codebase(str(resolved_path), driver)
            if _services.get("pipeline"):
                if hasattr(_services["pipeline"], "cache"):
                    _services["pipeline"].cache.clear()
                if hasattr(_services["pipeline"], "corpus_index"):
                    _services["pipeline"].corpus_index.refresh_from_neo4j(driver)
                if hasattr(_services["pipeline"], "graph_rag"):
                    _services["pipeline"].graph_rag._root = resolved_path.resolve().as_posix()
            return summary

        summary = await run_in_threadpool(_execute_code_ingest)
        skipped_count = summary.get("skipped_non_python", 0)
        skipped_exts = summary.get("skipped_extensions", [])
        msg = f"Ingested {summary.get('files', 0)} Python files."
        if skipped_count > 0:
            ext_str = f" ({', '.join(skipped_exts)})" if skipped_exts else ""
            msg += f" Skipped {skipped_count} non-Python file{'s' if skipped_count != 1 else ''}{ext_str}."

        return {"status": "success", "message": msg, "summary": summary}
    except Exception as e:
        logger.error("Failed to ingest codebase: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest codebase: {str(e)}",
        )


async def _process_single_document_file(
    upload_file: UploadFile,
    max_bytes: int,
    mem_rag,
    embed_model,
) -> tuple[Optional[int], Optional[str], int]:
    """
    Stream, validate, and ingest a single document file into ChromaDB.

    Returns
    -------
    tuple[Optional[int], Optional[str], int]
        (chunk_count, error_message, http_status_code)
    """
    filename = upload_file.filename or ""
    if not filename:
        return None, "Missing file name", status.HTTP_400_BAD_REQUEST

    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        return (
            None,
            f"Unsupported file type '{suffix}'. Allowed: .txt, .md, .pdf",
            status.HTTP_400_BAD_REQUEST,
        )

    total_bytes = 0
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            while chunk := await upload_file.read(65536):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    tmp.close()
                    Path(tmp.name).unlink(missing_ok=True)
                    return (
                        None,
                        f"File exceeds maximum allowed upload size of {max_bytes} bytes",
                        status.HTTP_413_CONTENT_TOO_LARGE,
                    )
                tmp.write(chunk)
            tmp_path = Path(tmp.name)

        def _execute_doc_ingest():
            return ingest_document(
                file_path=str(tmp_path),
                chroma_collection=mem_rag.collection,
                embedding_model=embed_model,
                source_name=filename,
            )

        chunk_count = await run_in_threadpool(_execute_doc_ingest)
        if chunk_count == 0:
            return (
                None,
                "No extractable text was found in this file. Image-only PDFs are not supported.",
                status.HTTP_400_BAD_REQUEST,
            )
        return chunk_count, None, status.HTTP_200_OK

    except ValueError as e:
        logger.warning("Document ingestion rejected for %s: %s", filename, e)
        return None, str(e), status.HTTP_400_BAD_REQUEST
    except Exception as e:
        logger.error("Document ingestion failed for %s: %s", filename, e, exc_info=True)
        return None, f"Document ingestion failed: {str(e)}", status.HTTP_500_INTERNAL_SERVER_ERROR
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


async def _handle_documents_ingest(
    file_list: list[UploadFile],
    is_batch_endpoint: bool = False,
) -> dict:
    valid_files = [f for f in file_list if f is not None and f.filename]
    if not valid_files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing file name")

    if _services["mem_rag"] is None:
        init_services()

    mem_rag = _services["mem_rag"]
    embed_model = _get_embedding_model()
    settings = get_settings()

    # Single-file mode: if called from /ingest/document with exactly 1 file,
    # preserve the exact response schema and direct HTTP exception semantics.
    if len(valid_files) == 1 and not is_batch_endpoint:
        f = valid_files[0]
        chunk_count, err_msg, err_code = await _process_single_document_file(
            f, settings.max_upload_bytes, mem_rag, embed_model
        )
        if err_msg is not None:
            raise HTTPException(status_code=err_code, detail=err_msg)

        if _services.get("pipeline") and hasattr(_services["pipeline"], "cache"):
            _services["pipeline"].cache.clear()

        return {
            "status": "success",
            "filename": f.filename,
            "chunks_stored": chunk_count,
            "chunks_created": chunk_count,
        }

    # Multi-file or batch mode
    results = []
    successful = 0
    total_chunks = 0

    for f in valid_files:
        chunk_count, err_msg, err_code = await _process_single_document_file(
            f, settings.max_upload_bytes, mem_rag, embed_model
        )
        if err_msg is not None:
            results.append({
                "filename": f.filename,
                "status": "error",
                "error": err_msg,
                "status_code": err_code,
            })
        else:
            successful += 1
            total_chunks += chunk_count
            results.append({
                "filename": f.filename,
                "status": "success",
                "chunks_stored": chunk_count,
                "chunks_created": chunk_count,
            })

    if successful > 0 and _services.get("pipeline") and hasattr(_services["pipeline"], "cache"):
        _services["pipeline"].cache.clear()

    failed_count = len(valid_files) - successful
    if successful == len(valid_files):
        overall_status = "success"
    elif successful > 0:
        overall_status = "partial_success"
    else:
        overall_status = "error"

    if successful == 0:
        err_details = "; ".join(f"{r['filename']}: {r.get('error', 'Failed')}" for r in results)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"All {len(valid_files)} file(s) failed ingestion: {err_details}",
        )

    return {
        "status": overall_status,
        "total_files": len(valid_files),
        "successful": successful,
        "failed": failed_count,
        "total_chunks_stored": total_chunks,
        "results": results,
    }


@app.post("/ingest/document", summary="Ingest Document File(s)")
async def ingest_doc(
    file: Optional[list[UploadFile]] = File(None),
    files: Optional[list[UploadFile]] = File(None),
) -> dict:
    """
    Upload and ingest one or more .txt, .md, or .pdf documents into Chroma's memory collection.
    Accepts single or multiple files under 'file' or 'files'.
    """
    file_list = [f for f in ((file or []) + (files or [])) if f is not None]
    return await _handle_documents_ingest(file_list, is_batch_endpoint=False)


@app.post("/ingest/documents", summary="Batch Ingest Multiple Documents")
async def ingest_docs(
    files: Optional[list[UploadFile]] = File(None),
    file: Optional[list[UploadFile]] = File(None),
) -> dict:
    """
    Upload and batch ingest multiple .txt, .md, or .pdf documents into Chroma's memory collection.
    Returns structured results for each file.
    """
    file_list = [f for f in ((files or []) + (file or [])) if f is not None]
    return await _handle_documents_ingest(file_list, is_batch_endpoint=True)


@app.get("/health", response_model=HealthResponse, summary="System Health Check")
async def health_check() -> HealthResponse:
    """
    Verify Neo4j connectivity and ChromaDB readiness.
    """
    def _execute_health():
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

        llm_ok = False
        llm_client = _services.get("llm_client")
        if llm_client and hasattr(llm_client, "is_configured"):
            llm_ok = llm_client.is_configured()
        elif _services.get("pipeline") and hasattr(_services["pipeline"], "llm"):
            llm_ok = _services["pipeline"].llm.is_configured()

        settings = get_settings()
        active_model = settings.gemini_model if settings.llm_provider == "gemini" else settings.groq_model

        return HealthResponse(
            status=overall_status,
            neo4j_connected=neo4j_ok,
            chroma_ready=chroma_ok,
            llm_configured=llm_ok,
            provider=settings.llm_provider,
            model=active_model,
            classifier_mode=settings.classifier_mode,
        )

    return await run_in_threadpool(_execute_health)
