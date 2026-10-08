"""
Live End-to-End Demo Script & Transcript Generator.

Usage:
    python -m scripts.demo_run --provider gemini|groq [--url http://localhost:8000]

Executes the full acceptance sequence and writes docs/e2e_transcript_<provider>.md:
  1. Health check (/health)
  2. Codebase ingestion (data/demo_codebase)
  3. Document ingestion (data/demo_documents/pricing.md)
  4. Code Query 1: "What does the login function call?"
  5. Code Query 2: "Why might login fail?"
  6. Memory Query: "What are the pricing tiers?"
  7. Simple Query (miss): "What is the capital of France?"
  8. Simple Query (hit):  "What is the capital of France?"
  9. Session Isolation: Session B memory query does not access Session A history.
"""

import argparse
import datetime
import os
import sys
import uuid
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def run_demo(provider: str, url: str | None = None) -> bool:
    settings = get_settings()
    # Override provider if explicitly passed
    if provider:
        object.__setattr__(settings, "llm_provider", provider)

    active_key = settings.gemini_api_key if settings.llm_provider == "gemini" else settings.groq_api_key
    if not active_key or active_key.startswith("your_"):
        print(f"Error: No live API key configured in .env for provider '{settings.llm_provider}'.", file=sys.stderr)
        return False

    print("=" * 70)
    print(f" ACEF Live End-to-End Verification -- Provider: {settings.llm_provider.upper()}")
    print("=" * 70)

    # 1. Connect to Neo4j
    from neo4j import GraphDatabase
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        with driver.session() as s:
            s.run("RETURN 1").single()
        print(f"[PASS] Step 1: Neo4j reachable at {settings.neo4j_uri}")
    except Exception as e:
        print(f"[FAIL] Step 1: Neo4j connection failed: {e}", file=sys.stderr)
        return False

    # 2. Setup Chroma & Modules
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

    pipeline = Pipeline(
        cache=cache,
        graph_rag=graph_rag,
        mem_rag=mem_rag,
        llm_client=llm,
        corpus_index=corpus_index,
    )
    print("[PASS] Step 2: Shared services and modules initialized")

    transcript_entries = []

    # 3. Ingest Demo Codebase
    code_path = Path(settings.demo_codebase_path).resolve()
    code_summary = ingest_codebase(str(code_path), driver)
    pipeline.corpus_index.refresh_from_neo4j(driver)
    assert code_summary["files_scanned"] >= 3
    assert code_summary["symbols_created"] >= 3
    print(f"[PASS] Step 3: Ingested codebase -- files={code_summary['files_scanned']}, symbols={code_summary['symbols_created']}")

    # 4. Ingest Demo Document
    doc_path = Path("data/demo_documents/pricing.md").resolve()
    chunks_count = ingest_document(str(doc_path), mem_rag.collection, embed_model)
    assert chunks_count >= 1
    print(f"[PASS] Step 4: Ingested document '{doc_path.name}' -- chunks={chunks_count}")

    session_a = uuid.uuid4()

    # Step 5: Code Query 1
    q1 = "What does the login function call?"
    res1 = pipeline.handle_query(QueryRequest(query=q1, session_id=session_a))
    assert res1.query_type == QueryType.CODE
    assert "hash_password" in res1.response.lower()
    print(f"[PASS] Step 5: Code query 1 -- type={res1.query_type.value}, latency={res1.latency_ms:.1f}ms")
    transcript_entries.append({"step": "Step 5 · Code Query (calls)", "query": q1, "resp": res1})

    # Step 6: Code Query 2
    q2 = "Why might login fail?"
    res2 = pipeline.handle_query(QueryRequest(query=q2, session_id=session_a))
    assert res2.query_type == QueryType.CODE
    print(f"[PASS] Step 6: Code query 2 (body reasoning) -- latency={res2.latency_ms:.1f}ms")
    transcript_entries.append({"step": "Step 6 · Code Query (body reasoning)", "query": q2, "resp": res2})

    # Step 7: Memory Query
    q3 = "What are the pricing tiers?"
    res3 = pipeline.handle_query(QueryRequest(query=q3, session_id=session_a))
    assert res3.query_type == QueryType.MEMORY
    print(f"[PASS] Step 7: Memory query -- type={res3.query_type.value}, latency={res3.latency_ms:.1f}ms")
    transcript_entries.append({"step": "Step 7 · Memory Query (document)", "query": q3, "resp": res3})

    # Step 8: Simple Query Miss
    q4 = "What is the capital of France?"
    res4 = pipeline.handle_query(QueryRequest(query=q4, session_id=session_a))
    assert res4.query_type == QueryType.SIMPLE
    assert res4.from_cache is False
    print(f"[PASS] Step 8: Simple query (cache miss) -- latency={res4.latency_ms:.1f}ms")
    transcript_entries.append({"step": "Step 8 · Simple Query (cache miss)", "query": q4, "resp": res4})

    # Step 9: Simple Query Hit
    res5 = pipeline.handle_query(QueryRequest(query=q4, session_id=session_a))
    assert res5.query_type == QueryType.SIMPLE
    assert res5.from_cache is True
    print(f"[PASS] Step 9: Simple query (cache hit) -- latency={res5.latency_ms:.1f}ms")
    transcript_entries.append({"step": "Step 9 · Simple Query (cache hit)", "query": q4, "resp": res5})

    # Step 10: Session Isolation
    session_b = uuid.uuid4()
    q6 = "What did we discuss earlier?"
    res6 = pipeline.handle_query(QueryRequest(query=q6, session_id=session_b))
    assert "capital of france" not in res6.response.lower()
    print(f"[PASS] Step 10: Session isolation verified")
    transcript_entries.append({"step": "Step 10 · Session Isolation Check", "query": q6, "resp": res6})

    # Write transcript
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    transcript_file = docs_dir / f"e2e_transcript_{settings.llm_provider}.md"

    model_name = settings.gemini_model if settings.llm_provider == "gemini" else settings.groq_model
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# Live E2E Verification Transcript — {settings.llm_provider.upper()}",
        f"",
        f"- **Date:** {now_str}",
        f"- **Provider:** {settings.llm_provider}",
        f"- **Model:** `{model_name}`",
        f"- **Status:** All 10 live acceptance steps passed",
        f"",
        f"---",
        f"",
    ]

    for item in transcript_entries:
        resp = item["resp"]
        lines.append(f"### {item['step']}")
        lines.append(f"- **Query:** `{item['query']}`")
        lines.append(f"- **Classified:** `{resp.query_type.value}` | **From Cache:** `{resp.from_cache}` | **Latency:** `{resp.latency_ms:.1f} ms`")
        if resp.sources:
            lines.append(f"- **Sources:** {', '.join(resp.sources)}")
        lines.append(f"- **Response:**\n> {resp.response.strip()}")
        lines.append(f"")

    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("=" * 70)
    print(f"SUCCESS: Transcript saved to {transcript_file}")
    driver.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Run ACEF live end-to-end verification")
    parser.add_argument("--provider", choices=["gemini", "groq"], default="gemini")
    parser.add_argument("--url", type=str, default=None)
    args = parser.parse_args()

    success = run_demo(provider=args.provider, url=args.url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
