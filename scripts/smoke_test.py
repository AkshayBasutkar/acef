"""
Smoke Test Script — verifies system readiness.

Modes:
  1. Default (CI / offline mode):
     Runs fast, self-contained checks without requiring running external
     services (Docker / Neo4j or external LLM API keys).
     Usage: python -m scripts.smoke_test

  2. Live verification (--require-live):
     Verifies real Neo4j connectivity, real ChromaDB persistence,
     real live LLM call ("pong" check), and real query pipeline.
     Usage: python -m scripts.smoke_test --require-live [--url http://localhost:8000]
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from core.schemas import QueryRequest, QueryType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smoke_test")


def run_offline_checks() -> bool:
    """Run fast in-memory checks for CI without requiring external dependencies."""
    print("=" * 65)
    print(" ACEF Smoke Test -- Offline / Self-Contained Mode")
    print("=" * 65)
    all_ok = True

    # 1. Config Check
    try:
        settings = get_settings()
        print(f"[PASS] Configuration: loaded successfully (provider: {settings.llm_provider})")
    except Exception as e:
        print(f"[FAIL] Configuration: failed to load settings ({e})")
        all_ok = False

    # 2. Chroma in-memory check
    try:
        import chromadb
        client = chromadb.EphemeralClient()
        col = client.get_or_create_collection("ci_smoke_test")
        col.add(ids=["t1"], documents=["test doc"], metadatas=[{"type": "test"}])
        assert col.count() == 1
        client.delete_collection("ci_smoke_test")
        print("[PASS] ChromaDB: in-memory client read/write verified")
    except Exception as e:
        print(f"[FAIL] ChromaDB: failed in-memory test ({e})")
        all_ok = False

    # 3. Embeddings check
    try:
        # Check that sentence_transformers is importable and works
        from sentence_transformers import SentenceTransformer
        settings = get_settings()
        model = SentenceTransformer(settings.embedding_model)
        vec = model.encode("smoke test query")
        assert len(vec) > 0
        print(f"[PASS] Embeddings: model '{settings.embedding_model}' loaded and vector generated (dim={len(vec)})")
    except Exception as e:
        print(f"[FAIL] Embeddings: failed to encode ({e})")
        all_ok = False

    # 4. Neo4j mock / simulation check
    try:
        from core.corpus_index import CorpusIndex
        idx = CorpusIndex(["login", "hash_password", "authenticate"])
        assert idx.contains("login")
        print("[PASS] Graph Index: symbol index logic verified")
    except Exception as e:
        print(f"[FAIL] Graph Index: failed ({e})")
        all_ok = False

    # 5. Classifier heuristic check
    try:
        from core.query_analyzer import classify
        assert classify("What is the capital of France?") == QueryType.SIMPLE
        assert classify("What does login call?", known_symbols=frozenset({"login"})) == QueryType.CODE
        print("[PASS] Query Classifier: heuristic signal evaluation verified")
    except Exception as e:
        print(f"[FAIL] Query Classifier: failed ({e})")
        all_ok = False

    # 6. In-process pipeline flow check (with mocks)
    try:
        from core.pipeline import Pipeline
        from modules.cache import RAGCacheLite
        from modules.graph_rag import GraphRAG
        from modules.mem_rag import MemRAG

        mock_chroma = MagicMock()
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_col.get.return_value = {"ids": [], "metadatas": []}
        mock_chroma.get_or_create_collection.return_value = mock_col
        mock_embed = MagicMock()
        mock_embed.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)

        cache = RAGCacheLite(mock_chroma, mock_embed)
        mem_rag = MemRAG(mock_chroma, mock_embed)
        graph_rag = MagicMock(spec=GraphRAG)
        llm = MagicMock()
        llm.generate.return_value = "Paris is the capital of France."

        pipeline = Pipeline(cache=cache, graph_rag=graph_rag, mem_rag=mem_rag, llm_client=llm)
        import uuid
        res = pipeline.handle_query(QueryRequest(query="What is the capital of France?", session_id=uuid.uuid4()))
        assert res.response == "Paris is the capital of France."
        print("[PASS] Pipeline: end-to-end query processing verified in-process")
    except Exception as e:
        print(f"[FAIL] Pipeline: in-process test failed ({e})")
        all_ok = False

    print("=" * 65)
    if all_ok:
        print("RESULT: All offline checks PASSED.")
    else:
        print("RESULT: One or more offline checks FAILED.")
    return all_ok


def run_live_checks(url: str | None = None) -> bool:
    """Run real checks against live Neo4j, live ChromaDB, and real LLM API."""
    print("=" * 65)
    print(" ACEF Smoke Test — LIVE Verification Mode (--require-live)")
    print("=" * 65)
    settings = get_settings()
    all_ok = True

    # 1. Neo4j Live Ping
    print("Checking live Neo4j connection...")
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        with driver.session() as session:
            record = session.run("RETURN 1 AS ping").single()
            if record and record["ping"] == 1:
                print(f"[PASS] Neo4j: reachable at {settings.neo4j_uri}")
            else:
                print(f"[FAIL] Neo4j: query returned unexpected result ({record})")
                all_ok = False
        driver.close()
    except Exception as e:
        print(f"[FAIL] Neo4j: connection failed ({type(e).__name__}: {e})")
        print("       Hint: Ensure Neo4j is running via: docker compose up -d")
        all_ok = False

    # 2. ChromaDB Live Persistence Check
    print("Checking live ChromaDB persistence...")
    try:
        import chromadb
        persist_dir = Path(settings.chroma_persist_dir).resolve()
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir))
        col = client.get_or_create_collection("smoke_live_test")
        col.add(ids=["test_1"], documents=["Live smoke test doc"], metadatas=[{"type": "smoke"}])
        assert col.count() >= 1
        client.delete_collection("smoke_live_test")
        print(f"[PASS] ChromaDB: persistent client verified at {persist_dir}")
    except Exception as e:
        print(f"[FAIL] ChromaDB: persistent check failed ({e})")
        all_ok = False

    # 3. Live LLM Client "pong" check
    print(f"Checking live LLM client ({settings.llm_provider})...")
    active_key = (
        settings.gemini_api_key
        if settings.llm_provider == "gemini"
        else settings.groq_api_key
    )
    if not active_key or active_key.startswith("your_") or active_key.startswith("placeholder"):
        print(f"[FAIL] LLM Client: No valid API key configured in .env for provider '{settings.llm_provider}'.")
        print("       Hint: Provide a valid GEMINI_API_KEY or GROQ_API_KEY in .env")
        all_ok = False
    else:
        try:
            from modules.llm_client import LLMClient
            llm_client = LLMClient(settings)
            resp = llm_client.generate(
                query="Say the word 'pong' and nothing else.",
                query_type="simple",
            )
            if "pong" in resp.lower():
                print(f"[PASS] LLM Client: live API returned expected 'pong' response ({resp.strip()})")
            else:
                print(f"[FAIL] LLM Client: response did not contain 'pong' (got: {resp.strip()})")
                all_ok = False
        except Exception as e:
            print(f"[FAIL] LLM Client: API call failed ({type(e).__name__}: {e})")
            all_ok = False

    # 4. End-to-End Query Verification
    print("Checking end-to-end query processing...")
    if url:
        try:
            import httpx
            with httpx.Client(base_url=url, timeout=10.0) as client:
                h_res = client.get("/health")
                if h_res.status_code == 200:
                    print(f"[PASS] Gateway: /health OK ({h_res.json()})")
                else:
                    print(f"[FAIL] Gateway: /health returned {h_res.status_code}")
                    all_ok = False

                q_res = client.post("/query", json={"query": "What is 2 + 2?"})
                if q_res.status_code == 200 and q_res.json().get("response"):
                    print(f"[PASS] Gateway: /query successful ({q_res.json()['response'][:60]}...)")
                else:
                    print(f"[FAIL] Gateway: /query failed ({q_res.status_code}: {q_res.text})")
                    all_ok = False
        except Exception as e:
            print(f"[FAIL] Gateway: request to {url} failed ({e})")
            all_ok = False
    else:
        # Run in-process live pipeline if Neo4j and LLM succeeded
        if all_ok:
            try:
                from api.gateway import init_services
                services = init_services()
                pipeline = services["pipeline"]
                import uuid
                res = pipeline.handle_query(QueryRequest(query="What is 2 + 2?", session_id=uuid.uuid4()))
                if res.response:
                    print(f"[PASS] In-Process Pipeline: query succeeded ({res.response.strip()[:60]}...)")
                else:
                    print("[FAIL] In-Process Pipeline: empty response received")
                    all_ok = False
            except Exception as e:
                print(f"[FAIL] In-Process Pipeline: query failed ({e})")
                all_ok = False

    print("=" * 65)
    if all_ok:
        print("RESULT: All live checks PASSED.")
    else:
        print("RESULT: One or more live checks FAILED.")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="ACEF Smoke Test")
    parser.add_argument(
        "--require-live",
        action="store_true",
        help="Enforce live verification of Neo4j, Chroma, and LLM API",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL of running ACEF gateway for live API testing (e.g. http://localhost:8000)",
    )
    args = parser.parse_args()

    if args.require_live:
        success = run_live_checks(url=args.url)
    else:
        success = run_offline_checks()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
