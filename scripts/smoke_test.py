"""
Smoke Test Script — verifies Neo4j, ChromaDB, SentenceTransformer, and LLM Client.

Usage:
    python -m scripts.smoke_test
"""

import logging
import sys
from pathlib import Path

from config.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smoke_test")


def check_config():
    logger.info("── 1. Checking Configuration ──")
    settings = get_settings()
    logger.info("  Config loaded. LLM Provider: %s", settings.llm_provider)
    logger.info("  Chroma persist dir: %s", settings.chroma_persist_dir)
    return settings


def check_neo4j(settings):
    logger.info("── 2. Checking Neo4j Connectivity ──")
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        with driver.session() as session:
            result = session.run("RETURN 1 AS ping").single()
            if result and result["ping"] == 1:
                logger.info("  ✓ Neo4j is ONLINE and reachable at %s", settings.neo4j_uri)
                driver.close()
                return True
        driver.close()
    except Exception as e:
        logger.warning("  ✗ Neo4j is OFFLINE or unreachable (%s: %s)", type(e).__name__, e)
        logger.warning("    (Start it with: docker compose up -d)")
        return False
    return False


def check_chroma(settings):
    logger.info("── 3. Checking ChromaDB ──")
    try:
        import chromadb
        persist_dir = Path(settings.chroma_persist_dir).resolve()
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir))
        col = client.get_or_create_collection("smoke_test_collection")
        col.add(
            ids=["test_1"],
            documents=["Smoke test document"],
            metadatas=[{"type": "test"}],
        )
        count = col.count()
        client.delete_collection("smoke_test_collection")
        logger.info("  ✓ ChromaDB initialized successfully at %s (rw verified)", persist_dir)
        return True
    except Exception as e:
        logger.error("  ✗ ChromaDB initialization failed: %s", e)
        return False


def check_embeddings(settings):
    logger.info("── 4. Checking Embedding Model ──")
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("  Loading SentenceTransformer('%s')...", settings.embedding_model)
        model = SentenceTransformer(settings.embedding_model)
        vec = model.encode("ACEF smoke test query")
        logger.info("  ✓ Embedding model loaded and produced vector shape: %s", vec.shape)
        return True
    except Exception as e:
        logger.error("  ✗ Embedding model error: %s", e)
        return False


def check_llm(settings):
    logger.info("── 5. Checking LLM Client ──")
    try:
        from modules.llm_client import LLMClient
        client = LLMClient(settings)

        active_key = settings.gemini_api_key if settings.llm_provider == "gemini" else settings.groq_api_key
        if not active_key or active_key.startswith("placeholder"):
            logger.info("  ! No live %s API key found in .env; skipped live API completion test.", settings.llm_provider)
            return True

        logger.info("  Calling %s (%s)...", settings.llm_provider, client.model)
        response = client.generate(
            query="Respond with the single word 'OK'.",
            query_type="simple",
        )
        logger.info("  ✓ LLM Response received: %s", response.strip()[:100])
        return True
    except Exception as e:
        logger.warning("  ! LLM call failed or key invalid: %s", e)
        return False


def main():
    settings = check_config()
    neo4j_ok = check_neo4j(settings)
    chroma_ok = check_chroma(settings)
    embed_ok = check_embeddings(settings)
    llm_ok = check_llm(settings)

    logger.info("══════════════════════════════════")
    logger.info("Smoke Test Summary:")
    logger.info("  Neo4j:       %s", "PASS" if neo4j_ok else "OFFLINE (run docker compose up -d)")
    logger.info("  ChromaDB:    %s", "PASS" if chroma_ok else "FAIL")
    logger.info("  Embeddings:  %s", "PASS" if embed_ok else "FAIL")
    logger.info("  LLM:         %s", "PASS" if llm_ok else "WARN")
    logger.info("══════════════════════════════════")

    if not (chroma_ok and embed_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
