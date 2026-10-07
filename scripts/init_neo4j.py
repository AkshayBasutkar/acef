"""
Neo4j Initialization Script — creates indexes and constraints.

Run this once after starting the Neo4j container (or as part of the
code ingestion pipeline). Idempotent — safe to run multiple times
thanks to 'IF NOT EXISTS' clauses.

Usage:
    python -m scripts.init_neo4j
"""

import logging
from neo4j import GraphDatabase

from config.settings import get_settings

logger = logging.getLogger(__name__)


# ── Index & constraint definitions ───────────────────────────────────────────
INIT_STATEMENTS = [
    # Full-text index over Symbol nodes — used by GraphRAG's retrieval to
    # anchor queries to code symbols via keyword/fuzzy matching.
    """
    CREATE FULLTEXT INDEX symbolSearch IF NOT EXISTS
    FOR (s:Symbol) ON EACH [s.name, s.docstring]
    """,
    # Uniqueness constraint on File.path — prevents duplicate File nodes
    # if the same codebase is ingested more than once.
    """
    CREATE CONSTRAINT file_path_unique IF NOT EXISTS
    FOR (f:File) REQUIRE f.path IS UNIQUE
    """,
]


def init_neo4j(driver=None) -> None:
    """
    Create the full-text index and uniqueness constraint in Neo4j.

    Parameters
    ----------
    driver : neo4j.Driver, optional
        An existing Neo4j driver instance.  If None, one is created from
        the current settings and closed after use.
    """
    settings = get_settings()
    close_driver = False

    if driver is None:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        close_driver = True

    try:
        with driver.session() as session:
            for statement in INIT_STATEMENTS:
                session.run(statement.strip())
                logger.info("Executed: %s", statement.strip()[:60] + "...")
        logger.info("Neo4j initialization complete — indexes and constraints ready.")
    finally:
        if close_driver:
            driver.close()


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    init_neo4j()
    print("✓ Neo4j indexes and constraints created successfully.")
