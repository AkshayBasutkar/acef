"""
Corpus Index — in-memory cache of known codebase symbols for corpus-aware routing.

Extracts symbol names (length >= 4) from Neo4j (:Symbol) nodes so that queries
referencing actual symbols (e.g. "What does login call?") can be detected by the
classifier even without explicit code syntax or keywords.
"""

import logging
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


class CorpusIndex:
    """In-memory index of lowercased symbol names from the ingested codebase."""

    def __init__(self, symbols: Optional[Iterable[str]] = None):
        """
        Parameters
        ----------
        symbols : Iterable[str], optional
            Initial symbol names (strings).
        """
        if symbols:
            self.symbols: frozenset[str] = frozenset(
                s.lower() for s in symbols if isinstance(s, str) and len(s) >= 4
            )
        else:
            self.symbols: frozenset[str] = frozenset()

    def refresh_from_neo4j(self, neo4j_driver) -> int:
        """
        Query Neo4j for all (:Symbol) names and refresh the cached symbols set.
        Never raises: if Neo4j is unavailable or fails, gracefully logs and
        preserves existing symbols.

        Parameters
        ----------
        neo4j_driver : neo4j.Driver
            Active Neo4j driver.

        Returns
        -------
        int
            Number of indexed symbols.
        """
        if neo4j_driver is None:
            logger.warning("CorpusIndex: Neo4j driver is None; keeping existing symbols")
            return len(self.symbols)

        try:
            with neo4j_driver.session() as session:
                result = session.run("MATCH (s:Symbol) RETURN s.name AS name")
                names = [record["name"] for record in result if record.get("name")]
                self.symbols = frozenset(
                    s.lower() for s in names if isinstance(s, str) and len(s) >= 4
                )
                logger.info("CorpusIndex refreshed — %d symbols indexed", len(self.symbols))
                return len(self.symbols)
        except Exception as e:
            logger.warning("CorpusIndex: Failed to query symbols from Neo4j: %s", e)
            return len(self.symbols)

    def add_symbols(self, names: Iterable[str]) -> None:
        """Add new symbol names to the index."""
        current = set(self.symbols)
        for s in names:
            if isinstance(s, str) and len(s) >= 4:
                current.add(s.lower())
        self.symbols = frozenset(current)

    def contains(self, symbol: str) -> bool:
        """Check if symbol (case-insensitive) is known."""
        return symbol.lower() in self.symbols
