"""
GraphRAG — code-aware retrieval using Neo4j full-text search and graph traversal.

Unlike embedding-based retrieval, GraphRAG leverages the explicit structure of
code: files import modules, functions call other functions.  That structure IS
the knowledge graph.

Retrieval strategy:
  1. Extract identifier-like tokens from the query (snake_case, camelCase,
     PascalCase, or quoted strings; fallback to the 3 longest words).
  2. Full-text search against the 'symbolSearch' index to anchor to code symbols.
  3. 1-hop Cypher traversal from each matched symbol to discover callers,
     callees, and the defining file.
  4. Format results as RetrievedChunk objects for the pipeline.
"""

import logging
import re

from core.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

# ── Regex for identifier-like tokens ─────────────────────────────────────────
# Matches: snake_case, camelCase, PascalCase words, and "quoted strings"
IDENTIFIER_PATTERN = re.compile(
    r'"([^"]+)"'           # double-quoted substrings
    r"|'([^']+)'"          # single-quoted substrings
    r"|([a-zA-Z_]\w*(?:_\w+)+)"   # snake_case (at least one underscore)
    r"|([A-Z][a-z]+(?:[A-Z][a-z]+)+)"  # PascalCase / camelCase (at least 2 segments)
)


def _extract_search_terms(query: str) -> str:
    """
    Extract identifier-like tokens from a query for Neo4j full-text search.

    Strategy:
      1. Look for snake_case, camelCase, PascalCase words and quoted substrings.
      2. If none found, fall back to the 3 longest words in the query
         (crude but functional anchor).
      3. Join matches with ' OR ' for Neo4j's Lucene full-text query syntax.

    Returns
    -------
    str
        A Lucene query string like "hash_password OR login OR auth"
    """
    matches = []
    for match in IDENTIFIER_PATTERN.finditer(query):
        # Each group corresponds to a different pattern variant
        token = match.group(1) or match.group(2) or match.group(3) or match.group(4)
        if token:
            matches.append(token)

    if not matches:
        # Fallback: use the 3 longest words (strip punctuation first)
        words = re.findall(r"[a-zA-Z_]\w*", query)
        words.sort(key=len, reverse=True)
        matches = words[:3]

    if not matches:
        return query  # last resort: use the raw query

    search_terms = " OR ".join(matches)
    logger.debug("Extracted search terms: %s", search_terms)
    return search_terms


class GraphRAG:
    """Code-graph retrieval using Neo4j full-text index + Cypher traversal."""

    def __init__(self, neo4j_driver):
        """
        Parameters
        ----------
        neo4j_driver : neo4j.Driver
            An active Neo4j driver instance (shared at the app level).
        """
        self.driver = neo4j_driver

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """
        Search the code graph for symbols relevant to the query.

        Steps:
          1. Extract identifier tokens from the query.
          2. Full-text search on the 'symbolSearch' index.
          3. For each matched symbol, traverse 1 hop to get callers,
             callees, and the defining file.
          4. Return formatted RetrievedChunk objects.

        Parameters
        ----------
        query : str
            The user's code-related question.
        top_k : int
            Maximum number of symbol matches to return (default 3).

        Returns
        -------
        list[RetrievedChunk]
            Context chunks with code symbol information and relationships.
            Empty list if no matches found (pipeline handles gracefully).
        """
        search_terms = _extract_search_terms(query)
        chunks: list[RetrievedChunk] = []

        try:
            with self.driver.session() as session:
                # ── Step 1: Full-text search for symbol anchors ──────────
                anchor_result = session.run(
                    """
                    CALL db.index.fulltext.queryNodes('symbolSearch', $terms)
                    YIELD node, score
                    RETURN node.name AS name, score
                    LIMIT $top_k
                    """,
                    terms=search_terms,
                    top_k=top_k,
                )
                matched_symbols = [
                    {"name": record["name"], "score": record["score"]}
                    for record in anchor_result
                ]

                if not matched_symbols:
                    logger.debug("GraphRAG: no symbol matches for terms='%s'", search_terms)
                    return []

                logger.debug(
                    "GraphRAG: found %d symbol matches: %s",
                    len(matched_symbols),
                    [s["name"] for s in matched_symbols],
                )

                # ── Step 2: 1-hop traversal for each matched symbol ──────
                for symbol in matched_symbols:
                    traversal_result = session.run(
                        """
                        MATCH (s:Symbol {name: $name})
                        OPTIONAL MATCH (s)-[:CALLS]->(callee:Symbol)
                        OPTIONAL MATCH (caller:Symbol)-[:CALLS]->(s)
                        OPTIONAL MATCH (f:File)-[:DEFINES]->(s)
                        RETURN s.name AS name,
                               s.kind AS kind,
                               s.docstring AS docstring,
                               s.lineno AS lineno,
                               f.path AS file_path,
                               collect(DISTINCT callee.name) AS calls,
                               collect(DISTINCT caller.name) AS called_by
                        """,
                        name=symbol["name"],
                    )

                    for record in traversal_result:
                        file_path = record["file_path"] or "unknown"
                        calls = [c for c in record["calls"] if c is not None]
                        called_by = [c for c in record["called_by"] if c is not None]

                        text = (
                            f"Function/Class: {record['name']} "
                            f"({record['kind']}) in {file_path}\n"
                            f"Line: {record['lineno']}\n"
                            f"Docstring: {record['docstring'] or 'none'}\n"
                            f"Calls: {', '.join(calls) or 'none'}\n"
                            f"Called by: {', '.join(called_by) or 'none'}"
                        )

                        chunk = RetrievedChunk(text=text, source=file_path)
                        chunks.append(chunk)

        except Exception as e:
            # Log the error but return empty — the pipeline proceeds with
            # no context rather than crashing the entire request.
            logger.error("GraphRAG retrieval failed: %s", e, exc_info=True)
            return []

        logger.info("GraphRAG: returning %d context chunks", len(chunks))
        return chunks
