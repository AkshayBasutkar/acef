"""
GraphRAG — code-aware retrieval using Neo4j full-text search and graph traversal.

Unlike embedding-based retrieval, GraphRAG leverages the explicit structure of
code: files import modules, functions call other functions.  That structure IS
the knowledge graph.

Retrieval strategy:
  1. Extract identifier-like tokens from the query (snake_case, camelCase,
     PascalCase, or quoted strings; fallback to the 3 longest words).
  2. Full-text search against the 'symbolSearch' index to anchor to code symbols.
  3. Batched multi-hop Cypher traversal from each matched symbol to discover
     callers, callees (up to configurable depth), the defining file, file
     imports, and parent class (if the symbol is a method).
  4. Score-based filtering removes low-relevance matches.
  5. Format results as RetrievedChunk objects for the pipeline.
"""

import logging
from pathlib import Path
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


def escape_lucene(term: str) -> str:
    """
    Escape Lucene special characters:
    + - & | ! ( ) { } [ ] ^ " ~ * ? : \\ /
    """
    lucene_chars = r'([+\-&|!(){}\[\]^"~*?:\\/])'
    return re.sub(lucene_chars, r'\\\1', term)


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
        return escape_lucene(query)  # last resort: use the escaped raw query

    # Escape Lucene special characters for each term
    escaped = [escape_lucene(m) for m in matches if m.strip()]
    if not escaped:
        return "*"
    search_terms = " OR ".join(escaped)
    logger.debug("Extracted search terms: %s", search_terms)
    return search_terms


class GraphRAG:
    """Code-graph retrieval using Neo4j full-text index + Cypher traversal.

    Improvements over the baseline:
      - Single batched Cypher query (was N+1 round trips)
      - Configurable multi-hop traversal depth (was fixed at 1)
      - Score-based filtering of low-relevance matches
      - Import-chain context (file dependencies)
      - Parent class context for methods
    """

    def __init__(
        self,
        neo4j_driver,
        max_hops: int = 2,
        min_score: float = 0.0,
        root: str | None = None,
    ):
        """
        Parameters
        ----------
        neo4j_driver : neo4j.Driver
            An active Neo4j driver instance (shared at the app level).
        max_hops : int
            Maximum traversal depth for call chains (default 2).
            1 = direct callers/callees only, 2 = includes indirect.
        min_score : float
            Minimum full-text search score to keep a symbol match (default 0.0).
            Symbols below this score are filtered out as noise.
        root : str, optional
            Codebase root path to isolate full-text anchors and traversal.
        """
        if max_hops < 1:
            raise ValueError(f"max_hops must be >= 1, got {max_hops}")
        if min_score < 0.0:
            raise ValueError(f"min_score must be >= 0.0, got {min_score}")
        self.driver = neo4j_driver
        self._max_hops = max_hops
        self._min_score = min_score
        self._root = Path(root).resolve().as_posix() if root else None

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        root: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Search the code graph for symbols relevant to the query.

        Uses a single batched Cypher query that:
          1. Anchors to symbols via full-text search (with score & root filtering)
          2. Traverses up to max_hops call chain depth
          3. Collects callers, defining file, file imports, and parent class

        Parameters
        ----------
        query : str
            The user's code-related question.
        top_k : int
            Maximum number of symbol matches to return (default 3).
        root : str, optional
            Override codebase root path for multi-codebase isolation.

        Returns
        -------
        list[RetrievedChunk]
            Context chunks with code symbol information and relationships.
            Empty list if no matches found (pipeline handles gracefully).
        """
        search_terms = _extract_search_terms(query)
        chunks: list[RetrievedChunk] = []
        active_root = Path(root).resolve().as_posix() if root else self._root

        try:
            with self.driver.session() as session:
                # ── Single batched query: anchor + traversal + context ────
                # Variable-length path depth is interpolated safely (integer)
                max_hops = int(self._max_hops)
                cypher = f"""
                    CALL db.index.fulltext.queryNodes('symbolSearch', $terms)
                    YIELD node, score
                    WITH node, score
                    WHERE score >= $min_score AND ($root IS NULL OR node.root = $root)
                    ORDER BY score DESC
                    LIMIT $top_k
                    OPTIONAL MATCH (f:File)-[:DEFINES]->(node)
                    WITH node, score, f
                    OPTIONAL MATCH (node)-[:CALLS*1..{max_hops}]->(callee:Symbol)
                    WITH node, score, f, collect(DISTINCT callee.name) AS calls
                    OPTIONAL MATCH (caller:Symbol)-[:CALLS]->(node)
                    WITH node, score, f, calls, collect(DISTINCT caller.name) AS called_by
                    OPTIONAL MATCH (f)-[:IMPORTS]->(imported:File)
                    WITH node, score, f, calls, called_by,
                         collect(DISTINCT imported.name) AS file_imports
                    OPTIONAL MATCH (parent:Symbol)-[:CONTAINS]->(node)
                    RETURN node.name AS name,
                           node.qualname AS qualname,
                           node.kind AS kind,
                           node.docstring AS docstring,
                           node.lineno AS lineno,
                           node.end_lineno AS end_lineno,
                           node.source AS source_code,
                           f.path AS file_path,
                           score,
                           calls,
                           called_by,
                           file_imports,
                           parent.name AS parent_class,
                           parent.qualname AS parent_qualname
                """

                result = session.run(
                    cypher,
                    terms=search_terms,
                    top_k=top_k,
                    min_score=self._min_score,
                    root=active_root,
                )

                records = list(result)

                if not records:
                    logger.debug(
                        "GraphRAG: no symbol matches for terms='%s'", search_terms
                    )
                    return []

                logger.debug(
                    "GraphRAG: found %d symbol matches: %s",
                    len(records),
                    [r["name"] for r in records],
                )

                # ── Format each record into a RetrievedChunk ─────────
                for record in records:
                    file_path = record.get("file_path") or "unknown"
                    calls = [c for c in (record.get("calls") or []) if c is not None]
                    called_by = [c for c in (record.get("called_by") or []) if c is not None]
                    file_imports = [
                        f for f in (record.get("file_imports") or []) if f is not None
                    ]
                    source_code = record.get("source_code") or ""
                    lineno = record.get("lineno", 0)
                    end_lineno = record.get("end_lineno") or lineno
                    parent_class = record.get("parent_qualname") or record.get("parent_class")
                    score = record.get("score", 0)
                    qualname = record.get("qualname") or record.get("name") or "unknown"

                    lines_str = (
                        f"{lineno}-{end_lineno}"
                        if end_lineno != lineno
                        else str(lineno)
                    )
                    source_block = (
                        f"\nSource:\n```python\n{source_code.strip()}\n```"
                        if source_code.strip()
                        else ""
                    )

                    # Build the kind label (e.g. "method of UserManager")
                    kind_label = record.get("kind") or "symbol"
                    if parent_class:
                        if kind_label == "method":
                            kind_label = f"method of {parent_class}"
                        elif kind_label == "nested_function":
                            kind_label = f"nested function of {parent_class}"

                    docstring = record.get("docstring") or "none"

                    text = (
                        f"Function/Class: {qualname} ({kind_label}) "
                        f"in {file_path}\n"
                        f"Lines: {lines_str}  |  Relevance score: {score:.2f}\n"
                        f"Docstring: {docstring}\n"
                        f"Calls: {', '.join(calls) or 'none'}\n"
                        f"Called by: {', '.join(called_by) or 'none'}"
                    )

                    if file_imports:
                        text += f"\nFile imports: {', '.join(file_imports)}"

                    text += source_block

                    chunk = RetrievedChunk(text=text, source=file_path)
                    chunks.append(chunk)

        except Exception as e:
            # Log the error but return empty — the pipeline proceeds with
            # no context rather than crashing the entire request.
            logger.error("GraphRAG retrieval failed: %s", e, exc_info=True)
            return []

        logger.info("GraphRAG: returning %d context chunks", len(chunks))
        return chunks

