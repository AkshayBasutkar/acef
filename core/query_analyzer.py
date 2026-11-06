"""
Query Analyzer — deterministic retrieval-needs classifier.

The labels identify the evidence source required to answer, not the topic:
  1. CODE — needs facts from the indexed local codebase / GraphRAG
  2. MEMORY — needs an uploaded document or prior conversation / MemRAG
  3. SIMPLE — answerable from the current query and general model knowledge

Consequently, generic programming questions and self-contained pasted code are
SIMPLE.  Technical vocabulary alone must never route a query to GraphRAG.
"""

import logging
import re
from typing import Callable, Optional

from core.schemas import QueryType
from config.settings import get_settings

logger = logging.getLogger(__name__)

# A concrete source filename can support a repository-specific request, but is
# not sufficient alone (for example, a pasted traceback remains SIMPLE).
RE_SOURCE_EXT = re.compile(
    r"\b[a-zA-Z0-9_\-\./\\]+\.(?:py|js|ts|jsx|tsx|java|cpp|c|h|cs|go|rs|rb|php|swift|kt)\b",
    re.IGNORECASE,
)

# ── Strong MEMORY phrases ────────────────────────────────────────────────────
STRONG_MEMORY_PHRASES: list[str] = [
    "you said",
    "what did you say",
    "you mentioned",
    "we discussed",
    "we discuss",
    "we talked",
    "we agreed",
    "we agree",
    "i told you",
    "my previous",
    "last time",
    "remind me what",
    "do you remember",
    "as mentioned",
    "the document",
    "the pdf",
    "the file i uploaded",
    "uploaded",
    "according to the",
    "in the document",
    "in the pdf",
    "my notes",
]

# Explicit evidence-source and intent signals.  These are deliberately narrower
# than generic programming vocabulary: they indicate that repository retrieval
# can materially contribute to the answer.
RE_LOCAL_CODEBASE_SCOPE = re.compile(
    r"\b(?:this|the|our|my|current|local|indexed)\s+"
    r"(?:codebase|code\s+base|repo(?:sitory)?|project|source\s+tree|source\s+code)\b",
    re.IGNORECASE,
)
RE_REPOSITORY_NAVIGATION = re.compile(
    r"\b(?:where\s+is\s+.+\s+defined|which\s+(?:functions?|methods?|files?)\s+"
    r"(?:calls?|uses?|imports?|references?)|what\s+does\s+.+\s+(?:calls?|uses?|imports?)|"
    r"what\s+(?:calls?|uses?|imports?)\s+.+|"
    r"callers?\s+of|callees?\s+of|find\s+(?:the\s+)?(?:definition|implementation|"
    r"references?)|trace\s+(?:the\s+)?call\s+(?:graph|path)|dependency\s+graph)\b",
    re.IGNORECASE,
)
RE_GENERIC_PROGRAMMING_REQUEST = re.compile(
    r"^\s*(?:how\s+(?:do|can|should)\s+i|how\s+to|what\s+is|explain|teach\s+me|"
    r"show\s+me\s+how|write\s+(?:a|an|the)?|give\s+me\s+(?:a|an)?\s*example)",
    re.IGNORECASE,
)
RE_FILE_FOCUSED_REQUEST = re.compile(
    r"\b(?:inspect|explain|summarize|refactor|find|locate|what\s+does|why\s+does)\b",
    re.IGNORECASE,
)


def classify(
    query: str,
    session_turn_count: int = 0,
    *,
    known_symbols: frozenset[str] = frozenset(),
    doc_similarity_fn: Optional[Callable[[], float]] = None,
    last_turn_type: Optional[QueryType] = None,
) -> QueryType:
    """
    Classify a query into exactly one of: CODE, MEMORY, or SIMPLE.

    Parameters
    ----------
    query : str
        The user's raw query text.
    session_turn_count : int
        Number of prior conversation turns in this session.
    known_symbols : frozenset[str], optional
        Lowercase symbol names from the ingested code graph (length >= 4).
    doc_similarity_fn : Callable[[], float], optional
        Lazy evaluation returning maximum semantic similarity against ingested docs.
    last_turn_type : QueryType, optional
        Classification of the preceding turn in the same session.

    Returns
    -------
    QueryType
        The classified query type.
    """
    ql = query.lower()
    settings = get_settings()

    tokens = set(re.findall(r"\b[a-zA-Z0-9_]{4,}\b", ql))
    matched_symbols = tokens.intersection(known_symbols)
    has_memory_reference = any(phrase in ql for phrase in STRONG_MEMORY_PHRASES)
    has_local_scope = bool(RE_LOCAL_CODEBASE_SCOPE.search(query))
    has_repo_navigation = bool(RE_REPOSITORY_NAVIGATION.search(query))
    has_generic_programming_form = bool(RE_GENERIC_PROGRAMMING_REQUEST.search(query))
    has_project_file_request = bool(
        RE_SOURCE_EXT.search(query) and RE_FILE_FOCUSED_REQUEST.search(query)
    )

    # Explicit recall requests remain MEMORY even when the recalled subject is
    # code.  A concrete repository action (find callers/definition, inspect this
    # project, etc.) instead requires current GraphRAG evidence.
    if has_memory_reference and not (has_local_scope or has_repo_navigation):
        logger.debug("Classified as MEMORY — explicit stored-context reference")
        return QueryType.MEMORY

    # CODE requires local-corpus evidence. A known symbol is useful evidence,
    # except in an explicitly generic tutorial/how-to formulation.
    code_required = (
        has_local_scope
        or has_repo_navigation
        or has_project_file_request
        or (bool(matched_symbols) and not has_generic_programming_form)
    )
    if code_required:
        logger.debug(
            "Classified as CODE — local_scope=%s repo_navigation=%s file_request=%s symbols=%s",
            has_local_scope,
            has_repo_navigation,
            has_project_file_request,
            sorted(matched_symbols),
        )
        return QueryType.CODE

    # ── MEMORY Signals ────────────────────────────────────────────────────────
    matched_memory_phrase = any(phrase in ql for phrase in STRONG_MEMORY_PHRASES)
    if matched_memory_phrase:
        logger.debug("Classified as MEMORY — matched strong memory phrase")
        return QueryType.MEMORY

    # Lazy document similarity probe
    if doc_similarity_fn is not None:
        try:
            doc_sim = doc_similarity_fn()
            threshold = getattr(settings, "doc_route_threshold", 0.45)
            if doc_sim >= threshold:
                logger.debug(
                    "Classified as MEMORY — document similarity probe %.3f >= %.3f",
                    doc_sim,
                    threshold,
                )
                return QueryType.MEMORY
        except Exception as e:
            logger.warning("Document similarity probe failed: %s", e)

    # ── 3. Follow-up Rule ────────────────────────────────────────────────────
    word_count = len(query.split())
    if (
        session_turn_count >= settings.memory_long_conversation_turns
        and word_count <= 5
    ):
        if last_turn_type == QueryType.CODE:
            logger.debug(
                "Classified as CODE — short follow-up (%d words) inheriting CODE turn in session (%d turns)",
                word_count,
                session_turn_count,
            )
            return QueryType.CODE
        logger.debug(
            "Classified as MEMORY — short follow-up (%d words) in long session (%d turns)",
            word_count,
            session_turn_count,
        )
        return QueryType.MEMORY

    # ── 4. Default: SIMPLE ───────────────────────────────────────────────────
    logger.debug("Classified as SIMPLE — no heuristic matches")
    return QueryType.SIMPLE
