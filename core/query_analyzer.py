"""
Query Analyzer — three-way heuristic classifier.

No ML model, no LLM call — pure keyword/pattern rules evaluated in priority
order (first match wins):

  1. CODE  — query contains code-related keywords
  2. MEMORY — query contains memory/document-related keywords
  3. MEMORY (fallback) — short follow-up in an already-long conversation
  4. SIMPLE — everything else

The classifier intentionally uses heuristics rather than an LLM to avoid
adding latency and API cost to every single request.  If it misclassifies
in practice, the cheapest fix is adding keywords to the lists below.
"""

import logging

from core.schemas import QueryType
from config.settings import get_settings

logger = logging.getLogger(__name__)

# ── Keyword lists ────────────────────────────────────────────────────────────
CODE_KEYWORDS: list[str] = [
    "function",
    "class ",
    "method",
    "variable",
    "bug",
    "error",
    "exception",
    "stack trace",
    "traceback",
    "refactor",
    "import",
    "def ",
    "repository",
    "repo",
    "script",
    "compile",
    "syntax",
    "codebase",
    "this code",
    ".py",
    ".js",
    ".ts",
    ".java",
    ".cpp",
]

MEMORY_KEYWORDS: list[str] = [
    "earlier",
    "previously",
    "before",
    "you said",
    "we discussed",
    "mentioned",
    "in the document",
    "according to",
    "that file",
    "the pdf",
    "the doc",
    "last time",
    "remind me",
    "summarize what",
]


def classify(query: str, session_turn_count: int = 0) -> QueryType:
    """
    Classify a query into exactly one of: SIMPLE, CODE, or MEMORY.

    Parameters
    ----------
    query : str
        The user's raw query text.
    session_turn_count : int
        Number of prior conversation turns in this session (from MemRAG's
        conversation collection).  Used for the short-follow-up heuristic.

    Returns
    -------
    QueryType
        The classified query type.
    """
    ql = query.lower()

    # 1. Code-related?
    if any(kw in ql for kw in CODE_KEYWORDS):
        logger.debug("Classified as CODE — matched keyword in query")
        return QueryType.CODE

    # 2. Memory/document-related?
    if any(kw in ql for kw in MEMORY_KEYWORDS):
        logger.debug("Classified as MEMORY — matched keyword in query")
        return QueryType.MEMORY

    # 3. Short follow-up in an already-long conversation?
    settings = get_settings()
    if (
        session_turn_count >= settings.memory_long_conversation_turns
        and len(query.split()) <= 8
    ):
        logger.debug(
            "Classified as MEMORY — short follow-up (%d words) in long "
            "session (%d turns)",
            len(query.split()),
            session_turn_count,
        )
        return QueryType.MEMORY

    # 4. Default
    logger.debug("Classified as SIMPLE — no keyword/heuristic match")
    return QueryType.SIMPLE
