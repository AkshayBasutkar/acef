"""
Query Analyzer — heuristic classifier with signal tiers.

No ML model or LLM call — deterministic, fast heuristics evaluated in priority order:
  1. CODE — >= 1 strong CODE signal OR >= 2 distinct weak CODE signals
  2. MEMORY — strong MEMORY phrase OR document probe >= threshold
  3. FOLLOW-UP — query <= 5 words in long session (>= threshold turns):
                 routes to CODE if last_turn_type == CODE, else MEMORY
  4. SIMPLE — default / general knowledge

Signal Tiers:
  - Strong CODE:
      * Traceback header ("traceback (most recent call last)")
      * Source-file extension (.py, .js, .ts, .java, .cpp, .go, .rs, etc.)
      * Backtick quotes (`foo`)
      * Call syntax (name() / name(args))
      * snake_case containing an underscore (foo_bar)
      * Line-anchored statements (def f, class X, import x, from x import y)
  - Weak CODE:
      * function, method, variable, bug, error, exception, refactor, compile,
        syntax, codebase, repo/repository, script, stack trace, type hint
      * camelCase/PascalCase tokens not in brand allowlist
      * relationship words (call, calls, called, caller, callee, defined, definition)
      * known symbols from ingested codebase (length >= 4)
  - Strong MEMORY phrases:
      * you said, you mentioned, we discussed, we discuss, we talked, i told you,
        my previous, last time, remind me what, as mentioned, the document,
        the pdf, the file i uploaded, uploaded, according to the, in the document,
        in the pdf, my notes
"""

import logging
import re
from typing import Callable, Optional

from core.schemas import QueryType
from config.settings import get_settings

logger = logging.getLogger(__name__)

# ── Brand Allowlist (common brands with camelCase/PascalCase) ────────────────
BRAND_ALLOWLIST: frozenset[str] = frozenset({
    "iphone", "ipad", "ios", "macos", "youtube", "ebay", "powerpoint",
    "javascript", "typescript", "github", "linkedin", "openai", "chatgpt",
    "android", "google", "microsoft", "apple", "facebook", "meta", "twitter",
    "netflix", "amazon", "aws", "docker", "redis", "postgres", "mysql", "mongodb",
})

# ── Strong CODE regex patterns ───────────────────────────────────────────────
RE_TRACEBACK = re.compile(r"traceback\s+\(most\s+recent\s+call\s+last\)", re.IGNORECASE)
RE_SOURCE_EXT = re.compile(
    r"\b[a-zA-Z0-9_\-\./\\]+\.(?:py|js|ts|jsx|tsx|java|cpp|c|h|cs|go|rs|rb|php|swift|kt)\b",
    re.IGNORECASE,
)
RE_BACKTICKS = re.compile(r"`[^`]+`")
RE_CALL_SYNTAX_EMPTY = re.compile(r"\b[a-zA-Z_]\w*\s*\(\s*\)")
RE_CALL_SYNTAX_ARGS = re.compile(
    r"\b(?!(?:if|for|while|with|this|that|these|those|or|and|in|on|at|to|for|with|from|by|about|like|such|as|eg|ie|between|not|is|are|was|were|see|also|including)\b)"
    r"[a-zA-Z_]\w*\s*\([a-zA-Z0-9_\s,\'\"*+-/]+\)",
    re.IGNORECASE,
)
RE_SNAKE_CASE = re.compile(r"\b[a-z0-9]+_[a-z0-9_]+\b")
RE_LINE_ANCHORED = re.compile(
    r"(?m)^\s*(?:import\s+[a-zA-Z_]|from\s+[a-zA-Z_]\w*\s+import|def\s+[a-zA-Z_]|class\s+[a-zA-Z_])"
)

# ── Weak CODE regex patterns ─────────────────────────────────────────────────
WEAK_CODE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("function", re.compile(r"\bfunctions?\b", re.IGNORECASE)),
    ("method", re.compile(r"\bmethods?\b", re.IGNORECASE)),
    ("variable", re.compile(r"\bvariables?\b", re.IGNORECASE)),
    ("bug", re.compile(r"\bbugs?\b", re.IGNORECASE)),
    ("error", re.compile(r"\berrors?\b", re.IGNORECASE)),
    ("exception", re.compile(r"\bexceptions?\b", re.IGNORECASE)),
    ("refactor", re.compile(r"\brefactors?|refactoring|refactored\b", re.IGNORECASE)),
    ("compile", re.compile(r"\bcompil(?:e|es|ed|ing|ation|er|ers)\b", re.IGNORECASE)),
    ("syntax", re.compile(r"\bsyntax\b", re.IGNORECASE)),
    ("codebase", re.compile(r"\bcodebases?\b", re.IGNORECASE)),
    ("repo", re.compile(r"\brepos?(?:itor(?:y|ies))?\b", re.IGNORECASE)),
    ("script", re.compile(r"\bscripts?\b", re.IGNORECASE)),
    ("stack_trace", re.compile(r"\bstack\s+traces?\b", re.IGNORECASE)),
    ("type_hint", re.compile(r"\btype\s+hints?\b", re.IGNORECASE)),
    ("relationship", re.compile(r"\b(?:calls?|called|caller|callees?|defined|definitions?)\b", re.IGNORECASE)),
]

RE_CAMEL_CASE = re.compile(r"\b(?:[a-z]+[A-Z][a-zA-Z0-9]*|[A-Z][a-z0-9]+[A-Z][a-zA-Z0-9]*)\b")

# ── Strong MEMORY phrases ────────────────────────────────────────────────────
STRONG_MEMORY_PHRASES: list[str] = [
    "you said",
    "you mentioned",
    "we discussed",
    "we discuss",
    "we talked",
    "i told you",
    "my previous",
    "last time",
    "remind me what",
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

    # ── 1. CODE Signals ──────────────────────────────────────────────────────
    strong_code_signals: list[str] = []
    if RE_TRACEBACK.search(query):
        strong_code_signals.append("traceback")
    if RE_SOURCE_EXT.search(query):
        strong_code_signals.append("file_extension")
    if RE_BACKTICKS.search(query):
        strong_code_signals.append("backticks")
    if RE_CALL_SYNTAX_EMPTY.search(query) or RE_CALL_SYNTAX_ARGS.search(query):
        strong_code_signals.append("call_syntax")
    if RE_SNAKE_CASE.search(query):
        strong_code_signals.append("snake_case")
    if RE_LINE_ANCHORED.search(query):
        strong_code_signals.append("line_anchored")

    distinct_weak_signals: set[str] = set()
    for name, pattern in WEAK_CODE_PATTERNS:
        if pattern.search(ql):
            distinct_weak_signals.add(name)

    # camelCase / PascalCase not in brand allowlist
    for match in RE_CAMEL_CASE.findall(query):
        if match.lower() not in BRAND_ALLOWLIST:
            distinct_weak_signals.add(f"camel_case:{match.lower()}")

    # Known codebase symbols matching query words
    if known_symbols:
        tokens = re.findall(r"\b[a-zA-Z0-9_]{4,}\b", ql)
        for token in tokens:
            if token in known_symbols:
                distinct_weak_signals.add(f"known_symbol:{token}")

    if len(strong_code_signals) >= 1 or len(distinct_weak_signals) >= 2:
        logger.debug(
            "Classified as CODE — strong signals: %s, distinct weak signals: %s",
            strong_code_signals,
            list(distinct_weak_signals),
        )
        return QueryType.CODE

    # ── 2. MEMORY Signals ────────────────────────────────────────────────────
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
