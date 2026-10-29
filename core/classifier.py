"""
Query Classifier — strategy-pattern wrapper around classification backends.

Three backends, selected via settings.classifier_mode:

  heuristic  Fast, deterministic regex heuristics (core/query_analyzer.py).
             Zero latency, zero API cost, always available. Default.

  llm        Lightweight LLM-based classifier. Sends a single-turn prompt to
             the configured Gemini or Groq provider using a small, cheap model
             (gemini-2.0-flash-lite / llama-3.1-8b-instant). Falls back to
             HeuristicClassifier automatically on any error.

  local      Reserved for a locally-deployed fine-tuned MiniLM model.
             Not yet implemented — set CLASSIFIER_MODE=local only after
             training and exporting the model.

Usage in pipeline:
    from core.classifier import get_classifier, ClassificationContext
    classifier = get_classifier(settings, llm_client)
    query_type = classifier.classify(query, ClassificationContext(...))
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from core.schemas import QueryType

logger = logging.getLogger(__name__)

# ── Default lightweight models per provider ───────────────────────────────────
_DEFAULT_CLASSIFIER_MODELS: dict[str, str] = {
    "gemini": "gemini-2.0-flash-lite",
    "groq": "llama-3.1-8b-instant",
}


# ── Classification context ────────────────────────────────────────────────────

@dataclass
class ClassificationContext:
    """
    All contextual signals that a classifier backend may use.

    Not every backend uses every field — the LLM classifier works purely from
    the query text, while the heuristic classifier uses all fields.
    Defaults are safe for zero-context use.
    """
    session_turn_count: int = 0
    known_symbols: frozenset[str] = field(default_factory=frozenset)
    doc_similarity_fn: Optional[Callable[[], float]] = None
    last_turn_type: Optional[QueryType] = None


# ── System prompt for LLM classifier ─────────────────────────────────────────
#
# Design principles:
#   1. Explicit positive examples for CODE that pure syntax heuristics miss
#      (algorithms, design patterns, conceptual CS questions).
#   2. Explicit hard negatives for CODE to prevent false positives from
#      parenthetical English, everyday underscored names, and ambiguous words.
#   3. Explicit MEMORY anchors so the model doesn't confuse "document" (a DB)
#      with "the document I uploaded".
#   4. Priority order stated: when signals conflict, CODE wins unless the user
#      is *explicitly* referencing prior context or an uploaded file.
#   5. Constrained output: one word only, no explanation.
#
_CLASSIFIER_SYSTEM_PROMPT = """\
You are a query router for a software assistant. Your only task is to classify \
the user query into exactly one of three routing categories. Reply with ONE WORD \
only — nothing else, no punctuation, no explanation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CATEGORIES
──────────

CODE — Route to the code knowledge graph (Neo4j).
  Use when the query is about software, programming, or technical engineering:
  • Any programming language syntax, constructs, or idioms
    (Python, JS, Go, Rust, SQL, HTML/CSS, shell scripts, regex, etc.)
  • Debugging, tracebacks, exceptions, stack traces, error codes (404, 500, …)
  • Named code entities: functions, classes, methods, variables, constants,
    modules, APIs, endpoints, decorators, hooks, types, interfaces
  • Software concepts: algorithms, data structures, design patterns,
    dependency injection, recursion, closures, async/await, concurrency,
    memory management, ORM, migrations, schemas
  • Development workflow: testing, CI/CD, linting, building, deploying,
    containerization, version control (git), package managers (npm, pip, etc.)
  • Architecture: microservices, REST, GraphQL, WebSockets, message queues,
    caching strategies, database queries

MEMORY — Route to document and conversation memory (ChromaDB).
  Use when the query explicitly references something the user has already
  provided in this session:
  • Uploaded files / documents: "the PDF", "the file I uploaded", "in the doc"
  • Prior conversation turns: "you said", "we discussed", "I told you",
    "last time", "remind me what", "as mentioned", "my previous"
  • Recall / summarization of prior context: "according to the document",
    "in my notes", "what was the plan we agreed on"

SIMPLE — Route to the LLM with no retrieval context.
  Use for everything else: general knowledge, factual questions, writing help,
  math, science, history, advice, or any non-technical topic.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMMON PITFALLS — read carefully
─────────────────────────────────

❌ Do NOT classify as CODE:
  • "What is the Federal Aviation Administration (FAA)?" — parentheses are not
    function calls; this is a general knowledge question.
  • "My username is cool_dude_99" — underscores in names/usernames are not
    code identifiers.
  • "There was an error in my tax return" — "error" in a non-technical context
    is not a programming error.
  • "Can I make a call on WhatsApp?" — product names and everyday verbs
    ("call", "function", "script") are not code signals in isolation.
  • "Recommend a book about the history of YouTube" — brand names are not code.
  • "What class should I take to learn pottery?" — "class" in English context.

✅ DO classify as CODE even without explicit syntax:
  • "How do I invert a binary tree?" — algorithmic question.
  • "Explain dependency injection" — software design concept.
  • "What is the time complexity of quicksort?" — CS theory question.
  • "How to write unit tests?" — development practice.
  • "Debug the issue with my code" — development action.
  • "How does polymorphism work?" — programming concept.
  • "What is a closure in JavaScript?" — language concept.
  • "Where is MAX_RETRIES defined?" — constant identifier (UPPER_SNAKE_CASE).
  • "SELECT * FROM users WHERE active = 1" — SQL is code.

❌ Do NOT classify as MEMORY unless the user explicitly references prior
   context. "The document" meaning "the documentation" is not MEMORY.
   "We discussed" about a general topic is not MEMORY unless it clearly
   refers to this conversation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRIORITY: CODE > MEMORY > SIMPLE
When a query has signals from multiple categories, prefer CODE if there is a
genuine programming/technical question, even if memory phrases are present
(e.g. "You mentioned there's a bug in auth.py" → CODE because auth.py is a
code file and the core question is about a bug).

Reply with exactly one word: CODE, MEMORY, or SIMPLE.\
"""

# Regex to extract a label from potentially noisy LLM output
_LABEL_RE = re.compile(r"\b(CODE|MEMORY|SIMPLE)\b", re.IGNORECASE)

_LABEL_MAP: dict[str, QueryType] = {
    "code": QueryType.CODE,
    "memory": QueryType.MEMORY,
    "simple": QueryType.SIMPLE,
}


# ── Classifier implementations ────────────────────────────────────────────────

class HeuristicClassifier:
    """
    Wraps the existing deterministic heuristic classifier.
    Zero latency, zero cost, always available. Used as fallback.
    """

    def classify(self, query: str, context: ClassificationContext) -> QueryType:
        # Import here to avoid circular imports at module load time
        from core.query_analyzer import classify as heuristic_classify
        return heuristic_classify(
            query,
            context.session_turn_count,
            known_symbols=context.known_symbols,
            doc_similarity_fn=context.doc_similarity_fn,
            last_turn_type=context.last_turn_type,
        )


class LLMClassifier:
    """
    Lightweight LLM-based query classifier.

    Uses the already-configured OpenAI-compatible client (Gemini / Groq) with
    a small, cheap model and temperature=0 to get deterministic single-word
    classifications. Falls back to HeuristicClassifier on any error without
    raising, so the pipeline is never blocked by a classification failure.
    """

    def __init__(self, llm_client, model_override: str = "", provider: str = "gemini"):
        """
        Parameters
        ----------
        llm_client : LLMClient
            Fully initialised LLMClient instance (for its .client and .model).
        model_override : str
            Explicit model name for classification calls. If empty, a
            provider-appropriate lightweight model is chosen automatically.
        provider : str
            The llm_provider value from settings ("gemini" or "groq"), used to
            pick the right default lightweight model.
        """
        self._openai_client = llm_client.client
        self._fallback = HeuristicClassifier()

        if model_override:
            self._model = model_override
        else:
            self._model = _DEFAULT_CLASSIFIER_MODELS.get(provider, llm_client.model)

        logger.info(
            "LLMClassifier initialised — model=%s (provider=%s)",
            self._model,
            provider,
        )

    def classify(self, query: str, context: ClassificationContext) -> QueryType:
        """
        Classify *query* via LLM. Returns a QueryType.
        On any failure (timeout, API error, parse error) falls back to the
        heuristic classifier and logs a warning — never raises.
        """
        try:
            response = self._openai_client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=5,     # "CODE", "MEMORY", "SIMPLE" — a few tokens is plenty
                timeout=5.0,      # classifier must be fast; fail fast and use heuristic
            )

            raw = (response.choices[0].message.content or "").strip()
            match = _LABEL_RE.search(raw)

            if match:
                label = _LABEL_MAP[match.group(1).lower()]
                logger.debug(
                    "LLMClassifier: query=%r → %s (raw=%r, model=%s)",
                    query[:80],
                    label.value,
                    raw,
                    self._model,
                )
                return label

            # Response not parseable — fall back
            logger.warning(
                "LLMClassifier: unparseable response %r for query %r — falling back to heuristic",
                raw,
                query[:80],
            )
            return self._fallback.classify(query, context)

        except Exception as exc:
            logger.warning(
                "LLMClassifier: call failed (%s: %s) — falling back to heuristic",
                type(exc).__name__,
                exc,
            )
            return self._fallback.classify(query, context)


class LocalClassifier:
    """
    Reserved for a locally-deployed fine-tuned MiniLM classifier.

    Workflow (future):
      1. Collect (query, label) pairs from LLMClassifier logs or manual annotation.
      2. Fine-tune a sentence-transformers classification head on those pairs.
      3. Export the model (ONNX or keep as PyTorch state dict).
      4. Set CLASSIFIER_MODE=local and CLASSIFIER_LOCAL_MODEL_PATH=./models/classifier
      5. Implement this class to load the model and run inference (~2ms per call).
    """

    def __init__(self, model_path: str):
        if not model_path:
            raise ValueError(
                "CLASSIFIER_LOCAL_MODEL_PATH must be set when CLASSIFIER_MODE=local. "
                "Train and export a model first, then provide its path."
            )
        raise NotImplementedError(
            "LocalClassifier is not yet implemented. "
            "Train a fine-tuned MiniLM model and then implement this class."
        )

    def classify(self, query: str, context: ClassificationContext) -> QueryType:  # pragma: no cover
        raise NotImplementedError


# ── Factory ───────────────────────────────────────────────────────────────────

def get_classifier(settings, llm_client=None):
    """
    Return the appropriate classifier instance based on settings.classifier_mode.

    Parameters
    ----------
    settings : Settings
        The application settings instance.
    llm_client : LLMClient, optional
        Required when classifier_mode == "llm". Ignored otherwise.

    Returns
    -------
    HeuristicClassifier | LLMClassifier | LocalClassifier
    """
    mode = (settings.classifier_mode or "heuristic").lower().strip()

    if mode == "llm":
        if llm_client is None:
            logger.warning(
                "classifier_mode=llm but no llm_client was provided — "
                "falling back to heuristic"
            )
            return HeuristicClassifier()
        return LLMClassifier(
            llm_client,
            model_override=settings.classifier_model or "",
            provider=settings.llm_provider,
        )

    if mode == "local":
        return LocalClassifier(settings.classifier_local_model_path or "")

    if mode != "heuristic":
        logger.warning(
            "Unknown classifier_mode %r — falling back to heuristic", mode
        )

    return HeuristicClassifier()

