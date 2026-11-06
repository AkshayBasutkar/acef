"""
RAGCache-Lite — semantic cache backed by a Chroma collection named "cache".

Embeds the query, searches for the nearest cached query, and returns the
stored response if cosine similarity exceeds the configured threshold.  The
cache is write-through: every non-cached response is stored immediately
after the LLM call completes.

Both this module and mem_rag.py share the same Chroma client (one embedded
database, two collections) and the same SentenceTransformer instance (loaded
once at the app level and passed in — never reloaded here).
"""

import logging
import re
import threading
from typing import Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Compact stop-words set for lowercase content-term extraction
STOP_WORDS: set[str] = {
    # Pronouns, determiners, conjunctions
    "what", "which", "where", "when", "who", "whom", "whose", "why", "how",
    "that", "this", "these", "those",
    "a", "an", "the", "and", "or", "but", "nor", "for", "yet", "so",
    "both", "either", "neither", "each", "every", "all", "any", "some",
    "few", "more", "most", "other", "such", "than", "too", "very",
    # Prepositions
    "about", "above", "across", "after", "against", "along", "among", "around",
    "at", "before", "behind", "below", "beneath", "beside", "between", "beyond",
    "by", "down", "during", "except", "from", "in", "inside", "into", "near",
    "off", "on", "onto", "out", "outside", "over", "through", "throughout",
    "till", "to", "toward", "under", "until", "up", "upon", "with", "within", "without",
    # Verbs / auxiliaries
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "can", "could", "shall", "should", "will", "would", "may", "might", "must",
    # Common conversational / query framing verbs & fillers
    "tell", "show", "give", "explain", "describe", "find", "list", "name",
    "please", "help", "know", "mean", "want", "like", "just", "also", "city",
}


def extract_content_terms(text: str) -> set[str]:
    """
    Extract meaningful lowercase content terms (nouns, lowercase entities,
    identifiers, code terms) excluding common English stop words.

    Captures:
      - Identifier-like tokens with '.' or '_' (e.g. 'auth.py', 'user_id', 'hash_password')
      - General content words of length >= 4 not in STOP_WORDS (e.g. 'france', 'germany', 'developer', 'enterprise')
    """
    raw_words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_.-]*", text.lower())
    content_terms = set()
    for w in raw_words:
        cleaned = w.strip(".-")
        if not cleaned:
            continue
        if ("." in cleaned or "_" in cleaned) and cleaned not in STOP_WORDS:
            content_terms.add(cleaned)
        elif len(cleaned) >= 4 and cleaned not in STOP_WORDS:
            content_terms.add(cleaned)
    return content_terms


def cache_terms_compatible(current_query: str, cached_query: str) -> bool:
    """
    Verify that the meaningful content terms between the current query and cached query match.

    Prevents false cache hits when embedding similarity is high but key lowercase
    entities or identifiers differ (e.g. 'france' vs 'germany', 'auth.py' vs 'main.py').
    """
    current_terms = extract_content_terms(current_query)
    cached_terms = extract_content_terms(cached_query)
    if current_terms or cached_terms:
        return current_terms == cached_terms
    return True


def extract_salient_tokens(text: str) -> set[str]:
    """
    Extract salient tokens (digits/numbers, quoted strings, and capitalized words
    not at the start of a sentence) from a query string.

    Parameters
    ----------
    text : str
        Input query text.

    Returns
    -------
    set[str]
        Set of normalized salient tokens.
    """
    salient: set[str] = set()

    # 1. Digits and numbers
    for num in re.findall(r"\b\d+(?:\.\d+)?\b", text):
        salient.add(num)

    # 2. Quoted strings (single, double, or backticks)
    for quoted in re.findall(r"[\"'\`]([^\"'\`]+)[\"'\`]", text):
        cleaned = quoted.strip().lower()
        if cleaned:
            salient.add(f"quote:{cleaned}")

    # 3. Capitalised words not at the start of a sentence
    # Split text into sentence segments
    sentences = re.split(r"[.?!]\s*", text)
    for sent in sentences:
        words = re.findall(r"\b[A-Za-z0-9_'\-]+\b", sent)
        if not words:
            continue
        # Skip the first word of the sentence
        for w in words[1:]:
            clean_w = w.rstrip("'s").strip("'\"-_")
            if clean_w and clean_w[0].isupper():
                salient.add(clean_w.lower())

    return salient


class RAGCacheLite:
    """Semantic similarity cache using ChromaDB."""

    def __init__(self, chroma_client, embedding_model):
        """
        Parameters
        ----------
        chroma_client : chromadb.ClientAPI
            Shared Chroma client (persistent, local directory).
        embedding_model : sentence_transformers.SentenceTransformer
            Shared embedding model — loaded ONCE at app level.
        """
        self._chroma_client = chroma_client
        self._collection = chroma_client.get_or_create_collection(
            name="cache",
            metadata={"hnsw:space": "cosine"},
        )
        self._embed = embedding_model
        self._threshold = get_settings().cache_similarity_threshold
        self._lock = threading.Lock()
        logger.info(
            "RAGCacheLite ready — threshold=%.3f  existing_items=%d",
            self._threshold,
            self._collection.count(),
        )

    def count(self) -> int:
        """Return the number of items stored in the cache."""
        return self._collection.count()

    def clear(self) -> None:
        """Clear all entries from the cache collection by recreating it."""
        with self._lock:
            try:
                self._chroma_client.delete_collection(name="cache")
            except Exception as e:
                logger.debug("delete_collection on clear: %s", e)
            self._collection = self._chroma_client.get_or_create_collection(
                name="cache",
                metadata={"hnsw:space": "cosine"},
            )
        logger.info("RAGCacheLite cleared — existing_items=%d", self._collection.count())

    # ─────────────────────────────────────────────────────────────────────────
    def lookup(
        self,
        query: str,
        query_vec: Optional[list[float]] = None,
    ) -> Optional[str]:
        """
        Search for a semantically near-identical cached query.

        Returns the cached response string if similarity >= threshold,
        otherwise None.

        Chroma returns *cosine distance* by default when the collection's
        hnsw:space is "cosine".  Similarity = 1 - distance.
        """
        if self._collection.count() == 0:
            return None

        if query_vec is not None:
            query_embedding = query_vec
        else:
            emb = self._embed.encode(query)
            query_embedding = emb.tolist() if hasattr(emb, "tolist") else list(emb)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents", "distances", "metadatas"],
        )

        # results shape: {"ids": [[...]], "documents": [[...]], "distances": [[...]], "metadatas": [[...]]}
        if not results["ids"] or not results["ids"][0]:
            return None

        distance = results["distances"][0][0]
        similarity = 1.0 - distance

        if similarity >= self._threshold:
            # I-06 Salient-token guard: prevent confident wrong answers on entity/number swaps
            metas = results.get("metadatas", [[]])
            cached_query = metas[0][0].get("query", "") if (metas and metas[0] and isinstance(metas[0][0], dict)) else ""
            if cached_query:
                # 1. Salient-token guard (digits/numbers, quoted strings, capitalized words)
                tokens_curr = extract_salient_tokens(query)
                tokens_cached = extract_salient_tokens(cached_query)
                if tokens_curr != tokens_cached:
                    logger.info(
                        "Cache miss (salient guard) — similarity=%.4f >= %.3f, but salient tokens mismatch: %s vs %s",
                        similarity,
                        self._threshold,
                        tokens_curr,
                        tokens_cached,
                    )
                    return None

                # 2. Content-term guard (lowercase entities, code identifiers, important terms)
                if not cache_terms_compatible(query, cached_query):
                    logger.info(
                        "Cache miss (content guard) — similarity=%.4f >= %.3f, but content terms mismatch: %s vs %s",
                        similarity,
                        self._threshold,
                        extract_content_terms(query),
                        extract_content_terms(cached_query),
                    )
                    return None

            cached_response = results["documents"][0][0]
            logger.info(
                "Cache HIT — similarity=%.4f (threshold=%.3f)",
                similarity,
                self._threshold,
            )
            return cached_response

        logger.debug(
            "Cache miss — best similarity=%.4f < threshold=%.3f",
            similarity,
            self._threshold,
        )
        return None

    # ─────────────────────────────────────────────────────────────────────────
    def store(self, query: str, response: str, query_id: str) -> None:
        """
        Store a (query, response) pair in the cache.

        The query is embedded and stored as the vector; the response is stored
        as the Chroma document (returned on cache hits).  The original query
        text is kept in metadata for debugging/inspection.
        """
        raw_emb = self._embed.encode(query)
        query_embedding = raw_emb.tolist() if hasattr(raw_emb, "tolist") else list(raw_emb)

        with self._lock:
            self._collection.add(
                ids=[query_id],
                embeddings=[query_embedding],
                documents=[response],
                metadatas=[{"query": query, "query_type": "simple"}],
            )
        logger.debug("Cached response for query_id=%s", query_id)
