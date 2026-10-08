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
from typing import Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)


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
        query_embedding = self._embed.encode(query).tolist()

        self._collection.add(
            ids=[query_id],
            embeddings=[query_embedding],
            documents=[response],
            metadatas=[{"query": query, "query_type": "simple"}],
        )
        logger.debug("Cached response for query_id=%s", query_id)
