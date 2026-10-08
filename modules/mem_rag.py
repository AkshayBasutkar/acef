"""
MemRAG — document retrieval and conversation memory backed by ChromaDB.

Manages the "memory" collection in the shared Chroma instance:
  1. Document chunks (tagged with type="document", source=filename)
  2. Conversation turns (tagged with type="conversation", session_id=...)

Retrieval isolates conversation turns to the current session_id while
allowing shared access to all ingested documents.

Both this module and cache.py share the same Chroma client (one embedded
database, two collections) and the same SentenceTransformer instance (loaded
once at the app level and passed in).
"""

import logging
import time
import uuid
from typing import Optional

from core.schemas import QueryType, RetrievedChunk

logger = logging.getLogger(__name__)


class MemRAG:
    """Document and conversation memory retrieval using ChromaDB."""

    def __init__(self, chroma_client, embedding_model):
        """
        Parameters
        ----------
        chroma_client : chromadb.ClientAPI
            Shared Chroma client (persistent local directory).
        embedding_model : sentence_transformers.SentenceTransformer
            Shared embedding model instance.
        """
        self._collection = chroma_client.get_or_create_collection(
            name="memory",
            metadata={"hnsw:space": "cosine"},
        )
        self._embed = embedding_model
        logger.info(
            "MemRAG ready — existing_items=%d in 'memory' collection",
            self._collection.count(),
        )

    @property
    def collection(self):
        """Access the underlying Chroma collection."""
        return self._collection

    def embed_query(self, query: str) -> list[float]:
        """Compute query embedding vector using the shared embedding model."""
        emb = self._embed.encode(query)
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

    # ─────────────────────────────────────────────────────────────────────────
    def best_document_similarity(
        self,
        query: Optional[str] = None,
        query_vec: Optional[list[float]] = None,
    ) -> float:
        """
        Return the highest cosine similarity between the query and ingested documents.
        Returns 0.0 if no documents have been ingested or if the probe fails.
        """
        try:
            if self._collection.count() == 0:
                return 0.0

            if query_vec is None:
                if not query:
                    return 0.0
                query_vec = self.embed_query(query)

            doc_res = self._collection.query(
                query_embeddings=[query_vec],
                n_results=1,
                where={"type": "document"},
                include=["distances"],
            )

            if (
                not doc_res.get("ids")
                or not doc_res["ids"][0]
                or not doc_res.get("distances")
                or not doc_res["distances"][0]
            ):
                return 0.0

            dist = doc_res["distances"][0][0]
            similarity = max(0.0, 1.0 - dist)
            return float(similarity)
        except Exception as e:
            logger.warning("Error calculating best document similarity: %s", e)
            return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
        query_vec: Optional[list[float]] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant document excerpts and conversation history for this session.

        Executes two queries to guarantee compatibility across all Chroma versions:
          1. Documents: where={"type": "document"}
          2. Conversation: where={"$and": [{"type": "conversation"}, {"session_id": session_id}]}
        Merges results by cosine distance and returns the overall top_k chunks.

        Parameters
        ----------
        query : str
            User's query.
        session_id : str
            Session identifier for isolating conversation memory.
        top_k : int
            Maximum number of combined chunks to return (default 5).
        query_vec : list[float], optional
            Precomputed query embedding vector to avoid redundant model encoding.

        Returns
        -------
        list[RetrievedChunk]
            Matched chunks with text and source annotations.
        """
        if self._collection.count() == 0:
            return []

        if query_vec is not None:
            emb_list = query_vec
        else:
            emb_list = self.embed_query(query)

        candidates = []

        # 1. Query document chunks
        try:
            doc_res = self._collection.query(
                query_embeddings=[emb_list],
                n_results=top_k,
                where={"type": "document"},
                include=["documents", "metadatas", "distances"],
            )
            if doc_res["ids"] and doc_res["ids"][0]:
                for doc, meta, dist in zip(
                    doc_res["documents"][0],
                    doc_res["metadatas"][0],
                    doc_res["distances"][0],
                ):
                    candidates.append({
                        "text": doc,
                        "source": meta.get("source", "document"),
                        "distance": dist,
                    })
        except Exception as e:
            logger.warning("Error querying document chunks: %s", e)

        # 2. Query conversation turns for this session
        try:
            conv_res = self._collection.query(
                query_embeddings=[emb_list],
                n_results=top_k,
                where={
                    "$and": [
                        {"type": "conversation"},
                        {"session_id": str(session_id)},
                    ]
                },
                include=["documents", "metadatas", "distances"],
            )
            if conv_res["ids"] and conv_res["ids"][0]:
                for doc, meta, dist in zip(
                    conv_res["documents"][0],
                    conv_res["metadatas"][0],
                    conv_res["distances"][0],
                ):
                    candidates.append({
                        "text": doc,
                        "source": f"conversation turn ({session_id})",
                        "distance": dist,
                    })
        except Exception as e:
            logger.warning("Error querying conversation turns: %s", e)

        if not candidates:
            return []

        # Sort candidates ascending by distance (lower distance = closer match)
        candidates.sort(key=lambda x: x["distance"])
        top_candidates = candidates[:top_k]

        return [
            RetrievedChunk(text=c["text"], source=c["source"])
            for c in top_candidates
        ]

    # ─────────────────────────────────────────────────────────────────────────
    def append_turn(
        self,
        session_id: str,
        query: str,
        response: str,
        query_type: str = "simple",
    ) -> None:
        """
        Store a conversation turn in the memory collection.

        Parameters
        ----------
        session_id : str
            Session identifier.
        query : str
            User query.
        response : str
            Assistant response.
        query_type : str
            The classification type of the query (simple, code, memory).
        """
        turn_text = f"User asked: {query}\nAssistant answered: {response}"
        turn_id = f"turn_{session_id}_{uuid.uuid4().hex[:8]}"

        raw_emb = self._embed.encode(turn_text)
        emb_list = raw_emb.tolist() if hasattr(raw_emb, "tolist") else list(raw_emb)

        self._collection.add(
            ids=[turn_id],
            embeddings=[emb_list],
            documents=[turn_text],
            metadatas=[{
                "type": "conversation",
                "session_id": str(session_id),
                "query_type": str(query_type),
                "timestamp": time.time(),
            }],
        )
        logger.debug(
            "Appended turn for session_id=%s (turn_id=%s, type=%s)",
            session_id, turn_id, query_type,
        )

    # ─────────────────────────────────────────────────────────────────────────
    def last_turn_type(self, session_id: str) -> Optional[QueryType]:
        """
        Return the QueryType of the most recent turn in the given session, or None.

        Parameters
        ----------
        session_id : str
            Session identifier.

        Returns
        -------
        QueryType | None
            QueryType of the last turn, or None if no turns exist for this session.
        """
        try:
            res = self._collection.get(
                where={
                    "$and": [
                        {"type": "conversation"},
                        {"session_id": str(session_id)},
                    ]
                },
                include=["metadatas"],
            )
            metas = res.get("metadatas", [])
            if not metas:
                return None

            latest = max(metas, key=lambda m: m.get("timestamp", 0))
            raw_type = latest.get("query_type", "simple")
            try:
                return QueryType(raw_type)
            except ValueError:
                return QueryType.SIMPLE
        except Exception as e:
            logger.warning("Error getting last turn type for session %s: %s", session_id, e)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    def session_turn_count(self, session_id: str) -> int:
        """
        Count the number of conversation turns recorded for a given session.

        Parameters
        ----------
        session_id : str
            Session identifier.

        Returns
        -------
        int
            Turn count.
        """
        try:
            res = self._collection.get(
                where={
                    "$and": [
                        {"type": "conversation"},
                        {"session_id": str(session_id)},
                    ]
                }
            )
            return len(res.get("ids", []))
        except Exception as e:
            logger.warning("Error counting turns for session %s: %s", session_id, e)
            return 0
