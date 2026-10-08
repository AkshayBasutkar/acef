"""
Unit tests for MemRAG and document ingestion.

Tests text extraction, sliding-window chunking, document ingestion,
session-isolated conversation memory, and retrieval.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from core.schemas import QueryType, RetrievedChunk
from ingestion.document_ingest import _chunk_text, _extract_text, ingest_document
from modules.mem_rag import MemRAG


# ── Text Extraction Tests ───────────────────────────────────────────────────

class TestExtractText:
    def test_extract_txt(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("Hello text world", encoding="utf-8")
        assert _extract_text(f) == "Hello text world"

    def test_extract_md(self, tmp_path):
        f = tmp_path / "sample.md"
        f.write_text("# Heading\nContent in markdown", encoding="utf-8")
        assert "# Heading" in _extract_text(f)

    def test_unsupported_format(self, tmp_path):
        f = tmp_path / "sample.xyz"
        f.write_text("unknown", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported document format"):
            _extract_text(f)


# ── Text Chunking Tests ─────────────────────────────────────────────────────

class TestChunkText:
    def test_empty_text(self):
        assert _chunk_text("") == []
        assert _chunk_text("   \n\t  ") == []

    def test_small_text_single_chunk(self):
        text = "word1 word2 word3"
        chunks = _chunk_text(text, chunk_size=10, chunk_overlap=2)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_sliding_window_overlap(self):
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size=40, chunk_overlap=10)

        assert len(chunks) > 1
        # Check first chunk has 40 words
        assert len(chunks[0].split()) == 40
        # Check second chunk starts at index 30 (40 - 10 overlap)
        second_chunk_first_word = chunks[1].split()[0]
        assert second_chunk_first_word == "word30"


# ── Ingest Document Tests ───────────────────────────────────────────────────

class TestIngestDocument:
    def test_file_not_found(self):
        col = MagicMock()
        embed = MagicMock()
        with pytest.raises(FileNotFoundError):
            ingest_document("nonexistent_file_abc123.md", col, embed)

    def test_ingest_document_successful(self, tmp_path):
        doc = tmp_path / "test_doc.md"
        doc.write_text("CloudScale AI pricing tiers include Free, Pro, and Enterprise.", encoding="utf-8")

        mock_col = MagicMock()
        mock_embed = MagicMock()
        mock_embed.encode.return_value = [0.1, 0.2, 0.3]

        count = ingest_document(str(doc), mock_col, mock_embed, chunk_size=300)
        assert count == 1

        # Check call to upsert or add
        call_method = mock_col.upsert if hasattr(mock_col, "upsert") else mock_col.add
        call_method.assert_called_once()
        kwargs = call_method.call_args[1]
        assert kwargs["ids"] == ["test_doc.md_0"]
        assert kwargs["metadatas"][0]["source"] == "test_doc.md"
        assert kwargs["metadatas"][0]["type"] == "document"


class TestIdempotentDocumentIngest:
    """I-12: Test that re-ingesting documents clears stale chunks."""

    def test_reingest_shorter_document_cleans_stale_chunks(self, tmp_path):
        import chromadb
        client = chromadb.EphemeralClient()
        col = client.create_collection("test_mem")

        class SimpleFakeEmbed:
            def encode(self, text):
                return [0.1, 0.2, 0.3]

        doc = tmp_path / "manual.txt"
        # Long document -> multiple chunks with chunk_size=10
        doc.write_text("word " * 50, encoding="utf-8")

        count1 = ingest_document(str(doc), col, SimpleFakeEmbed(), chunk_size=10, chunk_overlap=2)
        assert count1 > 1
        assert col.count() == count1

        # Ingest twice -> count is still count1 (not doubled)
        count2 = ingest_document(str(doc), col, SimpleFakeEmbed(), chunk_size=10, chunk_overlap=2)
        assert count2 == count1
        assert col.count() == count1

        # Replace with short document -> 1 chunk
        doc.write_text("short single chunk", encoding="utf-8")
        count3 = ingest_document(str(doc), col, SimpleFakeEmbed(), chunk_size=10, chunk_overlap=2)
        assert count3 == 1
        # Crucial: stale chunks from previous run were deleted, so count is exactly 1!
        assert col.count() == 1



# ── MemRAG Retrieval & Conversation Memory Tests ─────────────────────────────

class TestMemRAG:
    @pytest.fixture
    def mock_components(self):
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_col
        mock_embed = MagicMock()
        mock_embed.encode.return_value = [0.1, 0.2, 0.3]
        return mock_client, mock_col, mock_embed

    def test_init_creates_memory_collection(self, mock_components):
        mock_client, mock_col, mock_embed = mock_components
        rag = MemRAG(mock_client, mock_embed)
        mock_client.get_or_create_collection.assert_called_once_with(
            name="memory",
            metadata={"hnsw:space": "cosine"},
        )
        assert rag.collection == mock_col

    def test_append_turn(self, mock_components):
        mock_client, mock_col, mock_embed = mock_components
        rag = MemRAG(mock_client, mock_embed)

        rag.append_turn("sess-1", "What is the plan?", "Pro tier is $49", query_type="code")

        mock_col.add.assert_called_once()
        kwargs = mock_col.add.call_args[1]
        assert "User asked: What is the plan?" in kwargs["documents"][0]
        assert "Assistant answered: Pro tier is $49" in kwargs["documents"][0]
        assert kwargs["metadatas"][0]["type"] == "conversation"
        assert kwargs["metadatas"][0]["session_id"] == "sess-1"
        assert kwargs["metadatas"][0]["query_type"] == "code"

    def test_last_turn_type(self, mock_components):
        mock_client, mock_col, mock_embed = mock_components
        # Return two turns with different timestamps
        mock_col.get.return_value = {
            "metadatas": [
                {"type": "conversation", "session_id": "sess-1", "query_type": "simple", "timestamp": 100.0},
                {"type": "conversation", "session_id": "sess-1", "query_type": "code", "timestamp": 200.0},
            ]
        }
        rag = MemRAG(mock_client, mock_embed)
        last_type = rag.last_turn_type("sess-1")
        assert last_type == QueryType.CODE

    def test_last_turn_type_empty_returns_none(self, mock_components):
        mock_client, mock_col, mock_embed = mock_components
        mock_col.get.return_value = {"metadatas": []}
        rag = MemRAG(mock_client, mock_embed)
        assert rag.last_turn_type("sess-empty") is None

    def test_session_turn_count(self, mock_components):
        mock_client, mock_col, mock_embed = mock_components
        mock_col.get.return_value = {"ids": ["turn1", "turn2", "turn3"]}

        rag = MemRAG(mock_client, mock_embed)
        count = rag.session_turn_count("sess-1")

        assert count == 3
        mock_col.get.assert_called_once_with(
            where={"$and": [{"type": "conversation"}, {"session_id": "sess-1"}]}
        )

    def test_retrieve_empty_collection(self, mock_components):
        mock_client, mock_col, mock_embed = mock_components
        mock_col.count.return_value = 0

        rag = MemRAG(mock_client, mock_embed)
        chunks = rag.retrieve("Tell me about pricing", session_id="sess-1")
        assert chunks == []

    def test_retrieve_merges_and_sorts_docs_and_conversation(self, mock_components):
        mock_client, mock_col, mock_embed = mock_components
        mock_col.count.return_value = 10

        # Document query returns a match with distance 0.2
        doc_return = {
            "ids": [["doc_1"]],
            "documents": [["Pricing is $49/mo"]],
            "metadatas": [[{"source": "pricing.md", "type": "document"}]],
            "distances": [[0.20]],
        }
        # Conversation query returns a turn with closer distance 0.10
        conv_return = {
            "ids": [["turn_1"]],
            "documents": [["Earlier turn discussing discounts"]],
            "metadatas": [[{"type": "conversation", "session_id": "sess-1"}]],
            "distances": [[0.10]],
        }

        # Mock query return based on where arg
        def side_effect_query(*args, **kwargs):
            where = kwargs.get("where", {})
            if where.get("type") == "document":
                return doc_return
            return conv_return

        mock_col.query.side_effect = side_effect_query

        rag = MemRAG(mock_client, mock_embed)
        chunks = rag.retrieve("What is the cost?", session_id="sess-1", top_k=2)

        assert len(chunks) == 2
        # First chunk should be the closer match (distance 0.10 -> conversation turn)
        assert chunks[0].source == "conversation turn (sess-1)"
        assert chunks[0].text == "Earlier turn discussing discounts"
        # Second chunk should be pricing.md
        assert chunks[1].source == "pricing.md"
        assert chunks[1].text == "Pricing is $49/mo"

    def test_retrieve_failure_gracefully_degrades(self, mock_components):
        mock_client, mock_col, mock_embed = mock_components
        mock_col.count.return_value = 5
        mock_col.query.side_effect = RuntimeError("Chroma query failed")

        rag = MemRAG(mock_client, mock_embed)
        chunks = rag.retrieve("query", session_id="sess-1")
        assert chunks == []

    def test_retrieve_drops_chunks_below_relevance_floor(self, mock_components):
        mock_client, mock_col, mock_embed = mock_components
        mock_col.count.return_value = 2

        # Distance 0.85 -> similarity 0.15 < relevance_floor (0.30)
        mock_col.query.side_effect = [
            {
                "ids": [["doc1"]],
                "documents": [["Irrelevant doc excerpt"]],
                "metadatas": [[{"source": "random.txt"}]],
                "distances": [[0.85]],
            },
            {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            },
        ]

        rag = MemRAG(mock_client, mock_embed, relevance_floor=0.30)
        chunks = rag.retrieve("quantum physics", session_id="sess-1")
        assert chunks == []
