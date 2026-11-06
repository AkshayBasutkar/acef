"""
Unit tests for RAGCacheLite (modules/cache.py).

Tests store, lookup, clearing, and metadata tagging.
"""

from unittest.mock import MagicMock
import pytest

from modules.cache import RAGCacheLite


@pytest.fixture
def mock_cache_setup():
    mock_chroma = MagicMock()
    mock_col = MagicMock()
    mock_chroma.get_or_create_collection.return_value = mock_col
    mock_embed = MagicMock()
    mock_embed.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2, 0.3])
    return mock_chroma, mock_col, mock_embed


def test_cache_initialization(mock_cache_setup):
    mock_chroma, mock_col, mock_embed = mock_cache_setup
    mock_col.count.return_value = 0

    cache = RAGCacheLite(mock_chroma, mock_embed)
    assert cache.count() == 0
    mock_chroma.get_or_create_collection.assert_called_once_with(
        name="cache",
        metadata={"hnsw:space": "cosine"},
    )


def test_cache_store_stores_simple_query_type_metadata(mock_cache_setup):
    mock_chroma, mock_col, mock_embed = mock_cache_setup
    mock_col.count.return_value = 0

    cache = RAGCacheLite(mock_chroma, mock_embed)
    cache.store("What is 2+2?", "4", "q-123")

    mock_col.add.assert_called_once()
    kwargs = mock_col.add.call_args[1]
    assert kwargs["ids"] == ["q-123"]
    assert kwargs["documents"] == ["4"]
    assert kwargs["metadatas"] == [{"query": "What is 2+2?", "query_type": "simple"}]


def test_cache_clear_recreates_collection(mock_cache_setup):
    mock_chroma, mock_col, mock_embed = mock_cache_setup
    cache = RAGCacheLite(mock_chroma, mock_embed)

    cache.clear()

    mock_chroma.delete_collection.assert_called_once_with(name="cache")
    assert mock_chroma.get_or_create_collection.call_count == 2


def test_cache_lookup_hit_above_threshold(mock_cache_setup):
    mock_chroma, mock_col, mock_embed = mock_cache_setup
    mock_col.count.return_value = 1
    # distance = 0.05 -> similarity = 0.95 >= 0.92
    mock_col.query.return_value = {
        "ids": [["q-123"]],
        "documents": [["Cached Paris answer"]],
        "distances": [[0.05]],
    }

    cache = RAGCacheLite(mock_chroma, mock_embed)
    result = cache.lookup("What is France capital?")
    assert result == "Cached Paris answer"


def test_cache_lookup_miss_below_threshold(mock_cache_setup):
    mock_chroma, mock_col, mock_embed = mock_cache_setup
    mock_col.count.return_value = 1
    # distance = 0.20 -> similarity = 0.80 < 0.92
    mock_col.query.return_value = {
        "ids": [["q-123"]],
        "documents": [["Cached Paris answer"]],
        "distances": [[0.20]],
    }

    cache = RAGCacheLite(mock_chroma, mock_embed)
    result = cache.lookup("What is German capital?")
    assert result is None


def test_extract_salient_tokens():
    from modules.cache import extract_salient_tokens

    # Numbers
    assert extract_salient_tokens("Give me 3 tips and 42 tricks") == {"3", "42"}

    # Quoted strings
    assert extract_salient_tokens("Inspect 'auth.py' and `main`") == {"quote:auth.py", "quote:main"}

    # Capitalized words not at start of sentence
    t = extract_salient_tokens("What is the capital of France? Tell me.")
    assert "france" in t
    assert "what" not in t  # Start of first sentence
    assert "tell" not in t  # Start of second sentence


def test_salient_guard_rejects_entity_swap(mock_cache_setup):
    """Even if similarity is 0.98, France vs Germany must be rejected as a miss."""
    mock_chroma, mock_col, mock_embed = mock_cache_setup
    mock_col.count.return_value = 1
    mock_col.query.return_value = {
        "ids": [["q-123"]],
        "documents": [["Paris is the capital of France."]],
        "distances": [[0.02]],  # similarity 0.98 >= 0.92
        "metadatas": [[{"query": "What is the capital of Germany?", "query_type": "simple"}]],
    }

    cache = RAGCacheLite(mock_chroma, mock_embed)
    result = cache.lookup("What is the capital of France?")
    assert result is None  # Guard treated it as a miss!


def test_salient_guard_rejects_number_swap(mock_cache_setup):
    """Even if similarity is 0.98, 3 tips vs 5 tips must be rejected as a miss."""
    mock_chroma, mock_col, mock_embed = mock_cache_setup
    mock_col.count.return_value = 1
    mock_col.query.return_value = {
        "ids": [["q-123"]],
        "documents": [["Here are 5 tips..."]],
        "distances": [[0.02]],  # similarity 0.98 >= 0.92
        "metadatas": [[{"query": "Give me 5 tips for a job interview.", "query_type": "simple"}]],
    }

    cache = RAGCacheLite(mock_chroma, mock_embed)
    result = cache.lookup("Give me 3 tips for a job interview.")
    assert result is None  # Guard treated it as a miss!


def test_salient_guard_accepts_paraphrase_with_same_salient_tokens(mock_cache_setup):
    """Paraphrases sharing identical salient tokens hit the cache."""
    mock_chroma, mock_col, mock_embed = mock_cache_setup
    mock_col.count.return_value = 1
    mock_col.query.return_value = {
        "ids": [["q-123"]],
        "documents": [["Paris is the capital of France."]],
        "distances": [[0.02]],  # similarity 0.98 >= 0.92
        "metadatas": [[{"query": "Which city is France's capital?", "query_type": "simple"}]],
    }

    cache = RAGCacheLite(mock_chroma, mock_embed)
    result = cache.lookup("What is the capital of France?")
    assert result == "Paris is the capital of France."


# ── Lowercase Content-Term Guard Tests (Change 1) ───────────────────────────

def test_extract_content_terms():
    from modules.cache import extract_content_terms

    # Basic entity extraction with stop-words removed
    terms = extract_content_terms("What is the capital of france?")
    assert terms == {"capital", "france"}

    # Identifiers with dots and underscores preserved
    code_terms = extract_content_terms("Explain auth.py and hash_password")
    assert "auth.py" in code_terms
    assert "hash_password" in code_terms

    # Short stop words discarded
    assert extract_content_terms("what is in it for me?") == set()


def test_content_guard_rejects_lowercase_country_swap(mock_cache_setup):
    """Change 1: Lowercase 'france' vs 'germany' rejected despite high similarity."""
    mock_chroma, mock_col, mock_embed = mock_cache_setup
    mock_col.count.return_value = 1
    mock_col.query.return_value = {
        "ids": [["q-123"]],
        "documents": [["Paris is the capital of France."]],
        "distances": [[0.02]],  # similarity 0.98 >= 0.92
        "metadatas": [[{"query": "what is the capital of germany?", "query_type": "simple"}]],
    }

    cache = RAGCacheLite(mock_chroma, mock_embed)
    result = cache.lookup("what is the capital of france?")
    assert result is None


def test_content_guard_rejects_plan_tier_swap(mock_cache_setup):
    """Change 1: 'developer' plan vs 'enterprise' plan rejected despite high similarity."""
    mock_chroma, mock_col, mock_embed = mock_cache_setup
    mock_col.count.return_value = 1
    mock_col.query.return_value = {
        "ids": [["q-123"]],
        "documents": [["The Developer tier is free."]],
        "distances": [[0.02]],  # similarity 0.98 >= 0.92
        "metadatas": [[{"query": "What does the enterprise plan cost?", "query_type": "simple"}]],
    }

    cache = RAGCacheLite(mock_chroma, mock_embed)
    result = cache.lookup("What does the developer plan cost?")
    assert result is None


def test_content_guard_rejects_file_identifier_swap(mock_cache_setup):
    """Change 1: 'auth.py' vs 'main.py' rejected despite high similarity."""
    mock_chroma, mock_col, mock_embed = mock_cache_setup
    mock_col.count.return_value = 1
    mock_col.query.return_value = {
        "ids": [["q-123"]],
        "documents": [["auth.py handles authentication."]],
        "distances": [[0.02]],  # similarity 0.98 >= 0.92
        "metadatas": [[{"query": "Explain main.py", "query_type": "simple"}]],
    }

    cache = RAGCacheLite(mock_chroma, mock_embed)
    result = cache.lookup("Explain auth.py")
    assert result is None


def test_content_guard_rejects_function_identifier_swap(mock_cache_setup):
    """Change 1: 'hash_password' vs 'validate_username' rejected despite high similarity."""
    mock_chroma, mock_col, mock_embed = mock_cache_setup
    mock_col.count.return_value = 1
    mock_col.query.return_value = {
        "ids": [["q-123"]],
        "documents": [["hash_password hashes user passwords."]],
        "distances": [[0.02]],  # similarity 0.98 >= 0.92
        "metadatas": [[{"query": "What does validate_username do?", "query_type": "simple"}]],
    }

    cache = RAGCacheLite(mock_chroma, mock_embed)
    result = cache.lookup("What does hash_password do?")
    assert result is None


def test_content_guard_accepts_paraphrase_with_same_content_terms(mock_cache_setup):
    """Paraphrases sharing the same meaningful content terms hit the cache."""
    mock_chroma, mock_col, mock_embed = mock_cache_setup
    mock_col.count.return_value = 1
    mock_col.query.return_value = {
        "ids": [["q-123"]],
        "documents": [["Paris is the capital of France."]],
        "distances": [[0.02]],  # similarity 0.98 >= 0.92
        "metadatas": [[{"query": "tell me the capital of france please", "query_type": "simple"}]],
    }

    cache = RAGCacheLite(mock_chroma, mock_embed)
    result = cache.lookup("what is the capital of france?")
    assert result == "Paris is the capital of France."


# ── Real Ephemeral ChromaDB Test (Verification Requirement) ──────────────────

def test_real_chroma_cache_lifecycle_and_content_guard():
    """
    Real in-memory Chroma test using chromadb.EphemeralClient():
    1. Stores an entry.
    2. Retrieves an equivalent query (cache hit).
    3. Rejects an entity-swapped query (france vs germany) even when similarity = 1.0.
    4. Clears the cache collection and confirms it is empty.
    """
    import chromadb

    class FixedEmbedder:
        """Deterministic embedder returning a fixed vector for predictable testing."""
        def encode(self, text):
            return [1.0, 0.0, 0.0]

    client = chromadb.EphemeralClient()
    cache = RAGCacheLite(client, FixedEmbedder())

    # 1. Store
    cache.store("what is the capital of france?", "Paris is the capital of France.", "id-france")
    assert cache.count() == 1

    # 2. Retrieve equivalent query with same content terms (HIT)
    hit = cache.lookup("what is the capital of france?")
    assert hit == "Paris is the capital of France."

    # 3. Reject entity-swapped query: vector similarity is 1.0, but content terms differ ('germany' vs 'france')
    miss = cache.lookup("what is the capital of germany?")
    assert miss is None

    # 4. Clear and confirm empty
    cache.clear()
    assert cache.count() == 0
    assert cache.lookup("what is the capital of france?") is None
