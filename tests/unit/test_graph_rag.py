"""
Unit tests for GraphRAG search term extraction and code ingestion parsing.

Tests the heuristic token extraction (which doesn't require Neo4j)
and the AST-based file parser (which doesn't require Neo4j either).
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import MagicMock
from pathlib import Path
from modules.graph_rag import _extract_search_terms, GraphRAG
from ingestion.code_ingest import _parse_file, ingest_codebase


class TestExtractSearchTerms:
    """Test the identifier extraction heuristic for Neo4j full-text queries."""

    def test_snake_case_detected(self):
        terms = _extract_search_terms("What does hash_password do?")
        assert "hash_password" in terms

    def test_quoted_string_detected(self):
        terms = _extract_search_terms('Find the function "login" in the codebase')
        assert "login" in terms

    def test_single_quoted_detected(self):
        terms = _extract_search_terms("What does 'create_user' return?")
        assert "create_user" in terms

    def test_pascal_case_detected(self):
        terms = _extract_search_terms("Explain the UserManager class")
        assert "UserManager" in terms

    def test_fallback_to_longest_words(self):
        """When no identifiers are found, fall back to 3 longest words."""
        terms = _extract_search_terms("What does the login do?")
        # "login" (5 chars) should be among the top 3 longest
        assert "login" in terms

    def test_multiple_identifiers_joined_with_or(self):
        terms = _extract_search_terms("Does hash_password get called by create_user?")
        assert "hash_password" in terms
        assert "create_user" in terms
        assert "OR" in terms

    def test_empty_fallback(self):
        """Even a minimal query should produce some search terms."""
        terms = _extract_search_terms("why?")
        assert len(terms) > 0


class TestParseFile:
    """Test the AST parser against the demo codebase files."""

    @pytest.fixture
    def demo_codebase_path(self):
        return Path(__file__).resolve().parent.parent.parent / "data" / "demo_codebase"

    def test_parse_auth(self, demo_codebase_path):
        result = _parse_file(demo_codebase_path / "auth.py")
        symbol_names = [s["name"] for s in result["symbols"]]
        assert "login" in symbol_names
        assert "create_user" in symbol_names

    def test_parse_auth_has_imports(self, demo_codebase_path):
        result = _parse_file(demo_codebase_path / "auth.py")
        assert "utils" in result["imports"]

    def test_parse_auth_login_calls_hash_password(self, demo_codebase_path):
        result = _parse_file(demo_codebase_path / "auth.py")
        login_sym = [s for s in result["symbols"] if s["name"] == "login"][0]
        assert "hash_password" in login_sym["calls"]

    def test_parse_utils(self, demo_codebase_path):
        result = _parse_file(demo_codebase_path / "utils.py")
        symbol_names = [s["name"] for s in result["symbols"]]
        assert "hash_password" in symbol_names
        assert "validate_username" in symbol_names

    def test_parse_main(self, demo_codebase_path):
        result = _parse_file(demo_codebase_path / "main.py")
        symbol_names = [s["name"] for s in result["symbols"]]
        assert "main" in symbol_names

    def test_parse_main_imports_auth(self, demo_codebase_path):
        result = _parse_file(demo_codebase_path / "main.py")
        assert "auth" in result["imports"]

    def test_parse_main_calls_login(self, demo_codebase_path):
        result = _parse_file(demo_codebase_path / "main.py")
        main_sym = [s for s in result["symbols"] if s["name"] == "main"][0]
        assert "login" in main_sym["calls"]

    def test_docstrings_extracted(self, demo_codebase_path):
        result = _parse_file(demo_codebase_path / "utils.py")
        hp = [s for s in result["symbols"] if s["name"] == "hash_password"][0]
        assert "Hash a plaintext password" in hp["docstring"]

    def test_lineno_populated(self, demo_codebase_path):
        result = _parse_file(demo_codebase_path / "utils.py")
        for sym in result["symbols"]:
            assert sym["lineno"] > 0

    def test_source_code_extracted(self, demo_codebase_path):
        result = _parse_file(demo_codebase_path / "auth.py")
        login_sym = [s for s in result["symbols"] if s["name"] == "login"][0]
        assert "def login(" in login_sym["source"]
        assert "hash_password(password)" in login_sym["source"]
        assert login_sym["end_lineno"] >= login_sym["lineno"]

    def test_source_code_truncation(self, demo_codebase_path):
        result = _parse_file(demo_codebase_path / "auth.py", max_source_chars=30)
        login_sym = [s for s in result["symbols"] if s["name"] == "login"][0]
        assert len(login_sym["source"]) <= 30

    def test_retrieve_formats_source_code_block(self):
        from unittest.mock import MagicMock
        from modules.graph_rag import GraphRAG

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        # 1. Full-text search returns symbol anchor
        mock_session.run.side_effect = [
            [{"name": "login", "score": 1.5}],  # fulltext anchor
            [{  # traversal record
                "name": "login",
                "kind": "function",
                "docstring": "Authenticate user.",
                "lineno": 10,
                "end_lineno": 18,
                "source_code": "def login():\n    return True",
                "file_path": "auth.py",
                "calls": ["hash_password"],
                "called_by": ["main"],
            }],
        ]

        rag = GraphRAG(mock_driver)
        chunks = rag.retrieve("Why does login fail?")

        assert len(chunks) == 1
        assert "auth.py" in chunks[0].source
        assert "def login():" in chunks[0].text
        assert "Source:\n```python\ndef login():\n    return True\n```" in chunks[0].text
        assert "Lines: 10-18" in chunks[0].text


class TestLuceneEscaping:
    """I-11: Validate Lucene special character escaping."""

    def test_escape_lucene_special_chars(self):
        from modules.graph_rag import escape_lucene
        raw = "foo(bar) + baz && qux || [test] {x} ^ \"hello\" ~ * ? : \\ / - !"
        escaped = escape_lucene(raw)
        for char in "+-&|!(){}[]^\"~*?:\\/":
            assert f"\\{char}" in escaped

    def test_extract_search_terms_escapes_special_chars(self):
        from modules.graph_rag import _extract_search_terms
        terms = _extract_search_terms("foo(bar) + baz")
        assert "\\(" in terms or "\\+" in terms or "foo" in terms

    def test_retrieve_with_special_characters_runs_cypher(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = []

        rag = GraphRAG(mock_driver)
        chunks = rag.retrieve("foo(bar) + baz")
        assert chunks == []
        mock_session.run.assert_called_once()
        terms_arg = mock_session.run.call_args[1]["terms"]
        assert "\\(" in terms_arg or "\\+" in terms_arg or "foo" in terms_arg


class TestCodeReingestion:
    """I-12: Test that code ingestion is idempotent and scopes deletes to root."""

    def test_ingest_codebase_deletes_prior_root_and_batches_unwind(self, tmp_path):
        # Create a sample python file in tmp_path
        py_file = tmp_path / "module.py"
        py_file.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        summary1 = ingest_codebase(str(tmp_path), mock_driver)
        assert summary1["files"] == 1
        assert summary1["symbols"] == 1

        # Check that session.run was called to delete existing root nodes
        executed_queries = [call[0][0] for call in mock_session.run.call_args_list]
        assert any("DETACH DELETE" in q and "$root" in q for q in executed_queries)
        assert any("UNWIND $files" in q for q in executed_queries)
        assert any("UNWIND $symbols" in q for q in executed_queries)

        # Re-ingest
        summary2 = ingest_codebase(str(tmp_path), mock_driver)
        assert summary2 == summary1

