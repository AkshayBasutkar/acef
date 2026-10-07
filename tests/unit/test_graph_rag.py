"""
Unit tests for GraphRAG search term extraction and code ingestion parsing.

Tests the heuristic token extraction (which doesn't require Neo4j)
and the AST-based file parser (which doesn't require Neo4j either).
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from modules.graph_rag import _extract_search_terms
from ingestion.code_ingest import _parse_file
from pathlib import Path


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
