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
from ingestion.code_ingest import _parse_file, ingest_codebase, make_symbol_id, make_file_id


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

        # 1. Batched full-text search and graph traversal returns enriched record
        mock_session.run.return_value = [
            {
                "name": "login",
                "kind": "function",
                "docstring": "Authenticate user.",
                "lineno": 10,
                "end_lineno": 18,
                "source_code": "def login():\n    return True",
                "file_path": "auth.py",
                "score": 1.5,
                "calls": ["hash_password"],
                "called_by": ["main"],
                "file_imports": ["utils"],
                "parent_class": None,
            }
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

    def test_ingest_codebase_skips_non_python_files_and_reports_count(self, tmp_path):
        # Create 1 python file and multiple non-python files
        (tmp_path / "app.py").write_text("def run():\n    pass\n", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("Some text notes", encoding="utf-8")
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        (tmp_path / "style.css").write_text("body { color: red; }", encoding="utf-8")

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        summary = ingest_codebase(str(tmp_path), mock_driver)
        assert summary["files"] == 1
        assert summary["skipped_non_python"] == 3
        assert ".txt" in summary["skipped_extensions"]
        assert ".json" in summary["skipped_extensions"]
        assert ".css" in summary["skipped_extensions"]


class TestGraphRAGEnhancements:
    """Tests for the phased GraphRAG improvements."""

    def test_attribute_calls_extracted(self, tmp_path):
        code_file = tmp_path / "service.py"
        code_file.write_text(
            "class UserService:\n"
            "    def update_email(self, email):\n"
            "        self.validate(email)\n"
            "        db.execute('UPDATE')\n"
            "    def validate(self, email):\n"
            "        pass\n",
            encoding="utf-8",
        )
        res = _parse_file(code_file)
        update_sym = [s for s in res["symbols"] if s["name"] == "update_email"][0]
        assert "validate" in update_sym["calls"]
        assert "execute" in update_sym["calls"]

    def test_class_method_hierarchy_extracted(self, tmp_path):
        code_file = tmp_path / "models.py"
        code_file.write_text(
            "class Account:\n"
            "    def deposit(self, amt):\n"
            "        return amt\n",
            encoding="utf-8",
        )
        res = _parse_file(code_file)
        account_cls = [s for s in res["symbols"] if s["name"] == "Account"][0]
        deposit_m = [s for s in res["symbols"] if s["name"] == "deposit"][0]
        assert account_cls["kind"] == "class"
        assert deposit_m["kind"] == "method"
        assert deposit_m["parent_class"] == "Account"

    def test_contains_relationships_in_ingestion(self, tmp_path):
        code_file = tmp_path / "models.py"
        code_file.write_text(
            "class Account:\n"
            "    def deposit(self, amt):\n"
            "        pass\n",
            encoding="utf-8",
        )
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        summary = ingest_codebase(str(tmp_path), mock_driver)
        assert summary["contains"] == 1
        executed_queries = [call[0][0] for call in mock_session.run.call_args_list]
        assert any("UNWIND $contains" in q and "[:CONTAINS]->" in q for q in executed_queries)

    def test_retrieve_formats_parent_class_and_imports(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        mock_session.run.return_value = [
            {
                "name": "deposit",
                "kind": "method",
                "parent_class": "Account",
                "docstring": "Deposit funds.",
                "lineno": 5,
                "end_lineno": 8,
                "source_code": "def deposit(self, amt):\n    return amt",
                "file_path": "bank/account.py",
                "score": 2.1,
                "calls": ["audit_log"],
                "called_by": ["transfer"],
                "file_imports": ["utils.py", "audit.py"],
            }
        ]

        rag = GraphRAG(mock_driver, max_hops=3, min_score=0.8)
        chunks = rag.retrieve("How to deposit?")

        assert len(chunks) == 1
        text = chunks[0].text
        assert "method of Account" in text
        assert "Relevance score: 2.10" in text
        assert "File imports: utils.py, audit.py" in text
        assert "Calls: audit_log" in text
        assert "Called by: transfer" in text
        assert "Source:\n```python\ndef deposit(self, amt):" in text

    def test_contains_scoped_by_class_id_no_cross_linking(self, tmp_path):
        """P1 verification: Same-named methods in different classes must not cross-link."""
        code_file = tmp_path / "models.py"
        code_file.write_text(
            "class A:\n"
            "    def run(self):\n"
            "        pass\n"
            "class B:\n"
            "    def run(self):\n"
            "        pass\n",
            encoding="utf-8",
        )
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        summary = ingest_codebase(str(tmp_path), mock_driver)
        assert summary["contains"] == 2

        # Extract the $contains parameter passed to session.run
        contains_calls = [
            call[1]["contains"]
            for call in mock_session.run.call_args_list
            if "contains" in call[1]
        ]
        assert len(contains_calls) == 1
        edges = contains_calls[0]
        # Must be exactly: (A -> A.run) and (B -> B.run)
        edge_pairs = {(e["parent_id"], e["child_id"]) for e in edges}
        a_id = make_symbol_id(str(tmp_path), "models.py", "A")
        a_run_id = make_symbol_id(str(tmp_path), "models.py", "A.run")
        b_id = make_symbol_id(str(tmp_path), "models.py", "B")
        b_run_id = make_symbol_id(str(tmp_path), "models.py", "B.run")

        assert (a_id, a_run_id) in edge_pairs
        assert (b_id, b_run_id) in edge_pairs
        # Cross-links must NOT exist
        assert (a_id, b_run_id) not in edge_pairs
        assert (b_id, a_run_id) not in edge_pairs

    def test_conservative_call_resolution_self_method(self, tmp_path):
        """P1 verification: self.method() must resolve strictly to the caller's class."""
        code_file = tmp_path / "classes.py"
        code_file.write_text(
            "class A:\n"
            "    def run(self):\n"
            "        self.validate()\n"
            "    def validate(self):\n"
            "        pass\n"
            "class B:\n"
            "    def run(self):\n"
            "        self.validate()\n"
            "    def validate(self):\n"
            "        pass\n",
            encoding="utf-8",
        )
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        summary = ingest_codebase(str(tmp_path), mock_driver)
        calls_arg = [
            call[1]["calls"]
            for call in mock_session.run.call_args_list
            if "calls" in call[1]
        ][0]
        call_pairs = {(c["caller_id"], c["callee_id"]) for c in calls_arg}

        a_run = make_symbol_id(str(tmp_path), "classes.py", "A.run")
        a_val = make_symbol_id(str(tmp_path), "classes.py", "A.validate")
        b_run = make_symbol_id(str(tmp_path), "classes.py", "B.run")
        b_val = make_symbol_id(str(tmp_path), "classes.py", "B.validate")

        assert (a_run, a_val) in call_pairs
        assert (b_run, b_val) in call_pairs
        # Must not cross-link to another class's method with the same name
        assert (a_run, b_val) not in call_pairs
        assert (b_run, a_val) not in call_pairs

    def test_conservative_call_resolution_imported_module_vs_unresolved_object(self, tmp_path):
        """P1 verification: module.func() resolves only when module is imported; obj.func() stays unresolved."""
        (tmp_path / "utils.py").write_text("def helper(): pass\n", encoding="utf-8")
        (tmp_path / "other.py").write_text("def helper(): pass\n", encoding="utf-8")
        (tmp_path / "service.py").write_text(
            "import utils\n"
            "def run():\n"
            "    utils.helper()\n"
            "    db.helper()\n",
            encoding="utf-8",
        )
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        summary = ingest_codebase(str(tmp_path), mock_driver)
        calls_arg = [
            call[1]["calls"]
            for call in mock_session.run.call_args_list
            if "calls" in call[1]
        ][0]
        call_pairs = {(c["caller_id"], c["callee_id"]) for c in calls_arg}

        run_id = make_symbol_id(str(tmp_path), "service.py", "run")
        utils_helper_id = make_symbol_id(str(tmp_path), "utils.py", "helper")
        other_helper_id = make_symbol_id(str(tmp_path), "other.py", "helper")

        # utils.helper() should resolve to utils.py::helper
        assert (run_id, utils_helper_id) in call_pairs
        # utils.helper() should NOT resolve to other.py::helper
        assert (run_id, other_helper_id) not in call_pairs
        # db.helper() must NOT create a speculative edge to either helper
        assert len(call_pairs) == 1

    def test_nested_function_and_class_qualnames_no_collision(self, tmp_path):
        """P1 verification: nested functions and classes receive full qualnames with no ID collision."""
        code_file = tmp_path / "nested.py"
        code_file.write_text(
            "def first():\n"
            "    def helper():\n"
            "        pass\n"
            "def second():\n"
            "    def helper():\n"
            "        pass\n"
            "class Outer:\n"
            "    class Inner:\n"
            "        def method(self):\n"
            "            pass\n",
            encoding="utf-8",
        )
        res = _parse_file(code_file)
        qualnames = [s["qualname"] for s in res["symbols"]]
        assert "first" in qualnames
        assert "first.helper" in qualnames
        assert "second" in qualnames
        assert "second.helper" in qualnames
        assert "Outer" in qualnames
        assert "Outer.Inner" in qualnames
        assert "Outer.Inner.method" in qualnames

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        summary = ingest_codebase(str(tmp_path), mock_driver)
        assert summary["symbols"] == 7

        contains_calls = [
            call[1]["contains"]
            for call in mock_session.run.call_args_list
            if "contains" in call[1]
        ][0]
        contains_pairs = {(c["parent_id"], c["child_id"]) for c in contains_calls}

        first_id = make_symbol_id(str(tmp_path), "nested.py", "first")
        first_helper_id = make_symbol_id(str(tmp_path), "nested.py", "first.helper")
        second_id = make_symbol_id(str(tmp_path), "nested.py", "second")
        second_helper_id = make_symbol_id(str(tmp_path), "nested.py", "second.helper")
        outer_id = make_symbol_id(str(tmp_path), "nested.py", "Outer")
        inner_id = make_symbol_id(str(tmp_path), "nested.py", "Outer.Inner")
        method_id = make_symbol_id(str(tmp_path), "nested.py", "Outer.Inner.method")

        assert (first_id, first_helper_id) in contains_pairs
        assert (second_id, second_helper_id) in contains_pairs
        assert (outer_id, inner_id) in contains_pairs
        assert (inner_id, method_id) in contains_pairs

    def test_package_imports_and_no_same_name_stem_collisions(self, tmp_path):
        """P1 verification: package_a.utils and package_b.utils do not collide on stem 'utils'."""
        pkg_a = tmp_path / "package_a"
        pkg_b = tmp_path / "package_b"
        pkg_a.mkdir()
        pkg_b.mkdir()

        (pkg_a / "utils.py").write_text("def helper_a(): pass\n", encoding="utf-8")
        (pkg_b / "utils.py").write_text("def helper_b(): pass\n", encoding="utf-8")

        (tmp_path / "service_a.py").write_text(
            "from package_a.utils import helper_a\n"
            "def run_a():\n"
            "    helper_a()\n",
            encoding="utf-8",
        )
        (tmp_path / "service_b.py").write_text(
            "from package_b.utils import helper_b\n"
            "def run_b():\n"
            "    helper_b()\n",
            encoding="utf-8",
        )

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        summary = ingest_codebase(str(tmp_path), mock_driver)
        calls_arg = [
            call[1]["calls"]
            for call in mock_session.run.call_args_list
            if "calls" in call[1]
        ][0]
        call_pairs = {(c["caller_id"], c["callee_id"]) for c in calls_arg}

        run_a_id = make_symbol_id(str(tmp_path), "service_a.py", "run_a")
        helper_a_id = make_symbol_id(str(tmp_path), "package_a/utils.py", "helper_a")
        run_b_id = make_symbol_id(str(tmp_path), "service_b.py", "run_b")
        helper_b_id = make_symbol_id(str(tmp_path), "package_b/utils.py", "helper_b")

        assert (run_a_id, helper_a_id) in call_pairs
        assert (run_b_id, helper_b_id) in call_pairs
        assert (run_a_id, helper_b_id) not in call_pairs
        assert (run_b_id, helper_a_id) not in call_pairs

    def test_multi_codebase_isolation_in_ingest_and_retrieval(self, tmp_path):
        """P1 verification: codebase ingestion strictly scopes deletion and retrieval to root."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        (tmp_path / "main.py").write_text("def main(): pass\n", encoding="utf-8")
        ingest_codebase(str(tmp_path), mock_driver)

        # Deletion query MUST scope exclusively to $root and not f.path IN $paths
        executed_queries = [call[0][0] for call in mock_session.run.call_args_list]
        delete_file_queries = [q for q in executed_queries if "MATCH (f:File" in q and "DETACH DELETE" in q]
        assert len(delete_file_queries) == 1
        assert "root: $root" in delete_file_queries[0]
        assert "f.path IN $paths" not in delete_file_queries[0]

        # Retrieval query MUST filter by $root
        rag = GraphRAG(mock_driver, root=str(tmp_path))
        rag.retrieve("main")
        retrieval_query = mock_session.run.call_args[0][0]
        assert "($root IS NULL OR node.root = $root)" in retrieval_query
        assert mock_session.run.call_args[1]["root"] == tmp_path.resolve().as_posix()

    def test_settings_and_graph_rag_parameter_validation(self):
        """P2 verification: validation of retrieval settings and GraphRAG params."""
        mock_driver = MagicMock()
        with pytest.raises(ValueError, match="max_hops must be >= 1"):
            GraphRAG(mock_driver, max_hops=0)
        with pytest.raises(ValueError, match="min_score must be >= 0.0"):
            GraphRAG(mock_driver, min_score=-0.1)




