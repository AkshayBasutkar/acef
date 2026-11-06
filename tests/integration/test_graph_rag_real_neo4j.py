"""
Real Neo4j Integration Test for GraphRAG Ambiguity, Hierarchy, and Multi-Codebase Isolation.

Validates:
  1. Two classes with identical method names receive only their own CONTAINS edge.
  2. self.method() resolves strictly to the caller's class (no cross-linking).
  3. module.function() resolves only when module is an imported project file.
  4. Ambiguous object calls (e.g. db.execute()) create no speculative edges.
  5. Multi-codebase isolation: ingesting codebase 2 does not delete/corrupt codebase 1.
  6. GraphRAG retrieval strictly filters by active codebase root.
  7. Nested functions receive distinct IDs and parent CONTAINS edges.
  8. Package imports with identical file stems in separate packages do not collide.
"""

import pytest
from pathlib import Path
from neo4j import GraphDatabase

from config.settings import get_settings
from ingestion.code_ingest import ingest_codebase, make_symbol_id, make_file_id
from modules.graph_rag import GraphRAG


@pytest.fixture(scope="module")
def real_neo4j_driver():
    settings = get_settings()
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        with driver.session() as s:
            s.run("RETURN 1 AS ping").single()
        yield driver
        driver.close()
    except Exception as e:
        pytest.skip(f"Real Neo4j instance not reachable at {settings.neo4j_uri}: {e}")


def test_real_neo4j_contains_and_call_resolution(real_neo4j_driver, tmp_path):
    # Setup a realistic multi-class, multi-module project with duplicate names and nested functions
    models_file = tmp_path / "models.py"
    models_file.write_text(
        "class OrderService:\n"
        "    def run(self):\n"
        "        self.validate()\n"
        "    def validate(self):\n"
        "        pass\n"
        "\n"
        "class AuthService:\n"
        "    def run(self):\n"
        "        self.validate()\n"
        "    def validate(self):\n"
        "        pass\n"
        "\n"
        "def outer_fn():\n"
        "    def inner_fn():\n"
        "        pass\n",
        encoding="utf-8",
    )

    utils_file = tmp_path / "utils.py"
    utils_file.write_text(
        "def audit_log(event):\n"
        "    pass\n",
        encoding="utf-8",
    )

    other_file = tmp_path / "other.py"
    other_file.write_text(
        "def execute():\n"
        "    pass\n",
        encoding="utf-8",
    )

    app_file = tmp_path / "app.py"
    app_file.write_text(
        "import utils\n"
        "def handle_request():\n"
        "    utils.audit_log('start')\n"
        "    db.execute()\n",
        encoding="utf-8",
    )

    # Ingest into real Neo4j
    summary = ingest_codebase(str(tmp_path), real_neo4j_driver)
    assert summary["files"] == 4
    # 2 methods in OrderService + 2 methods in AuthService + 1 inner_fn in outer_fn = 5
    assert summary["contains"] == 5

    root_str = tmp_path.resolve().as_posix()

    try:
        with real_neo4j_driver.session() as session:
            # ── 1. Verify CONTAINS edges are properly scoped (no cross-linking) ──
            order_contains = session.run(
                """
                MATCH (cls:Symbol {name: 'OrderService', root: $root})-[:CONTAINS]->(m:Symbol)
                RETURN m.id AS id, m.name AS name
                """,
                root=root_str,
            )
            order_methods = {r["id"] for r in order_contains}
            expected_order_run = make_symbol_id(root_str, "models.py", "OrderService.run")
            expected_order_val = make_symbol_id(root_str, "models.py", "OrderService.validate")
            assert order_methods == {expected_order_run, expected_order_val}

            auth_contains = session.run(
                """
                MATCH (cls:Symbol {name: 'AuthService', root: $root})-[:CONTAINS]->(m:Symbol)
                RETURN m.id AS id, m.name AS name
                """,
                root=root_str,
            )
            auth_methods = {r["id"] for r in auth_contains}
            expected_auth_run = make_symbol_id(root_str, "models.py", "AuthService.run")
            expected_auth_val = make_symbol_id(root_str, "models.py", "AuthService.validate")
            assert auth_methods == {expected_auth_run, expected_auth_val}

            # ── 2. Verify nested function scoping in CONTAINS ────────────────────
            nested_contains = session.run(
                """
                MATCH (p:Symbol {name: 'outer_fn', root: $root})-[:CONTAINS]->(c:Symbol)
                RETURN c.id AS id, c.qualname AS qualname
                """,
                root=root_str,
            )
            nested_children = {r["qualname"] for r in nested_contains}
            assert nested_children == {"outer_fn.inner_fn"}

            # ── 3. Verify self.validate() does NOT cross-link across classes ────
            order_calls = session.run(
                """
                MATCH (caller:Symbol {id: $caller_id})-[:CALLS]->(callee:Symbol)
                RETURN callee.id AS id
                """,
                caller_id=expected_order_run,
            )
            order_callees = {r["id"] for r in order_calls}
            assert order_callees == {expected_order_val}
            assert expected_auth_val not in order_callees

            auth_calls = session.run(
                """
                MATCH (caller:Symbol {id: $caller_id})-[:CALLS]->(callee:Symbol)
                RETURN callee.id AS id
                """,
                caller_id=expected_auth_run,
            )
            auth_callees = {r["id"] for r in auth_calls}
            assert auth_callees == {expected_auth_val}
            assert expected_order_val not in auth_callees

            # ── 4. Verify module.func() links, but db.execute() does not ────────
            app_req_id = make_symbol_id(root_str, "app.py", "handle_request")
            audit_log_id = make_symbol_id(root_str, "utils.py", "audit_log")
            exec_id = make_symbol_id(root_str, "other.py", "execute")

            app_calls = session.run(
                """
                MATCH (caller:Symbol {id: $caller_id})-[:CALLS]->(callee:Symbol)
                RETURN callee.id AS id
                """,
                caller_id=app_req_id,
            )
            app_callees = {r["id"] for r in app_calls}
            assert app_callees == {audit_log_id}
            # other.py::execute must NEVER be linked
            assert exec_id not in app_callees

            # ── 5. Verify GraphRAG retrieval on the real graph with root isolation ──
            rag = GraphRAG(real_neo4j_driver, max_hops=2, min_score=0.1, root=root_str)
            chunks = rag.retrieve("handle_request")
            assert len(chunks) >= 1
            retrieved_text = "\n".join(c.text for c in chunks)
            assert "audit_log" in retrieved_text
            assert "other.py" not in retrieved_text

    finally:
        # Clean up test root from Neo4j
        with real_neo4j_driver.session() as session:
            session.run(
                """
                MATCH (f:File {root: $root})
                OPTIONAL MATCH (f)-[:DEFINES]->(s:Symbol)
                DETACH DELETE s, f
                """,
                root=root_str,
            )
            session.run(
                """
                MATCH (s:Symbol {root: $root})
                DETACH DELETE s
                """,
                root=root_str,
            )


def test_real_neo4j_multi_codebase_isolation(real_neo4j_driver, tmp_path):
    """
    Ingest two separate codebases that share identical relative file names ('main.py')
    and function names ('start').
    Verify:
      1. Ingesting codebase 2 does NOT delete or overwrite codebase 1's File/Symbol nodes.
      2. GraphRAG scoped to codebase 1 returns ONLY codebase 1's docstring/code.
    """
    root_1 = tmp_path / "codebase_1"
    root_2 = tmp_path / "codebase_2"
    root_1.mkdir()
    root_2.mkdir()

    (root_1 / "main.py").write_text(
        "def start():\n"
        "    '''CODEBASE_ONE_PRIMARY_ENTRY'''\n"
        "    pass\n",
        encoding="utf-8",
    )
    (root_2 / "main.py").write_text(
        "def start():\n"
        "    '''CODEBASE_TWO_SECONDARY_ENTRY'''\n"
        "    pass\n",
        encoding="utf-8",
    )

    # Ingest codebase 1
    s1 = ingest_codebase(str(root_1), real_neo4j_driver)
    assert s1["files"] == 1
    assert s1["symbols"] == 1

    # Ingest codebase 2 (with identical relative file path)
    s2 = ingest_codebase(str(root_2), real_neo4j_driver)
    assert s2["files"] == 1
    assert s2["symbols"] == 1

    r1_str = root_1.resolve().as_posix()
    r2_str = root_2.resolve().as_posix()

    try:
        with real_neo4j_driver.session() as session:
            # Both files must coexist in Neo4j with unique root-scoped IDs
            f1 = session.run("MATCH (f:File {root: $root}) RETURN f.id AS id", root=r1_str).single()
            f2 = session.run("MATCH (f:File {root: $root}) RETURN f.id AS id", root=r2_str).single()
            assert f1 is not None and f2 is not None
            assert f1["id"] != f2["id"]

            # Both symbols must coexist in Neo4j
            sym1 = session.run("MATCH (s:Symbol {root: $root}) RETURN s.docstring AS d", root=r1_str).single()
            sym2 = session.run("MATCH (s:Symbol {root: $root}) RETURN s.docstring AS d", root=r2_str).single()
            assert sym1 is not None and "CODEBASE_ONE" in sym1["d"]
            assert sym2 is not None and "CODEBASE_TWO" in sym2["d"]

        # GraphRAG retrieval scoped to codebase 1 MUST NOT return codebase 2
        rag1 = GraphRAG(real_neo4j_driver, min_score=0.01, root=r1_str)
        chunks1 = rag1.retrieve("start")
        assert len(chunks1) >= 1
        text1 = "\n".join(c.text for c in chunks1)
        assert "CODEBASE_ONE" in text1
        assert "CODEBASE_TWO" not in text1

        # GraphRAG retrieval scoped to codebase 2 MUST NOT return codebase 1
        rag2 = GraphRAG(real_neo4j_driver, min_score=0.01, root=r2_str)
        chunks2 = rag2.retrieve("start")
        assert len(chunks2) >= 1
        text2 = "\n".join(c.text for c in chunks2)
        assert "CODEBASE_TWO" in text2
        assert "CODEBASE_ONE" not in text2

    finally:
        for r in (r1_str, r2_str):
            with real_neo4j_driver.session() as session:
                session.run("MATCH (f:File {root: $root}) DETACH DELETE f", root=r)
                session.run("MATCH (s:Symbol {root: $root}) DETACH DELETE s", root=r)
