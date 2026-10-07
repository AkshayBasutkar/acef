"""
Code Ingestion — parse a Python codebase and build a Neo4j graph.

Walks a directory of .py files, parses each with the `ast` module, and
writes nodes and edges to Neo4j:

  - (:File {path, name})  nodes for each .py file
  - (:Symbol {name, kind, docstring, file_path, lineno})  nodes for each
    function, async function, or class definition
  - (:File)-[:DEFINES]->(:Symbol)  edges linking files to their symbols
  - (:File)-[:IMPORTS]->(:File)  edges for intra-project imports
  - (:Symbol)-[:CALLS]->(:Symbol)  edges for function calls (best-effort,
    same-run resolution only)

Python-only for this build (see §11 of the build guide).

Usage:
    from ingestion.code_ingest import ingest_codebase
    summary = ingest_codebase("./data/demo_codebase", neo4j_driver)
"""

import ast
import logging
import os
from pathlib import Path

from scripts.init_neo4j import init_neo4j

logger = logging.getLogger(__name__)


def _get_docstring(node: ast.AST) -> str:
    """Extract the docstring from an AST node, or return empty string."""
    try:
        return ast.get_docstring(node) or ""
    except TypeError:
        return ""


def _collect_calls(node: ast.AST) -> list[str]:
    """
    Walk the body of a function/class and collect all direct call names.
    Only captures simple Name calls (e.g. `foo()`) — not attribute calls
    (e.g. `self.foo()` or `module.bar()`).
    """
    calls = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            # Simple function call: foo()
            if isinstance(child.func, ast.Name):
                calls.append(child.func.id)
    return calls


def _parse_file(file_path: Path) -> dict:
    """
    Parse a single Python file and extract symbols and their relationships.

    Returns a dict with:
      - "symbols": list of {name, kind, docstring, lineno, calls}
      - "imports": list of module names imported via `import X` or `from X import ...`
    """
    source = file_path.read_text(encoding="utf-8", errors="replace")

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        logger.warning("Skipping %s — syntax error: %s", file_path, e)
        return {"symbols": [], "imports": []}

    symbols = []
    imports = []

    for node in ast.walk(tree):
        # ── Symbols: functions, async functions, classes ─────────────
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            symbol = {
                "name": node.name,
                "kind": kind,
                "docstring": _get_docstring(node),
                "lineno": node.lineno,
                "calls": _collect_calls(node),
            }
            symbols.append(symbol)

        # ── Imports ──────────────────────────────────────────────────
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    return {"symbols": symbols, "imports": imports}


def ingest_codebase(root_path: str, neo4j_driver) -> dict:
    """
    Walk a directory of .py files, parse each, and build a Neo4j code graph.

    Parameters
    ----------
    root_path : str
        Path to the root directory of the Python codebase to ingest.
    neo4j_driver : neo4j.Driver
        An active Neo4j driver instance.

    Returns
    -------
    dict
        Summary: {"files": N, "symbols": N, "calls": N, "imports": N}
    """
    root = Path(root_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Codebase path not found: {root}")

    # Ensure indexes exist before ingesting
    init_neo4j(neo4j_driver)

    # ── Phase 1: Discover and parse all .py files ────────────────────────
    py_files = sorted(root.rglob("*.py"))
    if not py_files:
        logger.warning("No .py files found in %s", root)
        return {"files": 0, "symbols": 0, "calls": 0, "imports": 0}

    # Map file stems to file paths (for resolving intra-project imports)
    stem_to_path: dict[str, str] = {}
    for f in py_files:
        rel_path = str(f.relative_to(root))
        stem_to_path[f.stem] = rel_path

    # Parse all files
    file_data: dict[str, dict] = {}  # rel_path -> parse result
    for f in py_files:
        rel_path = str(f.relative_to(root))
        file_data[rel_path] = _parse_file(f)
        logger.debug("Parsed %s — %d symbols", rel_path, len(file_data[rel_path]["symbols"]))

    # Collect all symbol names for call resolution
    all_symbol_names: set[str] = set()
    for data in file_data.values():
        for sym in data["symbols"]:
            all_symbol_names.add(sym["name"])

    # ── Phase 2: Write everything to Neo4j ───────────────────────────────
    counters = {"files": 0, "symbols": 0, "calls": 0, "imports": 0}

    with neo4j_driver.session() as session:
        # Clear existing data for a clean re-ingestion
        session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared existing Neo4j data for fresh ingestion.")

        # Create File nodes
        for rel_path in file_data:
            file_name = Path(rel_path).name
            session.run(
                "MERGE (f:File {path: $path}) SET f.name = $name",
                path=rel_path, name=file_name,
            )
            counters["files"] += 1

        # Create Symbol nodes and DEFINES edges
        for rel_path, data in file_data.items():
            for sym in data["symbols"]:
                session.run(
                    """
                    MATCH (f:File {path: $file_path})
                    CREATE (s:Symbol {
                        name: $name,
                        kind: $kind,
                        docstring: $docstring,
                        file_path: $file_path,
                        lineno: $lineno
                    })
                    CREATE (f)-[:DEFINES]->(s)
                    """,
                    file_path=rel_path,
                    name=sym["name"],
                    kind=sym["kind"],
                    docstring=sym["docstring"],
                    lineno=sym["lineno"],
                )
                counters["symbols"] += 1

        # Create IMPORTS edges (intra-project only)
        for rel_path, data in file_data.items():
            for imported_module in data["imports"]:
                # Check if the imported module resolves to a file in this project
                # Match on file stem (e.g., "auth" matches "auth.py")
                module_base = imported_module.split(".")[-1]
                if module_base in stem_to_path:
                    target_path = stem_to_path[module_base]
                    if target_path != rel_path:  # don't self-import
                        session.run(
                            """
                            MATCH (source:File {path: $source_path})
                            MATCH (target:File {path: $target_path})
                            MERGE (source)-[:IMPORTS]->(target)
                            """,
                            source_path=rel_path,
                            target_path=target_path,
                        )
                        counters["imports"] += 1

        # Create CALLS edges (best-effort, same-run resolution)
        for rel_path, data in file_data.items():
            for sym in data["symbols"]:
                for called_name in sym["calls"]:
                    # Only create edge if the callee is a known symbol
                    # from this same ingestion run
                    if called_name in all_symbol_names and called_name != sym["name"]:
                        session.run(
                            """
                            MATCH (caller:Symbol {name: $caller_name, file_path: $file_path})
                            MATCH (callee:Symbol {name: $callee_name})
                            MERGE (caller)-[:CALLS]->(callee)
                            """,
                            caller_name=sym["name"],
                            file_path=rel_path,
                            callee_name=called_name,
                        )
                        counters["calls"] += 1

    logger.info(
        "Ingestion complete — files=%d, symbols=%d, calls=%d, imports=%d",
        counters["files"], counters["symbols"], counters["calls"], counters["imports"],
    )
    return counters
