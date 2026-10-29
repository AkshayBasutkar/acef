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


def _parse_file(file_path: Path, max_source_chars: int = 1200) -> dict:
    """
    Parse a single Python file and extract symbols and their relationships.

    Returns a dict with:
      - "symbols": list of {name, kind, docstring, lineno, end_lineno, source, calls}
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

            # I-03: Extract source code segment for LLM context
            source_segment = ""
            try:
                raw_segment = ast.get_source_segment(source, node)
                if raw_segment:
                    source_segment = raw_segment[:max_source_chars]
            except Exception as e:
                logger.debug("Could not extract source segment for %s: %s", node.name, e)

            end_lineno = getattr(node, "end_lineno", node.lineno)

            symbol = {
                "name": node.name,
                "kind": kind,
                "docstring": _get_docstring(node),
                "lineno": node.lineno,
                "end_lineno": end_lineno,
                "source": source_segment,
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
    root_str = str(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Codebase path not found: {root}")

    # Ensure indexes exist before ingesting
    init_neo4j(neo4j_driver)

    # ── Phase 1: Discover all files and filter Python vs non-Python ──────
    ignore_dirs = {
        ".git", ".venv", "venv", "env", "node_modules",
        "__pycache__", ".pytest_cache", ".idea", ".vscode"
    }
    all_files = [
        f for f in sorted(root.rglob("*"))
        if f.is_file() and not any(part in ignore_dirs for part in f.parts)
    ]
    py_files = [
        f for f in all_files
        if f.suffix == ".py" and f.stat().st_size <= 1_048_576
    ]
    non_py_files = [f for f in all_files if f.suffix != ".py"]

    skipped_extensions = {f.suffix.lower() for f in non_py_files if f.suffix}
    if any(not f.suffix for f in non_py_files):
        skipped_extensions.add("(no extension)")

    if non_py_files:
        logger.info(
            "Found %d non-Python files in %s; skipping them. Extensions: %s",
            len(non_py_files),
            root_str,
            ", ".join(sorted(skipped_extensions)),
        )

    if not py_files:
        logger.warning(
            "No .py files found in %s (%d non-Python files skipped)",
            root, len(non_py_files)
        )
        return {
            "files": 0,
            "files_scanned": 0,
            "symbols": 0,
            "symbols_created": 0,
            "calls": 0,
            "call_relationships": 0,
            "imports": 0,
            "import_relationships": 0,
            "skipped_non_python": len(non_py_files),
            "skipped_extensions": sorted(list(skipped_extensions)),
        }

    # Map file stems to file paths (for resolving intra-project imports)
    stem_to_path: dict[str, str] = {}
    for f in py_files:
        rel_path = str(f.relative_to(root))
        stem_to_path[f.stem] = rel_path

    # Parse all files
    from config.settings import get_settings
    max_source_chars = get_settings().graph_source_max_chars

    file_data: dict[str, dict] = {}  # rel_path -> parse result
    for f in py_files:
        rel_path = str(f.relative_to(root))
        file_data[rel_path] = _parse_file(f, max_source_chars=max_source_chars)
        logger.debug("Parsed %s — %d symbols", rel_path, len(file_data[rel_path]["symbols"]))

    # Collect all symbol names for call resolution
    all_symbol_names: set[str] = set()
    for data in file_data.values():
        for sym in data["symbols"]:
            all_symbol_names.add(sym["name"])

    root_str = str(root)

    # ── Phase 2: Write everything to Neo4j ───────────────────────────────
    with neo4j_driver.session() as session:
        # I-12: Delete all File and Symbol nodes associated with this root before re-ingestion
        all_paths = list(file_data.keys())
        session.run(
            """
            MATCH (f:File)
            WHERE f.root = $root OR f.path IN $paths
            OPTIONAL MATCH (f)-[:DEFINES]->(s:Symbol)
            DETACH DELETE s, f
            """,
            root=root_str,
            paths=all_paths,
        )
        session.run(
            """
            MATCH (s:Symbol {root: $root})
            DETACH DELETE s
            """,
            root=root_str,
        )
        logger.info("Cleared existing Neo4j data for root %s.", root_str)

        # Batch create File nodes via UNWIND
        file_payload = [
            {"path": rel_path, "name": Path(rel_path).name}
            for rel_path in file_data
        ]
        session.run(
            """
            UNWIND $files AS f_data
            MERGE (f:File {path: f_data.path})
            SET f.name = f_data.name, f.root = $root
            """,
            files=file_payload,
            root=root_str,
        )

        # Batch create Symbol nodes and DEFINES edges via UNWIND
        symbol_payload = []
        for rel_path, data in file_data.items():
            for sym in data["symbols"]:
                symbol_payload.append({
                    "file_path": rel_path,
                    "name": sym["name"],
                    "kind": sym["kind"],
                    "docstring": sym["docstring"],
                    "lineno": sym["lineno"],
                    "end_lineno": sym["end_lineno"],
                    "source": sym["source"],
                })

        if symbol_payload:
            session.run(
                """
                UNWIND $symbols AS sym
                MATCH (f:File {path: sym.file_path, root: $root})
                CREATE (s:Symbol {
                    name: sym.name,
                    kind: sym.kind,
                    docstring: sym.docstring,
                    file_path: sym.file_path,
                    lineno: sym.lineno,
                    end_lineno: sym.end_lineno,
                    source: sym.source,
                    root: $root
                })
                CREATE (f)-[:DEFINES]->(s)
                """,
                symbols=symbol_payload,
                root=root_str,
            )

        # Batch create IMPORTS edges via UNWIND
        import_payload = []
        for rel_path, data in file_data.items():
            for imported_module in data["imports"]:
                module_base = imported_module.split(".")[-1]
                if module_base in stem_to_path:
                    target_path = stem_to_path[module_base]
                    if target_path != rel_path:
                        import_payload.append({
                            "source_path": rel_path,
                            "target_path": target_path,
                        })

        if import_payload:
            session.run(
                """
                UNWIND $imports AS imp
                MATCH (source:File {path: imp.source_path, root: $root})
                MATCH (target:File {path: imp.target_path, root: $root})
                MERGE (source)-[:IMPORTS]->(target)
                """,
                imports=import_payload,
                root=root_str,
            )

        # Batch create CALLS edges via UNWIND
        call_payload = []
        for rel_path, data in file_data.items():
            for sym in data["symbols"]:
                for called_name in sym["calls"]:
                    if called_name in all_symbol_names and called_name != sym["name"]:
                        call_payload.append({
                            "caller_name": sym["name"],
                            "file_path": rel_path,
                            "callee_name": called_name,
                        })

        if call_payload:
            session.run(
                """
                UNWIND $calls AS call
                MATCH (caller:Symbol {name: call.caller_name, file_path: call.file_path, root: $root})
                MATCH (callee:Symbol {name: call.callee_name, root: $root})
                MERGE (caller)-[:CALLS]->(callee)
                """,
                calls=call_payload,
                root=root_str,
            )

        counters = {
            "files": len(file_payload),
            "files_scanned": len(file_payload),
            "symbols": len(symbol_payload),
            "symbols_created": len(symbol_payload),
            "calls": len(call_payload),
            "call_relationships": len(call_payload),
            "imports": len(import_payload),
            "import_relationships": len(import_payload),
            "skipped_non_python": len(non_py_files),
            "skipped_extensions": sorted(list(skipped_extensions)),
        }

    logger.info(
        "Ingestion complete — files=%d, symbols=%d, calls=%d, imports=%d, skipped_non_python=%d",
        counters["files"], counters["symbols"], counters["calls"], counters["imports"], counters["skipped_non_python"],
    )
    return counters
