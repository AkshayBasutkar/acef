"""
Code Ingestion — parse a Python codebase and build a Neo4j graph.

Walks a directory of .py files, parses each with the `ast` module, and
writes nodes and edges to Neo4j with complete root-isolation, canonical
package-aware module resolution, and hierarchical lexical scoping:

  - (:File {id, path, name, root})  nodes for each .py file
  - (:Symbol {id, name, qualname, kind, docstring, file_path, lineno, end_lineno, source, root})
  - (:File)-[:DEFINES]->(:Symbol)  edges linking files to their symbols
  - (:File)-[:IMPORTS]->(:File)  edges for intra-project imports
  - (:Symbol)-[:CONTAINS]->(:Symbol)  edges linking classes to methods/nested classes and functions to nested functions
  - (:Symbol)-[:CALLS]->(:Symbol)  conservative, evidence-based function and method calls
"""

import ast
import logging
from pathlib import Path

from scripts.init_neo4j import init_neo4j

logger = logging.getLogger(__name__)


def make_file_id(root: str, rel_path: str) -> str:
    """Generate a globally unique, root-qualified identifier for a File node."""
    norm_root = Path(root).resolve().as_posix()
    norm_path = rel_path.replace("\\", "/")
    return f"{norm_root}::{norm_path}"


def make_symbol_id(root: str, rel_path: str, qualname: str) -> str:
    """Generate a globally unique, root-qualified identifier for a Symbol node."""
    norm_root = Path(root).resolve().as_posix()
    norm_path = rel_path.replace("\\", "/")
    return f"{norm_root}::{norm_path}::{qualname}"


def _get_docstring(node: ast.AST) -> str:
    """Extract docstring or return empty string."""
    try:
        return ast.get_docstring(node) or ""
    except TypeError:
        return ""


def _collect_calls(node: ast.AST) -> tuple[list[str], list[dict]]:
    """
    Extract calls with syntactic provenance:
      - 'bare': foo()
      - 'self': self.foo()
      - 'attr': mod.foo() where value is Name
      - 'other': obj.method()
    """
    calls = []
    call_sites = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                calls.append(child.func.id)
                call_sites.append({"type": "bare", "name": child.func.id})
            elif isinstance(child.func, ast.Attribute):
                calls.append(child.func.attr)
                if isinstance(child.func.value, ast.Name):
                    if child.func.value.id == "self":
                        call_sites.append({"type": "self", "name": child.func.attr})
                    else:
                        call_sites.append({
                            "type": "attr",
                            "prefix": child.func.value.id,
                            "name": child.func.attr,
                        })
                else:
                    call_sites.append({"type": "other", "name": child.func.attr})
    return calls, call_sites


def _canonical_module_name(rel_path: str) -> str:
    """Convert a relative file path (e.g. 'pkg/utils.py') to its canonical Python module name ('pkg.utils')."""
    p = Path(rel_path)
    parts = list(p.parts)
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_relative_import(current_file_rel: str, level: int, base_module: str | None) -> str | None:
    """Resolve a relative import (level > 0) to a canonical dotted module name."""
    p = Path(current_file_rel)
    pkg_parts = list(p.parent.parts)
    pop_count = level - 1
    if pop_count > len(pkg_parts):
        return None
    if pop_count > 0:
        pkg_parts = pkg_parts[:-pop_count]

    result_parts = list(pkg_parts)
    if base_module:
        result_parts.extend(base_module.split("."))
    return ".".join(result_parts) if result_parts else None


class HierarchicalSymbolVisitor(ast.NodeVisitor):
    """
    AST Visitor that builds full lexical scopes (qualnames) for classes,
    methods, and nested functions.
    """
    def __init__(self, source: str, max_source_chars: int):
        self.source = source
        self.max_source_chars = max_source_chars
        self.scope_stack: list[tuple[str, str]] = []  # (name, "class" | "function")
        self.symbols: list[dict] = []

    def _qualname(self) -> str:
        return ".".join(name for name, _ in self.scope_stack)

    def _parent_qualname(self) -> str | None:
        if len(self.scope_stack) > 1:
            return ".".join(name for name, _ in self.scope_stack[:-1])
        return None

    def _parent_class(self) -> str | None:
        classes = [name for name, kind in self.scope_stack[:-1] if kind == "class"]
        return ".".join(classes) if classes else None

    def _process_node(self, node: ast.AST, is_class: bool):
        qualname = self._qualname()
        parent_qualname = self._parent_qualname()
        parent_class = self._parent_class()

        if is_class:
            kind = "class"
        elif len(self.scope_stack) > 1 and self.scope_stack[-2][1] == "class":
            kind = "method"
        elif len(self.scope_stack) > 1 and self.scope_stack[-2][1] == "function":
            kind = "nested_function"
        else:
            kind = "function"

        # Line-aware source truncation
        source_segment = ""
        try:
            raw_segment = ast.get_source_segment(self.source, node)
            if raw_segment:
                if len(raw_segment) > self.max_source_chars:
                    suffix = "\n# ... truncated"
                    if self.max_source_chars > len(suffix):
                        budget = self.max_source_chars - len(suffix)
                        truncated = raw_segment[:budget]
                        last_newline = truncated.rfind('\n')
                        if last_newline > 0:
                            source_segment = truncated[:last_newline] + suffix
                        else:
                            source_segment = truncated + suffix
                    else:
                        source_segment = raw_segment[:self.max_source_chars]
                else:
                    source_segment = raw_segment
        except Exception:
            pass

        end_lineno = getattr(node, "end_lineno", node.lineno)
        calls_list, call_sites_list = _collect_calls(node)

        self.symbols.append({
            "name": node.name,
            "qualname": qualname,
            "parent_qualname": parent_qualname,
            "parent_class": parent_class,
            "kind": kind,
            "docstring": _get_docstring(node),
            "lineno": node.lineno,
            "end_lineno": end_lineno,
            "source": source_segment,
            "calls": calls_list,
            "call_sites": call_sites_list,
        })

    def visit_ClassDef(self, node: ast.ClassDef):
        self.scope_stack.append((node.name, "class"))
        self._process_node(node, is_class=True)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.scope_stack.append((node.name, "function"))
        self._process_node(node, is_class=False)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.scope_stack.append((node.name, "function"))
        self._process_node(node, is_class=False)
        self.generic_visit(node)
        self.scope_stack.pop()


def _parse_file(file_path: Path, max_source_chars: int = 1200, rel_path: str = "") -> dict:
    source = file_path.read_text(encoding="utf-8", errors="replace")

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        logger.warning("Skipping %s — syntax error: %s", file_path, e)
        return {
            "symbols": [],
            "imports": [],
            "imported_modules": {},
            "imported_symbols": {},
        }

    visitor = HierarchicalSymbolVisitor(source, max_source_chars)
    visitor.visit(tree)

    imports = []
    imported_modules: dict[str, str] = {}
    imported_symbols: dict[str, tuple[str, str]] = {}

    rel_str = rel_path or file_path.name

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
                local_name = alias.asname or alias.name
                imported_modules[local_name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            level = getattr(node, "level", 0)
            if level > 0:
                resolved_mod = _resolve_relative_import(rel_str, level, node.module)
            else:
                resolved_mod = node.module

            if resolved_mod:
                imports.append(resolved_mod)
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    imported_modules[local_name] = f"{resolved_mod}.{alias.name}"
                    imported_symbols[local_name] = (resolved_mod, alias.name)

    return {
        "symbols": visitor.symbols,
        "imports": imports,
        "imported_modules": imported_modules,
        "imported_symbols": imported_symbols,
    }


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
        Summary: {"files": N, "symbols": N, "calls": N, "imports": N, "contains": N}
    """
    root = Path(root_path).resolve()
    root_str = root.as_posix()
    if not root.is_dir():
        raise FileNotFoundError(f"Codebase path not found: {root}")

    # Ensure indexes and constraints exist before ingesting
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
            "contains": 0,
            "contains_relationships": 0,
            "skipped_non_python": len(non_py_files),
            "skipped_extensions": sorted(list(skipped_extensions)),
        }

    # Map canonical dotted module names to relative file paths
    canonical_module_to_file: dict[str, str] = {}
    file_to_canonical_module: dict[str, str] = {}
    stem_to_file: dict[str, str | None] = {}

    for f in py_files:
        rel_path = str(f.relative_to(root)).replace("\\", "/")
        can_name = _canonical_module_name(rel_path)
        canonical_module_to_file[can_name] = rel_path
        file_to_canonical_module[rel_path] = can_name

        # Also support unique stem lookup if not colliding
        stem = f.stem
        if stem in stem_to_file:
            stem_to_file[stem] = None  # ambiguous!
        else:
            stem_to_file[stem] = rel_path

    # Parse all files
    from config.settings import get_settings
    max_source_chars = get_settings().graph_source_max_chars

    file_data: dict[str, dict] = {}
    for f in py_files:
        rel_path = str(f.relative_to(root)).replace("\\", "/")
        file_data[rel_path] = _parse_file(f, max_source_chars=max_source_chars, rel_path=rel_path)
        logger.debug("Parsed %s — %d symbols", rel_path, len(file_data[rel_path]["symbols"]))

    # Assign deterministic IDs and build resolution indices
    all_symbols_by_id: dict[str, dict] = {}
    file_top_level_symbols: dict[str, set[str]] = {}

    for rel_path, data in file_data.items():
        file_top_level_symbols[rel_path] = set()
        for sym in data["symbols"]:
            sym_id = make_symbol_id(root_str, rel_path, sym["qualname"])
            sym["id"] = sym_id
            sym["file_id"] = make_file_id(root_str, rel_path)
            sym["file_path"] = rel_path
            all_symbols_by_id[sym_id] = sym

            if sym.get("parent_qualname") is None:
                file_top_level_symbols[rel_path].add(sym["name"])

    # ── Phase 2: Write everything to Neo4j ───────────────────────────────
    with neo4j_driver.session() as session:
        # Multi-codebase isolation: strictly delete nodes belonging to THIS root only
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
        logger.info("Cleared existing Neo4j data strictly for root %s.", root_str)

        # Batch create File nodes via UNWIND with root-scoped id
        file_payload = [
            {
                "id": make_file_id(root_str, rel_path),
                "path": rel_path,
                "name": Path(rel_path).name,
            }
            for rel_path in file_data
        ]
        session.run(
            """
            UNWIND $files AS f_data
            MERGE (f:File {id: f_data.id})
            SET f.path = f_data.path, f.name = f_data.name, f.root = $root
            """,
            files=file_payload,
            root=root_str,
        )

        # Batch create Symbol nodes and DEFINES edges via UNWIND
        symbol_payload = []
        for rel_path, data in file_data.items():
            for sym in data["symbols"]:
                symbol_payload.append({
                    "id": sym["id"],
                    "file_id": sym["file_id"],
                    "file_path": rel_path,
                    "name": sym["name"],
                    "qualname": sym["qualname"],
                    "kind": sym["kind"],
                    "parent_class": sym.get("parent_class"),
                    "docstring": sym["docstring"],
                    "lineno": sym["lineno"],
                    "end_lineno": sym["end_lineno"],
                    "source": sym["source"],
                })

        if symbol_payload:
            session.run(
                """
                UNWIND $symbols AS sym
                MATCH (f:File {id: sym.file_id})
                CREATE (s:Symbol {
                    id: sym.id,
                    name: sym.name,
                    qualname: sym.qualname,
                    kind: sym.kind,
                    parent_class: sym.parent_class,
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

        # Batch create IMPORTS edges via UNWIND (canonical module resolution)
        import_payload = []
        seen_imports: set[tuple[str, str]] = set()

        for rel_path, data in file_data.items():
            source_id = make_file_id(root_str, rel_path)
            for imported_module in data["imports"]:
                target_rel = canonical_module_to_file.get(imported_module)
                if not target_rel and imported_module in stem_to_file:
                    target_rel = stem_to_file[imported_module]

                if target_rel and target_rel != rel_path:
                    target_id = make_file_id(root_str, target_rel)
                    edge = (source_id, target_id)
                    if edge not in seen_imports:
                        seen_imports.add(edge)
                        import_payload.append({
                            "source_id": source_id,
                            "target_id": target_id,
                        })

        if import_payload:
            session.run(
                """
                UNWIND $imports AS imp
                MATCH (source:File {id: imp.source_id})
                MATCH (target:File {id: imp.target_id})
                MERGE (source)-[:IMPORTS]->(target)
                """,
                imports=import_payload,
            )

        # Conservative, evidence-based CALLS edge resolution
        call_payload = []
        seen_calls: set[tuple[str, str]] = set()

        for rel_path, data in file_data.items():
            imported_mods = data.get("imported_modules", {})
            imported_syms = data.get("imported_symbols", {})

            for sym in data["symbols"]:
                caller_id = sym["id"]
                caller_parent_class = sym.get("parent_class")

                for call in sym.get("call_sites", []):
                    callee_id = None
                    call_type = call.get("type")
                    call_name = call.get("name")

                    if call_type == "self":
                        # self.method(): resolve strictly to caller_parent_class in this file
                        if caller_parent_class:
                            cand_qualname = f"{caller_parent_class}.{call_name}"
                            cand_id = make_symbol_id(root_str, rel_path, cand_qualname)
                            if cand_id in all_symbols_by_id:
                                callee_id = cand_id

                    elif call_type == "attr":
                        # prefix.method(): ONLY resolve if prefix is a known imported module in this file
                        prefix = call.get("prefix")
                        if prefix in imported_mods:
                            target_mod = imported_mods[prefix]
                            target_rel = canonical_module_to_file.get(target_mod)
                            if not target_rel and target_mod in stem_to_file:
                                target_rel = stem_to_file[target_mod]

                            if target_rel:
                                cand_id = make_symbol_id(root_str, target_rel, call_name)
                                if cand_id in all_symbols_by_id:
                                    callee_id = cand_id

                    elif call_type == "bare":
                        # bare func():
                        # Priority A: explicitly imported symbol
                        if call_name in imported_syms:
                            mod_name, orig_name = imported_syms[call_name]
                            target_rel = canonical_module_to_file.get(mod_name)
                            if not target_rel and mod_name in stem_to_file:
                                target_rel = stem_to_file[mod_name]

                            if target_rel:
                                cand_id = make_symbol_id(root_str, target_rel, orig_name)
                                if cand_id in all_symbols_by_id:
                                    callee_id = cand_id

                        # Priority B: top-level symbol in the same file
                        if not callee_id and call_name in file_top_level_symbols.get(rel_path, set()):
                            cand_id = make_symbol_id(root_str, rel_path, call_name)
                            if cand_id != caller_id and cand_id in all_symbols_by_id:
                                callee_id = cand_id

                    # If resolved, record edge
                    if callee_id and callee_id != caller_id:
                        edge = (caller_id, callee_id)
                        if edge not in seen_calls:
                            seen_calls.add(edge)
                            call_payload.append({
                                "caller_id": caller_id,
                                "callee_id": callee_id,
                            })

        if call_payload:
            session.run(
                """
                UNWIND $calls AS call
                MATCH (caller:Symbol {id: call.caller_id})
                MATCH (callee:Symbol {id: call.callee_id})
                MERGE (caller)-[:CALLS]->(callee)
                """,
                calls=call_payload,
            )

        # Batch create CONTAINS edges (hierarchical lexical scoping)
        contains_payload = []
        for rel_path, data in file_data.items():
            for sym in data["symbols"]:
                parent_qualname = sym.get("parent_qualname")
                if parent_qualname:
                    parent_id = make_symbol_id(root_str, rel_path, parent_qualname)
                    if parent_id in all_symbols_by_id:
                        contains_payload.append({
                            "parent_id": parent_id,
                            "child_id": sym["id"],
                        })

        if contains_payload:
            session.run(
                """
                UNWIND $contains AS c
                MATCH (parent:Symbol {id: c.parent_id})
                MATCH (child:Symbol {id: c.child_id})
                MERGE (parent)-[:CONTAINS]->(child)
                """,
                contains=contains_payload,
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
            "contains": len(contains_payload),
            "contains_relationships": len(contains_payload),
            "skipped_non_python": len(non_py_files),
            "skipped_extensions": sorted(list(skipped_extensions)),
        }

    logger.info(
        "Ingestion complete — files=%d, symbols=%d, calls=%d, imports=%d, contains=%d, skipped_non_python=%d",
        counters["files"], counters["symbols"], counters["calls"], counters["imports"],
        counters["contains"], counters["skipped_non_python"],
    )
    return counters
