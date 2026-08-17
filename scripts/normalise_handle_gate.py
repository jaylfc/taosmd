#!/usr/bin/env python3
"""Duplicate-definition gate.

Scans ``taosmd/`` and ``tests/`` for any top-level function or method name
defined more than once in the same module (scope) and fails if rival
definitions are found.  The original gate only looked for the internal helper
``_normalise_handle``; this one examines every name so that shadowed test
functions -- and any other accidental redefinition -- are reported.

``@typing.overload`` stubs, ``@property`` getters, and ``@<name>.setter`` /
``@<name>.getter`` / ``@<name>.deleter`` accessories are legitimate same-name
pairs and are therefore excluded.  Files that cannot be decoded as UTF-8 are
skipped with a warning rather than allowed to crash the gate.

Definitions inside module-level ``if`` / ``try`` / ``for`` / ``while`` /
``with`` blocks are treated as module-scope, matching Python's binding rules.
Same-name closures inside one parent function are also reported; same-name
closures in different parents remain legal.  The ``try:`` / ``except
ImportError:`` fallback pattern is recognised and left silent.
"""
from __future__ import annotations

import ast
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TARGET_PATTERNS = ("taosmd/**/*.py", "tests/**/*.py")


@dataclass
class Duplicate:
    name: str
    scope: str
    lines: list[int] = field(default_factory=list)


@dataclass
class _Def:
    scope: str
    name: str
    lineno: int
    in_try: bool = False
    in_import_error_except: bool = False


def _has_overload_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name):
            if decorator.id == "overload":
                return True
        elif isinstance(decorator, ast.Attribute):
            if decorator.attr == "overload":
                return True
        elif isinstance(decorator, ast.Call):
            func = decorator.func
            if isinstance(func, ast.Name) and func.id == "overload":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "overload":
                return True
    return False


def _has_property_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """``@property`` getters and ``@<name>.setter`` / ``.getter`` /
    ``.deleter`` accessories always share their name with the companion
    accessor, so they are never rival definitions."""
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "property":
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr in (
            "setter",
            "getter",
            "deleter",
        ):
            return True
    return False


def _is_exempt(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return _has_overload_decorator(node) or _has_property_decorator(node)


def _is_import_error_handler(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return False
    if isinstance(handler.type, ast.Name):
        return handler.type.id == "ImportError"
    if isinstance(handler.type, ast.Tuple):
        return any(
            isinstance(elt, ast.Name) and elt.id == "ImportError"
            for elt in handler.type.elts
        )
    return False


def _is_try_import_error_fallback(defs: list[_Def]) -> bool:
    if not defs:
        return False
    if not all(d.in_try for d in defs):
        return False
    try_body_defs = [d for d in defs if not d.in_import_error_except]
    import_error_defs = [d for d in defs if d.in_import_error_except]
    return len(try_body_defs) == 1 and len(import_error_defs) == 1


def _collect_definitions(
    body: list[ast.stmt],
    scope: str = "module",
    class_path: tuple[str, ...] = (),
    in_try: bool = False,
    in_import_error_except: bool = False,
) -> list[_Def]:
    """Yield ``_Def`` for every non-exempt function/method.

    ``scope`` is ``"module"`` for a top-level function, ``"class Foo.Bar"``
    for a method of ``Foo.Bar``, and ``"module > outer"`` for a closure
    inside ``outer``.  Definitions inside module-level ``if`` / ``try`` /
    ``for`` / ``while`` / ``with`` blocks are collected with ``"module"``
    scope, matching Python's binding rules.  Closures are collected with a
    scope that identifies their parent function, so that same-name closures
    in one parent are reported while same-name closures in different parents
    remain legal.
    """
    found: list[_Def] = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_exempt(node):
                found.append(
                    _Def(scope, node.name, node.lineno, in_try, in_import_error_except)
                )
            closure_scope = (
                f"{scope} > {node.name}"
                if scope != "module"
                else f"module > {node.name}"
            )
            found.extend(
                _collect_definitions(
                    node.body,
                    closure_scope,
                    class_path=(),
                    in_try=False,
                    in_import_error_except=False,
                )
            )
        elif isinstance(node, ast.ClassDef):
            if scope == "module":
                new_class_path = class_path + (node.name,)
                new_scope = "class " + ".".join(new_class_path)
                found.extend(
                    _collect_definitions(
                        node.body,
                        new_scope,
                        new_class_path,
                        in_try=False,
                        in_import_error_except=False,
                    )
                )
        elif isinstance(
            node,
            (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith),
        ):
            if scope == "module":
                found.extend(
                    _collect_definitions(
                        node.body,
                        scope,
                        class_path,
                        in_try,
                        in_import_error_except,
                    )
                )
                if hasattr(node, "orelse") and node.orelse:
                    found.extend(
                        _collect_definitions(
                            node.orelse,
                            scope,
                            class_path,
                            in_try,
                            in_import_error_except,
                        )
                    )
        elif isinstance(node, ast.Try):
            if scope == "module":
                found.extend(
                    _collect_definitions(
                        node.body,
                        scope,
                        class_path,
                        in_try=True,
                        in_import_error_except=in_import_error_except,
                    )
                )
                for handler in node.handlers:
                    is_ie = _is_import_error_handler(handler)
                    found.extend(
                        _collect_definitions(
                            handler.body,
                            scope,
                            class_path,
                            in_try=True,
                            in_import_error_except=is_ie,
                        )
                    )
                if node.orelse:
                    found.extend(
                        _collect_definitions(
                            node.orelse,
                            scope,
                            class_path,
                            in_try=True,
                            in_import_error_except=in_import_error_except,
                        )
                    )
                if node.finalbody:
                    found.extend(
                        _collect_definitions(
                            node.finalbody,
                            scope,
                            class_path,
                            in_try=True,
                            in_import_error_except=in_import_error_except,
                        )
                    )
    return found


def _duplicate_definitions(file_path: Path) -> list[Duplicate]:
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        print(
            f"normalise-handle-gate: skipping {file_path}: unreadable",
            file=sys.stderr,
        )
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    defs: dict[tuple[str, str], list[_Def]] = defaultdict(list)
    for defn in _collect_definitions(tree.body):
        defs[(defn.scope, defn.name)].append(defn)

    duplicates: list[Duplicate] = []
    for (scope, name), def_list in defs.items():
        if len(def_list) <= 1:
            continue
        if _is_try_import_error_fallback(def_list):
            continue
        duplicates.append(
            Duplicate(name=name, scope=scope, lines=[d.lineno for d in def_list])
        )
    duplicates.sort(key=lambda d: d.lines[0])
    return duplicates


def _iter_target_files() -> list[Path]:
    files: list[Path] = []
    for pattern in TARGET_PATTERNS:
        files.extend(REPO_ROOT.glob(pattern))
    return sorted(set(files))


def main(argv: list[str] | None = None) -> int:
    failures: list[tuple[str, Duplicate]] = []
    for file_path in _iter_target_files():
        for dup in _duplicate_definitions(file_path):
            failures.append((str(file_path.relative_to(REPO_ROOT)), dup))

    if failures:
        print("DUPLICATE-DEFINITION GATE FAIL:")
        for path, dup in failures:
            count = len(dup.lines)
            print(
                f"  {path}: {dup.scope} '{dup.name}' defined on lines "
                f"{dup.lines}; found {count} definitions; at most 1 is allowed"
            )
        return 1

    print("normalise-handle-gate: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
