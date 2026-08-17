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


def _collect_definitions(
    body: list[ast.stmt], class_path: tuple[str, ...] = ()
) -> list[tuple[str, str, int]]:
    """Yield ``(scope, name, lineno)`` for every non-exempt function/method.

    ``scope`` is ``"module"`` for a top-level function, ``"class Foo.Bar"``
    for a method of ``Foo.Bar``.  Nested functions (closures) are
    intentionally NOT collected: they are neither top-level functions nor
    methods, and same-name closures in different parents are legal.
    """
    scope = "module" if not class_path else "class " + ".".join(class_path)
    found: list[tuple[str, str, int]] = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_exempt(node):
                found.append((scope, node.name, node.lineno))
        elif isinstance(node, ast.ClassDef):
            found.extend(_collect_definitions(node.body, class_path + (node.name,)))
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

    defs: dict[tuple[str, str], list[int]] = defaultdict(list)
    for scope, name, lineno in _collect_definitions(tree.body):
        defs[(scope, name)].append(lineno)

    duplicates: list[Duplicate] = [
        Duplicate(name=name, scope=scope, lines=lines)
        for (scope, name), lines in defs.items()
        if len(lines) > 1
    ]
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
