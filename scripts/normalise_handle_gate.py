#!/usr/bin/env python3
"""Normalise handle gate.

Scans ``taosmd/`` for duplicate definitions of the internal helper
``_normalise_handle`` and fails if any module contains more than one.
``@typing.overload`` stubs are excluded: they are not rival copies.

Files that cannot be decoded as UTF-8 are skipped with a warning rather
than allowed to crash the gate.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def _count_definitions(file_path: Path) -> int:
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        print(
            f"normalise-handle-gate: skipping {file_path}: unreadable",
            file=sys.stderr,
        )
        return 0

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0

    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "_normalise_handle" and not _has_overload_decorator(node):
                count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    failures: list[tuple[str, int]] = []
    for file_path in sorted(REPO_ROOT.glob("taosmd/**/*.py")):
        count = _count_definitions(file_path)
        if count > 1:
            rel = file_path.relative_to(REPO_ROOT)
            failures.append((str(rel), count))

    if failures:
        print("NORMALISE_HANDLE GATE FAIL:")
        for path, count in failures:
            print(
                f"  {path}: found {count} definitions; at most 1 is allowed"
            )
        return 1

    print("normalise-handle-gate: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
