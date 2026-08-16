#!/usr/bin/env python3
"""Guard: exactly one _normalise_handle definition in taosmd/.

Fails when the count of `def _normalise_handle` across taosmd/ is not exactly 1.
Matches the pattern of the existing deleted-symbols-gate.
"""
from __future__ import annotations

import ast
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TAOSMD_DIR = REPO_ROOT / "taosmd"


def _extract_normalise_handles(source: str) -> list[str]:
    """Extract _normalise_handle definition names from Python source."""
    names: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return names

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_normalise_handle":
            names.append(node.name)
    return names


def _count_definitions() -> int:
    """Count all `def _normalise_handle` occurrences under taosmd/."""
    count = 0
    for py_file in sorted(TAOSMD_DIR.rglob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        count += len(_extract_normalise_handles(source))
    return count


def main() -> int:
    count = _count_definitions()
    if count != 1:
        print(
            f"NORMALISE_HANDLE GATE FAIL: expected exactly 1 "
            f"definition of `def _normalise_handle` under taosmd/, "
            f"found {count}"
        )
        return 1
    print(
        f"NORMALISE_HANDLE GATE PASS: exactly 1 definition of "
        f"`def _normalise_handle` found under taosmd/"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())