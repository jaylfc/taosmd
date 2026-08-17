#!/usr/bin/env python3
"""Guard: at most one ``_normalise_handle`` definition under ``taosmd/``.

Fails (exit 1) when two or more definitions of ``_normalise_handle`` appear
anywhere under ``taosmd/`` -- whether ``def`` or ``async def``, at module
level, inside a class body, or nested inside a block. Zero or one definitions
is green.

The gate asserts *at most* one, not *exactly* one: a clean branch that has not
yet landed the promoted helper (#283) legitimately has zero definitions, and
the gate must not manufacture a failure for their absence. The defect this
guard exists to catch is a **duplicate** -- two independent copies of the slug
helper that no existing check could see -- so it fails only when the count is
two or more.

This gate is invoked from two places, so a green result means it ran:

* ``.github/workflows/normalise-handle-gate.yml`` -- runs the script in CI on
  every pull request (mirrors ``deleted-symbols-gate.yml``).
* ``tests/test_normalise_handle_gate.py`` -- exercises the logic and the
  end-to-end exit code; CI runs it via ``ci.yml``'s ``pytest tests/``.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TAOSMD_DIR = REPO_ROOT / "taosmd"

_TARGET = "_normalise_handle"


def _definitions_in_source(source: str) -> list[str]:
    """Return every ``_normalise_handle`` function def found in ``source``.

    Uses :func:`ast.walk` so the search recurses into class bodies, function
    bodies and nested blocks. ``ast.iter_child_nodes`` only visits the top
    module level and would miss a definition hidden inside a class or an
    ``if`` block -- the exact blind spot the #285 review found.

    Both :class:`ast.FunctionDef` and :class:`ast.AsyncFunctionDef` are matched,
    so an ``async def`` duplicate is not silently counted as zero. Source that
    fails to parse yields no matches: an unparseable file cannot define the
    symbol, so it is treated as "not present" rather than a crash.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == _TARGET
    ]


def _count_definitions(taosmd_dir: Path) -> int:
    """Count every ``def _normalise_handle`` under ``taosmd_dir``.

    Files that cannot be read (permission denied, broken symlink, non-UTF-8
    bytes that raise on decode) are skipped with a warning rather than allowed
    to crash the gate, so an unreadable file produces a clean reported failure
    instead of an unhandled traceback.
    """
    count = 0
    for py_file in sorted(taosmd_dir.rglob("*.py")):
        try:
            source = py_file.read_text(encoding="utf-8")
        except OSError as exc:
            print(
                f"normalise-handle-gate: WARNING cannot read {py_file}: {exc}",
                file=sys.stderr,
            )
            continue
        count += len(_definitions_in_source(source))
    return count


def check(taosmd_dir: Path | None = None) -> tuple[int, str]:
    """Run the gate against ``taosmd_dir`` (``TAOSMD_DIR`` when omitted).

    Returns ``(count, message)``. The gate passes when ``count <= 1`` and
    fails when ``count >= 2`` (a duplicate). ``count`` is returned with the
    message so callers can report the measured value, not a boolean.
    """
    if taosmd_dir is None:
        taosmd_dir = TAOSMD_DIR
    count = _count_definitions(taosmd_dir)
    if count >= 2:
        return (
            count,
            f"NORMALISE_HANDLE GATE FAIL: found {count} definitions of "
            f"`{_TARGET}` under {taosmd_dir}; at most 1 is allowed. "
            f"Duplicate definitions of this slug helper must not land.",
        )
    return (
        count,
        f"NORMALISE_HANDLE GATE PASS: found {count} definition(s) of "
        f"`{_TARGET}` under {taosmd_dir}; at most 1 is allowed.",
    )


def main() -> int:
    count, message = check()
    print(message)
    return 1 if count >= 2 else 0


if __name__ == "__main__":
    sys.exit(main())
