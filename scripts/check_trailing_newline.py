#!/usr/bin/env python3
"""Trailing-newline gate.

Verifies that every file in scope ends with exactly one ``0x0a`` (newline).
A file missing its trailing newline silently concatenates the next appended
entry onto the last line, corrupting release notes in ``changelog.d/`` and
the pinned-data ``README.md``.

Scope (deliberately narrow):
  - ``changelog.d/*.md``
  - ``benchmarks/data/README.md``

An optional ``TRAILING_NEWLINE_ROOT`` environment variable overrides the repo
root the gate scans (defaulting to the tree containing this script). This lets
the gate target an arbitrary checkout or a fixture directory -- useful for CI
on staged changes and for reproducible local demos.

Usage:
    python scripts/check_trailing_newline.py
    TRAILING_NEWLINE_ROOT=/tmp/fixture python scripts/check_trailing_newline.py
    (exits 0 when every file in scope ends in exactly one newline, 1 otherwise)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = Path(os.environ.get("TRAILING_NEWLINE_ROOT") or _REPO_ROOT)

SCOPE_GLOBS = (
    "changelog.d/*.md",
    "benchmarks/data/README.md",
)


def _last_byte_hex(file_path: Path) -> str | None:
    """Return the last byte of *file_path* as a two-digit hex string, or None
    if the file has fewer than 1 byte."""
    try:
        data = file_path.read_bytes()
    except (OSError,):
        return None
    if len(data) == 0:
        return None
    return f"{(data[-1]):02x}"


def _check_file(file_path: Path) -> tuple[bool, str | None]:
    """Check whether *file_path* ends with exactly one newline.

    Returns ``(pass, last_byte_hex)``:
      - ``pass`` is ``True`` when the file ends with exactly one ``0x0a``.
      - ``last_byte_hex`` is the hex value of the actual last byte when
        ``pass`` is ``False`` (``None`` when the file is empty or when pass
        is True).
    """
    try:
        data = file_path.read_bytes()
    except (OSError,):
        return True, None

    # Empty file: not an offender (nothing to concatenate onto).
    if len(data) == 0:
        return True, None

    # Last byte must be 0x0a.
    if data[-1] != 0x0a:
        last_hex = f"{(data[-1]):02x}"
        return False, last_hex

    # Last byte is 0x0a. Check that the second-to-last byte is NOT 0x0a
    # (which would be a blank line at EOF, i.e. more than one trailing newline).
    if len(data) >= 2 and data[-2] == 0x0a:
        # Blank line at EOF: the file ends with 0x0a0a or more.
        # The last byte is still 0x0a, but it's not "exactly one".
        last_hex = "0a"
        return False, last_hex

    # Exactly one trailing newline (0x0a) -> clean.
    return True, None


def _scan_scope() -> list[tuple[Path, bool, str | None]]:
    """Scan all files in the scope and return (path, passes, last_byte_hex)."""
    results: list[tuple[Path, bool, str | None]] = []
    for glob in SCOPE_GLOBS:
        for path in sorted(REPO_ROOT.glob(glob)):
            passed, last_byte = _check_file(path)
            results.append((path, passed, last_byte))
    return results


def main(argv: list[str] | None = None) -> int:
    del argv  # unused, kept for interface consistency
    violations: list[str] = []

    for path, passed, last_byte in _scan_scope():
        if not passed:
            violations.append(f"  {path}: last byte 0x{last_byte}")

    if violations:
        print("trailing-newline-gate FAIL:")
        for v in violations:
            print(v)
        return 1

    # Still need to report empty-file cases.
    # Print nothing extra for empty files — they are not offenders.
    print("trailing-newline-gate: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())