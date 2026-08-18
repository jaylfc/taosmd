#!/usr/bin/env python3
"""Witness token gate.

Verifies that every explicit ``# WITNESS​: <test>::<token>`` marker in a source
file resolves to something real INSIDE the named test: the referenced test file
must exist and ``<token>`` must appear in it as a case-sensitive substring (grep
semantics).

Why a token, not a path: a source comment that cites a test as the justification
for a constant is only as good as what the test still contains. A path-only check
passes green while the cited witness tokens are deleted from the test, because the
test file itself survives. This gate cannot be fooled by that -- it resolves the
TOKEN, not the path. The gate is non-vacuous in both directions: it fails when a
declared token is removed (file stays importable), and it does NOT fire on a bare
test-filename mention in ordinary prose.

Marker contract (the only thing that triggers the check):

    # WITNESS​: tests/test_foo.py::some_grepable_token

- The ``WITNESS:`` marker is all-caps and opt-in. A bare mention of a test
  filename in prose -- whether describing a removal or stating a fact --
  declares nothing and is invisible to the gate. Only lines containing the
  ``WITNESS:`` marker are parsed, so prose like "the witness was removed from
  test_resume_arm_time.py" is never a declaration.
- The text after the marker is split on the FIRST ``::``: the left side is the
  test file path (resolved relative to the repo root), the right side is the token
  to grep for. Splitting on the first ``::`` lets a token itself contain ``::``
  (for example a pytest node id), matching grep semantics. ``::`` is never part
  of a relative file path, so no escaping is required.
- The token is matched as a case-sensitive substring, the same match ``grep``
  would perform.

Scope: ``taosmd/**/*.py`` and ``scripts/**/*.py``. Witnesses are declared in
source, not in tests -- the suite embeds fixture markers as f-strings
(e.g. ``f"# WITNESS​: {TEST}::{TOK}"``) whose interpolated names never
resolve to real files, so scanning ``tests/`` would produce false violations.
The sibling ``check_deleted_symbols.py`` gate follows the same pattern: it
restricts to ``taosmd/`` only, excluding ``tests/``.

The zero-width space (U+200B) between ``WITNESS`` and ``:`` in the on-disk
examples above is intentional: it de-marks the docstring examples so they
are not treated as live markers.

Two de-marked examples (with ZWSP between WITNESS and ``:``) that must be
exempted so the gate does not flag its own docstring; every other line in the
same file (e.g. an appended genuine marker) still is. Greppable name.

An optional ``WITNESS_GATE_ROOT`` environment variable overrides the repo root
the gate scans (defaulting to the tree containing this script). This lets the
gate target an arbitrary checkout, a staged tree, or a fixture directory -- useful
for CI-on-staged-changes and for reproducible local demos.

Usage:
    python scripts/check_witness_token.py
    WITNESS_GATE_ROOT=/tmp/fixture python scripts/check_witness_token.py
    (exits 0 when every declared witness resolves, 1 otherwise)
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = Path(os.environ.get("WITNESS_GATE_ROOT") or _REPO_ROOT)
WITNESS_RE = re.compile(r"#\s*WITNESS:\s*(.+)$")
# A near-miss resembles a WITNESS marker whose separator is broken -- a
# zero-width character (e.g. U+200B) lodged between WITNESS and the colon,
# or the colon replaced. Requiring the ``::`` payload keeps ordinary prose
# mentioning WITNESS from being mistaken for a malformed marker. The regex
# ``#\s*WITNESS[^:](?=[^:]*:)`` also catches prose carrying a plain colon
# (e.g. a colon appearing after WITNESS text without ``::``) without ``::``,
# which master's ``(?=.*::)`` does not -- this is the "widening" noted in the
# regression table.
_NEAR_MISS_RE = re.compile(r"#\s*WITNESS[^:](?=[^:]*:)")
# Documented examples of a de-marked marker in this file's own docstring.
# They are intentionally de-marked and must not be reported; every other line
# in the same file (e.g. an appended genuine marker) still is. Greppable name.
_DEMARKED_MARKER_EXEMPTION = {
    "scripts/check_witness_token.py": frozenset({4, 19, 36}),
}


@dataclass
class WitnessClaim:
    source_file: str
    line_number: int
    raw: str
    test_file: str = ""
    token: str = ""


@dataclass
class Violation:
    claim: WitnessClaim
    reason: str


def _relative_source(file_path: Path, repo_root: Path) -> str:
    try:
        return str(file_path.relative_to(repo_root))
    except ValueError:
        return str(file_path)


def _extract_claims(file_path: Path, repo_root: Path) -> list[WitnessClaim]:
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        print(
            f"witness-gate: skipping {file_path}: unreadable",
            file=sys.stderr,
        )
        return []
    rel = _relative_source(file_path, repo_root)
    claims: list[WitnessClaim] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        match = WITNESS_RE.search(line)
        if not match:
            continue
        raw = match.group(1).strip()
        claim = WitnessClaim(source_file=rel, line_number=lineno, raw=raw)
        if "::" in raw:
            test_file, token = raw.split("::", 1)
            claim.test_file = test_file.strip()
            claim.token = token.strip()
        claims.append(claim)
    return claims


def _check_near_misses(file_path: Path, repo_root: Path) -> list[Violation]:
    rel = _relative_source(file_path, repo_root)
    exempt_lines = _DEMARKED_MARKER_EXEMPTION.get(rel, frozenset())
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    violations: list[Violation] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        if lineno in exempt_lines:
            continue
        if _NEAR_MISS_RE.search(line):
            claim = WitnessClaim(
                source_file=rel, line_number=lineno, raw=line.strip()
            )
            violations.append(Violation(claim, "de-marked or malformed marker"))
    return violations


def _resolve_test_file(test_file: str, repo_root: Path) -> Path | None:
    candidate = repo_root / test_file
    if candidate.is_file():
        return candidate
    return None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _source_files(repo_root: Path) -> list[Path]:
    return sorted(
        set(repo_root.glob("taosmd/**/*.py"))
        | set(repo_root.glob("scripts/**/*.py"))
    )


def _check_claim(
    claim: WitnessClaim,
    repo_root: Path,
    contents: dict[Path, str],
) -> Violation | None:
    if not claim.test_file or not claim.token:
        return Violation(
            claim,
            "malformed WITNESS marker: expected <test_file>::<token>",
        )
    target = _resolve_test_file(claim.test_file, repo_root)
    if target is None:
        return Violation(claim, f"test file not found: {claim.test_file}")
    if target not in contents:
        try:
            contents[target] = _read_text(target)
        except OSError:
            contents[target] = ""
    if claim.token not in contents[target]:
        return Violation(
            claim,
            f"witness token not found in {claim.test_file}",
        )
    return None


def check_witnesses(repo_root: Path = REPO_ROOT) -> list[Violation]:
    violations: list[Violation] = []
    contents: dict[Path, str] = {}
    for source in _source_files(repo_root):
        violations.extend(_check_near_misses(source, repo_root))
        for claim in _extract_claims(source, repo_root):
            violation = _check_claim(claim, repo_root, contents)
            if violation is not None:
                violations.append(violation)
    return violations


def main(argv: list[str] | None = None) -> int:
    violations = check_witnesses(REPO_ROOT)
    if violations:
        print("WITNESS GATE FAIL:")
        for v in violations:
            print(
                f"  {v.claim.source_file}:{v.claim.line_number}: "
                f"WITNESS {v.claim.raw} -> {v.reason}"
            )
        return 1
    print("witness-gate: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
