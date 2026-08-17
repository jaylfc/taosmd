#!/usr/bin/env python3
"""Witness token gate.

Verifies that every explicit ``# WITNESS: <test>::<token>`` marker in a source
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

    # WITNESS: tests/test_foo.py::some_grepable_token

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

Scope: ``taosmd/**/*.py``. Witnesses are declared in source, not in tests --
scanning ``tests/`` for markers would let a test's own example markers (which
name fixture trees that do not exist in the checked-out repo) produce false
violations. The sibling normalise-handle gate uses the same ``taosmd/`` scope.

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


def _resolve_test_file(test_file: str, repo_root: Path) -> Path | None:
    candidate = repo_root / test_file
    if candidate.is_file():
        return candidate
    return None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _source_files(repo_root: Path) -> list[Path]:
    return sorted(set(repo_root.glob("taosmd/**/*.py")))


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
