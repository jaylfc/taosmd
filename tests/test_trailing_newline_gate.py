"""Tests for scripts/check_trailing_newline.py.

Proves the trailing-newline gate in both directions:

* GREEN, passing case: all files in scope end with exactly one ``0x0a``.
* RED, failing case: a file missing its trailing newline exits non-zero and
  names the offending path and its last byte.
* RED, vacuous-fixture case: an empty file is not an offender.
* RED, blank-line-at-EOF case: a file ending in two newlines (``0x0a0a``)
  exits non-zero.
* RED, two offenders in one run: both paths are reported.
* SCOPE, file outside scope: a file with no newline outside the globs is
  ignored (exit 0).
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import scripts.check_trailing_newline as tng
from scripts.check_trailing_newline import main as check_main

REPO_ROOT = Path(__file__).resolve().parent.parent
GATE_SCRIPT = REPO_ROOT / "scripts" / "check_trailing_newline.py"


def _write(repo: Path, rel: str, text: str) -> Path:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


# ----------------------------------------------------------------------
# Integration tests: the six required scenarios
# ----------------------------------------------------------------------


def _repo(tmp_path: Path) -> Path:
    """Create a minimal repo checkout under *tmp_path*."""
    repo = tmp_path / "repo"
    (repo / "changelog.d").mkdir(parents=True)
    (repo / "benchmarks" / "data").mkdir(parents=True)
    return repo


def _run_check(repo: Path) -> tuple[int, str]:
    """Run the gate against *repo* and return (exit_code, stdout)."""
    original = tng.REPO_ROOT
    env = os.environ.copy()
    env["TRAILING_NEWLINE_ROOT"] = str(repo)
    buf = io.StringIO()
    try:
        tng.REPO_ROOT = repo
        with redirect_stdout(buf):
            rc = check_main([])
    finally:
        tng.REPO_ROOT = original
    return rc, buf.getvalue()


class TestTrailingNewlineGate:
    def test_green_all_end_0x0a(self, tmp_path):
        """fixture ends 0x0a -> exit 0"""
        repo = _repo(tmp_path)
        _write(repo, "changelog.d/test.md", "### Release notes\n")
        _write(repo, "benchmarks/data/README.md", "Pinned data\n")
        rc, out = _run_check(repo)
        assert rc == 0, f"expected exit 0, got {rc}: {out!r}"
        assert "trailing-newline-gate: clean" in out

    def test_red_no_newline(self, tmp_path):
        """fixture ends 0x2e (no newline) -> exit 1, offending path in output"""
        repo = _repo(tmp_path)
        _write(repo, "changelog.d/test.md", "### Release notes")
        _write(repo, "benchmarks/data/README.md", "Pinned data")
        rc, out = _run_check(repo)
        assert rc == 1, f"expected exit 1, got {rc}: {out!r}"
        assert "changelog.d/test.md" in out

    def test_red_blank_line_at_eof(self, tmp_path):
        """fixture ends 0x0a0a (blank line at EOF) -> exit 1 (exactly one newline)"""
        repo = _repo(tmp_path)
        _write(repo, "changelog.d/test.md", "### Release notes\n\n")
        _write(repo, "benchmarks/data/README.md", "Pinned data\n\n")
        rc, out = _run_check(repo)
        assert rc == 1, f"expected exit 1, got {rc}: {out!r}"
        assert "trailing-newline-gate FAIL" in out

    def test_green_empty_fixture(self, tmp_path):
        """empty fixture -> exit 0"""
        repo = _repo(tmp_path)
        changelog = repo / "changelog.d" / "empty.md"
        changelog.write_text("")
        readme = repo / "benchmarks" / "data" / "README.md"
        readme.write_text("")
        rc, out = _run_check(repo)
        assert rc == 0, f"expected exit 0 for empty fixtures, got {rc}: {out!r}"
        assert "trailing-newline-gate: clean" in out

    def test_two_offenders_one_run(self, tmp_path):
        """two offenders in one run -> exit 1 and BOTH paths reported"""
        repo = _repo(tmp_path)
        _write(repo, "changelog.d/test1.md", "### Release notes")
        _write(repo, "changelog.d/test2.md", "### More notes")
        _write(repo, "benchmarks/data/README.md", "Pinned data\n")
        rc, out = _run_check(repo)
        assert rc == 1, f"expected exit 1, got {rc}: {out!r}"
        assert "changelog.d/test1.md" in out
        assert "changelog.d/test2.md" in out

    def test_scope_respected_outside_globs(self, tmp_path):
        """file outside the scope globs, no newline -> exit 0 (scope is respected)"""
        repo = _repo(tmp_path)
        outside = repo / "src" / "orphan.md"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_text("orphan content")
        rc, out = _run_check(repo)
        assert rc == 0, f"expected exit 0 when outside scope, got {rc}: {out!r}"
        assert "trailing-newline-gate: clean" in out

    def test_last_byte_hex_report(self, tmp_path):
        """Verify the exact hex byte is reported for a no-newline file."""
        repo = _repo(tmp_path)
        # Content ending with 0x2e (period) and no trailing newline.
        _write(repo, "changelog.d/test.md", "### Release notes.")
        rc, out = _run_check(repo)
        assert rc == 1
        # The output should contain the path and the hex byte
        lines = out.strip().splitlines()
        # Find lines containing the path
        path_lines = [l for l in lines if "changelog.d/test.md" in l]
        assert len(path_lines) >= 1, f"Expected path in output, got: {out!r}"
        # The hex byte should be 0x2e (period, the last byte).
        assert "0x2e" in out

    def test_passing_with_exact_one_newline(self, tmp_path):
        """Verify files with exactly one trailing newline pass."""
        repo = _repo(tmp_path)
        _write(repo, "changelog.d/test.md", "### Release notes\n")
        readme = repo / "benchmarks" / "data" / "README.md"
        readme.write_text("Pinned data\n")
        rc, out = _run_check(repo)
        assert rc == 0, f"expected exit 0 for files with exactly one newline, got {rc}: {out!r}"
        assert "trailing-newline-gate: clean" in out

    def test_benchmarks_readme_no_newline(self, tmp_path):
        """RED: benchmarks/data/README.md missing trailing newline."""
        repo = _repo(tmp_path)
        _write(repo, "changelog.d/test.md", "### Release notes\n")
        _write(repo, "benchmarks/data/README.md", "Pinned data")
        rc, out = _run_check(repo)
        assert rc == 1, f"expected exit 1 for missing README trailing newline, got {rc}: {out!r}"
        assert "benchmarks/data/README.md" in out

    def test_changelog_multiple_offenders(self, tmp_path):
        """RED: multiple changelog fragments all missing newlines."""
        repo = _repo(tmp_path)
        _write(repo, "changelog.d/fix1.md", "Fix one")
        _write(repo, "changelog.d/fix2.md", "Fix two")
        _write(repo, "changelog.d/fix3.md", "Fix three")
        _write(repo, "benchmarks/data/README.md", "Pinned data\n")
        rc, out = _run_check(repo)
        assert rc == 1, f"expected exit 1 for multiple changelog offenders, got {rc}: {out!r}"
        assert "changelog.d/fix1.md" in out
        assert "changelog.d/fix2.md" in out
        assert "changelog.d/fix3.md" in out