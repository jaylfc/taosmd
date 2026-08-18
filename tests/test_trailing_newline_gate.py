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
* OVERRIDE: ``TRAILING_NEWLINE_ROOT`` actually redirects the scan, proven from
  a fresh interpreter because the variable is read at import time.
* REAL TREE: the gate is clean against this checkout, so a bad fragment turns a
  plain ``pytest`` run red as well as CI.
"""
from __future__ import annotations

import io
import os
import subprocess
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
    """Run the gate against *repo* in-process and return (exit_code, stdout).

    This rebinds ``tng.REPO_ROOT`` directly. It does NOT exercise the
    ``TRAILING_NEWLINE_ROOT`` environment variable, which is only read at import
    time and therefore only reachable from a fresh interpreter; see
    ``TestTrailingNewlineRootOverride`` for that.
    """
    original = tng.REPO_ROOT
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


# ----------------------------------------------------------------------
# The TRAILING_NEWLINE_ROOT override, which is documented in the gate's
# docstring and read at import time. Every test above rebinds tng.REPO_ROOT
# in-process, so none of them touch the environment variable at all: dropping
# the override entirely (``REPO_ROOT = _REPO_ROOT``) leaves them all green.
# These tests spawn a real interpreter so the override is actually read.
# ----------------------------------------------------------------------


def _run_gate_subprocess(cwd: Path, root: str | None) -> subprocess.CompletedProcess[str]:
    """Invoke the gate script in a fresh interpreter from *cwd*.

    When *root* is not None it is passed as ``TRAILING_NEWLINE_ROOT``; when it
    is None the variable is removed, so the gate falls back to the tree
    containing the script.
    """
    env = os.environ.copy()
    if root is None:
        env.pop("TRAILING_NEWLINE_ROOT", None)
    else:
        env["TRAILING_NEWLINE_ROOT"] = root
    return subprocess.run(
        [sys.executable, str(GATE_SCRIPT)],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )


class TestTrailingNewlineRootOverride:
    def test_override_redirects_the_scan_to_a_clean_tree(self, tmp_path):
        """TRAILING_NEWLINE_ROOT=<clean fixture> -> exit 0 against the fixture.

        Fails when the override is dropped: without it the gate scans the real
        repository instead of the fixture, so the offender planted below is
        never seen.
        """
        repo = _repo(tmp_path)
        _write(repo, "changelog.d/test.md", "### Release notes\n")
        _write(repo, "benchmarks/data/README.md", "Pinned data\n")
        result = _run_gate_subprocess(tmp_path, str(repo))
        assert result.returncode == 0, f"expected exit 0, got {result.returncode}: {result.stdout!r} {result.stderr!r}"
        assert "trailing-newline-gate: clean" in result.stdout

    def test_override_redirects_the_scan_to_an_offending_tree(self, tmp_path):
        """TRAILING_NEWLINE_ROOT=<offending fixture> -> exit 1 naming the fixture path.

        This is the assertion that kills the ``REPO_ROOT = _REPO_ROOT`` mutant:
        the reported path must be the one under *tmp_path*, which can only
        happen if the environment variable was honoured.
        """
        repo = _repo(tmp_path)
        _write(repo, "changelog.d/offender.md", "no trailing newline")
        _write(repo, "benchmarks/data/README.md", "Pinned data\n")
        result = _run_gate_subprocess(tmp_path, str(repo))
        assert result.returncode == 1, f"expected exit 1, got {result.returncode}: {result.stdout!r} {result.stderr!r}"
        assert "trailing-newline-gate FAIL" in result.stdout
        assert str(repo / "changelog.d" / "offender.md") in result.stdout
        assert "last byte 0x65" in result.stdout


class TestRealRepositoryTree:
    def test_gate_is_clean_against_the_real_repository(self, tmp_path):
        """The gate exits 0 against this checkout, with no override set.

        This is the only test that reads the repository's own files, and so the
        only one that makes a plain ``pytest`` run go red when a fragment is
        committed without its trailing newline. It is intended to fail when a
        fragment is added badly; that is the point of it.
        """
        result = _run_gate_subprocess(tmp_path, None)
        assert result.returncode == 0, (
            "the real tree has a file in scope that does not end in exactly one "
            f"newline:\n{result.stdout}{result.stderr}"
        )
        assert "trailing-newline-gate: clean" in result.stdout
