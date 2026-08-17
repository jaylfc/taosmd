"""Tests for scripts/check_witness_token.py.

Proves the witness-token gate in both directions:

* RED, motivating: a source file declares ``# WITNESS: <test>::<token>`` but the
  token is absent from that test -- the gate exits non-zero and names the source
  file, the test and the token.
* RED, non-vacuity: the passing case (token present) passes, then ONLY the token
  is deleted from the test file (the file stays present and importable) and the
  gate flips to non-zero. A path-only check could not see this.
* GREEN, true positive: a declared witness whose token IS present exits 0.
* GREEN, false positive: a source file that merely mentions a test filename in
  ordinary prose, with no ``WITNESS:`` marker, is not flagged.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import scripts.check_witness_token as wtg
from scripts.check_witness_token import (
    _extract_claims,
    _resolve_test_file,
    check_witnesses,
    main,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
GATE_SCRIPT = REPO_ROOT / "scripts" / "check_witness_token.py"

TEST_SRC = "taosmd/svc.py"
TEST_TEST = "tests/test_foo.py"
TOKEN = "MAX_FIRE_TO_DELETE"

GREEN_SRC = (
    "# Constants justified by a live witness in the suite.\n"
    f"# WITNESS: {TEST_TEST}::{TOKEN}\n"
    "RETRY_LEAD_SECONDS = 60\n"
)
GREEN_TEST = (
    f"{TOKEN} = 1\n"
    "\n"
    "def test_arm_time():\n"
    f"    assert {TOKEN} == 1\n"
)
RED_TEST = (
    "# The witness token was removed by a revert; the citation stayed.\n"
    "\n"
    "def test_arm_time():\n"
    "    assert 1 == 1\n"
)
PROSE_SRC = (
    "# NOTE: test_foo.py\"; that witness does not exist in that file, and the\n"
    "# suite stayed green through the revert.\n"
)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "taosmd").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "taosmd" / "__init__.py").write_text("")
    return repo


def _write(repo: Path, rel: str, text: str) -> Path:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _green_repo(tmp_path: Path) -> Path:
    repo = _repo(tmp_path)
    _write(repo, TEST_SRC, GREEN_SRC)
    _write(repo, TEST_TEST, GREEN_TEST)
    return repo


def _run_main(repo: Path) -> tuple[int, str]:
    original = wtg.REPO_ROOT
    wtg.REPO_ROOT = repo
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = main([])
    finally:
        wtg.REPO_ROOT = original
    return rc, buf.getvalue()


def _run_main_clean(repo: Path) -> None:
    rc, out = _run_main(repo)
    assert rc == 0, out


def _run_cli(repo: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["WITNESS_GATE_ROOT"] = str(repo)
    return subprocess.run(
        [sys.executable, str(GATE_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
    )


# ----------------------------------------------------------------------
# unit tests for the parser
# ----------------------------------------------------------------------


class TestExtractClaims:
    def test_parses_bare_marker(self, tmp_path):
        f = _write(tmp_path, TEST_SRC, f"# WITNESS: {TEST_TEST}::{TOKEN}\n")
        claims = _extract_claims(f, tmp_path)
        assert len(claims) == 1
        c = claims[0]
        assert c.source_file == TEST_SRC
        assert c.line_number == 1
        assert c.test_file == TEST_TEST
        assert c.token == TOKEN

    def test_inline_comment_marker(self, tmp_path):
        f = _write(
            tmp_path,
            TEST_SRC,
            f"RETRY_LEAD_SECONDS = 60  # WITNESS: {TEST_TEST}::{TOKEN}\n",
        )
        claims = _extract_claims(f, tmp_path)
        assert len(claims) == 1
        assert claims[0].test_file == TEST_TEST
        assert claims[0].token == TOKEN
        assert claims[0].line_number == 1

    def test_no_space_after_colon(self, tmp_path):
        f = _write(tmp_path, TEST_SRC, f"# WITNESS:{TEST_TEST}::{TOKEN}\n")
        claims = _extract_claims(f, tmp_path)
        assert len(claims) == 1
        assert claims[0].test_file == TEST_TEST
        assert claims[0].token == TOKEN

    def test_token_may_contain_double_colon(self, tmp_path):
        f = _write(tmp_path, TEST_SRC, f"# WITNESS: {TEST_TEST}::node::inner\n")
        claims = _extract_claims(f, tmp_path)
        assert len(claims) == 1
        assert claims[0].token == "node::inner"

    def test_missing_separator_is_malformed(self, tmp_path):
        f = _write(tmp_path, TEST_SRC, f"# WITNESS: {TEST_TEST}\n")
        claims = _extract_claims(f, tmp_path)
        assert len(claims) == 1
        assert claims[0].test_file == ""
        assert claims[0].token == ""

    def test_empty_payload_ignored(self, tmp_path):
        f = _write(tmp_path, TEST_SRC, "# WITNESS:\n")
        assert _extract_claims(f, tmp_path) == []

    def test_lowercase_witness_in_prose_ignored(self, tmp_path):
        f = _write(tmp_path, TEST_SRC, f"# witness: {TEST_TEST}::{TOKEN}\n")
        assert _extract_claims(f, tmp_path) == []

    def test_bare_filename_in_prose_ignored(self, tmp_path):
        f = _write(
            tmp_path,
            TEST_SRC,
            "# See test_foo.py for the sample above.\n",
        )
        assert _extract_claims(f, tmp_path) == []

    def test_unreadable_file_skipped(self, tmp_path):
        f = _write(tmp_path, TEST_SRC, f"# WITNESS: {TEST_TEST}::{TOKEN}\n")
        f.chmod(0o000)
        try:
            assert _extract_claims(f, tmp_path) == []
        finally:
            f.chmod(0o644)


class TestResolveTestFile:
    def test_resolves_repo_relative(self, tmp_path):
        repo = _repo(tmp_path)
        _write(repo, TEST_TEST, f"{TOKEN} = 1\n")
        assert _resolve_test_file(TEST_TEST, repo) == repo / TEST_TEST

    def test_returns_none_when_absent(self, tmp_path):
        repo = _repo(tmp_path)
        assert _resolve_test_file(TEST_TEST, repo) is None


# ----------------------------------------------------------------------
# integration tests: the four required scenarios
# ----------------------------------------------------------------------


class TestWitnessGateIntegration:
    def test_red_motivating_token_absent(self, tmp_path):
        repo = _green_repo(tmp_path)
        # The witness token is absent from the test file (the revert case).
        _write(repo, TEST_TEST, RED_TEST)

        violations = check_witnesses(repo)
        assert len(violations) == 1
        v = violations[0]
        assert v.claim.source_file == TEST_SRC
        assert v.claim.test_file == TEST_TEST
        assert v.claim.token == TOKEN
        assert "not found" in v.reason

        rc, out = _run_main(repo)
        assert rc == 1
        assert "WITNESS GATE FAIL" in out
        assert TEST_SRC in out
        assert TEST_TEST in out
        assert TOKEN in out

    def test_red_non_vacuity_flips_when_token_removed(self, tmp_path):
        repo = _green_repo(tmp_path)

        # GREEN baseline: token present -> clean.
        assert check_witnesses(repo) == []
        _run_main_clean(repo)

        # Delete ONLY the token from the test file. The file stays present and
        # importable (it still defines test_arm_time), so a path-only check
        # could never see this -- the gate must flip to non-zero.
        _write(repo, TEST_TEST, RED_TEST)
        violations = check_witnesses(repo)
        assert len(violations) == 1
        assert violations[0].claim.token == TOKEN

        rc, _out = _run_main(repo)
        assert rc == 1

    def test_green_true_positive(self, tmp_path):
        repo = _green_repo(tmp_path)
        assert check_witnesses(repo) == []
        rc, out = _run_main(repo)
        assert rc == 0
        assert "witness-gate: clean" in out

    def test_green_prose_mention_ignored(self, tmp_path):
        repo = _green_repo(tmp_path)
        _write(repo, "taosmd/prose.py", PROSE_SRC)
        # No WITNESS marker anywhere; the prose mention of the test filename
        # must not be treated as a declaration.
        assert check_witnesses(repo) == []
        rc, out = _run_main(repo)
        assert rc == 0
        assert "witness-gate: clean" in out


# ----------------------------------------------------------------------
# CLI subprocess tests: prove the real script exit codes (Layer A path)
# ----------------------------------------------------------------------


class TestWitnessGateCLI:
    def test_cli_red_exit_nonzero(self, tmp_path):
        repo = _green_repo(tmp_path)
        _write(repo, TEST_TEST, RED_TEST)
        result = _run_cli(repo)
        assert result.returncode == 1
        assert "WITNESS GATE FAIL" in result.stdout
        assert TEST_SRC in result.stdout
        assert TEST_TEST in result.stdout
        assert TOKEN in result.stdout

    def test_cli_green_exit_zero(self, tmp_path):
        repo = _green_repo(tmp_path)
        result = _run_cli(repo)
        assert result.returncode == 0
        assert "witness-gate: clean" in result.stdout
