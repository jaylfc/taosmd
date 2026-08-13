"""Tests for scripts/check_deleted_symbols.py.

Proves both directions:
- A real deletion without the trailer exits non-zero and names the symbol.
- The same deletion with the trailer exits zero and says it was waived.
- The waiver works after a PR-body edit (parsed from PR body text).
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

import scripts.check_deleted_symbols as cds
from scripts.check_deleted_symbols import (
    TRAILER,
    Violation,
    _extract_symbols,
    _find_adding_commit,
    _get_symbols_at_ref,
    _run_git,
    check_deleted_symbols,
    find_signal_symbols,
    main,
    parse_waived_symbols,
)


# ----------------------------------------------------------------------
# unit tests for pure functions
# ----------------------------------------------------------------------

class TestExtractSymbols:
    def test_top_level_def(self):
        src = "def foo():\n    pass\n"
        syms = _extract_symbols(src, "pkg/mod.py")
        assert "pkg/mod.py:foo" in syms
        assert syms["pkg/mod.py:foo"] == "def"

    def test_top_level_class(self):
        src = "class Foo:\n    pass\n"
        syms = _extract_symbols(src, "pkg/mod.py")
        assert "pkg/mod.py:Foo" in syms
        assert syms["pkg/mod.py:Foo"] == "class"

    def test_nested_method(self):
        src = "class Foo:\n    def bar(self):\n        pass\n"
        syms = _extract_symbols(src, "pkg/mod.py")
        assert "pkg/mod.py:Foo.bar" in syms
        assert syms["pkg/mod.py:Foo.bar"] == "def"

    def test_async_def(self):
        src = "async def foo():\n    pass\n"
        syms = _extract_symbols(src, "pkg/mod.py")
        assert "pkg/mod.py:foo" in syms
        assert syms["pkg/mod.py:foo"] == "def"

    def test_syntax_error_returns_empty(self):
        syms = _extract_symbols("def foo(\n", "pkg/mod.py")
        assert syms == {}


class TestFindSignalSymbols:
    def test_deleted_symbol_detected(self):
        base = {"taosmd/foo.py:bar": "def"}
        head = {}
        signal = find_signal_symbols(base, head)
        assert signal == {"taosmd/foo.py:bar": "def"}

    def test_new_symbol_not_signal(self):
        base = {}
        head = {"taosmd/foo.py:bar": "def"}
        signal = find_signal_symbols(base, head)
        assert signal == {}

    def test_unchanged_symbol_not_signal(self):
        base = {"taosmd/foo.py:bar": "def"}
        head = {"taosmd/foo.py:bar": "def"}
        signal = find_signal_symbols(base, head)
        assert signal == {}


class TestParseWaivedSymbols:
    def test_no_trailer(self):
        assert parse_waived_symbols("just a description") == set()

    def test_single_symbol(self):
        body = "Some text\nRemoves-Intentionally: taosmd/foo.py:bar\n"
        assert parse_waived_symbols(body) == {"taosmd/foo.py:bar"}

    def test_multiple_symbols(self):
        body = "Removes-Intentionally: taosmd/foo.py:bar, taosmd/baz.py:qux"
        assert parse_waived_symbols(body) == {"taosmd/foo.py:bar", "taosmd/baz.py:qux"}

    def test_none_body(self):
        assert parse_waived_symbols(None) == set()

    def test_trailer_with_extra_whitespace(self):
        body = "Removes-Intentionally:   taosmd/foo.py:bar  ,  taosmd/baz.py:qux  "
        assert parse_waived_symbols(body) == {"taosmd/foo.py:bar", "taosmd/baz.py:qux"}

    def test_partial_match_not_trailer(self):
        body = "Removes-Intentionallyity: taosmd/foo.py:bar"
        assert parse_waived_symbols(body) == set()


# ----------------------------------------------------------------------
# integration tests with a temporary git repo
# ----------------------------------------------------------------------

def _init_repo(tmp_path):
    """Create a git repo with a base branch and a head commit."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)

    # base commit
    base_file = tmp_path / "taosmd" / "service.py"
    base_file.parent.mkdir(parents=True, exist_ok=True)
    base_file.write_text(
        "def a2a_send():\n    pass\n\ndef dashboard_stats():\n    pass\n"
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=tmp_path, check=True, capture_output=True)

    # head commit: delete a2a_send
    head_file = tmp_path / "taosmd" / "service.py"
    head_file.write_text("def dashboard_stats():\n    pass\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "head"], cwd=tmp_path, check=True, capture_output=True)

    return tmp_path


class TestDeletedSymbolsIntegration:
    def test_deletion_fails_without_waiver(self, tmp_path, monkeypatch):
        repo = _init_repo(tmp_path)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        violations, waived = check_deleted_symbols(
            base_ref="HEAD~1",
            repo_root=repo,
            pr_body=None,
            waived=None,
        )
        assert len(violations) == 1
        assert violations[0].symbol == "taosmd/service.py:a2a_send"
        assert waived == set()

    def test_deletion_passes_with_waiver(self, tmp_path, monkeypatch):
        repo = _init_repo(tmp_path)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        violations, waived = check_deleted_symbols(
            base_ref="HEAD~1",
            repo_root=repo,
            pr_body="Removes-Intentionally: taosmd/service.py:a2a_send",
            waived=None,
        )
        assert violations == []
        assert waived == {"taosmd/service.py:a2a_send"}

    def test_main_exits_nonzero_on_violation(self, tmp_path, monkeypatch):
        repo = _init_repo(tmp_path)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        rc = main(["--base", "HEAD~1"])
        assert rc == 1

    def test_main_exits_zero_with_waiver(self, tmp_path, monkeypatch):
        repo = _init_repo(tmp_path)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        rc = main([
            "--base", "HEAD~1",
            "--pr-body", "Removes-Intentionally: taosmd/service.py:a2a_send",
        ])
        assert rc == 0

    def test_main_prints_waived_message(self, tmp_path, monkeypatch, capsys):
        repo = _init_repo(tmp_path)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        main([
            "--base", "HEAD~1",
            "--pr-body", "Removes-Intentionally: taosmd/service.py:a2a_send",
        ])
        captured = capsys.readouterr()
        assert "deleted-symbols-guard: waived via Removes-Intentionally: taosmd/service.py:a2a_send" in captured.out
        assert "deleted-symbols-guard: clean" in captured.out

    def test_main_prints_violation_details(self, tmp_path, monkeypatch, capsys):
        repo = _init_repo(tmp_path)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        rc = main(["--base", "HEAD~1"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "DELETED-SYMBOLS FAIL" in captured.out
        assert "taosmd/service.py:a2a_send" in captured.out
