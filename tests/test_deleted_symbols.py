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
    _extract_all_exports,
    _extract_symbols,
    _find_adding_commit,
    _get_symbols_at_ref,
    _run_git,
    check_deleted_symbols,
    find_removed_all_entries,
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


class TestExtractAllExports:
    def test_single_line_all(self):
        src = 'def foo():\n    pass\n\ndef bar():\n    pass\n\n__all__ = ["foo", "bar"]\n'
        exports = _extract_all_exports(src, "pkg/mod.py")
        assert "pkg/mod.py:foo" in exports
        assert "pkg/mod.py:bar" in exports
        assert len(exports) == 2

    def test_multi_line_all(self):
        src = (
            'def foo():\n    pass\n\n'
            'def bar():\n    pass\n\n'
            '__all__ = ["foo",\n    "bar"]\n'
        )
        exports = _extract_all_exports(src, "pkg/mod.py")
        assert "pkg/mod.py:foo" in exports
        assert "pkg/mod.py:bar" in exports

    def test_tuple_all(self):
        src = '__all__ = ("foo", "bar")\n'
        exports = _extract_all_exports(src, "pkg/mod.py")
        assert "pkg/mod.py:foo" in exports
        assert "pkg/mod.py:bar" in exports

    def test_continuation_at_column_zero(self):
        src = '__all__ = ["foo", "bar",\n"baz"]\n'
        exports = _extract_all_exports(src, "pkg/mod.py")
        assert "pkg/mod.py:foo" in exports
        assert "pkg/mod.py:bar" in exports
        assert "pkg/mod.py:baz" in exports

    def test_no_all(self):
        src = "def foo():\n    pass\n"
        exports = _extract_all_exports(src, "pkg/mod.py")
        assert exports == {}

    def test_syntax_error_returns_empty(self):
        exports = _extract_all_exports("def foo(\n", "pkg/mod.py")
        assert exports == {}

    def test_augmented_all(self):
        src = '__all__ = ["foo"]\n__all__ += ["bar"]\n'
        exports = _extract_all_exports(src, "pkg/mod.py")
        assert "pkg/mod.py:foo" in exports
        assert "pkg/mod.py:bar" in exports


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


class TestFindRemovedAllEntries:
    def test_removed_all_but_def_survives(self):
        base_exports = {"pkg/mod.py:foo": "export", "pkg/mod.py:bar": "export"}
        head_exports = {"pkg/mod.py:foo": "export"}
        head_symbols = {"pkg/mod.py:foo": "def", "pkg/mod.py:bar": "def"}
        signal = find_removed_all_entries(base_exports, head_exports, head_symbols)
        assert signal == {"pkg/mod.py:bar": "def"}

    def test_removed_all_and_def_removed_is_not_signal(self):
        base_exports = {"pkg/mod.py:foo": "export", "pkg/mod.py:bar": "export"}
        head_exports = {"pkg/mod.py:foo": "export"}
        head_symbols = {"pkg/mod.py:foo": "def"}
        signal = find_removed_all_entries(base_exports, head_exports, head_symbols)
        assert signal == {}

    def test_no_removal(self):
        base_exports = {"pkg/mod.py:foo": "export"}
        head_exports = {"pkg/mod.py:foo": "export"}
        head_symbols = {"pkg/mod.py:foo": "def"}
        signal = find_removed_all_entries(base_exports, head_exports, head_symbols)
        assert signal == {}

    def test_added_to_all_is_not_signal(self):
        base_exports = {"pkg/mod.py:foo": "export"}
        head_exports = {"pkg/mod.py:foo": "export", "pkg/mod.py:bar": "export"}
        head_symbols = {"pkg/mod.py:foo": "def", "pkg/mod.py:bar": "def"}
        signal = find_removed_all_entries(base_exports, head_exports, head_symbols)
        assert signal == {}

    def test_removed_export_no_def_is_not_signal(self):
        base_exports = {"pkg/mod.py:bar": "export"}
        head_exports = {}
        head_symbols = {}
        signal = find_removed_all_entries(base_exports, head_exports, head_symbols)
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


def _init_repo_with_n_symbols(tmp_path, n):
    """Create a git repo with n top-level defs, all deleted in head."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)

    base_file = tmp_path / "taosmd" / "service.py"
    base_file.parent.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(f"def s{i}():\n    pass\n" for i in range(n))
    base_file.write_text(lines)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=tmp_path, check=True, capture_output=True)

    head_file = tmp_path / "taosmd" / "service.py"
    head_file.write_text("")
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

    def test_main_prints_waiver_hint_on_failure(self, tmp_path, monkeypatch, capsys):
        repo = _init_repo(tmp_path)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        rc = main(["--base", "HEAD~1"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "Removes-Intentionally:" in captured.out
        assert "If these deletions are intentional" in captured.out
        assert f"{TRAILER} taosmd/service.py:a2a_send" in captured.out

    def test_main_omits_waiver_hint_when_clean(self, tmp_path, monkeypatch, capsys):
        repo = _init_repo(tmp_path)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        rc = main([
            "--base", "HEAD~1",
            "--pr-body", "Removes-Intentionally: taosmd/service.py:a2a_send",
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert "If these deletions are intentional" not in captured.out

    def test_main_prints_single_line_waiver_for_five_violations(self, tmp_path, monkeypatch, capsys):
        repo = _init_repo_with_n_symbols(tmp_path, 5)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        rc = main(["--base", "HEAD~1"])
        assert rc == 1
        captured = capsys.readouterr()
        trailer_lines = [
            line for line in captured.out.splitlines() if line.strip().startswith(TRAILER)
        ]
        assert len(trailer_lines) == 1

    def test_main_prints_multi_line_waiver_for_six_violations(self, tmp_path, monkeypatch, capsys):
        repo = _init_repo_with_n_symbols(tmp_path, 6)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        rc = main(["--base", "HEAD~1"])
        assert rc == 1
        captured = capsys.readouterr()
        trailer_lines = [
            line for line in captured.out.splitlines() if line.strip().startswith(TRAILER)
        ]
        assert len(trailer_lines) == 6

    def test_main_multi_line_waiver_feeds_back_to_rc0(self, tmp_path, monkeypatch, capsys):
        repo = _init_repo_with_n_symbols(tmp_path, 6)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        rc = main(["--base", "HEAD~1"])
        assert rc == 1
        captured = capsys.readouterr()
        trailer_lines = [
            line.strip() for line in captured.out.splitlines() if line.strip().startswith(TRAILER)
        ]
        pr_body = "\n".join(trailer_lines)

        rc = main(["--base", "HEAD~1", "--pr-body", pr_body])
        assert rc == 0

    def test_wrong_symbol_still_fails_at_scale(self, tmp_path, monkeypatch):
        repo = _init_repo_with_n_symbols(tmp_path, 6)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        rc = main([
            "--base", "HEAD~1",
            "--pr-body", "Removes-Intentionally: wrong/path:wrong_sym",
        ])
        assert rc == 1

    def test_waiving_some_of_n_still_fails_for_rest(self, tmp_path, monkeypatch):
        repo = _init_repo_with_n_symbols(tmp_path, 6)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        waived = "Removes-Intentionally: taosmd/service.py:s0, taosmd/service.py:s1, taosmd/service.py:s2"
        rc = main(["--base", "HEAD~1", "--pr-body", waived])
        assert rc == 1


# ----------------------------------------------------------------------
# __all__ export-removal: a name dropped from __all__ while its def survives
# ----------------------------------------------------------------------

def _init_repo_all_removed(tmp_path):
    """A name is removed from __all__ while its def survives at HEAD."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)

    base_file = tmp_path / "taosmd" / "service.py"
    base_file.parent.mkdir(parents=True, exist_ok=True)
    base_file.write_text(
        'def func_a():\n    pass\n\n'
        'def func_b():\n    pass\n\n'
        '__all__ = ["func_a", "func_b"]\n'
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=tmp_path, check=True, capture_output=True)

    head_file = tmp_path / "taosmd" / "service.py"
    head_file.write_text(
        'def func_a():\n    pass\n\n'
        'def func_b():\n    pass\n\n'
        '__all__ = ["func_a"]\n'
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "head"], cwd=tmp_path, check=True, capture_output=True)

    return tmp_path


def _init_repo_legitimate_deletion(tmp_path):
    """A name is removed from both __all__ and the file (legitimate deletion)."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)

    base_file = tmp_path / "taosmd" / "service.py"
    base_file.parent.mkdir(parents=True, exist_ok=True)
    base_file.write_text(
        'def func_a():\n    pass\n\n'
        'def func_b():\n    pass\n\n'
        '__all__ = ["func_a", "func_b"]\n'
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=tmp_path, check=True, capture_output=True)

    head_file = tmp_path / "taosmd" / "service.py"
    head_file.write_text(
        'def func_a():\n    pass\n\n'
        '__all__ = ["func_a"]\n'
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "head"], cwd=tmp_path, check=True, capture_output=True)

    return tmp_path


def _init_repo_noop(tmp_path):
    """A PR that touches neither symbols nor __all__."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)

    base_file = tmp_path / "taosmd" / "service.py"
    base_file.parent.mkdir(parents=True, exist_ok=True)
    base_file.write_text(
        'def func_a():\n    pass\n\n'
        '__all__ = ["func_a"]\n'
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=tmp_path, check=True, capture_output=True)

    head_file = tmp_path / "taosmd" / "service.py"
    head_file.write_text(
        '# harmless comment\n'
        'def func_a():\n    pass\n\n'
        '__all__ = ["func_a"]\n'
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "head"], cwd=tmp_path, check=True, capture_output=True)

    return tmp_path


class TestAllRemovalIntegration:
    def test_all_removal_fails_when_def_survives(self, tmp_path, monkeypatch):
        repo = _init_repo_all_removed(tmp_path)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        violations, waived = check_deleted_symbols(
            base_ref="HEAD~1",
            repo_root=repo,
            pr_body=None,
            waived=None,
        )
        assert len(violations) == 1
        assert violations[0].symbol == "taosmd/service.py:func_b"
        assert violations[0].kind == "export-removed"
        assert waived == set()

    def test_all_removal_exits_nonzero(self, tmp_path, monkeypatch):
        repo = _init_repo_all_removed(tmp_path)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        rc = main(["--base", "HEAD~1"])
        assert rc == 1

    def test_all_removal_passes_with_waiver(self, tmp_path, monkeypatch):
        repo = _init_repo_all_removed(tmp_path)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        rc = main([
            "--base", "HEAD~1",
            "--pr-body", "Removes-Intentionally: taosmd/service.py:func_b",
        ])
        assert rc == 0

    def test_all_removal_prints_violation_details(self, tmp_path, monkeypatch, capsys):
        repo = _init_repo_all_removed(tmp_path)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        rc = main(["--base", "HEAD~1"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "DELETED-SYMBOLS FAIL" in captured.out
        assert "from __all__" in captured.out
        assert "taosmd/service.py:func_b" in captured.out


class TestAllRemovalControl:
    def test_legitimate_deletion_has_no_export_violation(self, tmp_path, monkeypatch):
        """A name removed from both __all__ and the file is a legitimate deletion.
        The existing def-deletion check catches it; the __all__ check must not."""
        repo = _init_repo_legitimate_deletion(tmp_path)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        violations, waived = check_deleted_symbols(
            base_ref="HEAD~1",
            repo_root=repo,
            pr_body=None,
            waived=None,
        )
        deleted = [v for v in violations if v.kind == "deleted"]
        assert len(deleted) == 1
        assert deleted[0].symbol == "taosmd/service.py:func_b"
        assert waived == set()
        export_removed = [v for v in violations if v.kind == "export-removed"]
        assert export_removed == []

    def test_legitimate_deletion_passes_with_def_waiver(self, tmp_path, monkeypatch):
        """With the def-deletion waived, the full gate is clean."""
        repo = _init_repo_legitimate_deletion(tmp_path)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        rc = main([
            "--base", "HEAD~1",
            "--pr-body", "Removes-Intentionally: taosmd/service.py:func_b",
        ])
        assert rc == 0

    def test_noop_change_is_clean(self, tmp_path, monkeypatch):
        """A PR that touches neither symbols nor __all__ stays green."""
        repo = _init_repo_noop(tmp_path)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(cds, "REPO_ROOT", repo)

        rc = main(["--base", "HEAD~1"])
        assert rc == 0
