"""Tests for scripts/normalise_handle_gate.py.

These tests are the wiring that makes the gate a gate instead of a file in a
folder: ``ci.yml`` collects everything under ``tests/``, so importing and
exercising ``normalise_handle_gate`` here means a green PR is a gate that ran,
not a gate that was never called.

They also pin down the shapes the original gate missed (async defs, class
methods, nested defs) -- all of which ``ast.iter_child_nodes`` +
``isinstance(..., ast.FunctionDef)`` silently ignored, the exact blind spots
flagged in the #285 review.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import scripts.normalise_handle_gate as nhg
from scripts.normalise_handle_gate import (
    check,
    main,
    _count_definitions,
    _definitions_in_source,
)


# ----------------------------------------------------------------------
# unit tests for the pure extraction function (shape coverage)
# ----------------------------------------------------------------------

class TestDefinitionsInSource:
    def test_zero_definitions(self):
        assert _definitions_in_source("def foo():\n    pass\n") == []

    def test_single_top_level_def(self):
        src = "def _normalise_handle():\n    pass\n"
        assert _definitions_in_source(src) == ["_normalise_handle"]

    def test_async_def_counted(self):
        # The original gate matched only ast.FunctionDef, so `async def`
        # slipped past and a duplicate could land undetected.
        src = "async def _normalise_handle():\n    pass\n"
        assert _definitions_in_source(src) == ["_normalise_handle"]

    def test_class_method_counted(self):
        # iter_child_nodes never descends into the class body, so a method
        # duplicate was invisible to the original gate.
        src = "class Foo:\n    def _normalise_handle(self):\n        pass\n"
        assert _definitions_in_source(src) == ["_normalise_handle"]

    def test_nested_in_block_counted(self):
        # Definitions nested under an `if`/with/for were missed because
        # iter_child_nodes only visits module-level nodes.
        src = "if True:\n    def _normalise_handle():\n        pass\n"
        assert _definitions_in_source(src) == ["_normalise_handle"]

    def test_two_top_level_defs(self):
        src = (
            "def _normalise_handle():\n    pass\n\n"
            "def _normalise_handle():\n    pass\n"
        )
        assert _definitions_in_source(src) == [
            "_normalise_handle",
            "_normalise_handle",
        ]

    def test_duplicate_one_async_one_sync(self):
        src = (
            "def _normalise_handle():\n    pass\n\n"
            "async def _normalise_handle():\n    pass\n"
        )
        assert _definitions_in_source(src) == [
            "_normalise_handle",
            "_normalise_handle",
        ]

    def test_duplicate_top_level_and_class_method(self):
        src = (
            "def _normalise_handle():\n    pass\n\n"
            "class Foo:\n    def _normalise_handle(self):\n        pass\n"
        )
        assert _definitions_in_source(src) == [
            "_normalise_handle",
            "_normalise_handle",
        ]

    def test_duplicate_top_level_and_nested(self):
        src = (
            "def _normalise_handle():\n    pass\n\n"
            "if True:\n    def _normalise_handle():\n        pass\n"
        )
        assert _definitions_in_source(src) == [
            "_normalise_handle",
            "_normalise_handle",
        ]

    def test_syntax_error_returns_empty(self):
        assert _definitions_in_source("def _normalise_handle(\n") == []

    def test_unrelated_names_not_counted(self):
        src = (
            "def _normalise_handle():\n    pass\n\n"
            "def _normalise_handle_helper():\n    pass\n"
        )
        assert _definitions_in_source(src) == ["_normalise_handle"]


# ----------------------------------------------------------------------
# helper: build a throwaway taosmd/ tree under tmp_path
# ----------------------------------------------------------------------

def _write_taosmd(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a fake ``taosmd/`` package under ``tmp_path`` with the given files."""
    taosmd = tmp_path / "taosmd"
    taosmd.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        path = taosmd / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return taosmd


def _src_one_def() -> str:
    return "def _normalise_handle(handle, *, mint_strip=False):\n    return handle\n"


# ----------------------------------------------------------------------
# check() -- the at-most-one assertion
# ----------------------------------------------------------------------

class TestCheck:
    def test_zero_definitions_passes(self, tmp_path):
        taosmd = _write_taosmd(tmp_path, {"service.py": "def foo():\n    pass\n"})
        count, message = check(taosmd)
        assert count == 0
        assert "PASS" in message

    def test_one_definition_passes(self, tmp_path):
        taosmd = _write_taosmd(tmp_path, {"service.py": _src_one_def()})
        count, message = check(taosmd)
        assert count == 1
        assert "PASS" in message

    def test_duplicate_top_level_fails(self, tmp_path):
        taosmd = _write_taosmd(
            tmp_path,
            {"service.py": _src_one_def() + _src_one_def()},
        )
        count, message = check(taosmd)
        assert count == 2
        assert "FAIL" in message

    def test_duplicate_spread_across_files_fails(self, tmp_path):
        taosmd = _write_taosmd(
            tmp_path,
            {"a.py": _src_one_def(), "b.py": _src_one_def()},
        )
        count, message = check(taosmd)
        assert count == 2
        assert "FAIL" in message

    def test_duplicate_via_async_def_fails(self, tmp_path):
        # The shape the original gate let through.
        src = (
            "def _normalise_handle():\n    pass\n\n"
            "async def _normalise_handle():\n    pass\n"
        )
        taosmd = _write_taosmd(tmp_path, {"service.py": src})
        count, message = check(taosmd)
        assert count == 2
        assert "FAIL" in message

    def test_duplicate_via_class_method_fails(self, tmp_path):
        src = (
            "def _normalise_handle():\n    pass\n\n"
            "class Foo:\n    def _normalise_handle(self):\n        pass\n"
        )
        taosmd = _write_taosmd(tmp_path, {"service.py": src})
        count, message = check(taosmd)
        assert count == 2
        assert "FAIL" in message

    def test_duplicate_via_nested_def_fails(self, tmp_path):
        src = (
            "def _normalise_handle():\n    pass\n\n"
            "if True:\n    def _normalise_handle():\n        pass\n"
        )
        taosmd = _write_taosmd(tmp_path, {"service.py": src})
        count, message = check(taosmd)
        assert count == 2
        assert "FAIL" in message


# ----------------------------------------------------------------------
# _count_definitions -- unreadable file is skipped, not a crash
# ----------------------------------------------------------------------

class TestCountDefinitions:
    def test_unreadable_file_is_skipped(self, tmp_path):
        taosmd = _write_taosmd(tmp_path, {"ok.py": _src_one_def()})
        bad = taosmd / "locked.py"
        bad.write_text(_src_one_def())
        bad.chmod(0o000)
        try:
            count = _count_definitions(taosmd)
        finally:
            bad.chmod(0o644)
        assert count == 1

    def test_empty_dir_is_zero(self, tmp_path):
        taosmd = _write_taosmd(tmp_path, {"service.py": "def foo():\n    pass\n"})
        assert _count_definitions(taosmd) == 0


# ----------------------------------------------------------------------
# main() -- end-to-end exit codes against a temp taosmd tree
# ----------------------------------------------------------------------

class TestMain:
    def test_main_exits_zero_on_master_clean(self, tmp_path, monkeypatch):
        # Master has zero definitions; the gate must not invent a failure.
        taosmd = _write_taosmd(tmp_path, {"service.py": "def foo():\n    pass\n"})
        monkeypatch.setattr(nhg, "TAOSMD_DIR", taosmd)
        assert main() == 0

    def test_main_exits_zero_on_single_definition(self, tmp_path, monkeypatch):
        taosmd = _write_taosmd(tmp_path, {"service.py": _src_one_def()})
        monkeypatch.setattr(nhg, "TAOSMD_DIR", taosmd)
        assert main() == 0

    def test_main_exits_nonzero_on_duplicate(self, tmp_path, monkeypatch):
        taosmd = _write_taosmd(
            tmp_path,
            {"a.py": _src_one_def(), "b.py": _src_one_def()},
        )
        monkeypatch.setattr(nhg, "TAOSMD_DIR", taosmd)
        assert main() == 1

    def test_main_reports_count_on_failure(self, tmp_path, monkeypatch, capsys):
        taosmd = _write_taosmd(
            tmp_path,
            {"a.py": _src_one_def(), "b.py": _src_one_def()},
        )
        monkeypatch.setattr(nhg, "TAOSMD_DIR", taosmd)
        rc = main()
        out = capsys.readouterr().out
        assert rc == 1
        assert "2" in out
        assert "FAIL" in out


# ----------------------------------------------------------------------
# Live scan of the real taosmd/ tree (this repo on master)
# ----------------------------------------------------------------------

class TestLiveTree:
    def test_live_tree_has_at_most_one(self):
        # This is the check CI actually runs. On a clean master with no
        # _normalise_handle definitions yet, the gate must be green rather
        # than failing for the absence of the promoted helper (that helper
        # lands on #283, this gate must not mandate it).
        count, message = check()
        assert count <= 1, message
