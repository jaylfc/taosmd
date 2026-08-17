"""Tests for scripts/normalise_handle_gate.py.

Proves both directions:
- A single _normalise_handle definition lets the gate pass.
- Two rival definitions cause the gate to fail with an explicit count.
- @typing.overload stubs are excluded from the count.
- try/except ImportError fallback must leave the gate silent.
- Files that cannot be decoded as UTF-8 are skipped with a warning.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import scripts.normalise_handle_gate as nhg
from scripts.normalise_handle_gate import (
    _count_definitions,
    _has_overload_decorator,
    main,
)


# ----------------------------------------------------------------------
# unit tests for pure helpers
# ----------------------------------------------------------------------

class TestHasOverloadDecorator:
    def test_name_overload(self):
        src = "@overload\ndef _normalise_handle(handle: str) -> str: ...\n"
        tree = __import__('ast').parse(src)
        funcs = [n for n in __import__('ast').walk(tree) if isinstance(n, __import__('ast').FunctionDef)]
        assert _has_overload_decorator(funcs[0]) is True

    def test_typing_overload(self):
        src = "import typing\n@typing.overload\ndef _normalise_handle(handle: str) -> str: ...\n"
        tree = __import__('ast').parse(src)
        funcs = [n for n in __import__('ast').walk(tree) if isinstance(n, __import__('ast').FunctionDef)]
        assert _has_overload_decorator(funcs[0]) is True

    def test_no_overload(self):
        src = "def _normalise_handle(handle: str) -> str:\n    return handle\n"
        tree = __import__('ast').parse(src)
        funcs = [n for n in __import__('ast').walk(tree) if isinstance(n, __import__('ast').FunctionDef)]
        assert _has_overload_decorator(funcs[0]) is False


class TestCountDefinitions:
    def test_single_def_returns_one(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("def _normalise_handle(handle):\n    pass\n")
        assert _count_definitions(f) == 1

    def test_no_def_returns_zero(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("def other():\n    pass\n")
        assert _count_definitions(f) == 0

    def test_two_defs_return_two(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "def _normalise_handle(a):\n    pass\n"
            "def _normalise_handle(b):\n    pass\n"
        )
        assert _count_definitions(f) == 2

    def test_overload_stubs_ignored(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "import typing\n\n"
            "@typing.overload\n"
            "def _normalise_handle(handle: str) -> str: ...\n"
            "\n"
            "def _normalise_handle(handle):\n    pass\n"
        )
        assert _count_definitions(f) == 1

    def test_name_overload_ignored(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "from typing import overload\n\n"
            "@overload\n"
            "def _normalise_handle(handle: str) -> str: ...\n"
            "\n"
            "def _normalise_handle(handle):\n    pass\n"
        )
        assert _count_definitions(f) == 1

    def test_try_except_import_error_fallback_is_silent(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "try:\n"
            "    from typing import overload\n"
            "except ImportError:\n"
            "    pass\n"
            "\n"
            "def _normalise_handle(handle):\n    pass\n"
        )
        assert _count_definitions(f) == 1

    def test_non_utf8_file_skipped(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_bytes(b"def _normalise_handle():\n    pass\n\xff\xfe\n")
        assert _count_definitions(f) == 0

    def test_unreadable_file_skipped(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("def _normalise_handle():\n    pass\n")
        f.chmod(0o000)
        try:
            assert _count_definitions(f) == 0
        finally:
            f.chmod(0o644)


# ----------------------------------------------------------------------
# integration tests with a temporary repo
# ----------------------------------------------------------------------

def _init_repo(tmp_path, single_def=True):
    repo = tmp_path / "repo"
    repo.mkdir()
    taosmd_dir = repo / "taosmd"
    taosmd_dir.mkdir()
    (taosmd_dir / "__init__.py").write_text("")

    mod = taosmd_dir / "service.py"
    if single_def:
        mod.write_text("def _normalise_handle(handle):\n    pass\n")
    else:
        mod.write_text(
            "def _normalise_handle(a):\n    pass\n"
            "def _normalise_handle(b):\n    pass\n"
        )
    return repo


class TestNormaliseHandleGateIntegration:
    def test_single_def_passes(self, tmp_path, monkeypatch):
        repo = _init_repo(tmp_path, single_def=True)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(nhg, "REPO_ROOT", repo)
        rc = main([])
        assert rc == 0

    def test_two_defs_fail_with_count(self, tmp_path, monkeypatch):
        repo = _init_repo(tmp_path, single_def=False)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(nhg, "REPO_ROOT", repo)
        rc = main([])
        assert rc == 1

    def test_main_prints_duplicate_count(self, tmp_path, monkeypatch, capsys):
        repo = _init_repo(tmp_path, single_def=False)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(nhg, "REPO_ROOT", repo)
        rc = main([])
        assert rc == 1
        captured = capsys.readouterr()
        assert "NORMALISE_HANDLE GATE FAIL" in captured.out
        assert "found 2 definitions" in captured.out

    def test_main_prints_clean_when_single(self, tmp_path, monkeypatch, capsys):
        repo = _init_repo(tmp_path, single_def=True)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(nhg, "REPO_ROOT", repo)
        rc = main([])
        assert rc == 0
        captured = capsys.readouterr()
        assert "normalise-handle-gate: clean" in captured.out
