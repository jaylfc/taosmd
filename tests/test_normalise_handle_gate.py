"""Tests for scripts/normalise_handle_gate.py.

Proves the duplicate-definition gate (the successor to the narrow
``_normalise_handle``-only check):

- It fires for ANY name defined more than once in the same scope, not just
  ``_normalise_handle`` -- including shadowed test functions.
- It is per-scope: the same name in two different classes is fine, as is the
  same name as a top-level function and a method.
- ``@typing.overload`` stubs, ``@property`` getters, and
  ``@<name>.setter`` / ``@<name>.getter`` / ``@<name>.deleter`` accessories
  are legitimate same-name pairs and do not fire.
- Definitions inside module-level ``if`` / ``try`` / ``for`` / ``while`` /
  ``with`` blocks share the module scope, matching Python's binding rules.
- Same-name closures inside one parent function are reported; same-name
  closures in different parents remain legal.
- The ``try:`` / ``except ImportError:`` fallback pattern is recognised and
  left silent.
- Files that cannot be decoded as UTF-8 are skipped with a warning, and
  files that fail to parse are left silent.
"""
from __future__ import annotations

import ast
from pathlib import Path

import scripts.normalise_handle_gate as nhg
from scripts.normalise_handle_gate import (
    _duplicate_definitions,
    _has_overload_decorator,
    _has_property_decorator,
    main,
)

# ----------------------------------------------------------------------
# unit tests for pure helpers
# ----------------------------------------------------------------------

class TestHasOverloadDecorator:
    def test_name_overload(self):
        src = "@overload\ndef _normalise_handle(handle: str) -> str: ...\n"
        tree = ast.parse(src)
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert _has_overload_decorator(funcs[0]) is True

    def test_typing_overload(self):
        src = "import typing\n@typing.overload\ndef _normalise_handle(handle: str) -> str: ...\n"
        tree = ast.parse(src)
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert _has_overload_decorator(funcs[0]) is True

    def test_no_overload(self):
        src = "def _normalise_handle(handle: str) -> str:\n    return handle\n"
        tree = ast.parse(src)
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert _has_overload_decorator(funcs[0]) is False


class TestHasPropertyDecorator:
    def test_property_getter_is_exempt(self):
        src = "class C:\n    @property\n    def x(self):\n        return 1\n"
        tree = ast.parse(src)
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert _has_property_decorator(funcs[0]) is True

    def test_setter_is_exempt(self):
        src = "class C:\n    @x.setter\n    def x(self, v):\n        pass\n"
        tree = ast.parse(src)
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert _has_property_decorator(funcs[0]) is True

    def test_getter_is_exempt(self):
        src = "class C:\n    @x.getter\n    def x(self):\n        pass\n"
        tree = ast.parse(src)
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert _has_property_decorator(funcs[0]) is True

    def test_deleter_is_exempt(self):
        src = "class C:\n    @x.deleter\n    def x(self):\n        pass\n"
        tree = ast.parse(src)
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert _has_property_decorator(funcs[0]) is True

    def test_plain_function_is_not_exempt(self):
        src = "def x():\n    pass\n"
        tree = ast.parse(src)
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert _has_property_decorator(funcs[0]) is False


class TestDuplicateDefinitions:
    def test_single_def_has_no_duplicates(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("def foo():\n    pass\n")
        assert _duplicate_definitions(f) == []

    def test_no_def_has_no_duplicates(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("CONST = 1\nclass C:\n    pass\n")
        assert _duplicate_definitions(f) == []

    def test_two_defs_of_any_name_reported(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "def some_other_name(a):\n    pass\n"
            "def some_other_name(b):\n    pass\n"
        )
        dups = _duplicate_definitions(f)
        assert len(dups) == 1
        assert dups[0].name == "some_other_name"
        assert dups[0].scope == "module"
        assert dups[0].lines == [1, 3]

    def test_generalises_beyond_normalise_handle(self, tmp_path):
        """The whole point of the gate: not keyed on one hardcoded name."""
        f = tmp_path / "mod.py"
        f.write_text(
            "def _normalise_handle(a):\n    pass\n"
            "def handle_request(b):\n    pass\n"
            "def handle_request(c):\n    pass\n"
        )
        dups = _duplicate_definitions(f)
        assert [d.name for d in dups] == ["handle_request"]

    def test_overload_pair_is_not_a_duplicate(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "from typing import overload\n\n"
            "@overload\n"
            "def foo(x: str) -> str: ...\n"
            "@overload\n"
            "def foo(x: int) -> int: ...\n"
            "def foo(x):\n    return x\n"
        )
        assert _duplicate_definitions(f) == []

    def test_property_setter_pair_is_not_a_duplicate(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "class C:\n"
            "    @property\n"
            "    def x(self):\n"
            "        return self._x\n"
            "    @x.setter\n"
            "    def x(self, v):\n"
            "        self._x = v\n"
        )
        assert _duplicate_definitions(f) == []

    def test_all_accessor_pairings_are_not_duplicates(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "class C:\n"
            "    @property\n"
            "    def x(self):\n"
            "        return self._x\n"
            "    @x.getter\n"
            "    def x(self):\n"
            "        return self._x\n"
            "    @x.setter\n"
            "    def x(self, v):\n"
            "        self._x = v\n"
            "    @x.deleter\n"
            "    def x(self):\n"
            "        del self._x\n"
        )
        assert _duplicate_definitions(f) == []

    def test_method_shadow_inside_a_class_is_reported(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "class C:\n"
            "    def bar(self):\n"
            "        return 1\n"
            "    def bar(self):\n"
            "        return 2\n"
        )
        dups = _duplicate_definitions(f)
        assert len(dups) == 1
        assert dups[0].name == "bar"
        assert dups[0].scope == "class C"

    def test_same_method_name_in_different_classes_is_fine(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "class A:\n"
            "    def bar(self):\n"
            "        return 1\n"
            "class B:\n"
            "    def bar(self):\n"
            "        return 2\n"
        )
        assert _duplicate_definitions(f) == []

    def test_top_level_fn_and_method_same_name_is_fine(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "def foo():\n"
            "    pass\n"
            "class C:\n"
            "    def foo(self):\n"
            "        pass\n"
        )
        assert _duplicate_definitions(f) == []

    def test_same_name_closures_in_one_parent_are_reported(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "def outer():\n"
            "    def inner():\n"
            "        pass\n"
            "    def inner():\n"
            "        pass\n"
            "    return inner\n"
        )
        dups = _duplicate_definitions(f)
        assert len(dups) == 1
        assert dups[0].name == "inner"
        assert dups[0].scope == "module > outer"
        assert dups[0].lines == [2, 4]

    def test_same_name_closures_in_different_parents_are_fine(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "def outer1():\n"
            "    def inner():\n"
            "        pass\n"
            "def outer2():\n"
            "    def inner():\n"
            "        pass\n"
        )
        assert _duplicate_definitions(f) == []

    def test_def_in_module_level_if_is_reported(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "if condition:\n"
            "    def foo():\n"
            "        pass\n"
            "def foo():\n"
            "    pass\n"
        )
        dups = _duplicate_definitions(f)
        assert len(dups) == 1
        assert dups[0].name == "foo"
        assert dups[0].scope == "module"
        assert dups[0].lines == [2, 4]

    def test_def_in_module_level_for_is_reported(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "for x in range(10):\n"
            "    def foo():\n"
            "        pass\n"
            "def foo():\n"
            "    pass\n"
        )
        dups = _duplicate_definitions(f)
        assert len(dups) == 1
        assert dups[0].name == "foo"
        assert dups[0].scope == "module"

    def test_def_in_module_level_while_is_reported(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "while True:\n"
            "    def foo():\n"
            "        pass\n"
            "    break\n"
            "def foo():\n"
            "    pass\n"
        )
        dups = _duplicate_definitions(f)
        assert len(dups) == 1
        assert dups[0].name == "foo"
        assert dups[0].scope == "module"

    def test_def_in_module_level_with_is_reported(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "with open('x') as f:\n"
            "    def foo():\n"
            "        pass\n"
            "def foo():\n"
            "    pass\n"
        )
        dups = _duplicate_definitions(f)
        assert len(dups) == 1
        assert dups[0].name == "foo"
        assert dups[0].scope == "module"

    def test_try_except_import_error_fallback_with_defs_is_silent(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "try:\n"
            "    def foo():\n"
            "        pass\n"
            "except ImportError:\n"
            "    def foo():\n"
            "        pass\n"
        )
        assert _duplicate_definitions(f) == []

    def test_try_except_value_error_fallback_is_reported(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text(
            "try:\n"
            "    def foo():\n"
            "        pass\n"
            "except ValueError:\n"
            "    def foo():\n"
            "        pass\n"
        )
        dups = _duplicate_definitions(f)
        assert len(dups) == 1
        assert dups[0].name == "foo"
        assert dups[0].scope == "module"
        f = tmp_path / "mod.py"
        f.write_text(
            "try:\n"
            "    from typing import overload\n"
            "except ImportError:\n"
            "    pass\n"
            "\n"
            "def foo():\n"
            "    pass\n"
        )
        assert _duplicate_definitions(f) == []

    def test_syntax_error_is_silent(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("def foo(:\n    pass\n")
        assert _duplicate_definitions(f) == []

    def test_non_utf8_file_skipped(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_bytes(b"def foo():\n    pass\n\xff\xfe\n")
        assert _duplicate_definitions(f) == []

    def test_unreadable_file_skipped(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("def foo():\n    pass\n")
        f.chmod(0o000)
        try:
            assert _duplicate_definitions(f) == []
        finally:
            f.chmod(0o644)


# ----------------------------------------------------------------------
# integration tests with a temporary repo
# ----------------------------------------------------------------------

def _init_repo(tmp_path, module_rel: str, content: str):
    repo = tmp_path / "repo"
    repo.mkdir()
    pkg = repo / Path(module_rel).parent
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    (repo / Path(module_rel)).write_text(content)
    return repo


class TestDuplicateDefinitionGateIntegration:
    def test_clean_when_no_duplicates(self, tmp_path, monkeypatch, capsys):
        repo = _init_repo(
            tmp_path, "taosmd/service.py", "def foo():\n    pass\n"
        )
        monkeypatch.setattr(nhg, "REPO_ROOT", repo)
        rc = main([])
        assert rc == 0
        captured = capsys.readouterr()
        assert "normalise-handle-gate: clean" in captured.out

    def test_shadowed_test_in_tests_tree_fails(self, tmp_path, monkeypatch, capsys):
        repo = _init_repo(
            tmp_path,
            "tests/test_smoke.py",
            "def test_a():\n    pass\n\ndef test_a():\n    pass\n",
        )
        monkeypatch.setattr(nhg, "REPO_ROOT", repo)
        rc = main([])
        assert rc == 1
        captured = capsys.readouterr()
        assert "DUPLICATE-DEFINITION GATE FAIL" in captured.out
        assert "test_a" in captured.out
        assert "tests/test_smoke.py" in captured.out
        assert "found 2 definitions" in captured.out

    def test_rival_defs_in_taosmd_tree_fail(self, tmp_path, monkeypatch, capsys):
        repo = _init_repo(
            tmp_path,
            "taosmd/service.py",
            "def _normalise_handle(a):\n    pass\n"
            "def _normalise_handle(b):\n    pass\n",
        )
        monkeypatch.setattr(nhg, "REPO_ROOT", repo)
        rc = main([])
        assert rc == 1
        captured = capsys.readouterr()
        assert "DUPLICATE-DEFINITION GATE FAIL" in captured.out
        assert "_normalise_handle" in captured.out
        assert "found 2 definitions" in captured.out

    def test_clean_output_when_passing(self, tmp_path, monkeypatch, capsys):
        repo = _init_repo(
            tmp_path, "taosmd/service.py", "def foo():\n    pass\n"
        )
        monkeypatch.setattr(nhg, "REPO_ROOT", repo)
        main([])
        captured = capsys.readouterr()
        assert "fail" not in captured.out.lower()
