#!/usr/bin/env python3
"""Deleted-symbols guard.

Detects PRs that silently delete code added to the target branch after the
merge base. This catches the "clean merge, no conflict" failure mode where a
PR branch cut before hardening commits landed on dev would silently delete
them when merged -- git reports "Automatic merge went well" with no conflict
because the PR branch simply wins on files dev touched after the branch point.

Two shapes of silent loss are detected:

  1. A def/class present on the target branch but gone at HEAD (definition
     removal).
  2. A name removed from a module __all__ while its top-level def/class still
     exists at HEAD (export removal). This is the shape a mechanical merge
     resolution produces: one side of a conflicted __all__ block is taken
     wholesale, dropping entries without touching the definitions.

Algorithm:
  1. Extract all Python def/class symbols at two points: the target branch
     head and HEAD (the merge ref / PR head).
  2. A symbol is "deleted by the PR" if it exists at the target head but not
     at HEAD. The signal is that set.
  3. Also compare __all__ membership: an entry present at the target head but
     absent at HEAD is a signal only when the corresponding top-level
     def/class still exists at HEAD. A genuine deletion removes both the
     definition and the __all__ entry, so it stays allowed.
  4. Fail with an explicit list of the deleted symbols and the commits that
     added them.
  5. A "Removes-Intentionally: <symbol>, ..." trailer in the PR body waives
     named symbols, making deliberate deletions a conscious, auditable act.

Narrow by design: Python def/class names only (test functions are defs).
Exports are compared via AST so continuation lines and indentation never
matter.

Usage:
    python scripts/check_deleted_symbols.py --base origin/master
    python scripts/check_deleted_symbols.py --base origin/master --pr-body "..."
    python scripts/check_deleted_symbols.py --base origin/master --waived "path/file.py:func"
"""
from __future__ import annotations

import argparse
import ast
import io
import os
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAILER = "Removes-Intentionally:"


@dataclass
class Violation:
    symbol: str
    added_by: str
    kind: str = "deleted"


def _run_git(args: list[str], cwd: str | Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True,
    )
    return result.stdout


def _extract_symbols(source: str, file_path: str) -> dict[str, str]:
    """Extract symbol identifiers from Python source code.

    Returns a dict mapping "file_path:qualified_name" to kind ("def" or
    "class"). Qualified names use dot notation for nested definitions
    (e.g. ClassName.method, Outer.Inner).
    """
    symbols: dict[str, str] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return symbols

    def visit(node: ast.AST, prefix: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}" if prefix else child.name
                symbols[f"{file_path}:{name}"] = "def"
                visit(child, f"{name}.")
            elif isinstance(child, ast.ClassDef):
                name = f"{prefix}{child.name}" if prefix else child.name
                symbols[f"{file_path}:{name}"] = "class"
                visit(child, f"{name}.")
            else:
                visit(child, prefix)

    visit(tree)
    return symbols


def _collect_all_names(value: ast.AST, file_path: str, exports: dict[str, str]) -> None:
    """Collect string-literal names from an __all__ value expression."""
    if isinstance(value, (ast.List, ast.Tuple)):
        for elt in value.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                exports[f"{file_path}:{elt.value}"] = "export"


def _extract_all_exports(source: str, file_path: str) -> dict[str, str]:
    """Extract __all__ export names from Python source.

    Returns a dict mapping "file_path:name" to "export". Uses AST (not
    regex) so continuation lines and indentation never matter. Handles list
    and tuple literals, including __all__ += [...] augmented assignment.
    """
    exports: dict[str, str] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return exports

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    _collect_all_names(node.value, file_path, exports)
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            _collect_all_names(node.value, file_path, exports)

    return exports


def _get_symbols_at_ref(ref: str, repo_root: Path = REPO_ROOT) -> dict[str, str]:
    """Get all Python symbols at a given git ref using git archive."""
    result = subprocess.run(
        ["git", "archive", ref],
        cwd=repo_root, capture_output=True, check=True,
    )
    symbols: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as tar:
        for member in tar.getmembers():
            if not member.name.startswith("taosmd/") or not member.name.endswith(".py"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            source = f.read().decode("utf-8", errors="ignore")
            symbols.update(_extract_symbols(source, member.name))
    return symbols


def _get_all_exports_at_ref(ref: str, repo_root: Path = REPO_ROOT) -> dict[str, str]:
    """Get all __all__ exports at a given git ref using git archive."""
    result = subprocess.run(
        ["git", "archive", ref],
        cwd=repo_root, capture_output=True, check=True,
    )
    exports: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as tar:
        for member in tar.getmembers():
            if not member.name.startswith("taosmd/") or not member.name.endswith(".py"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            source = f.read().decode("utf-8", errors="ignore")
            exports.update(_extract_all_exports(source, member.name))
    return exports


def _find_adding_commit(
    file_path: str,
    name: str,
    kind: str,
    base_ref: str,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Find the commit that added a symbol to the base branch."""
    leaf = name.split(".")[-1]
    search = f"{kind} {leaf}("
    out = _run_git(
        ["log", "--oneline", "--reverse", "-S", search, base_ref, "--", file_path],
        cwd=repo_root,
    )
    lines = out.splitlines()
    if lines:
        return lines[0]
    # Fallback: try without the opening paren (for classes without bases,
    # e.g. `class Foo:` rather than `class Foo(Bar):`).
    search = f"{kind} {leaf}"
    out = _run_git(
        ["log", "--oneline", "--reverse", "-S", search, base_ref, "--", file_path],
        cwd=repo_root,
    )
    lines = out.splitlines()
    if lines:
        return lines[0]
    return "unknown"


def parse_waived_symbols(pr_body: str | None) -> set[str]:
    """Parse Removes-Intentionally trailer from PR body text."""
    waived: set[str] = set()
    if not pr_body:
        return waived
    for line in pr_body.splitlines():
        line = line.strip()
        if line.startswith(TRAILER):
            symbols_str = line[len(TRAILER):].strip()
            for sym in symbols_str.split(","):
                sym = sym.strip()
                if sym:
                    waived.add(sym)
    return waived


def find_signal_symbols(
    base_symbols: dict[str, str],
    head_symbols: dict[str, str],
) -> dict[str, str]:
    """Find symbols at base that are not at HEAD."""
    base_keys = set(base_symbols.keys())
    head_keys = set(head_symbols.keys())
    signal_keys = base_keys - head_keys
    return {k: base_symbols[k] for k in signal_keys}


def find_removed_all_entries(
    base_exports: dict[str, str],
    head_exports: dict[str, str],
    head_symbols: dict[str, str],
) -> dict[str, str]:
    """Find __all__ entries removed at HEAD while their def/class still exists.

    An entry in base __all__ that is absent from head __all__ is a signal only
    when the corresponding top-level def/class still exists at HEAD. A genuine
    deletion removes both the definition and the __all__ entry, so it does not
    appear here -- it is caught by find_signal_symbols instead.
    """
    removed = set(base_exports) - set(head_exports)
    return {k: head_symbols[k] for k in removed if k in head_symbols}


def check_deleted_symbols(
    base_ref: str,
    repo_root: Path = REPO_ROOT,
    pr_body: str | None = None,
    waived: set[str] | None = None,
) -> tuple[list[Violation], set[str]]:
    """Main check function.

    Returns (violations, waived_symbols) where violations is a list of
    Violation objects for non-waived signal symbols, and waived_symbols is
    the set of signal symbols that were waived.
    """
    base_symbols = _get_symbols_at_ref(base_ref, repo_root)
    head_symbols = _get_symbols_at_ref("HEAD", repo_root)

    signal = find_signal_symbols(base_symbols, head_symbols)

    base_exports = _get_all_exports_at_ref(base_ref, repo_root)
    head_exports = _get_all_exports_at_ref("HEAD", repo_root)
    export_signal = find_removed_all_entries(base_exports, head_exports, head_symbols)

    waived_set: set[str] = set()
    if waived:
        waived_set.update(waived)
    waived_set.update(parse_waived_symbols(pr_body))

    violations: list[Violation] = []
    waived_in_signal: set[str] = set()
    for symbol, kind in sorted(signal.items()):
        if symbol in waived_set:
            waived_in_signal.add(symbol)
            continue
        file_path, name = symbol.rsplit(":", 1)
        added_by = _find_adding_commit(file_path, name, kind, base_ref, repo_root)
        violations.append(Violation(symbol=symbol, added_by=added_by, kind="deleted"))

    for symbol, kind in sorted(export_signal.items()):
        if symbol in waived_set:
            waived_in_signal.add(symbol)
            continue
        file_path, name = symbol.rsplit(":", 1)
        added_by = _find_adding_commit(file_path, name, kind, base_ref, repo_root)
        violations.append(Violation(symbol=symbol, added_by=added_by, kind="export-removed"))

    return violations, waived_in_signal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Target branch ref (e.g. origin/master)")
    parser.add_argument("--pr-body", default=None, help="PR body text (for Removes-Intentionally trailer)")
    parser.add_argument("--waived", default=None, help="Comma-separated list of waived symbols")
    args = parser.parse_args(argv)

    pr_body = args.pr_body
    if pr_body is None:
        pr_body = os.environ.get("PR_BODY")

    waived: set[str] = set()
    if args.waived:
        for sym in args.waived.split(","):
            sym = sym.strip()
            if sym:
                waived.add(sym)

    violations, waived_symbols = check_deleted_symbols(args.base, REPO_ROOT, pr_body, waived)

    if waived_symbols:
        for sym in sorted(waived_symbols):
            print(f"deleted-symbols-guard: waived via Removes-Intentionally: {sym}")

    if violations:
        deleted = [v for v in violations if v.kind == "deleted"]
        export_removed = [v for v in violations if v.kind == "export-removed"]

        if deleted:
            print(
                f"DELETED-SYMBOLS FAIL: this PR deletes {len(deleted)} symbol(s) that "
                f"exist on the base branch but are missing from the merge result:"
            )
            for v in deleted:
                print(f"  - {v.symbol} (added by {v.added_by})")

        if export_removed:
            print(
                f"DELETED-SYMBOLS FAIL: this PR removes {len(export_removed)} symbol(s) "
                f"from __all__ while the definition still exists at HEAD:"
            )
            for v in export_removed:
                print(f"  - {v.symbol} (added by {v.added_by})")

        print()
        print(
            "If these deletions are intentional, edit the PR body to add this trailer. "
            "Editing the PR body re-runs this gate automatically; Re-run job does not "
            "work because it replays the stale payload with the old body:"
        )
        print()
        if len(violations) > 5:
            for v in violations:
                print(f"    {TRAILER} {v.symbol}")
        else:
            trailer_symbols = ", ".join(v.symbol for v in violations)
            print(f"    {TRAILER} {trailer_symbols}")
        return 1

    print("deleted-symbols-guard: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
