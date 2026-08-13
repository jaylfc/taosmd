#!/usr/bin/env python3
"""RemoteClient routes gate.

Extracts every ``self._run("<METHOD>", "<path>"`` literal from
``taosmd/remote.py`` and requires each path to appear as a routed literal
in ``taosmd/http_server.py``.  Fails with the method and path named.

Paths built by interpolation (f-strings, ``%s``) cannot be extracted by a
literal scan.  They are counted as unparseable call sites and reported so
a silent drop in coverage is visible.

A ``Missing-Route-Intentionally:`` trailer in the PR body waives named
missing routes, making deliberate omissions a conscious, auditable act.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REMOTE = REPO_ROOT / "taosmd" / "remote.py"
DEFAULT_HTTP = REPO_ROOT / "taosmd" / "http_server.py"
TRAILER = "Missing-Route-Intentionally:"


@dataclass(frozen=True)
class RemoteCall:
    method: str
    path: str
    lineno: int


@dataclass(frozen=True)
class MissingRoute:
    method: str
    path: str
    lineno: int


def _parse_remote_calls(source: str) -> tuple[list[RemoteCall], int]:
    """Extract literal ``self._run(method, path)`` calls from remote.py.

    Returns (literal_calls, unparseable_count).
    """
    tree = ast.parse(source)
    calls: list[RemoteCall] = []
    unparseable = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "_run":
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != "self":
            continue
        if len(node.args) < 2:
            continue
        method_arg = node.args[0]
        path_arg = node.args[1]
        if not isinstance(method_arg, ast.Constant) or not isinstance(method_arg.value, str):
            continue
        method = method_arg.value.upper()
        if isinstance(path_arg, ast.Constant) and isinstance(path_arg.value, str):
            calls.append(RemoteCall(method=method, path=path_arg.value, lineno=path_arg.lineno))
        else:
            unparseable += 1

    return calls, unparseable


def _is_path_related(node: ast.AST) -> bool:
    """Return True if the node references the local variable ``path``."""
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == "path":
            return True
    return False


def _extract_route_literals(source: str) -> set[str]:
    """Extract routed path literals from http_server.py.

    Collects string literals that appear in comparisons or method calls
    involving the local variable ``path``.
    """
    tree = ast.parse(source)
    literals: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            if not _is_path_related(node):
                continue
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    literals.add(comparator.value)
                elif isinstance(comparator, (ast.Tuple, ast.List)):
                    for elt in comparator.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            literals.add(elt.value)
        elif isinstance(node, ast.Call):
            if not _is_path_related(node):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("startswith", "endswith"):
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    literals.add(node.args[0].value)

    return literals


def parse_waived_routes(pr_body: str | None) -> set[str]:
    """Parse Missing-Route-Intentionally trailer from PR body text."""
    waived: set[str] = set()
    if not pr_body:
        return waived
    for line in pr_body.splitlines():
        if TRAILER in line:
            routes_str = line.split(TRAILER, 1)[1].strip()
            for route in routes_str.split(","):
                route = route.strip()
                if route:
                    waived.add(route)
    return waived


def check_remote_routes(
    remote_path: Path = DEFAULT_REMOTE,
    http_path: Path = DEFAULT_HTTP,
    pr_body: str | None = None,
) -> tuple[list[MissingRoute], set[str], int]:
    """Main check function.

    Returns (missing_routes, waived_routes, unparseable_count).
    """
    remote_source = remote_path.read_text(encoding="utf-8")
    http_source = http_path.read_text(encoding="utf-8")

    calls, unparseable = _parse_remote_calls(remote_source)
    route_literals = _extract_route_literals(http_source)

    missing: list[MissingRoute] = []
    waived: set[str] = set()

    waived.update(parse_waived_routes(pr_body))

    for call in calls:
        key = f"{call.method} {call.path}"
        if call.path not in route_literals and key not in waived:
            missing.append(MissingRoute(method=call.method, path=call.path, lineno=call.lineno))

    return missing, waived, unparseable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default=str(DEFAULT_REMOTE), help="Path to remote.py")
    parser.add_argument("--http", default=str(DEFAULT_HTTP), help="Path to http_server.py")
    parser.add_argument("--pr-body", default=None, help="PR body text (for Missing-Route-Intentionally trailer)")
    args = parser.parse_args(argv)

    pr_body = args.pr_body
    if pr_body is None:
        pr_body = os.environ.get("PR_BODY")

    missing, waived, unparseable = check_remote_routes(
        remote_path=Path(args.remote),
        http_path=Path(args.http),
        pr_body=pr_body,
    )

    if unparseable:
        print(f"remote-routes-gate: {unparseable} unparseable call site(s) (interpolated paths not checked)")

    if waived:
        for route in sorted(waived):
            print(f"remote-routes-gate: waived via {TRAILER} {route}")

    if missing:
        print(
            f"REMOTE-ROUTES FAIL: {len(missing)} RemoteClient call path(s) have no matching "
            f"routed literal in http_server.py:"
        )
        for m in missing:
            print(f"  {m.method} {m.path} (remote.py:{m.lineno})")
        print()
        trailer_routes = ", ".join(f"{m.method} {m.path}" for m in missing)
        print(
            "If these omissions are intentional, add this trailer to the PR body and "
            "the gate will re-run automatically:"
        )
        print()
        print(f"    {TRAILER} {trailer_routes}")
        return 1

    print("remote-routes-gate: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
