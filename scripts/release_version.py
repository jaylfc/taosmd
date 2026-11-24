#!/usr/bin/env python3
"""Compute and stamp the 0.5.N release version from the v0.5.0 anchor tag.

Every merge to master that passes CI becomes a numbered release by itself
(tag + package publish). The version scheme is 0.5.N where N is the number of
first-parent commits on master after the anchor tag v0.5.0 (placed by hand on
the catch-up commit; the first automated release is 0.5.1). First-parent
counting means a merged PR with 12 commits adds 1, not 12.

Standard library only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_ANCHOR = "v0.5.0"
DEFAULT_REV = "HEAD"

_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_VERSION_LINE_RE = re.compile(r'^(\s*version\s*=\s*)"[^"]*"(\s*)$')


class ReleaseVersionError(Exception):
    """Raised when the version cannot be computed or stamped."""


def _run(repo_dir, *args):
    """Run a git command in *repo_dir* and return the CompletedProcess."""
    return subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        capture_output=True,
        text=True,
    )



def _resolve_repo_root(repo_dir):
    """Resolve *repo_dir* to its git toplevel, if it is inside a work tree."""
    repo_dir = Path(repo_dir)
    result = _run(repo_dir, "rev-parse", "--show-toplevel")
    if result.returncode != 0 or not result.stdout.strip():
        raise ReleaseVersionError(f"not a git repository: {repo_dir}")
    return Path(result.stdout.strip())


def compute_version(repo_dir, anchor=DEFAULT_ANCHOR, rev=DEFAULT_REV):
    """Return the 0.5.N version for *rev* relative to *anchor* in *repo_dir*."""
    repo_dir = _resolve_repo_root(repo_dir)

    # Anchor tag must exist.
    if _run(repo_dir, "rev-parse", "--verify", "--quiet", anchor).returncode != 0:
        raise ReleaseVersionError(
            f"anchor tag {anchor!r} not found in {repo_dir}"
        )

    # rev must be a descendant of the anchor.
    if _run(repo_dir, "merge-base", "--is-ancestor", anchor, rev).returncode != 0:
        raise ReleaseVersionError(
            f"rev {rev!r} is not a descendant of anchor {anchor!r}"
        )

    count = int(
        _run(
            repo_dir,
            "rev-list",
            "--count",
            "--first-parent",
            f"{anchor}..{rev}",
        ).stdout.strip()
    )
    if count == 0:
        raise ReleaseVersionError(
            f"rev {rev!r} is the anchor itself ({anchor}); that commit is "
            "0.5.0 itself and is released by hand"
        )
    return f"0.5.{count}"


def _project_version_indices(lines):
    """Return indices of `version = "..."` lines inside the [project] table."""
    in_project = False
    indices = []
    for index, line in enumerate(lines):
        section = _SECTION_RE.match(line)
        if section:
            in_project = section.group(1) == "project"
            continue
        if in_project and _VERSION_LINE_RE.match(line):
            indices.append(index)
    return indices


def stamp(repo_dir, version, commit, built_at):
    """Rewrite the [project] version line and write taosmd/_build_info.py."""
    repo_dir = _resolve_repo_root(repo_dir)
    pyproject = repo_dir / "pyproject.toml"
    try:
        lines = pyproject.read_text(encoding="utf-8").split("\n")
    except OSError as exc:
        raise ReleaseVersionError(f"cannot read {pyproject}: {exc}")

    indices = _project_version_indices(lines)
    if len(indices) != 1:
        raise ReleaseVersionError(
            f"expected exactly one version = \"...\" line in the [project] "
            f"table of {pyproject}, found {len(indices)}"
        )

    index = indices[0]
    match = _VERSION_LINE_RE.match(lines[index])
    lines[index] = f'{match.group(1)}"{version}"{match.group(2)}'
    try:
        pyproject.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        raise ReleaseVersionError(f"cannot write {pyproject}: {exc}")

    build_info = repo_dir / "taosmd" / "_build_info.py"
    try:
        build_info.parent.mkdir(parents=True, exist_ok=True)
        build_info.write_text(
            f"COMMIT = '{commit}'\nBUILT_AT = '{built_at}'\n", encoding="utf-8"
        )
    except OSError as exc:
        raise ReleaseVersionError(f"cannot write {build_info}: {exc}")


def _utc_now():
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="scripts/release_version.py",
        description="Compute and stamp the 0.5.N release version.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--print", action="store_true")
    group.add_argument("--stamp", action="store_true")
    parser.add_argument("--anchor", default=DEFAULT_ANCHOR)
    parser.add_argument("--repo", default=".")
    args = parser.parse_args(argv)

    try:
        version = compute_version(args.repo, anchor=args.anchor, rev=DEFAULT_REV)
    except ReleaseVersionError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.stamp:
        commit = _run(args.repo, "rev-parse", "HEAD").stdout.strip()
        try:
            stamp(args.repo, version, commit, _utc_now())
        except ReleaseVersionError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
