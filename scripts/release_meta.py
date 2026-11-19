#!/usr/bin/env python3
"""Release metadata generator.

Collects changelog fragments added between two git refs and parses trailer
lines to produce machine-readable release.json for the nightly updater.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ReleaseMetaError(Exception):
    """Raised when release metadata collection fails."""
    pass


@dataclass
class MigrationInfo:
    fragment: str
    kind: str  # "one-way" or "reversible"


def _run_git(args: list[str], cwd: Path) -> str:
    """Run git command and return stdout."""
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True,
    )
    return result.stdout


def _get_commit_sha(repo_dir: Path, rev: str) -> str:
    """Get full commit SHA for a revision."""
    return _run_git(["rev-parse", rev], repo_dir).strip()


def _get_added_fragments(repo_dir: Path, previous_ref: str | None, rev: str) -> list[Path]:
    """Get list of changelog fragments added between previous_ref and rev."""
    if previous_ref is None:
        # First release: list all fragments at rev
        output = _run_git(["ls-tree", "-r", "--name-only", rev, "--", "changelog.d/"], repo_dir)
        fragments = [Path(line) for line in output.strip().splitlines() if line]
    else:
        # Diff between previous_ref and rev, only added files (--diff-filter=A)
        output = _run_git(
            ["diff", "--name-only", "--diff-filter=A", previous_ref, rev, "--", "changelog.d/"],
            repo_dir,
        )
        fragments = [Path(line) for line in output.strip().splitlines() if line]

    # Filter out README.md
    fragments = [f for f in fragments if f.name != "README.md"]

    # Sort for deterministic output
    fragments.sort(key=lambda p: str(p))
    return fragments


def _read_fragment_at_rev(repo_dir: Path, fragment: Path, rev: str) -> str:
    """Read fragment content at a specific revision."""
    return _run_git(["show", f"{rev}:{fragment}"], repo_dir)


def _parse_trailers(content: str, fragment_path: str) -> tuple[list[MigrationInfo], list[str]]:
    """Parse trailer lines from fragment content.

    Returns:
        Tuple of (migrations, manual_steps).
    """
    migrations: list[MigrationInfo] = []
    manual_steps: list[str] = []

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("Migration:"):
            value = line[len("Migration:"):].strip()
            if value not in ("one-way", "reversible"):
                raise ReleaseMetaError(
                    f"{fragment_path}: invalid Migration value '{value}' (must be 'one-way' or 'reversible')"
                )
            migrations.append(MigrationInfo(fragment=fragment_path, kind=value))
        elif line.startswith("Manual-Step:"):
            value = line[len("Manual-Step:"):].strip()
            if not value:
                raise ReleaseMetaError(
                    f"{fragment_path}: empty Manual-Step value"
                )
            manual_steps.append(value)

    return migrations, manual_steps


def collect(repo_dir: str | Path, previous_ref: str | None, rev: str = "HEAD") -> dict[str, Any]:
    """Collect release metadata from changelog fragments.

    Args:
        repo_dir: Path to the git repository.
        previous_ref: Previous release tag/ref, or None for first release.
        rev: Git revision to collect from (default: HEAD).

    Returns:
        Dictionary with release metadata.

    Raises:
        ReleaseMetaError: If fragment parsing fails.
    """
    repo_path = Path(repo_dir)
    commit_sha = _get_commit_sha(repo_path, rev)
    fragments = _get_added_fragments(repo_path, previous_ref, rev)

    all_migrations: list[MigrationInfo] = []
    all_manual_steps: list[str] = []

    for fragment in fragments:
        content = _read_fragment_at_rev(repo_path, fragment, rev)
        fragment_str = str(fragment)
        migrations, manual_steps = _parse_trailers(content, fragment_str)
        all_migrations.extend(migrations)
        all_manual_steps.extend(manual_steps)

    one_way_migration = any(m.kind == "one-way" for m in all_migrations)
    manual_step = len(all_manual_steps) > 0

    # Use OrderedDict to ensure key order
    result = OrderedDict()
    result["schema"] = 1
    result["version"] = "0.0.0"  # placeholder, overridden by CLI
    result["commit"] = commit_sha
    result["previous_version"] = previous_ref
    result["one_way_migration"] = one_way_migration
    result["manual_step"] = manual_step
    result["manual_steps"] = all_manual_steps
    result["migrations"] = [{"fragment": m.fragment, "kind": m.kind} for m in all_migrations]
    result["fragments"] = [str(f) for f in fragments]

    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate release metadata from changelog fragments")
    parser.add_argument("--version", required=True, help="Release version (e.g., 0.5.1)")
    parser.add_argument("--previous-ref", dest="previous_ref", help="Previous release tag/ref")
    parser.add_argument("--out", required=True, help="Output JSON file path")
    parser.add_argument("--rev", default="HEAD", help="Git revision to collect from (default: HEAD)")
    args = parser.parse_args(argv)

    try:
        repo_dir = Path.cwd()
        data = collect(repo_dir, args.previous_ref, args.rev)
        data["version"] = args.version

        out_path = Path(args.out)
        out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

        print(f"one_way_migration={str(data['one_way_migration']).lower()} manual_step={str(data['manual_step']).lower()}")
        return 0

    except ReleaseMetaError as e:
        print(str(e), file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as e:
        print(f"git command failed: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())