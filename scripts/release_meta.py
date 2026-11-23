#!/usr/bin/env python3
"""Release metadata generator.

Collects changelog fragments added between two git refs and parses trailer
lines to produce machine-readable release.json for the nightly updater.
"""
from __future__ import annotations

import argparse
import json
import re
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

    # Filter out ONLY changelog.d/README.md at the top level
    fragments = [f for f in fragments if not (f.parent == Path("changelog.d") and f.name == "README.md")]

    # Sort for deterministic output
    fragments.sort(key=lambda p: str(p))
    return fragments


def _read_fragment_at_rev(repo_dir: Path, fragment: Path, rev: str) -> str:
    """Read fragment content at a specific revision."""
    try:
        return _run_git(["show", f"{rev}:{fragment}"], repo_dir)
    except subprocess.CalledProcessError as e:
        # Re-raise to be caught by main()
        raise
    except UnicodeDecodeError as e:
        raise ReleaseMetaError(
            f"{fragment}: - non-UTF-8 content (invalid byte sequence)"
        ) from e


def _strip_trailer_markers(line: str) -> str:
    """Strip leading whitespace, list markers and emphasis from a line."""
    # Strip leading whitespace
    stripped = line.strip()

    # Strip UTF-8 BOM only (not ZWSP - that should cause refusal)
    stripped = stripped.lstrip('\ufeff')

    # Strip emphasis FIRST: **bold**, *italic*, _underline_, `inline code`
    # Bold: **text** -> text
    stripped = re.sub(r'\*\*([^*]*)\*\*', r'\1', stripped)
    # Italic: *text* -> text (but not if part of ** already handled)
    stripped = re.sub(r'(?<!\*)\*([^*]*)\*(?!\*)', r'\1', stripped)
    # Underline: _text_ -> text
    stripped = re.sub(r'_+([^_]*)_+', r'\1', stripped)
    # Inline code: `code` -> code
    stripped = re.sub(r'`([^`]*)`', r'\1', stripped)

    # Strip ONE leading list marker (-, *, +, >) if present
    if stripped and stripped[0] in '-*+>':
        stripped = stripped[1:].strip()

    # Strip trailing whitespace
    stripped = stripped.strip()
    return stripped


def _is_real_fence(line: str) -> bool:
    """Check if a line is a real fence opener/closer (not inline code)."""
    stripped = line.lstrip()
    # Real fence: starts with ``` followed by whitespace, info string, or end of line
    # Not inline: ```text``` or ```text` (backticks appear again on same line)
    if not stripped.startswith("```"):
        return False
    rest = stripped[3:]
    # If there are more backticks on the line, it's inline code
    if "```" in rest:
        return False
    # Empty, whitespace, or info string (alphanum/-/_ followed by whitespace or end)
    return rest == "" or rest[0].isspace() or bool(re.match(r'^[\w-]+(?:\s|$)', rest))


def _looks_like_trailer(line: str) -> bool:
    """Check if a line looks like a trailer candidate (broad detection)."""
    s = line.strip()
    # Strip UTF-8 BOM and zero-width spaces for detection
    s = s.lstrip('\ufeff\u200b')
    # Remove markdown headings
    s = re.sub(r'^#+\s*', '', s)
    # Remove blockquotes
    s = re.sub(r'^>\s*', '', s)
    # Remove numbered lists
    s = re.sub(r'^\d+\.\s*', '', s)
    # Remove checkboxes
    s = re.sub(r'^-\s*\[\s*[x ]?\s*\]\s*', '', s)
    # Remove multiple list markers
    while s.startswith(('- ', '* ', '+ ', '> ')):
        s = s[2:]
    # Remove single leading list marker char (repeat)
    while s and s[0] in '-*+>':
        s = s[1:].lstrip()
    # Strip emphasis for detection
    s = re.sub(r'\*\*([^*]*)\*\*', r'\1', s)
    s = re.sub(r'(?<!\*)\*([^*]*)\*(?!\*)', r'\1', s)
    s = re.sub(r'_+([^_]*)_+', r'\1', s)
    s = re.sub(r'`([^`]*)`', r'\1', s)
    s = s.strip()
    # Check for trailer key (including plurals and full-width colon)
    return bool(re.match(r'^(migrations?|manual[-_ ]?steps?)\s*[:：]', s, re.IGNORECASE))


def _parse_trailers(content: str, fragment_path: str) -> tuple[list[MigrationInfo], list[str]]:
    """Parse trailer lines from fragment content.

    Returns:
        Tuple of (migrations, manual_steps).

    A line is a trailer candidate when, after stripping leading whitespace,
    list markers (-, *, +, >), and emphasis characters (*, _, backticks),
    it matches case-insensitively the pattern ^(migration|manual[-_ ]?step):.
    If a candidate is EXACTLY a canonical trailer line
    (Migration: one-way, Migration: reversible, Manual-Step: <non-empty text>),
    it is used. Any other candidate that matches the pattern but is not
    canonical raises ReleaseMetaError naming the file, line number and the
    canonical form to use (CLI exit 2, no file written).
    Prose that merely mentions a migration (no key: shape at the line start
    after stripping) is still ignored.
    Lines inside fenced code blocks (between ``` lines) are ignored entirely,
    neither used nor refused.
    """
    migrations: list[MigrationInfo] = []
    manual_steps: list[str] = []

    in_code_block = False

    for line_num, line in enumerate(content.splitlines(), start=1):
        # Check for fenced code block boundaries - only toggle on REAL fence lines
        if _is_real_fence(line):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        # Broad detection: does this line look like a trailer attempt?
        if not _looks_like_trailer(line):
            continue

        # It looks like a trailer - now do canonical stripping and validation
        stripped = _strip_trailer_markers(line)

        # Check for canonical Migration: form (case-sensitive key, exact value)
        if re.match(r'^Migration:\s+one-way$', stripped):
            migrations.append(MigrationInfo(fragment=fragment_path, kind="one-way"))
        elif re.match(r'^Migration:\s+reversible$', stripped):
            migrations.append(MigrationInfo(fragment=fragment_path, kind="reversible"))
        # Check for canonical Manual-Step: form (case-sensitive, exact case with hyphen)
        elif re.match(r'^Manual-Step:\s+.+$', stripped):
            value = stripped[len("Manual-Step:"):].strip()
            if value:
                manual_steps.append(value)
            # No else branch - regex ensures non-empty value after stripping
        else:
            # Candidate matches the pattern but is not canonical - raise loud refusal
            canonical = "Migration: one-way or Migration: reversible, Manual-Step: <non-empty text>"
            raise ReleaseMetaError(
                f"{fragment_path}:{line_num} - unknown trailer format; "
                f"use one of: {canonical}"
            )

    # Check for unclosed fence at end of file
    if in_code_block:
        raise ReleaseMetaError(
            f"{fragment_path}: - unclosed code fence"
        )

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

    # Strip leading 'v' from previous_ref for cleaner version output
    previous_version = previous_ref[1:] if previous_ref and previous_ref.startswith("v") else previous_ref

    # Use OrderedDict to ensure key order
    result = OrderedDict()
    result["schema"] = 1
    result["version"] = "0.0.0"  # placeholder, overridden by CLI
    result["commit"] = commit_sha
    result["previous_version"] = previous_version
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
    except UnicodeDecodeError as e:
        print(f"non-UTF-8 content: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
