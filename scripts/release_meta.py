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
import unicodedata
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
    """Get list of changelog fragments added or modified between previous_ref and rev."""
    if previous_ref is None:
        # First release: list all fragments at rev
        output = _run_git(["ls-tree", "-r", "--name-only", rev, "--", "changelog.d/"], repo_dir)
        fragments = [Path(line) for line in output.strip().splitlines() if line]
    else:
        # Diff between previous_ref and rev, added or modified files (--diff-filter=AM)
        output = _run_git(
            ["diff", "--name-only", "--diff-filter=AM", previous_ref, rev, "--", "changelog.d/"],
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
    except UnicodeDecodeError as e:
        raise ReleaseMetaError(
            f"{fragment}: - non-UTF-8 content (invalid byte sequence)"
        ) from e


def _normalize_line(line: str) -> str:
    """Normalize a line for trailer detection.

    Applies NFKC normalization, removes Unicode format (Cf) characters and soft hyphens,
    then strips whitespace.
    """
    normalized = unicodedata.normalize('NFKC', line)
    cleaned = ''.join(
        ch for ch in normalized
        if unicodedata.category(ch) != 'Cf' and ch != '\u00ad'
    )
    return cleaned.strip()


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
    """Check if a line is a real fence opener/closer (not inline code).

    A fence line must start with ``` or ~~~ after 0-3 spaces of indentation.
    4 or more spaces of indentation means it's a code block continuation, not a fence.
    Any info string is allowed after the fence markers.
    """
    # Check indentation: up to 3 spaces
    indent = 0
    for ch in line:
        if ch == ' ':
            indent += 1
            if indent > 3:
                return False
        else:
            break

    stripped = line[indent:]
    # Real fence: starts with ``` or ~~~ followed by whitespace, info string, or end of line
    # Not inline: ```text``` or ```text` (backticks appear again on same line)
    if not (stripped.startswith("```") or stripped.startswith("~~~")):
        return False
    rest = stripped[3:]
    # If there are more fence chars on the line, it's inline code
    if "```" in rest or "~~~" in rest:
        return False
    # Empty, whitespace, or any info string (non-whitespace followed by whitespace or end)
    return rest == "" or rest[0].isspace() or bool(re.match(r'^\S+(?:\s|$)', rest))


def _looks_like_trailer(line: str) -> bool:
    """Check if a line looks like a trailer candidate (broad detection).

    Uses normalized, case-folded text to detect any mention of migration or manual steps.
    """
    normalized = _normalize_line(line)
    folded = normalized.casefold()
    # Check for migration mention
    if 'migration' in folded:
        return True
    # Check for manual steps mention (manual step, manual steps, manual-step, manual_step, etc.)
    # \W matches non-word chars; underscore is a word char, so also check for underscore explicitly
    if re.search(r'manual[\W_]*steps?', folded):
        return True
    return False


def _parse_trailers(content: str, fragment_path: str) -> tuple[list[MigrationInfo], list[str]]:
    """Parse trailer lines from fragment content.

    Returns:
        Tuple of (migrations, manual_steps).

    A line is a trailer candidate when its normalized, case-folded text contains
    "migration" or matches "manual[\\W_]*steps?". If a candidate is EXACTLY a canonical
    trailer line (Migration: one-way, Migration: reversible, Manual-Step: <non-empty text>),
    after stripping markers, it is used. Any other candidate raises ReleaseMetaError
    naming the file, line number and the canonical form to use (CLI exit 2, no file written).
    Lines inside fenced code blocks (between ``` or ~~~ lines) are ignored entirely,
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


if __name__ == "__main__":
    sys.exit(main())