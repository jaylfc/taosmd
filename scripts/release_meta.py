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


# Confusables: Cyrillic and Greek lookalikes for a, c, e, i, o, p, x, y, M, T, H.
_CONFUSABLES = {
    "\u0430": "a",  # Cyrillic a
    "\u0441": "c",  # Cyrillic es
    "\u0435": "e",  # Cyrillic ie
    "\u0454": "e",  # Cyrillic Ukrainian ie
    "\u0456": "i",  # Cyrillic Ukrainian i
    "\u0457": "i",  # Cyrillic yi
    "\u0458": "j",
    "\u043e": "o",  # Cyrillic o
    "\u0440": "p",  # Cyrillic er
    "\u0445": "x",  # Cyrillic ha
    "\u0443": "y",  # Cyrillic u (looks like y)
    "\u041c": "M",  # Cyrillic Em
    "\u0422": "T",  # Cyrillic Te
    "\u041d": "H",  # Cyrillic En
    "\u0391": "A",  # Greek Alpha
    "\u0392": "B",  # Greek Beta
    "\u0395": "E",  # Greek Epsilon
    "\u0397": "H",  # Greek Eta
    "\u0399": "I",  # Greek Iota
    "\u039a": "K",  # Greek Kappa
    "\u039c": "M",  # Greek Mu
    "\u039d": "N",  # Greek Nu
    "\u039f": "O",  # Greek Omicron
    "\u03a1": "P",  # Greek Rho
    "\u03a4": "T",  # Greek Tau
    "\u03a7": "X",  # Greek Chi
    "\u03a5": "Y",  # Greek Upsilon
    "\u03b1": "a",  # Greek alpha
    "\u03bf": "o",  # Greek omicron
    "\u03c1": "p",  # Greek rho
    "\u03c7": "x",  # Greek chi
    "\u03c5": "y",  # Greek upsilon
    "\u03b5": "e",  # Greek epsilon
    "\u03b9": "i",  # Greek iota
    "\u03ba": "k",  # Greek kappa
    "\u03bd": "v",
    "\u03bc": "m",
}

_CANONICAL_MIGRATION = "Migration: one-way"
_CANONICAL_REVERSIBLE = "Migration: reversible"
_CANONICAL_MANUAL_PREFIX = "Manual-Step: "


def _canonicalize_trailer(stripped: str) -> str | None:
    """Return canonical trailer string when a trailer attempt is canonicalizable."""
    m = re.match(r"^(migrations?|manual[\W_]*steps?)\s*([:=]|[\u0080-\uffff])\s*(.*)$", stripped, re.IGNORECASE)
    if not m:
        return None
    key_raw, sep_raw, value = m.group(1), m.group(2), m.group(3)
    sep = unicodedata.normalize("NFKC", sep_raw)
    if sep not in ":=":
        if "COLON" not in unicodedata.name(sep_raw, ""):
            return None
        sep = ":"
    if sep != ":":
        return None
    key_norm = re.sub(r"[\W_]+", "", key_raw).casefold()
    if key_norm in ("migration", "migrations"):
        if value.strip() == "one-way":
            return _CANONICAL_MIGRATION
        if value.strip() == "reversible":
            return _CANONICAL_REVERSIBLE
        return None
    if key_norm == "manualstep":
        value = value.strip()
        if value:
            return f"{_CANONICAL_MANUAL_PREFIX}{value}"
    if key_norm == "manualsteps":
        value = value.strip()
        if value:
            return f"{_CANONICAL_MANUAL_PREFIX}{value}"
    return None


def _normalize_line(line: str) -> str:
    """Normalize a line for trailer detection.

    NFKC; remove every Unicode format (Cf) character and U+00AD; strip.
    """
    normalized = unicodedata.normalize("NFKC", line)
    cleaned = "".join(
        ch for ch in normalized
        if unicodedata.category(ch) != "Cf" and ch != "\u00ad"
    )
    return cleaned.strip()


def _is_colon_like(ch: str) -> bool:
    """Colon-like character: ':', fullwidth colon and other Unicode colon forms after NFKC, or '='."""
    if ch in ":=":
        return True
    normalized = unicodedata.normalize("NFKC", ch)
    if normalized in ":=":
        return True
    name = unicodedata.name(ch, "")
    return "COLON" in name and unicodedata.category(ch).startswith("P")


def _strip_leading_markup(line: str) -> str:
    """Strip a leading run of markup as defined by the trailer rule."""
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch.isspace() or ch in ">#-*+_~`|":
            i += 1
            continue
        if line.startswith("<!--", i):
            i += 4
            continue
        if ch == "[" and i + 2 < n and line[i + 1] in " xX" and line[i + 2] == "]":
            i += 3
            continue
        m = re.match(r"[0-9A-Za-z]+[.)]", line[i:])
        if m:
            i += m.end()
            continue
        break
    return line[i:].rstrip()


def _deconfuse(text: str) -> str:
    out = []
    for ch in text:
        if ch.isascii():
            out.append(ch)
        else:
            out.append(_CONFUSABLES.get(ch, ch))
    return "".join(out)


def _is_homoglyph_trailer_attempt(normalized_line: str) -> bool:
    r"""Refuse confusable-key attempts with a colon-like separator.

    Uses the same key pattern as the normal trailer rule (migrations? and
    manual[\W_]*steps?) after mapping confusables to ASCII. Only flags lines
    where the key portion contains non-ASCII characters that map to the
    canonical key pattern.
    """
    stripped = _strip_leading_markup(normalized_line)
    if not stripped:
        return False

    # First check if the stripped line (as-is) matches the trailer key pattern
    # with a colon-like separator. If it does, it's a normal attempt, not a
    # homoglyph attempt (the normal path will handle it).
    if _trailer_attempt_after_strip(stripped):
        return False

    # Now check if deconfusing the line makes it match the trailer key pattern.
    # We need to find the key portion in the original (with potential homoglyphs)
    # and verify it contained non-ASCII characters.
    deconfused = _deconfuse(stripped)
    m = _TRAILER_KEY_RE.match(deconfused)
    if not m:
        return False

    # The key matched in the deconfused version. Now check if the corresponding
    # portion in the original stripped line contained any non-ASCII characters.
    key_end = m.end()
    original_key_portion = stripped[:key_end]
    if original_key_portion.isascii():
        return False

    # Verify there's a colon-like separator after the key in the original
    rest = stripped[key_end:].lstrip()
    return bool(rest) and _is_colon_like(rest[0])


_TRAILER_KEY_RE = re.compile(r"(migrations?|manual[\W_]*steps?)", re.IGNORECASE)


def _trailer_attempt_after_strip(stripped: str) -> bool:
    m = _TRAILER_KEY_RE.match(stripped)
    if not m:
        return False
    rest = stripped[m.end():].lstrip()
    return bool(rest) and _is_colon_like(rest[0])


def _is_trailer_attempt(line: str) -> tuple[bool, str, str]:
    """Return (is_attempt, normalized_line, stripped_after_markup)."""
    normalized = _normalize_line(line)
    stripped = _strip_leading_markup(normalized)
    return _trailer_attempt_after_strip(stripped), normalized, stripped


def _strip_trailer_markers(line: str) -> str:
    """Strip leading/trailing markup for canonical trailer parsing."""
    normalized = _normalize_line(line)
    stripped = _strip_leading_markup(normalized)
    # Leading markup stripping can consume opening emphasis markers; remove any
    # remaining symmetric wrappers around the canonical trailer text.
    for _ in range(3):
        prev = stripped
        stripped = stripped.strip()
        if len(stripped) >= 4 and stripped.startswith("**") and stripped.endswith("**"):
            stripped = stripped[2:-2].strip()
        elif len(stripped) >= 4 and stripped.startswith("__") and stripped.endswith("__"):
            stripped = stripped[2:-2].strip()
        elif len(stripped) >= 2 and stripped.startswith("`") and stripped.endswith("`"):
            stripped = stripped[1:-1].strip()
        elif len(stripped) >= 2 and stripped[0] in "*_" and stripped.endswith(stripped[0]):
            stripped = stripped[1:-1].strip()
        stripped = stripped.rstrip()
        while stripped and stripped[-1] in "*_`~>|":
            stripped = stripped[:-1].rstrip()
        if stripped.endswith("-->"):
            stripped = stripped[:-3].rstrip()
        if stripped.endswith("--"):
            stripped = stripped[:-2].rstrip()
        if stripped == prev:
            break
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
    """Check if a line is a trailer attempt under the strict rule."""
    is_attempt, normalized, _ = _is_trailer_attempt(line)
    return is_attempt or _is_homoglyph_trailer_attempt(normalized)


def _parse_trailers(content: str, fragment_path: str) -> tuple[list[MigrationInfo], list[str]]:
    """Parse trailer lines from fragment content.

    A line is a trailer attempt if, after normalisation and markup stripping, it
    starts with migration/migrations or manual/manual-step/manual_step/manual steps
    followed by optional whitespace and a colon-like character. Canonical attempts
    parse; any other trailer attempt refuses with exit 2. A line that is not a
    trailer attempt is ignored.
    """
    migrations: list[MigrationInfo] = []
    manual_steps: list[str] = []

    in_code_block = False

    for line_num, line in enumerate(content.splitlines(), start=1):
        if _is_real_fence(line):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        is_attempt, normalized, _ = _is_trailer_attempt(line)
        if not is_attempt:
            if _is_homoglyph_trailer_attempt(normalized):
                canonical = "Migration: one-way or Migration: reversible, Manual-Step: <non-empty text>"
                raise ReleaseMetaError(
                    f"{fragment_path}:{line_num} - confusable trailer key; "
                    f"use one of: {canonical}"
                )
            continue

        stripped = _strip_trailer_markers(line)
        canonical = _canonicalize_trailer(stripped)
        if canonical == _CANONICAL_MIGRATION:
            migrations.append(MigrationInfo(fragment=fragment_path, kind="one-way"))
        elif canonical == _CANONICAL_REVERSIBLE:
            migrations.append(MigrationInfo(fragment=fragment_path, kind="reversible"))
        elif canonical and canonical.startswith(_CANONICAL_MANUAL_PREFIX):
            manual_steps.append(canonical[len(_CANONICAL_MANUAL_PREFIX):])
        else:
            canonical_form = "Migration: one-way or Migration: reversible, Manual-Step: <non-empty text>"
            raise ReleaseMetaError(
                f"{fragment_path}:{line_num} - unknown trailer format; "
                f"use one of: {canonical_form}"
            )

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