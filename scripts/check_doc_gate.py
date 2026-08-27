#!/usr/bin/env python3
"""Documentation-drift gate.

Blocks commits/PRs that change feature code without a matching doc update,
unless the change carries an explicit "Docs-Reviewed: <why>" trailer.

Two layers:
  invariants  -- deterministic sanity checks (Layer A). Currently: every
                 scripts/ tinyagentos/ docs/ desktop/ taosmd/ path mentioned in the
                 configured doc set actually exists on disk, AND every protected
                 doc still contains its declared required section headings.
                 Layer A runs regardless of what changed, so a doc emptied of
                 the sections it exists to hold fails the gate even when no code
                 rule fires (the "gutting hole").
  diff-gate   -- path -> doc rule engine (Layer B). A configured rule fires
                 when a *structural* change (a file added or deleted) matches
                 one of its `when_changed` globs. A rule with the opt-in
                 `on_modify = true` flag ALSO fires on a plain modification
                 (taOS card tsk-sxuexd): this is what lets a change to a
                 protected doc itself trip a content check, so a doc can never
                 be silently gutted. A fired rule is satisfied by either editing
                 one of its `require_doc` files in the same changeset, or by a
                 commit-message trailer line starting with the configured
                 trailer (default "Docs-Reviewed:"). Editing a require_doc is
                 NOT path-only: the touched doc's declared `required_headings`
                 are asserted, so a one-character edit cannot mask a gutted
                 section.

Config lives in docs/doc-gate.toml. Rules are data, not code: add more by
editing the TOML, no changes to this file required.

Usage:
    python scripts/check_doc_gate.py invariants
    python scripts/check_doc_gate.py diff-gate --staged
    python scripts/check_doc_gate.py diff-gate --base origin/master
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "docs" / "doc-gate.toml"
DEFAULT_TRAILER = "Docs-Reviewed:"

# A path-like token: one of the known repo prefixes followed by a run of
# non-whitespace / non-quoting characters. The negative lookbehind stops us
# matching a prefix that is actually embedded inside a larger path (e.g. the
# "tinyagentos/" inside "/home/<user>/tinyagentos/data/" in a deploy-layout
# table), which would otherwise falsely flag deploy-time paths that never
# exist in the repo itself.
_TOKEN_RE = re.compile(r"(?<![\w/])(?:scripts|tinyagentos|docs|desktop|taosmd)/[^\s`\"'|]+")

# Runtime data-dir paths and generated files under taosmd/ that must not be
# treated as repo-relative source references. These would otherwise match
# when the prefix is embedded in a larger path (e.g. ~/.taosmd/config.json),
# producing false positives for paths that never exist in the repo tree.
_EXCLUDED_TOKEN_RE = re.compile(
    r"^taosmd/(?:"
    r"archive|"
    r"archive-index\.db|"
    r"config\.json|"
    r"knowledge-graph\.db|"
    r"vector-memory\.db|"
    r"project\.toml|"
    r"_build_info\.py"
    r")(?:/|$)"
)

# Chars that mark a token as a glob pattern or a <placeholder> rather than a
# concrete repo path -- these are never asserted to exist.
_GLOB_OR_PLACEHOLDER_CHARS = set("*?[]{}<>$~")

# Trailing punctuation that is prose/markdown decoration, not part of the path.
_TRAILING_PUNCT = ".,;:!?)]}'\"`"


def _clean_token(raw: str) -> str | None:
    """Normalize a raw regex match into a bare repo-relative path, or None
    if it should be ignored (glob, placeholder, anchor/query fragment)."""
    token = raw
    for sep in ("#", "?"):
        if sep in token:
            token = token.split(sep, 1)[0]
    while token and token[-1] in _TRAILING_PUNCT:
        token = token[:-1]
    if not token:
        return None
    if any(c in _GLOB_OR_PLACEHOLDER_CHARS for c in token):
        return None
    return token


def extract_path_tokens(text: str) -> list[str]:
    """Pull candidate repo-relative path tokens out of doc prose."""
    tokens = []
    for m in _TOKEN_RE.finditer(text):
        cleaned = _clean_token(m.group(0))
        if cleaned and not _EXCLUDED_TOKEN_RE.match(cleaned):
            tokens.append(cleaned)
    return tokens


def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _read_headings(doc_path: Path) -> list[str]:
    """Return every markdown heading line (leading '#' preserved) in a doc.
    Lines starting with '#' inside fenced code blocks are code comments, but
    they only matter insofar as one happens to equal a declared required
    heading (see _missing_headings), which never occurs for the sections this
    gate protects."""
    if not doc_path.is_file():
        return []
    return [
        line
        for line in doc_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.lstrip().startswith("#")
    ]


def _missing_headings(repo_root: Path, doc: str, required: list[str]) -> list[str]:
    """Return the declared headings absent from `doc`. A missing or empty doc
    is not reported here -- existence is the referenced-paths job, and a doc
    that exists but is empty simply fails the (non-empty) headings it declares.
    The required list is matched as a substring of a heading line so that a
    real edit deleting a whole section is caught, while whitespace/prefix churn
    around an intact heading is not."""
    if not required:
        return []
    headings = _read_headings(repo_root / doc)
    if not headings:
        return list(required)
    missing: list[str] = []
    for req in required:
        if not any(req in h for h in headings):
            missing.append(req)
    return missing


def check_referenced_paths(repo_root: Path, files_to_scan: list[str], config: dict) -> list[str]:
    """Layer A: every scripts/tinyagentos/docs/desktop/taosmd path token mentioned in
    the configured doc set must exist on disk. A scan-target file that itself
    does not exist (e.g. a local-only, gitignored doc) is silently skipped
    rather than treated as a failure. Runtime data-dir paths and generated
    files under taosmd/ are excluded via _EXCLUDED_TOKEN_RE."""
    failures: list[str] = []
    for rel in files_to_scan:
        doc_path = repo_root / rel
        if not doc_path.is_file():
            continue
        text = doc_path.read_text(encoding="utf-8", errors="ignore")
        for token in extract_path_tokens(text):
            if not (repo_root / token).exists():
                failures.append(f"{rel} references '{token}' which does not exist in the repo")
    return failures


def check_required_headings(repo_root: Path, config: dict) -> list[str]:
    """Layer A, always-on: every protected doc must still contain its declared
    required section headings. Because this runs on the committed/working-tree
    content regardless of the changeset, gutting a doc (deleting a required
    section) fails the gate even when the change triggers no rule -- closing
    taOS #2383's escape hatches where a path-based gate passed an emptied doc."""
    failures: list[str] = []
    required_map = config.get("invariants", {}).get("required_headings", {})
    for doc, headings in required_map.items():
        for missing in _missing_headings(repo_root, doc, headings):
            failures.append(f"{doc} -- required section '{missing}' is absent")
    return failures


def _glob_match(path: str, pattern: str) -> bool:
    """Path-segment-aware glob match, unlike fnmatch (where `*` crosses `/`).

    `**` matches zero or more path segments (so a trailing `/**` also matches
    the bare parent, e.g. `a/**` matches `a`), a single `*` matches within one
    path segment only (`[^/]*`), `?` matches one non-separator character
    (`[^/]`), and every other character is matched literally.
    """
    regex_parts = []
    i = 0
    length = len(pattern)
    while i < length:
        char = pattern[i]
        if char == "*":
            if i + 1 < length and pattern[i + 1] == "*":
                # A trailing `/**` should also match the bare parent path, so
                # fold the preceding literal `/` into an optional group.
                if regex_parts and regex_parts[-1] == "/" and i + 2 == length:
                    regex_parts[-1] = "(?:/.*)?"
                else:
                    regex_parts.append(".*")
                i += 2
            else:
                regex_parts.append("[^/]*")
                i += 1
        elif char == "?":
            regex_parts.append("[^/]")
            i += 1
        else:
            regex_parts.append(re.escape(char))
            i += 1
    return re.fullmatch("".join(regex_parts), path) is not None


def _match_any(path: str, patterns: list[str]) -> bool:
    return any(_glob_match(path, pat) for pat in patterns)


def _is_test_path(path: str) -> bool:
    """A test file is never a structural feature change (a new app, route,
    etc.), so adding or removing one must not trip a doc-gate structural rule.
    Covers frontend co-located tests and Python test modules (#171)."""
    base = path.rsplit("/", 1)[-1]
    if "/__tests__/" in path or "__tests__/" in path:
        return True
    if base.startswith("test_") and base.endswith(".py"):
        return True
    return base.endswith(
        (
            ".test.tsx",
            ".test.ts",
            ".test.jsx",
            ".test.js",
            ".spec.tsx",
            ".spec.ts",
            ".spec.js",
        )
    )


def evaluate_rules(
    changed_status: list[tuple[str, str]],
    commit_messages: list[str],
    config: dict,
    repo_root: Path,
) -> list[str]:
    """Layer B: run every configured rule against a changeset.

    changed_status: list of (status, path) pairs as from `git diff
    --name-status`, e.g. [("A", "desktop/src/apps/Foo/Foo.tsx"), ("M", "x")].
    Only status "A" (added) or "D" (deleted) files count as structural change
    for the purposes of *triggering* a rule; plain modifications ("M") are
    ignored to keep the gate precise -- unless the rule opts in via
    `on_modify = true`, in which case "M" triggers too (tsk-sxuexd). Any
    changed file (any status) can *satisfy* a rule if it matches a require_doc
    glob. When a require_doc is touched to satisfy a fired rule, the touched
    doc's declared `required_headings` are asserted as well, so a path-based
    "doc was edited" can no longer mask a gutted section.
    commit_messages: full text of each commit message in range (empty list
    when there is no finalized commit yet, i.e. --staged mode).
    """
    trailer = get_trailer(config)
    rules = config.get("rules", [])
    required_headings = config.get("invariants", {}).get("required_headings", {})

    all_paths = [path for _status, path in changed_status]
    structural_paths_default = [
        path for status, path in changed_status
        if status in ("A", "D", "T") and not _is_test_path(path)
    ]

    trailer_present = any(
        line.strip().startswith(trailer) and line.strip()[len(trailer):].strip()
        for message in commit_messages
        for line in message.splitlines()
    )

    failures: list[str] = []
    for rule in rules:
        name = rule.get("name", "?")
        when_changed = rule.get("when_changed", [])
        require_doc = rule.get("require_doc", [])
        hint = rule.get("hint", "")
        on_modify = rule.get("on_modify", False)

        # Default structural set is add/delete only; on_modify adds "M" so a
        # modification to a protected doc (or handler) can trip a content check.
        if on_modify:
            structural_paths = [
                path for status, path in changed_status
                if status in ("A", "D", "M", "T") and not _is_test_path(path)
            ]
        else:
            structural_paths = structural_paths_default

        triggered = any(_match_any(p, when_changed) for p in structural_paths)
        if not triggered:
            continue

        doc_edited = any(_match_any(p, require_doc) for p in all_paths)
        # The trailer bypass is deliberate and must not be weakened: an explicit
        # "Docs-Reviewed: <why>" waiver skips the doc-update obligation
        # entirely, and with it the content assertion (the contributor swore the
        # doc was reviewed and waives it).
        if trailer_present:
            continue

        if doc_edited:
            # HOLE 2 fix: a touched require_doc is not "satisfied" by its path
            # alone. Assert the declared required sections are still present, so
            # a one-character edit cannot mask a gutted section.
            for doc in require_doc:
                if not any(_match_any(p, [doc]) for p in all_paths):
                    continue
                for missing in _missing_headings(repo_root, doc, required_headings.get(doc, [])):
                    failures.append(
                        f"{name} -- required section '{missing}' is absent from {doc}"
                    )
            continue

        failures.append(
            f"{name} -- {hint} (edit one of: {', '.join(require_doc)}, "
            f"or add a 'Docs-Reviewed: <why>' trailer)"
        )
    return failures


def _run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return result.stdout


def _parse_name_status(output: str) -> list[tuple[str, str]]:
    changed: list[tuple[str, str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        path = parts[-1]
        if status.startswith("R") or status.startswith("C"):
            old_path = parts[1]
            changed.append(("D", old_path))
            changed.append(("A", path))
        else:
            changed.append((status[0], path))
    return changed


def _git_changed_staged() -> list[tuple[str, str]]:
    return _parse_name_status(_run_git(["diff", "--cached", "--name-status"]))


def _git_changed_base(base_ref: str) -> list[tuple[str, str]]:
    return _parse_name_status(_run_git(["diff", "--name-status", f"{base_ref}...HEAD"]))


def _git_commit_messages(base_ref: str) -> list[str]:
    out = _run_git(["log", f"{base_ref}..HEAD", "--format=%B%x00"])
    return [m for m in out.split("\x00") if m.strip()]


def _check_conflict_markers(
    repo_root: Path, changed_paths: list[str], ref: str | None = None, cached: bool = False
) -> list[str]:
    """Greps each changed file for unresolved merge-conflict markers.

    For ``--staged`` the index is searched (``--cached``); for ``--base`` the
    given ref (usually ``HEAD``) is searched. A file that does not exist in the
    searched tree is silently skipped."""
    failures: list[str] = []
    for path in changed_paths:
        args = ["git", "grep", "-n", "-E", r"^(<<<<<<< |=======$|>>>>>>> )"]
        if cached:
            args.insert(4, "--cached")
        elif ref:
            args.append(ref)
        args.extend(["--", path])
        result = subprocess.run(args, cwd=repo_root, capture_output=True, text=True)
        if result.returncode == 0:
            failures.append(
                f"conflict-marker -- {path} contains unresolved merge conflict markers"
            )
    return failures


def get_trailer(config: dict) -> str:
    """Single source of truth for the commit-message trailer prefix, shared
    by the diff-gate check and the hooks (via the print-trailer command)."""
    return config.get("gate", {}).get("trailer", DEFAULT_TRAILER)


def _report(failures: list[str]) -> int:
    if not failures:
        print("doc-gate: clean")
        return 0
    for failure in failures:
        print(f"DOC-GATE FAIL: {failure}")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("invariants", help="Run Layer-A deterministic checks")
    sub.add_parser("print-trailer", help="Print the configured commit trailer prefix")

    diff_parser = sub.add_parser("diff-gate", help="Run Layer-B path->doc rule engine")
    group = diff_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--staged", action="store_true", help="Check the git index (pre-commit)")
    group.add_argument("--base", help="Compare <base>...HEAD (CI / commit-msg)")

    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "invariants":
        ic = config.get("invariants", {})
        files_to_scan = ic.get("referenced_paths_scan", [])
        failures = check_referenced_paths(REPO_ROOT, files_to_scan, config)
        failures.extend(check_required_headings(REPO_ROOT, config))
        return _report(failures)

    if args.command == "print-trailer":
        print(get_trailer(config))
        return 0

    # diff-gate
    if args.staged:
        changed = _git_changed_staged()
        commit_messages: list[str] = []
    else:
        changed = _git_changed_base(args.base)
        commit_messages = _git_commit_messages(args.base)

    failures = evaluate_rules(changed, commit_messages, config, REPO_ROOT)
    all_paths = [path for _status, path in changed]
    if args.staged:
        failures.extend(_check_conflict_markers(REPO_ROOT, all_paths, cached=True))
    else:
        failures.extend(_check_conflict_markers(REPO_ROOT, all_paths, ref="HEAD"))
    return _report(failures)


if __name__ == "__main__":
    sys.exit(main())
