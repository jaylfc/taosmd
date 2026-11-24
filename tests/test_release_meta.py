#!/usr/bin/env python3
"""Tests for scripts/release_meta.py.

Every test builds a REAL throwaway git repo in tmp_path (subprocess git;
tag a first commit, add fragments in later commits) and calls the real
functions and the real CLI.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "release_meta.py"


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


def _init_repo(repo: Path) -> None:
    _git(["init"], repo)
    _git(["config", "user.email", "test@test.com"], repo)
    _git(["config", "user.name", "Test"], repo)


def _write_fragment(repo: Path, name: str, content: str) -> Path:
    path = repo / "changelog.d" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _commit_all(repo: Path, message: str) -> None:
    _git(["add", "."], repo)
    _git(["commit", "-m", message], repo)


def _tag(repo: Path, tag: str) -> None:
    _git(["tag", tag], repo)


def _run_collect(repo: Path, previous_ref: str | None, rev: str = "HEAD") -> dict:
    from scripts.release_meta import collect
    return collect(repo, previous_ref, rev)


def _run_cli(repo: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    env = {**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)}
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        env=env,
    )


class TestCollect:
    def test_two_plain_fragments_after_tag(self, tmp_path):
        """(a) two plain fragments added after the tag -> both flags false,
        fragments lists exactly those two, a fragment that existed BEFORE the
        tag is not listed."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)

        # Create initial fragment BEFORE the tag
        _write_fragment(repo, "old-fragment.md", "### Fixed\n\nOld fix\n")
        _commit_all(repo, "initial commit with old fragment")
        _tag(repo, "v0.5.0")

        # Add two new fragments AFTER the tag
        _write_fragment(repo, "new-fragment-1.md", "### Added\n\nNew feature 1\n")
        _write_fragment(repo, "new-fragment-2.md", "### Fixed\n\nNew fix 2\n")
        _commit_all(repo, "add two new fragments")

        result = _run_collect(repo, "v0.5.0")

        assert result["schema"] == 1
        # version is a placeholder in collect(); CLI overrides it
        assert result["version"] == "0.0.0"
        assert result["previous_version"] == "0.5.0"
        assert result["one_way_migration"] is False
        assert result["manual_step"] is False
        assert result["manual_steps"] == []
        assert result["migrations"] == []
        assert sorted(result["fragments"]) == sorted([
            "changelog.d/new-fragment-1.md",
            "changelog.d/new-fragment-2.md",
        ])
        # old-fragment.md must NOT be in fragments
        assert "changelog.d/old-fragment.md" not in result["fragments"]

    def test_one_way_migration_and_manual_step(self, tmp_path):
        """(b) one fragment with `Migration: one-way` and one with
        `Manual-Step: stop the service first` -> one_way_migration true,
        manual_step true, manual_steps == ['stop the service first'],
        migrations has the one-way entry."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)

        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        _write_fragment(repo, "migration-one-way.md", "### Changed\n\nSchema change\n\nMigration: one-way\n")
        _write_fragment(repo, "manual-step.md", "### Added\n\nNew feature\n\nManual-Step: stop the service first\n")
        _commit_all(repo, "add migration and manual step fragments")

        result = _run_collect(repo, "v0.5.0")

        assert result["one_way_migration"] is True
        assert result["manual_step"] is True
        assert result["manual_steps"] == ["stop the service first"]
        assert len(result["migrations"]) == 1
        assert result["migrations"][0]["fragment"] == "changelog.d/migration-one-way.md"
        assert result["migrations"][0]["kind"] == "one-way"

    def test_reversible_migration_only(self, tmp_path):
        """(c) `Migration: reversible` only -> one_way_migration false and
        the migration is listed with kind reversible."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)

        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        _write_fragment(repo, "reversible.md", "### Changed\n\nSchema change\n\nMigration: reversible\n")
        _commit_all(repo, "add reversible migration fragment")

        result = _run_collect(repo, "v0.5.0")

        assert result["one_way_migration"] is False
        assert result["manual_step"] is False
        assert len(result["migrations"]) == 1
        assert result["migrations"][0]["fragment"] == "changelog.d/reversible.md"
        assert result["migrations"][0]["kind"] == "reversible"

    def test_migration_typo_exits_with_error(self, tmp_path):
        """(d) `Migration: oneway` (typo) -> CLI exit 2, stderr names the
        file and the value, no output file."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)

        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        _write_fragment(repo, "bad-migration.md", "### Changed\n\nSchema change\n\nMigration: oneway\n")
        _commit_all(repo, "add bad migration fragment")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        assert result.returncode == 2
        assert "bad-migration.md" in result.stderr
        assert "unknown trailer format" in result.stderr
        assert not out_file.exists()

    def test_empty_manual_step_exits_with_error(self, tmp_path):
        """(d) Empty `Manual-Step:` -> CLI exit 2, stderr names the file and
        the value, no output file."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)

        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        _write_fragment(repo, "empty-manual.md", "### Added\n\nNew feature\n\nManual-Step: \n")
        _commit_all(repo, "add empty manual step fragment")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        assert result.returncode == 2
        assert "empty-manual.md" in result.stderr
        assert "Manual-Step" in result.stderr
        assert not out_file.exists()

    def test_modified_fragment_counted(self, tmp_path):
        """(e) a fragment MODIFIED after the tag IS counted (--diff-filter=AM);
        changelog.d/README.md is ignored."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)

        # Create fragment and README.md BEFORE the tag
        _write_fragment(repo, "existing.md", "### Fixed\n\nOld fix\n")
        (repo / "changelog.d" / "README.md").write_text("# Changelog\n")
        _commit_all(repo, "initial commit with existing fragment and readme")
        _tag(repo, "v0.5.0")

        # MODIFY the existing fragment to add a trailer
        _write_fragment(repo, "existing.md", "### Fixed\n\nOld fix updated\n\nMigration: one-way\n")
        # Add a NEW fragment
        _write_fragment(repo, "new-fragment.md", "### Added\n\nNew feature\n")
        _commit_all(repo, "modify existing and add new")

        result = _run_collect(repo, "v0.5.0")

        # Both modified and newly added fragments should be listed
        assert sorted(result["fragments"]) == sorted([
            "changelog.d/existing.md",
            "changelog.d/new-fragment.md",
        ])
        # README.md must never be listed
        assert "changelog.d/README.md" not in result["fragments"]
        # The trailer in the modified fragment should be parsed
        assert result["one_way_migration"] is True
        assert len(result["migrations"]) == 1
        assert result["migrations"][0]["fragment"] == "changelog.d/existing.md"
        assert result["migrations"][0]["kind"] == "one-way"

    def test_no_previous_ref_first_release(self, tmp_path):
        """(f) no --previous-ref -> every fragment at HEAD is used and
        previous_version is null."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)

        # Add fragments without any tag
        _write_fragment(repo, "fragment-1.md", "### Added\n\nFeature 1\n")
        _write_fragment(repo, "fragment-2.md", "### Fixed\n\nFix 2\n")
        _commit_all(repo, "initial commit with two fragments")

        result = _run_collect(repo, None)

        assert result["previous_version"] is None
        assert sorted(result["fragments"]) == sorted([
            "changelog.d/fragment-1.md",
            "changelog.d/fragment-2.md",
        ])

    def test_prose_migration_mention_ignored(self, tmp_path):
        """Prose mentioning migration is not a trailer attempt; it is ignored."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")
        _write_fragment(repo, "prose.md", "### Changed\n\nThis is not a one-way migration\n")
        _commit_all(repo, "add prose fragment")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}: {result.stderr}"
        assert "one_way_migration=false" in result.stdout
        assert out_file.exists()

    def test_fenced_code_block_ignored(self, tmp_path):
        """(d) Migration: one-way inside a fenced code block is ignored
        (exit 0, flag false) and does not refuse."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")
        _write_fragment(repo, "fenced.md", "### Changed\n\n```\nMigration: one-way\n```\n")
        _commit_all(repo, "add fenced fragment")

        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(repo / "release.json")])

        assert result.returncode == 0
        assert "one_way_migration=false" in result.stdout

    def test_inline_backticks_do_not_open_a_fence(self, tmp_path):
        """(a) inline backticks do not open a fence: fragment lines 'use ```x``` here' then
        'Migration: one-way' -> exit 0, one_way_migration true."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        _write_fragment(repo, "inline-backticks.md", "### Changed\n\nuse ```x``` here\n\nMigration: one-way\n")
        _commit_all(repo, "add inline backticks fragment")

        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(repo / "release.json")])

        assert result.returncode == 0
        assert "one_way_migration=true" in result.stdout

    def test_real_fence_still_ignored(self, tmp_path):
        """(b) real fence lines are ignored: '```' line, 'Migration: one-way', '```' line
        -> exit 0, flag false."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        _write_fragment(repo, "real-fence.md", "### Changed\n\n```\nMigration: one-way\n```\n")
        _commit_all(repo, "add real fence fragment")

        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(repo / "release.json")])

        assert result.returncode == 0
        assert "one_way_migration=false" in result.stdout

    def test_fence_with_info_string(self, tmp_path):
        """(c) fence with info string is ignored: '```python' ... '```' around a trailer
        -> ignored."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        _write_fragment(repo, "fence-with-info.md", "### Changed\n\n```python\nMigration: one-way\n```\n")
        _commit_all(repo, "add fence with info fragment")

        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(repo / "release.json")])

        assert result.returncode == 0
        assert "one_way_migration=false" in result.stdout

    def test_unclosed_fence_refused(self, tmp_path):
        """(d) unclosed fence is refused: a '```' line with no closing fence -> exit 2,
        stderr names the file and 'unclosed', no output file."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        _write_fragment(repo, "unclosed-fence.md", "### Changed\n\n```\nMigration: one-way\n")
        _commit_all(repo, "add unclosed fence fragment")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        assert result.returncode == 2
        assert "unclosed-fence.md" in result.stderr
        assert "unclosed" in result.stderr.lower()
        assert not out_file.exists()

    # RED tests for tsk-v7rly2 - these should FAIL before the fix

    def test_bom_before_trailer(self, tmp_path):
        """(1) UTF-8 BOM before the trailer: strip the BOM and parse correctly."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        # Write fragment with UTF-8 BOM on the trailer line itself
        content = "### Changed\n\nSchema change\n\n\ufeffMigration: one-way\n"
        path = repo / "changelog.d" / "bom-fragment.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        _commit_all(repo, "add BOM fragment")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        # Should succeed (exit 0) and parse the trailer correctly
        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}: {result.stderr}"
        assert "one_way_migration=true" in result.stdout

    def test_subdir_readme_trailer_parsed(self, tmp_path):
        """(2) Trailer in changelog.d/<subdir>/README.md must be parsed (only top-level README.md skipped)."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        # Create subdir with README.md containing a trailer
        subdir_readme = repo / "changelog.d" / "subdir" / "README.md"
        subdir_readme.parent.mkdir(parents=True, exist_ok=True)
        subdir_readme.write_text("# Subdir\n\nMigration: one-way\n", encoding="utf-8")
        _commit_all(repo, "add subdir README with trailer")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        # Should succeed (exit 0) and parse the trailer correctly
        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}: {result.stderr}"
        assert "one_way_migration=true" in result.stdout

    def test_inline_backticks_not_fence(self, tmp_path):
        """(3) Inline ``` span does not toggle fence state: two inline ```a``` ```b``` should not hide content between them."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        # Two inline backtick spans with a trailer between them
        # Current bug: inline ``` toggles fence, so trailer is hidden
        content = "### Changed\n\n```a```\nMigration: one-way\n```b```\n"
        _write_fragment(repo, "inline-fence.md", content)
        _commit_all(repo, "add inline fence fragment")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        # Should parse the trailer (exit 0, one_way_migration=true)
        # Current bug: returns one_way_migration=false
        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}: {result.stderr}"
        assert "one_way_migration=true" in result.stdout, f"Expected one_way_migration=true, got stdout: {result.stdout}"

    @pytest.mark.parametrize("fragment_name, fragment_content, should_parse", [
        ("near-miss-gt-dash.md", "### Changed\n\n> - Migration: one-way\n", True),
        ("near-miss-numbered.md", "### Changed\n\n1. Migration: one-way\n", True),
        ("near-miss-checkbox.md", "### Changed\n\n- [ ] Migration: one-way\n", True),
        ("near-miss-h2.md", "### Changed\n\n## Migration: one-way\n", True),
        ("near-miss-fullwidth-colon.md", "### Changed\n\nMigration： one-way\n", True),
        ("near-miss-zwsp.md", "### Changed\n\n\u200bMigration: one-way\n", True),
        ("near-miss-plural-migrations.md", "### Changed\n\nMigrations: one-way\n", True),
        ("near-miss-plural-manual.md", "### Changed\n\nManual-Steps: do something\n", True),
        ("near-miss-not-attempt.md", "### Changed\n\nThis plan includes a migration step later\n", "ignore"),
        ("near-miss-migrating-prose.md", "### Changed\n\nmigrating old stores is handled automatically\n", "ignore"),
    ])
    def test_near_miss_prefixes(self, tmp_path, fragment_name, fragment_content, should_parse):
        """Trailer attempts that normalize to canonical parse; prose mentions are ignored."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        _write_fragment(repo, fragment_name, fragment_content)
        _commit_all(repo, f"add near-miss fragment {fragment_name}")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        if should_parse == "ignore":
            assert result.returncode == 0, f"Expected exit 0 for {fragment_name}: {result.stderr}"
            assert "one_way_migration=false" in result.stdout
            assert "manual_step=false" in result.stdout
        elif should_parse:
            assert result.returncode == 0, f"Expected exit 0 for {fragment_name}: {result.stderr}"
            assert "one_way_migration=true" in result.stdout or "manual_step=true" in result.stdout
        else:
            assert result.returncode == 2, f"Expected exit 2 for {fragment_name}, got {result.returncode}: {result.stdout}"
            assert not out_file.exists()

    def test_non_utf8_fragment_refused(self, tmp_path):
        """(5) Non-UTF8 fragment gives exit-2 refusal, not uncaught traceback."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        # Write fragment with invalid UTF-8 (latin-1 bytes)
        path = repo / "changelog.d" / "non-utf8.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"### Changed\n\nMigration: one-way\n\xff\xfe")  # invalid UTF-8
        _commit_all(repo, "add non-utf8 fragment")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        # Should refuse with exit 2, not crash with traceback (exit 1)
        assert result.returncode == 2, f"Expected exit 2, got {result.returncode}: {result.stderr}"
        assert "non-utf8.md" in result.stderr
        assert not out_file.exists()

    def test_empty_manual_step_value_reachable(self, tmp_path):
        """(6) Empty Manual-Step value branch (lines ~143-148) must be reachable and tested."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        # Manual-Step: with only whitespace after colon (empty value after stripping)
        _write_fragment(repo, "empty-manual.md", "### Added\n\nNew feature\n\nManual-Step:   \n")
        _commit_all(repo, "add empty manual step fragment")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        # Should refuse (exit 2) with proper error message
        assert result.returncode == 2, f"Expected exit 2, got {result.returncode}: {result.stderr}"
        assert "empty-manual.md" in result.stderr
        assert "Manual-Step" in result.stderr
        assert not out_file.exists()

    # Parametrised test for all variants from the WHY section (tsk-dtb6f6 / tsk-j7fnhc)
    @pytest.mark.parametrize("fragment_name, fragment_content, should_parse", [
        # Trailer attempts that normalize to canonical parse.
        ("numbered_list.md", "### Changed\n\n1) Migration: one-way\n", True),
        ("blockquote_heading.md", "### Changed\n\n> ## Migration: one-way\n", True),
        ("zwsp_prefix.md", "### Changed\n\n\u200bMigration: one-way\n", True),
        ("word_joiner_prefix.md", "### Changed\n\n\u2060Migration: one-way\n", True),
        ("zwnj_prefix.md", "### Changed\n\n\u200cMigration: one-way\n", True),
        ("soft_hyphen_prefix.md", "### Changed\n\n\u00adMigration: one-way\n", True),
        ("dash_numbered.md", "### Changed\n\n- 1. Migration: one-way\n", True),
        ("lettered_list.md", "### Changed\n\na. Migration: one-way\n", True),
        ("table_row.md", "### Changed\n\n| Migration: one-way |\n", True),
        ("html_comment.md", "### Changed\n\n<!-- Migration: one-way -->\n", True),
        ("manual_step_space.md", "### Changed\n\nManual Step: do something\n", True),
        ("manual_step_underscore.md", "### Changed\n\nManual_Step: do something\n", True),
        ("manual_step_nospace.md", "### Changed\n\nManualStep: do something\n", True),
        # Canonical forms.
        ("canonical_one_way.md", "### Changed\n\nMigration: one-way\n", True),
        ("canonical_reversible.md", "### Changed\n\nMigration: reversible\n", True),
        ("canonical_manual_step.md", "### Changed\n\nManual-Step: do something\n", True),
        # Fences.
        ("fence_cpp_inside.md", "### Changed\n\n```c++\nMigration: one-way\n```\n", "ignore"),
        ("fence_tilde_inside.md", "### Changed\n\n~~~cpp\nMigration: one-way\n~~~\n", "ignore"),
        ("fence_4space_indent.md", "### Changed\n\n    ```\nMigration: one-way\n    ```\n", True),
        ("fence_between_blocks.md", "### Changed\n\n```c++\n```\nMigration: one-way\n```python\n```\n", True),
        # Prose mentions are ignored by the new rule.
        ("prose_migrate.md", "### Changed\n\nRun migrate before restart\n", "ignore"),
        ("prose_migrating.md", "### Changed\n\nWe are migrating users gradually\n", "ignore"),
    ])
    def test_allowlist_variants(self, tmp_path, fragment_name, fragment_content, should_parse):
        """Parametrised test for all variants: canonical forms parse, non-canonical refuse,
        trailers inside valid fences are ignored."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        _write_fragment(repo, fragment_name, fragment_content)
        _commit_all(repo, f"add fragment {fragment_name}")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        if should_parse == "ignore":
            # Trailer inside valid fence: exit 0, no migration detected
            assert result.returncode == 0, f"Expected exit 0 for {fragment_name}, got {result.returncode}: {result.stderr}"
            assert "one_way_migration=false" in result.stdout
            assert "manual_step=false" in result.stdout
        elif should_parse:
            # Canonical form: exit 0, migration detected
            assert result.returncode == 0, f"Expected exit 0 for {fragment_name}, got {result.returncode}: {result.stderr}"
            if "one-way" in fragment_content:
                assert "one_way_migration=true" in result.stdout
            elif "reversible" in fragment_content:
                assert "one_way_migration=false" in result.stdout
            elif "Manual-Step" in fragment_content:
                assert "manual_step=true" in result.stdout
        else:
            # Non-canonical mention: exit 2
            assert result.returncode == 2, f"Expected exit 2 for {fragment_name}, got {result.returncode}: {result.stdout}"
            assert fragment_name in result.stderr
            assert not out_file.exists()

    # Mutant-killing tests (must fail if specific mutants are applied)

    def test_one_way_anchor_required(self, tmp_path):
        """Mutant killer: dropping `one-way$` anchor must fail.
        `Migration: one-way extra` must refuse, not parse as one-way."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        _write_fragment(repo, "bad-one-way.md", "### Changed\n\nMigration: one-way extra\n")
        _commit_all(repo, "add bad one-way fragment")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        # Must refuse (exit 2) - if anchor dropped, this would incorrectly parse
        assert result.returncode == 2, f"Expected exit 2, got {result.returncode}: {result.stdout}"
        assert "bad-one-way.md" in result.stderr
        assert not out_file.exists()

    def test_reversible_anchor_required(self, tmp_path):
        """Mutant killer: dropping `reversible$` anchor must fail.
        `Migration: reversible extra` must refuse.
        `Migration: reversible, but one-way for existing users` must NOT parse as reversible."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        # Test 1: extra text after reversible
        _write_fragment(repo, "bad-reversible.md", "### Changed\n\nMigration: reversible extra\n")
        _commit_all(repo, "add bad reversible fragment")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        assert result.returncode == 2, f"Expected exit 2, got {result.returncode}: {result.stdout}"
        assert "bad-reversible.md" in result.stderr
        assert not out_file.exists()

        # Test 2: "reversible, but one-way for existing users" must NOT parse as reversible
        repo2 = tmp_path / "repo2"
        repo2.mkdir()
        _init_repo(repo2)
        (repo2 / "README.md").write_text("# Test\n")
        _commit_all(repo2, "initial commit")
        _tag(repo2, "v0.5.0")

        _write_fragment(repo2, "reversible-but-one-way.md", "### Changed\n\nMigration: reversible, but one-way for existing users\n")
        _commit_all(repo2, "add reversible-but-one-way fragment")

        out_file2 = repo2 / "release.json"
        result2 = _run_cli(repo2, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file2)])

        # Must refuse (exit 2) - if anchor dropped, this would incorrectly parse as reversible
        assert result2.returncode == 2, f"Expected exit 2, got {result2.returncode}: {result2.stdout}"
        assert "reversible-but-one-way.md" in result2.stderr
        assert not out_file2.exists()

    def test_git_failure_returns_2(self, tmp_path):
        """Mutant killer: git failure handler must return 2, not 0."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        _write_fragment(repo, "fragment.md", "### Added\n\nNew feature\n")
        _commit_all(repo, "add fragment")

        # Corrupt the git repo to cause a git failure
        (repo / ".git").rename(repo / ".git.bak")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        # Must return 2 (git failure), not 0
        assert result.returncode == 2, f"Expected exit 2, got {result.returncode}: {result.stdout}"
        assert "git command failed" in result.stderr.lower()
        assert not out_file.exists()

    def test_top_level_readme_still_skipped(self, tmp_path):
        """Mutant killer: removing README filter must fail.
        Top-level changelog.d/README.md must still be skipped."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        # Add top-level README.md with a trailer - should be skipped
        (repo / "changelog.d").mkdir(exist_ok=True)
        (repo / "changelog.d" / "README.md").write_text("# Changelog\n\nMigration: one-way\n", encoding="utf-8")
        # Add a real fragment
        _write_fragment(repo, "real-fragment.md", "### Added\n\nReal feature\n")
        _commit_all(repo, "add README and real fragment")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        # Should succeed but only process real-fragment.md (README.md skipped)
        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}: {result.stderr}"
        assert "one_way_migration=false" in result.stdout
        
        # Verify fragments via collect()
        result_collect = _run_collect(repo, "v0.5.0")
        assert result_collect["fragments"] == ["changelog.d/real-fragment.md"]
        assert "changelog.d/README.md" not in result_collect["fragments"]

    def test_emphasis_stripping_required(self, tmp_path):
        """Mutant killer: removing bold/inline-code stripping must fail.
        `**Migration: one-way**` and `` `Migration: one-way` `` must still parse."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        # Bold
        _write_fragment(repo, "bold.md", "### Changed\n\n**Migration: one-way**\n")
        _commit_all(repo, "add bold fragment")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}: {result.stderr}"
        assert "one_way_migration=true" in result.stdout

        # Inline code
        repo2 = tmp_path / "repo2"
        repo2.mkdir()
        _init_repo(repo2)
        (repo2 / "README.md").write_text("# Test\n")
        _commit_all(repo2, "initial commit")
        _tag(repo2, "v0.5.0")

        _write_fragment(repo2, "inline.md", "### Changed\n\n`Migration: one-way`\n")
        _commit_all(repo2, "add inline fragment")

        out_file2 = repo2 / "release.json"
        result2 = _run_cli(repo2, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file2)])

        assert result2.returncode == 0, f"Expected exit 0, got {result2.returncode}: {result2.stderr}"
        assert "one_way_migration=true" in result2.stdout

    def test_output_json_keys_order_and_newline(self, tmp_path):
        """(g) the written file is valid JSON whose keys appear in the
        specified order and which ends in exactly one newline."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)

        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        _write_fragment(repo, "fragment.md", "### Added\n\nNew feature\n")
        _commit_all(repo, "add fragment")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        assert result.returncode == 0
        assert out_file.exists()

        content = out_file.read_text(encoding="utf-8")

        # Valid JSON
        data = json.loads(content)

        # Keys in specified order
        expected_keys = [
            "schema",
            "version",
            "commit",
            "previous_version",
            "one_way_migration",
            "manual_step",
            "manual_steps",
            "migrations",
            "fragments",
        ]
        assert list(data.keys()) == expected_keys

        # Ends with exactly one newline
        assert content.endswith("\n")
        assert not content.endswith("\n\n")

        # CLI prints the flags
        assert "one_way_migration=false" in result.stdout
        assert "manual_step=false" in result.stdout

    # Fence opener mutant-killing tests (L115, L118 cases)
    def test_fence_a_space_not_a_fence(self, tmp_path):
        """Mutant killer: ```a ``` (inline code on same line) must NOT be treated as a fence.
        Trailer after it should be exposed and parsed if canonical."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        # ```a ``` is inline code (backticks appear again on same line), not a fence
        # Trailer after it is OUTSIDE any fence
        _write_fragment(repo, "fence-a-space.md", "### Changed\n\n```a ```\nMigration: one-way\n")
        _commit_all(repo, "add fence-a-space fragment")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        # Should parse the canonical trailer (exit 0, one_way_migration=true)
        # Note: file ends without closing fence, but ```a ``` was not a fence opener
        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}: {result.stderr}"
        assert "one_way_migration=true" in result.stdout

    def test_fence_js_is_a_fence(self, tmp_path):
        """Mutant killer: ``` js (info string) MUST be treated as a fence.
        Trailer inside it must be ignored."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        # ``` js is a valid fence opener with info string "js"
        _write_fragment(repo, "fence-js.md", "### Changed\n\n``` js\nMigration: one-way\n```\n")
        _commit_all(repo, "add fence-js fragment")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        # Trailer inside fence must be ignored (exit 0, no migration)
        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}: {result.stderr}"
        assert "one_way_migration=false" in result.stdout

    def test_tilde_fence_works(self, tmp_path):
        """Mutant killer: ~~~ fences must work the same as ``` fences."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        _write_fragment(repo, "tilde-fence.md", "### Changed\n\n~~~\nMigration: one-way\n~~~\n")
        _commit_all(repo, "add tilde fence fragment")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        assert result.returncode == 0
        assert "one_way_migration=false" in result.stdout

    def test_tilde_fence_with_info_string(self, tmp_path):
        """Mutant killer: ~~~cpp fences must work."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        _write_fragment(repo, "tilde-fence-info.md", "### Changed\n\n~~~cpp\nMigration: one-way\n~~~\n")
        _commit_all(repo, "add tilde fence with info fragment")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        assert result.returncode == 0
        assert "one_way_migration=false" in result.stdout


class TestCLI:
    def test_cli_prints_flags(self, tmp_path):
        """CLI prints one_way_migration and manual_step flags."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)

        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        _write_fragment(repo, "one-way.md", "### Changed\n\nMigration: one-way\n")
        _write_fragment(repo, "manual.md", "### Added\n\nManual-Step: do something\n")
        _commit_all(repo, "add fragments")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        assert result.returncode == 0
        assert "one_way_migration=true" in result.stdout
        assert "manual_step=true" in result.stdout

    @pytest.mark.parametrize(
        "fragment_name, fragment_content",
        [
            ("near-miss-lowercase.md", "### Changed\n\nmigration: one-way\n"),
            ("near-miss-dash.md", "### Changed\n\n- Migration: one-way\n"),
            ("near-miss-bold.md", "### Changed\n\n**Migration: one-way**\n"),
            ("near-miss-space.md", "### Changed\n\nMigration : one-way\n"),
            ("near-miss-manual.md", "### Changed\n\nmanual-step: stop the service\n"),
        ],
    )
    def test_near_miss_lines(self, tmp_path, fragment_name, fragment_content):
        """(a) parametrize the five near-miss lines.

        Non-canonical near-misses raise ReleaseMetaError (exit 2, stderr names
        the fragment and the canonical form), while canonical near-misses are
        accepted exactly as before (exit 0, produce output JSON).
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")
        _write_fragment(repo, fragment_name, fragment_content)
        _commit_all(repo, "add near-miss fragment")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])

        # Canonicalizable near-misses parse after normalisation/markup stripping.
        canonicalizable = {
            "near-miss-lowercase.md",
            "near-miss-dash.md",
            "near-miss-bold.md",
            "near-miss-space.md",
            "near-miss-manual.md",
        }
        if fragment_name in canonicalizable:
            assert result.returncode == 0, (
                f"Expected exit 0 for {fragment_name}, got {result.returncode}: {result.stderr}"
            )
            assert out_file.exists()
            if "manual" in fragment_name:
                assert "manual_step=true" in result.stdout
            else:
                assert "one_way_migration=true" in result.stdout
        else:
            assert result.returncode == 2, (
                f"Expected exit 2 for {fragment_name}, got {result.returncode}"
            )
            assert fragment_name in result.stderr
            assert not out_file.exists()


class TestRepoChangelogD:
    def test_repo_changelog_d_self_check(self, tmp_path):
        """Every real changelog.d fragment must parse with exit 0."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")

        src = REPO_ROOT / "changelog.d"
        dst = repo / "changelog.d"
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.rglob("*"):
            if p.is_file():
                rel = p.relative_to(src)
                target = dst / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(p.read_bytes())
        _commit_all(repo, "add real changelog.d")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.0.0-test", "--previous-ref", "v0.5.0", "--out", str(out_file)])
        assert result.returncode == 0, result.stderr
        assert out_file.exists()


class TestNormalizationPins:
    @pytest.mark.parametrize("line, expected", [
        ("Migra\u00adtion: one-way", "parse"),
        ("\uff2digration: one-way", "parse"),
        ("Migra\u200btion: one-way", "parse"),
        ("Manua\u00adl-Step: do it", "parse"),
        ("\uff2danual-Step: do it", "parse"),
        ("Manual\u200b-Step: do it", "parse"),
    ])
    def test_normalization_pins(self, tmp_path, line, expected):
        """Normalised canonical trailers parse; never silently ignored."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)
        (repo / "README.md").write_text("# Test\n")
        _commit_all(repo, "initial commit")
        _tag(repo, "v0.5.0")
        _write_fragment(repo, "pin.md", f"### Fixed\n\n{line}\n")
        _commit_all(repo, "add pin fragment")

        out_file = repo / "release.json"
        result = _run_cli(repo, ["--version", "0.5.1", "--previous-ref", "v0.5.0", "--out", str(out_file)])
        if expected == "parse":
            assert result.returncode == 0, result.stderr
            data = json.loads(out_file.read_text(encoding="utf-8"))
            assert data["one_way_migration"] is True or data["manual_step"] is True
        else:
            assert result.returncode == 2
            assert not out_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])