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
        assert result["previous_version"] == "v0.5.0"
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
        assert "oneway" in result.stderr
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

    def test_modified_fragment_not_counted(self, tmp_path):
        """(e) a fragment MODIFIED (not added) after the tag is not counted;
        changelog.d/README.md is ignored."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(repo)

        # Create fragment and README.md BEFORE the tag
        _write_fragment(repo, "existing.md", "### Fixed\n\nOld fix\n")
        (repo / "changelog.d" / "README.md").write_text("# Changelog\n")
        _commit_all(repo, "initial commit with existing fragment and readme")
        _tag(repo, "v0.5.0")

        # MODIFY the existing fragment (not add new)
        _write_fragment(repo, "existing.md", "### Fixed\n\nOld fix updated\n")
        # Add a NEW fragment
        _write_fragment(repo, "new-fragment.md", "### Added\n\nNew feature\n")
        _commit_all(repo, "modify existing and add new")

        result = _run_collect(repo, "v0.5.0")

        # Only the newly added fragment should be listed
        assert result["fragments"] == ["changelog.d/new-fragment.md"]
        # README.md must never be listed
        assert "changelog.d/README.md" not in result["fragments"]

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])