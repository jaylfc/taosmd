"""Tests for scripts/release_version.py.

Every test builds a REAL throwaway git repository in ``tmp_path`` using
subprocess git and calls the real functions / real CLI. The module is loaded
via importlib from the file path (no reliance on ``scripts`` being on sys.path).
"""
from __future__ import annotations

import datetime as _dt
import importlib.util
import re
import subprocess
import sys
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "release_version.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("release_version", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rv = _load_module()


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )


def _write(repo, rel, text):
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _init_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    return repo


def _seed(repo):
    _write(repo, "README.md", "hello\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial commit")


def _run_cli(repo, *flags):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *flags, "--repo", str(repo)],
        capture_output=True,
        text=True,
    )


# ----------------------------------------------------------------------
# (a) anchor then 3 direct commits -> 0.5.3
# (b) anchor then side branch (5 commits) merged with --no-ff -> 0.5.1
# ----------------------------------------------------------------------


class TestComputeVersion:
    def test_anchor_then_three_direct_commits(self, tmp_path):
        repo = _init_repo(tmp_path)
        _seed(repo)
        _git(repo, "tag", "v0.5.0")
        for i in range(3):
            _write(repo, f"direct{i}.txt", f"{i}\n")
            _git(repo, "add", f"direct{i}.txt")
            _git(repo, "commit", "-m", f"direct commit {i}")
        assert rv.compute_version(repo) == "0.5.3"

    def test_first_parent_counting(self, tmp_path):
        repo = _init_repo(tmp_path)
        _seed(repo)
        _git(repo, "tag", "v0.5.0")
        _git(repo, "checkout", "-b", "side")
        for i in range(5):
            _write(repo, f"side{i}.txt", f"{i}\n")
            _git(repo, "add", f"side{i}.txt")
            _git(repo, "commit", "-m", f"side commit {i}")
        _git(repo, "checkout", "master")
        _git(repo, "merge", "--no-ff", "-m", "merge side", "side")
        assert rv.compute_version(repo) == "0.5.1"


# ----------------------------------------------------------------------
# (c) CLI refusals: exit 2 with a non-empty stderr line
# ----------------------------------------------------------------------


class TestCliRefusals:
    def test_rev_is_anchor_refused(self, tmp_path):
        repo = _init_repo(tmp_path)
        _seed(repo)
        _git(repo, "tag", "v0.5.0")  # HEAD is the anchor itself
        result = _run_cli(repo, "--print")
        assert result.returncode == 2, result.stderr
        assert result.stderr.strip()

    def test_anchor_missing_refused(self, tmp_path):
        repo = _init_repo(tmp_path)
        _seed(repo)  # no v0.5.0 tag
        result = _run_cli(repo, "--print")
        assert result.returncode == 2, result.stderr
        assert result.stderr.strip()

    def test_rev_not_descendant_of_anchor_refused(self, tmp_path):
        repo = _init_repo(tmp_path)
        _seed(repo)
        _git(repo, "checkout", "-b", "side")
        _git(repo, "tag", "v0.5.0")  # anchor only on the side branch
        _git(repo, "checkout", "master")  # HEAD does not contain the anchor
        result = _run_cli(repo, "--print")
        assert result.returncode == 2, result.stderr
        assert result.stderr.strip()


# ----------------------------------------------------------------------
# (d) --stamp changes only the [project] version line and writes
#     taosmd/_build_info.py with COMMIT == HEAD sha and parseable UTC BUILT_AT
# ----------------------------------------------------------------------


class TestStamp:
    @pytest.mark.parametrize("pyproject_content,description", [
        (
            '[project]\nname = "taosmd"\nversion = "0.4.1"\n\n[tool.x]\nversion = "9.9.9"\n',
            "master layout: stray version in [tool.x] after [project]",
        ),
        (
            'version = "9.9.9"\n\n[project]\nname = "taosmd"\nversion = "0.4.1"\n',
            "#734 layout: stray version before [project]",
        ),
    ])
    def test_stamp_changes_only_project_version(self, tmp_path, pyproject_content, description):
        repo = _init_repo(tmp_path)
        _seed(repo)
        _git(repo, "tag", "v0.5.0")
        _write(repo, "f.txt", "x\n")
        _git(repo, "add", "f.txt")
        _git(repo, "commit", "-m", "release commit")
        head = _git(repo, "rev-parse", "HEAD").stdout.strip()
        _write(repo, "pyproject.toml", pyproject_content)
        result = _run_cli(repo, "--stamp")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "0.5.1"
        pyproject = (repo / "pyproject.toml").read_text(encoding="utf-8")
        if "tool.x" in pyproject_content:
            # master layout: stray version in [tool.x] should be untouched
            assert pyproject == (
                '[project]\nname = "taosmd"\nversion = "0.5.1"\n\n[tool.x]\nversion = "9.9.9"\n'
            ), pyproject
        else:
            # #734 layout: stray version before [project] should be untouched
            assert pyproject == (
                'version = "9.9.9"\n\n[project]\nname = "taosmd"\nversion = "0.5.1"\n'
            ), pyproject
        build_info = (repo / "taosmd" / "_build_info.py").read_text(
            encoding="utf-8"
        )
        assert f"COMMIT = '{head}'" in build_info, build_info
        match = re.search(r"BUILT_AT = '([^']+)'", build_info)
        assert match, build_info
        built_at = match.group(1)
        assert built_at.endswith("Z"), built_at
        parsed = _dt.datetime.fromisoformat(built_at.replace("Z", "+00:00"))
        assert parsed.tzinfo == _dt.timezone.utc, built_at


# ----------------------------------------------------------------------
# (d2) #734 fix: --stamp works from a subdirectory of the repo
# ----------------------------------------------------------------------


class TestStampSubdirectory:
    def test_stamp_from_subdirectory(self, tmp_path):
        """Running the CLI from a subdirectory of the repo works correctly.
        This guards the #734 fix for the release_version subdirectory fix
        in scripts/release_version.py (_resolve_repo_root uses git toplevel).
        """
        repo = _init_repo(tmp_path)
        _seed(repo)
        _git(repo, "tag", "v0.5.0")
        _write(repo, "f.txt", "x\n")
        _git(repo, "add", "f.txt")
        _git(repo, "commit", "-m", "release commit")
        head = _git(repo, "rev-parse", "HEAD").stdout.strip()
        _write(
            repo,
            "pyproject.toml",
            '[project]\nname = "taosmd"\nversion = "0.4.1"\n',
        )
        # Create a subdirectory and run from there
        subdir = repo / "subdir"
        subdir.mkdir()
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--stamp", "--repo", str(subdir)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "0.5.1"
        pyproject = (repo / "pyproject.toml").read_text(encoding="utf-8")
        assert pyproject == (
            '[project]\nname = "taosmd"\nversion = "0.5.1"\n'
        ), pyproject
        build_info = (repo / "taosmd" / "_build_info.py").read_text(
            encoding="utf-8"
        )
        assert f"COMMIT = '{head}'" in build_info, build_info
        match = re.search(r"BUILT_AT = '([^']+)'", build_info)
        assert match, build_info
        built_at = match.group(1)
        assert built_at.endswith("Z"), built_at
        parsed = _dt.datetime.fromisoformat(built_at.replace("Z", "+00:00"))
        assert parsed.tzinfo == _dt.timezone.utc, built_at

    def test_compute_version_from_subdirectory(self, tmp_path):
        """compute_version() works when called with a subdirectory path."""
        repo = _init_repo(tmp_path)
        _seed(repo)
        _git(repo, "tag", "v0.5.0")
        _write(repo, "f.txt", "x\n")
        _git(repo, "add", "f.txt")
        _git(repo, "commit", "-m", "release commit")
        subdir = repo / "subdir"
        subdir.mkdir()
        version = rv.compute_version(subdir)
        assert version == "0.5.1"


# ----------------------------------------------------------------------
# (e) two [project] version lines or none -> refusal, file unchanged
# ----------------------------------------------------------------------


class TestStampRefusal:
    def test_two_project_version_lines_refused(self, tmp_path):
        repo = _init_repo(tmp_path)
        _seed(repo)
        _git(repo, "tag", "v0.5.0")
        _write(repo, "pyproject.toml", '[project]\nversion = "0.4.1"\nversion = "0.4.2"\n')
        original = (repo / "pyproject.toml").read_bytes()
        try:
            rv.stamp(repo, "0.5.1", "a" * 40, "2026-01-01T00:00:00Z")
        except rv.ReleaseVersionError:
            pass
        else:
            raise AssertionError("expected ReleaseVersionError")
        assert (repo / "pyproject.toml").read_bytes() == original

    def test_no_project_version_line_refused(self, tmp_path):
        repo = _init_repo(tmp_path)
        _seed(repo)
        _git(repo, "tag", "v0.5.0")
        _write(repo, "pyproject.toml", '[project]\nname = "taosmd"\n')
        original = (repo / "pyproject.toml").read_bytes()
        try:
            rv.stamp(repo, "0.5.1", "a" * 40, "2026-01-01T00:00:00Z")
        except rv.ReleaseVersionError:
            pass
        else:
            raise AssertionError("expected ReleaseVersionError")
        assert (repo / "pyproject.toml").read_bytes() == original
