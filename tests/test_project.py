"""Tests for project identity fingerprinting."""
from __future__ import annotations

import os
import subprocess

import pytest

from taosmd.project import (
    ProjectInfo,
    ProjectResolver,
    _get_git_remote,
    _hash_remote,
    _normalize_git_remote,
    _read_project_toml,
    get_project_id,
)


# --- _normalize_git_remote ---------------------------------------------------


class TestNormalizeGitRemote:
    def test_ssh_shorthand(self):
        assert _normalize_git_remote("git@github.com:owner/repo.git") == "https://github.com/owner/repo"

    def test_https_with_git_suffix(self):
        assert _normalize_git_remote("https://github.com/owner/repo.git") == "https://github.com/owner/repo"

    def test_https_without_suffix(self):
        assert _normalize_git_remote("https://github.com/owner/repo") == "https://github.com/owner/repo"

    def test_ssh_protocol(self):
        assert _normalize_git_remote("ssh://git@github.com/owner/repo.git") == "https://github.com/owner/repo"

    def test_case_normalization(self):
        assert _normalize_git_remote("HTTPS://GitHub.COM/Owner/Repo") == "https://github.com/owner/repo"

    def test_strips_whitespace(self):
        assert _normalize_git_remote("  git@github.com:a/b.git  ") == "https://github.com/a/b"

    def test_gitlab(self):
        assert _normalize_git_remote("git@gitlab.com:group/project.git") == "https://gitlab.com/group/project"

    def test_self_hosted(self):
        assert _normalize_git_remote("https://git.example.com/team/repo.git") == "https://git.example.com/team/repo"


# --- _hash_remote ------------------------------------------------------------


class TestHashRemote:
    def test_deterministic(self):
        assert _hash_remote("https://github.com/owner/repo") == _hash_remote("https://github.com/owner/repo")

    def test_length(self):
        h = _hash_remote("https://github.com/owner/repo")
        assert len(h) == 12

    def test_hex(self):
        h = _hash_remote("https://github.com/owner/repo")
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_urls_differ(self):
        a = _hash_remote("https://github.com/owner/repo-a")
        b = _hash_remote("https://github.com/owner/repo-b")
        assert a != b


# --- _get_git_remote ---------------------------------------------------------


class TestGetGitRemote:
    def test_returns_url_in_git_repo(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/test/repo.git"],
            cwd=tmp_path, capture_output=True,
        )
        assert _get_git_remote(str(tmp_path)) == "https://github.com/test/repo.git"

    def test_returns_none_in_non_git_dir(self, tmp_path):
        assert _get_git_remote(str(tmp_path)) is None

    def test_returns_none_without_remote(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        assert _get_git_remote(str(tmp_path)) is None


# --- _read_project_toml ------------------------------------------------------


class TestReadProjectToml:
    def test_reads_explicit_id(self, tmp_path):
        toml_dir = tmp_path / ".taosmd"
        toml_dir.mkdir()
        (toml_dir / "project.toml").write_text('project_id = "my-project"\n')
        assert _read_project_toml(str(tmp_path)) == "my-project"

    def test_reads_single_quoted(self, tmp_path):
        toml_dir = tmp_path / ".taosmd"
        toml_dir.mkdir()
        (toml_dir / "project.toml").write_text("project_id = 'my-project'\n")
        assert _read_project_toml(str(tmp_path)) == "my-project"

    def test_returns_none_without_file(self, tmp_path):
        assert _read_project_toml(str(tmp_path)) is None

    def test_returns_none_without_key(self, tmp_path):
        toml_dir = tmp_path / ".taosmd"
        toml_dir.mkdir()
        (toml_dir / "project.toml").write_text("other_key = 'value'\n")
        assert _read_project_toml(str(tmp_path)) is None


# --- ProjectResolver ---------------------------------------------------------


class TestProjectResolver:
    def test_explicit_id_wins(self, tmp_path):
        resolver = ProjectResolver(cwd=str(tmp_path), explicit_id="forced-id")
        info = resolver.describe()
        assert info.project_id == "forced-id"
        assert info.source == "explicit"

    def test_git_remote_preferred(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/owner/repo.git"],
            cwd=tmp_path, capture_output=True,
        )
        resolver = ProjectResolver(cwd=str(tmp_path))
        info = resolver.describe()
        assert info.source == "git_remote"
        assert info.project_id == _hash_remote("https://github.com/owner/repo")
        assert info.remote_url == "https://github.com/owner/repo.git"

    def test_toml_fallback(self, tmp_path):
        toml_dir = tmp_path / ".taosmd"
        toml_dir.mkdir()
        (toml_dir / "project.toml").write_text('project_id = "custom-id"\n')
        resolver = ProjectResolver(cwd=str(tmp_path))
        info = resolver.describe()
        assert info.source == "project_toml"
        assert info.project_id == "custom-id"
        assert info.warnings  # warns about no git remote

    def test_cwd_hash_last_resort(self, tmp_path):
        resolver = ProjectResolver(cwd=str(tmp_path))
        info = resolver.describe()
        assert info.source == "cwd_hash"
        assert len(info.project_id) == 12
        assert info.warnings  # warns about instability


# --- get_project_id (convenience) -------------------------------------------


class TestGetProjectId:
    def test_returns_string(self, tmp_path):
        pid = get_project_id(cwd=str(tmp_path), explicit_id="test")
        assert isinstance(pid, str)

    def test_length(self, tmp_path):
        # No explicit_id: a detected/derived fingerprint is always 12 hex chars.
        # (explicit_id is returned verbatim, covered by test_explicit.)
        pid = get_project_id(cwd=str(tmp_path))
        assert len(pid) == 12

    def test_explicit(self, tmp_path):
        pid = get_project_id(cwd=str(tmp_path), explicit_id="my-id")
        assert pid == "my-id"
