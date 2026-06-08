"""Project identity — deterministic, environment-derived project fingerprints.

The project fingerprint solves a real multi-agent problem: when Claude Code,
Kilo Code, Hermes, or any other agent framework works on the same codebase,
they need to share memory without relying on agent instructions being
identical. Instructions drift; git remotes don't.

The fingerprint is a short hash of the git remote origin URL (normalized).
It's stable across clones, machines, and agent sessions — as long as the
repo has a remote, the fingerprint is deterministic.

Fallback chain:
  1. ``git config --get remote.origin.url`` → sha256, 12 hex chars
  2. Explicit project ID from ``<project>/.taosmd/project.toml``
  3. Hash of ``os.getcwd()`` (unstable if moved — last resort)

Usage::

    from taosmd.project import get_project_id, ProjectResolver

    # Auto-detect from git remote
    pid = get_project_id()

    # Explicit override
    pid = get_project_id(explicit_id="my-project")

    # Full resolver with fallback chain
    resolver = ProjectResolver()
    pid = resolver.resolve()
    info = resolver.describe()  # includes source and warnings
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_FINGERPRINT_LEN = 12  # 48 bits — enough for billions of projects, short enough to read


class ProjectFingerprintError(RuntimeError):
    """Raised when no project identity can be determined."""


def _normalize_git_remote(url: str) -> str:
    """Normalize a git remote URL to a canonical form for hashing.

    Handles the common variants:
    - ``git@github.com:owner/repo.git`` → ``https://github.com/owner/repo``
    - ``https://github.com/owner/repo.git`` → ``https://github.com/owner/repo``
    - ``ssh://git@github.com/owner/repo.git`` → ``https://github.com/owner/repo``
    """
    url = url.strip()
    # Strip .git suffix
    url = re.sub(r"\.git$", "", url)
    # SSH shorthand: git@host:path → https://host/path
    ssh_match = re.match(r"git@([^:]+):(.+)$", url)
    if ssh_match:
        url = f"https://{ssh_match.group(1)}/{ssh_match.group(2)}"
    # ssh:// prefix
    url = re.sub(r"^ssh://git@", "https://", url)
    # Lowercase for consistency
    url = url.lower()
    return url


def _hash_remote(url: str) -> str:
    """SHA-256 hash of a normalized remote URL, truncated to _FINGERPRINT_LEN."""
    return hashlib.sha256(url.encode()).hexdigest()[:_FINGERPRINT_LEN]


def _get_git_remote(cwd: str | None = None) -> str | None:
    """Return the origin URL for the git repo at *cwd*, or None."""
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def _read_project_toml(cwd: str | None = None) -> str | None:
    """Read an explicit project ID from ``<cwd>/.taosmd/project.toml``."""
    toml_path = Path(cwd or ".") / ".taosmd" / "project.toml"
    if not toml_path.exists():
        return None
    try:
        text = toml_path.read_text()
        # Simple parse — look for ``project_id = "..."`` without toml dep
        match = re.search(r'project_id\s*=\s*["\']([^"\']+)["\']', text)
        if match:
            return match.group(1).strip()
    except OSError as exc:
        logger.warning("taosmd: failed to read %s: %s", toml_path, exc)
    return None


@dataclass
class ProjectInfo:
    """Result of project identity resolution."""
    project_id: str
    source: str  # "git_remote", "project_toml", "explicit", "cwd_hash"
    remote_url: str | None = None
    warnings: list[str] | None = None


class ProjectResolver:
    """Resolve project identity using the full fallback chain.

    Args:
        cwd: Working directory to resolve from. Defaults to ``os.getcwd()``.
        explicit_id: If given, skip detection and use this ID directly.
    """

    def __init__(self, cwd: str | None = None, explicit_id: str | None = None):
        self.cwd = cwd
        self.explicit_id = explicit_id

    def resolve(self) -> str:
        """Return the project fingerprint (12-char hex string)."""
        return self.describe().project_id

    def describe(self) -> ProjectInfo:
        """Return full resolution details including source and warnings."""
        warnings: list[str] = []

        # 1. Explicit override
        if self.explicit_id:
            return ProjectInfo(
                project_id=self.explicit_id,
                source="explicit",
            )

        # 2. Git remote
        remote = _get_git_remote(self.cwd)
        if remote:
            normalized = _normalize_git_remote(remote)
            return ProjectInfo(
                project_id=_hash_remote(normalized),
                source="git_remote",
                remote_url=remote,
            )

        # 3. .taosmd/project.toml
        toml_id = _read_project_toml(self.cwd)
        if toml_id:
            warnings.append(
                "No git remote detected — using .taosmd/project.toml. "
                "Consider adding a git remote for stable project identity."
            )
            return ProjectInfo(
                project_id=toml_id,
                source="project_toml",
                warnings=warnings,
            )

        # 4. Fallback: hash of cwd
        cwd = self.cwd or os.getcwd()
        warnings.append(
            "No git remote or .taosmd/project.toml found — using directory hash. "
            "This is unstable if the project is moved. "
            "Create .taosmd/project.toml with: project_id = \"my-project\""
        )
        return ProjectInfo(
            project_id=_hash_remote(cwd),
            source="cwd_hash",
            warnings=warnings,
        )


def get_project_id(
    *,
    cwd: str | None = None,
    explicit_id: str | None = None,
) -> str:
    """Convenience wrapper — return the project fingerprint as a string.

    Args:
        cwd: Working directory. Defaults to ``os.getcwd()``.
        explicit_id: Skip detection, use this ID directly.

    Returns:
        12-character hex string identifying the project.
    """
    return ProjectResolver(cwd=cwd, explicit_id=explicit_id).resolve()
