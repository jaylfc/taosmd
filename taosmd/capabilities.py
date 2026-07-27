"""Build identity and capability advertisement for ``GET /version``.

Why this module exists
----------------------
A taOSmd server answers unknown non-API paths with the dashboard SPA, so a
``GET /collections`` against a build with no collections code returns
``200 text/html``. An integrator checking "did the route exist?" by status code
therefore gets a confident yes from a server that cannot do the thing. And a
semver alone cannot answer "does this box actually speak collections", because
features land continuously between version bumps and a production box can sit a
month stale without anyone noticing.

So the server publishes a **capability list**: stable contract identifiers a
consumer can test membership against.

The naming contract
-------------------
Every identifier is ``<contract>.v<N>``, for example ``collections.v1``. The
suffix is the whole point: when the collections wire contract changes in a way
that breaks existing callers, the advertised identifier becomes
``collections.v2``. A consumer pinned to ``collections.v1`` sees the capability
disappear (a visible, actionable break) instead of ``collections`` silently
meaning something new. Additive, backwards-compatible changes keep the same
identifier. A build may advertise several versions of one contract at once
during a migration window.

Keeping the list honest
-----------------------
The list is never a hand-maintained constant of feature names. Each identifier
is declared next to a *probe*: the module and the symbols that implement it,
plus the route markers that expose it over HTTP. A capability is advertised
only if its probe resolves on the running build, so deleting or renaming the
implementation deletes the claim rather than leaving a stale boast. The route
markers are asserted against the real dispatcher by
``tests/test_version_capabilities.py``, so a declaration cannot drift away from
the surface it describes.
"""

from __future__ import annotations

import functools
import importlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "CAPABILITY_PROBES",
    "all_probes",
    "build_info",
    "capabilities",
    "probe_for",
    "reset_caches",
    "resolve_commit",
    "version_payload",
]


@dataclass(frozen=True)
class CapabilityProbe:
    """A contract identifier bound to the code that must exist for it to be true.

    ``module``/``symbols``  : imported and attribute-checked at advertisement
                              time. All symbols must be present.
    ``route_markers``       : substrings that must appear in the HTTP
                              dispatcher for this capability to be reachable.
                              Asserted by the divergence test, not at runtime.
    """

    name: str
    module: str
    symbols: tuple[str, ...]
    route_markers: tuple[str, ...]


# The capability table. Each entry sits immediately next to the symbols that
# make it true; nothing here is advertised without those symbols resolving.
# Adding a capability means adding its probe, and the divergence test in
# tests/test_version_capabilities.py fails if the routes named here are not in
# the dispatcher. When a contract breaks, add a new `.vN+1` entry (and keep or
# drop the old one depending on whether the build still serves it).
CAPABILITY_PROBES: tuple[CapabilityProbe, ...] = (
    CapabilityProbe(
        name="a2a.v1",
        module="taosmd.service",
        symbols=("a2a_send", "a2a_feed", "a2a_channels", "a2a_members"),
        route_markers=(
            '"/a2a/send"',
            '"/a2a/messages"',
            '"/a2a/stream"',
            '"/a2a/channels"',
            '"/a2a/members"',
        ),
    ),
    CapabilityProbe(
        name="collections.v1",
        module="taosmd.service",
        symbols=(
            "collections_create",
            "collections_list",
            "collections_get",
            "collections_index_start",
            "collections_link",
            "collections_unlink",
            "collections_archive",
        ),
        route_markers=('"/collections"', '"/collections/"', '"/index"', '"/link"'),
    ),
    CapabilityProbe(
        name="grants.v1",
        module="taosmd.service",
        symbols=("collections_grant", "collections_revoke"),
        route_markers=('"/grants"', '"/grants/"'),
    ),
    CapabilityProbe(
        name="graph.v1",
        module="taosmd.service",
        symbols=("graph", "graph_activations"),
        route_markers=('"/graph"', '"/graph/activations"'),
    ),
    CapabilityProbe(
        name="ingest.v1",
        module="taosmd.service",
        symbols=("ingest", "ingest_batch"),
        route_markers=('"/ingest"', '"/ingest/batch"'),
    ),
    CapabilityProbe(
        name="search.v1",
        module="taosmd.service",
        symbols=("search",),
        route_markers=('"/search"',),
    ),
    CapabilityProbe(
        name="shelves.v1",
        module="taosmd.service",
        symbols=(
            "list_shelves",
            "admin_shelf_create",
            "admin_shelf_archive",
            "admin_shelf_unarchive",
        ),
        route_markers=('"/shelves"', '"/shelves/"'),
    ),
    CapabilityProbe(
        name="tasks.v1",
        module="taosmd.service",
        symbols=(
            "task_create",
            "task_list",
            "task_ready",
            "task_prime",
            "task_update",
            "task_add_edge",
            "task_remove_edge",
        ),
        route_markers=('"/tasks"', '"/tasks/ready"', '"/tasks/prime"'),
    ),
    CapabilityProbe(
        # Time-travel over the temporal KG: ?as_of= on GET /graph, plus the
        # temporal parsing/filtering stage behind search.
        name="temporal.v1",
        module="taosmd.temporal",
        symbols=(
            "parse_temporal_expression",
            "extract_temporal_expression",
            "apply_temporal_stage",
        ),
        route_markers=('"as_of"', '"/graph"'),
    ),
)


def all_probes() -> tuple[CapabilityProbe, ...]:
    """Every declared capability probe, in table order."""
    return CAPABILITY_PROBES


def probe_for(name: str) -> CapabilityProbe:
    """Return the probe declaring ``name``; raises ``KeyError`` if undeclared."""
    for probe in CAPABILITY_PROBES:
        if probe.name == name:
            return probe
    raise KeyError(name)


def _resolves(probe: CapabilityProbe) -> bool:
    """True when the running build really implements ``probe``."""
    try:
        module = importlib.import_module(probe.module)
    except Exception:  # noqa: BLE001 - a missing/broken feature is just absent
        return False
    return all(hasattr(module, symbol) for symbol in probe.symbols)


@functools.lru_cache(maxsize=1)
def _resolved_capabilities() -> tuple[str, ...]:
    return tuple(sorted(probe.name for probe in CAPABILITY_PROBES if _resolves(probe)))


def capabilities() -> list[str]:
    """Contract identifiers this build actually implements, sorted.

    The probe result is cached (a public endpoint should not re-import on every
    request) but a fresh list is handed out each call so no caller can mutate
    the cached answer. Tests that mutate the probed modules call
    :func:`reset_caches`.
    """
    return list(_resolved_capabilities())


def reset_caches() -> None:
    """Drop the cached capability and build-identity answers (tests only)."""
    _resolved_capabilities.cache_clear()
    _resolved_build_info.cache_clear()


# ---------------------------------------------------------------------------
# build identity: commit + build/install time
# ---------------------------------------------------------------------------
#
# Resolved once, at import, and cached. Two hard rules:
#   1. never shell out. `git rev-parse` in a request path can block on a lock,
#      a slow filesystem, or a missing binary, and a monitoring endpoint must
#      not be able to hang. The git plumbing we need (HEAD -> ref -> sha) is
#      plain file reads, so we read the files directly.
#   2. never raise. A build with no resolvable commit reports null; it does not
#      turn /version into a 500.


def _build_stamp() -> dict | None:
    """Optional build-time stamp written by a packaging step.

    A wheel or container build may drop a ``taosmd/_build_info.py`` exporting
    ``COMMIT`` and/or ``BUILT_AT`` (ISO 8601). It is absent from a plain source
    checkout, which is why the git reader below exists.
    """
    try:
        module = importlib.import_module("taosmd._build_info")
    except Exception:  # noqa: BLE001 - unstamped build is the normal case
        return None
    commit = getattr(module, "COMMIT", None)
    built_at = getattr(module, "BUILT_AT", None)
    if not isinstance(commit, str) or not commit.strip():
        commit = None
    if not isinstance(built_at, str) or not built_at.strip():
        built_at = None
    if commit is None and built_at is None:
        return None
    return {"commit": commit, "built_at": built_at}


def _git_dir(package_dir: Path) -> Path | None:
    """Locate the ``.git`` directory for a source checkout, or None.

    Handles the worktree/submodule form where ``.git`` is a file containing
    ``gitdir: <path>`` rather than a directory.
    """
    dot_git = package_dir.parent / ".git"
    if dot_git.is_dir():
        return dot_git
    if dot_git.is_file():
        text = dot_git.read_text(encoding="utf-8", errors="replace").strip()
        if text.startswith("gitdir:"):
            target = Path(text[len("gitdir:") :].strip())
            if not target.is_absolute():
                target = (dot_git.parent / target).resolve()
            if target.is_dir():
                return target
    return None


def _read_ref(git_dir: Path, ref: str) -> str | None:
    """Resolve a ref name to a sha via loose ref, then packed-refs."""
    search_dirs = [git_dir]
    # A linked worktree keeps refs in the main repo, named by commondir.
    common = git_dir / "commondir"
    if common.is_file():
        rel = common.read_text(encoding="utf-8", errors="replace").strip()
        if rel:
            resolved = (git_dir / rel).resolve()
            if resolved.is_dir():
                search_dirs.append(resolved)

    for base in search_dirs:
        loose = base / ref
        if loose.is_file():
            value = loose.read_text(encoding="utf-8", errors="replace").strip()
            if _looks_like_sha(value):
                return value
        packed = base / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith(("#", "^")):
                    continue
                sha, _, name = line.partition(" ")
                if name.strip() == ref and _looks_like_sha(sha):
                    return sha
    return None


def _looks_like_sha(value: str) -> bool:
    return len(value) == 40 and all(c in "0123456789abcdef" for c in value.lower())


def resolve_commit(package_dir: Path) -> tuple[str | None, str | None]:
    """Best-effort ``(commit, source)`` for the build rooted at ``package_dir``.

    ``source`` is ``"build-stamp"``, ``"git"``, or None when unresolvable.
    Never raises and never spawns a process.
    """
    try:
        stamp = _build_stamp()
        if stamp and stamp.get("commit"):
            return stamp["commit"], "build-stamp"

        git_dir = _git_dir(Path(package_dir))
        if git_dir is None:
            return None, None
        head_file = git_dir / "HEAD"
        if not head_file.is_file():
            return None, None
        head = head_file.read_text(encoding="utf-8", errors="replace").strip()
        if head.startswith("ref:"):
            sha = _read_ref(git_dir, head[len("ref:") :].strip())
            return (sha, "git") if sha else (None, None)
        if _looks_like_sha(head):
            return head, "git"
    except Exception:  # noqa: BLE001 - identity is best-effort, never fatal
        return None, None
    return None, None


def _resolve_built_at(package_dir: Path) -> tuple[str | None, str | None]:
    """Best-effort ``(timestamp, source)``: build stamp, else install date."""
    try:
        stamp = _build_stamp()
        if stamp and stamp.get("built_at"):
            return stamp["built_at"], "build-stamp"

        # Installed distributions: the dist-info directory's mtime is when this
        # copy was installed, which is what an operator chasing a stale box
        # actually wants to know.
        from importlib import metadata  # noqa: PLC0415 - optional, import-time only

        dist = metadata.distribution("taosmd")
        dist_path = getattr(dist, "_path", None)
        candidate = Path(dist_path) if dist_path else None
        if candidate is None or not candidate.exists():
            candidate = Path(package_dir)
        stamped = datetime.fromtimestamp(os.stat(candidate).st_mtime, tz=timezone.utc)
        return stamped.strftime("%Y-%m-%dT%H:%M:%SZ"), "install"
    except Exception:  # noqa: BLE001 - best-effort
        return None, None


@functools.lru_cache(maxsize=1)
def _resolved_build_info() -> tuple[tuple[str, str | None], ...]:
    package_dir = Path(__file__).resolve().parent
    commit, commit_source = resolve_commit(package_dir)
    built_at, built_at_source = _resolve_built_at(package_dir)
    return (
        ("commit", commit),
        ("commit_source", commit_source),
        ("built_at", built_at),
        ("built_at_source", built_at_source),
    )


def build_info() -> dict:
    """Build identity: commit, its source, build/install time, its source.

    Resolved once on first call (server startup) so no request ever pays for the
    filesystem work; a fresh dict is returned each call so callers cannot mutate
    the cached answer.
    """
    return dict(_resolved_build_info())


def version_payload() -> dict:
    """The full ``GET /version`` body.

    Deliberately narrow: build identity and capability identifiers only. No
    paths, no tokens, no configuration, because this endpoint is unauthenticated
    by design so monitoring and drift probes keep working on a token-secured box.
    """
    from . import __version__  # noqa: PLC0415 - avoid a circular import at module load

    return {"version": __version__, **build_info(), "capabilities": capabilities()}
