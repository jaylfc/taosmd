"""Tests for the /version endpoint and the capability contract (taosmd.capabilities).

Why this exists: a taOSmd server answers unknown non-API paths with the
dashboard SPA (200 text/html), so "the route returned 200" proves nothing about
what a build supports. /version publishes stable contract identifiers so a
consumer can ask "does this box actually speak collections" and get a real
answer, and the tests below exist to make sure the answer cannot be a lie.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from taosmd import __version__, capabilities
from taosmd import api as taosmd_api
from taosmd import http_server


def _patch_embedder(stores: dict) -> None:
    """Deterministic 8-dim hash embedder so no ONNX/QMD model is needed."""
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


def _get(url: str) -> tuple[int, dict]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    """Base URL of a running token-less test server."""
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)

    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        for s in list(taosmd_api._stores_cache.values()):
            for store in (s.get("archive"), s.get("vector"), s.get("kg")):
                if store and hasattr(store, "close"):
                    try:
                        httpd.service_loop.run(store.close())
                    except Exception:
                        pass
        httpd.service_loop.close()


# ---------------------------------------------------------------------------
# module-level capability derivation
# ---------------------------------------------------------------------------


def test_capabilities_are_contract_identifiers_with_version_suffix():
    """Every advertised capability is `<name>.v<N>`, never a bare feature name.

    The suffix is the contract: a breaking change to collections must surface
    as `collections.v2`, a visible break consumers can pin against, rather than
    `collections` quietly meaning something new.
    """
    caps = capabilities.capabilities()
    assert caps, "expected a non-empty capability list"
    assert caps == sorted(caps), "capability list must be stably sorted"
    assert len(caps) == len(set(caps)), "capability list must not repeat entries"
    for cap in caps:
        name, _, ver = cap.rpartition(".")
        assert name, f"{cap!r} has no contract name"
        assert ver.startswith("v") and ver[1:].isdigit(), f"{cap!r} lacks a .vN suffix"


def test_capabilities_include_collections_and_grants_on_this_build():
    """This build ships collections, so it must say so."""
    caps = capabilities.capabilities()
    assert "collections.v1" in caps
    assert "grants.v1" in caps
    assert "a2a.v1" in caps
    assert "tasks.v1" in caps
    assert "temporal.v1" in caps


def test_capability_is_dropped_when_its_backing_symbol_is_missing(monkeypatch):
    """A capability cannot be advertised by a build that lacks the code.

    This is the anti-drift property: the list is derived by probing the real
    implementation, so deleting the code deletes the claim.
    """
    from taosmd import service

    probe = capabilities.probe_for("collections.v1")
    victim = probe.symbols[0]
    assert hasattr(service, victim)
    monkeypatch.delattr(service, victim)
    capabilities.reset_caches()

    caps = capabilities.capabilities()
    assert "collections.v1" not in caps
    # Unrelated capabilities are unaffected.
    assert "a2a.v1" in caps

    capabilities.reset_caches()


def test_capability_declarations_do_not_diverge_from_the_http_surface():
    """Each declared capability's routes must actually exist in the dispatcher.

    The declaration table sits next to nothing but this test; if someone adds a
    capability without wiring its routes (or renames a route out from under a
    capability) this fails.
    """
    source = Path(http_server.__file__).read_text(encoding="utf-8")
    dispatch = source[source.index("def _make_handler") :]
    for probe in capabilities.all_probes():
        assert probe.route_markers, f"{probe.name} declares no route markers"
        for marker in probe.route_markers:
            assert marker in dispatch, (
                f"capability {probe.name} claims marker {marker!r} but it is "
                "absent from the HTTP dispatcher"
            )


def test_every_declared_capability_probe_resolves_on_this_build():
    """No declared capability is unresolvable here (catches typo'd symbols)."""
    declared = {p.name for p in capabilities.all_probes()}
    assert set(capabilities.capabilities()) == declared


# ---------------------------------------------------------------------------
# commit / built_at resolution
# ---------------------------------------------------------------------------


def test_commit_is_null_when_the_package_is_not_a_git_checkout(tmp_path):
    """A pip install with no build stamp reports commit: null, never an error."""
    pkg = tmp_path / "site-packages" / "taosmd"
    pkg.mkdir(parents=True)
    commit, source = capabilities.resolve_commit(pkg)
    assert commit is None
    assert source is None


def test_commit_resolves_from_a_git_checkout(tmp_path):
    """A checkout with .git/HEAD pointing at a branch resolves that branch's sha."""
    root = tmp_path / "repo"
    pkg = root / "taosmd"
    pkg.mkdir(parents=True)
    git = root / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/master\n")
    sha = "0123456789abcdef0123456789abcdef01234567"
    (git / "refs" / "heads" / "master").write_text(sha + "\n")

    commit, source = capabilities.resolve_commit(pkg)
    assert commit == sha
    assert source == "git"


def test_commit_resolves_from_packed_refs(tmp_path):
    """Branch refs that have been packed still resolve (no loose ref file)."""
    root = tmp_path / "repo"
    pkg = root / "taosmd"
    pkg.mkdir(parents=True)
    git = root / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/master\n")
    sha = "89abcdef0123456789abcdef0123456789abcdef"
    (git / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted\n"
        f"{sha} refs/heads/master\n"
    )

    commit, source = capabilities.resolve_commit(pkg)
    assert commit == sha


def test_commit_resolves_detached_head(tmp_path):
    root = tmp_path / "repo"
    pkg = root / "taosmd"
    pkg.mkdir(parents=True)
    git = root / ".git"
    git.mkdir()
    sha = "abcdef0123456789abcdef0123456789abcdef01"
    (git / "HEAD").write_text(sha + "\n")

    commit, source = capabilities.resolve_commit(pkg)
    assert commit == sha


def test_commit_resolution_never_raises_on_a_corrupt_git_dir(tmp_path):
    """Garbage in .git degrades to null rather than breaking the endpoint."""
    root = tmp_path / "repo"
    pkg = root / "taosmd"
    pkg.mkdir(parents=True)
    git = root / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/gone\n")

    commit, source = capabilities.resolve_commit(pkg)
    assert commit is None
    assert source is None


def test_commit_prefers_a_build_stamp_over_the_checkout(tmp_path, monkeypatch):
    """A wheel built with a stamp reports the stamped sha."""
    monkeypatch.setattr(
        capabilities, "_build_stamp", lambda: {"commit": "f" * 40, "built_at": None}
    )
    commit, source = capabilities.resolve_commit(Path(tmp_path))
    assert commit == "f" * 40
    assert source == "build-stamp"


def test_build_info_is_cached_and_shaped(monkeypatch):
    """build_info() is resolved once at import and returns the documented keys."""
    info = capabilities.build_info()
    assert set(info) == {"commit", "commit_source", "built_at", "built_at_source"}
    assert info["commit"] is None or (
        isinstance(info["commit"], str) and len(info["commit"]) == 40
    )
    assert info["commit_source"] in (None, "git", "build-stamp")
    assert info["built_at"] is None or info["built_at"].endswith("Z")
    # Cached but copy-on-read: a caller cannot poison the cached answer.
    info["commit"] = "tampered"
    assert capabilities.build_info()["commit"] != "tampered"


def test_capabilities_are_copy_on_read():
    """Mutating the returned list cannot poison the cached capability answer."""
    caps = capabilities.capabilities()
    caps.append("fake.v9")
    assert "fake.v9" not in capabilities.capabilities()


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------


def test_version_endpoint_shape(live_server):
    status, body = _get(f"{live_server}/version")
    assert status == 200, body
    assert set(body) == {
        "version",
        "commit",
        "commit_source",
        "built_at",
        "built_at_source",
        "capabilities",
    }
    assert body["version"] == __version__
    assert isinstance(body["capabilities"], list)
    assert all(isinstance(c, str) for c in body["capabilities"])
    assert "collections.v1" in body["capabilities"]
    assert body["commit"] is None or isinstance(body["commit"], str)
    assert body["built_at"] is None or isinstance(body["built_at"], str)


def test_version_matches_the_module_derivation(live_server):
    """The endpoint must not maintain its own idea of the capability list."""
    _, body = _get(f"{live_server}/version")
    assert body["capabilities"] == capabilities.capabilities()


def test_health_keeps_its_existing_contract(live_server):
    """taOS and the dashboard already consume status+version; ADD only.

    This assertion is deliberately explicit so a future change cannot silently
    break the existing consumers.
    """
    status, body = _get(f"{live_server}/health")
    assert status == 200
    assert body["status"] == "ok"
    assert isinstance(body["version"], str) and body["version"]
    assert body["version"] == __version__


def test_health_gains_the_capability_list(live_server):
    _, body = _get(f"{live_server}/health")
    assert body["capabilities"] == capabilities.capabilities()


def test_version_and_health_leak_nothing_sensitive(live_server):
    """No paths, tokens, or config values beyond version + capabilities."""
    allowed_version = {
        "version",
        "commit",
        "commit_source",
        "built_at",
        "built_at_source",
        "capabilities",
    }
    _, version_body = _get(f"{live_server}/version")
    _, health_body = _get(f"{live_server}/health")
    assert set(version_body) <= allowed_version
    assert set(health_body) <= {"status", "version", "capabilities"}

    for body in (version_body, health_body):
        blob = json.dumps(body)
        assert "/" not in blob.replace("\\/", ""), f"looks like a path leaked: {blob}"
        for banned in ("token", "data_dir", "home", "secret", "password"):
            assert banned not in blob.lower(), f"{banned!r} leaked in {blob}"


# ---------------------------------------------------------------------------
# public by design, even with a server token configured
# ---------------------------------------------------------------------------


@pytest.fixture
def token_server(tmp_path, monkeypatch):
    """Server with a server_token configured; yields (base_url, token)."""
    data_dir = tmp_path / "token-data"
    data_dir.mkdir()
    token = "test-bearer-token-version"
    (data_dir / "config.json").write_text(json.dumps({"server_token": token}))

    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)

    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}", token
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        httpd.service_loop.close()


def _raw_get(url: str) -> tuple[int, dict]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def test_version_is_public_when_a_server_token_is_configured(token_server):
    base_url, _ = token_server
    status, body = _raw_get(f"{base_url}/version")
    assert status == 200, body
    assert body["version"] == __version__
    assert "collections.v1" in body["capabilities"]


def test_health_stays_public_when_a_server_token_is_configured(token_server):
    base_url, _ = token_server
    status, body = _raw_get(f"{base_url}/health")
    assert status == 200
    assert body["status"] == "ok"
    assert "capabilities" in body


def test_data_plane_is_still_gated_alongside_the_public_version(token_server):
    """Sanity: making /version public did not open the data plane."""
    base_url, _ = token_server
    status, _ = _raw_get(f"{base_url}/projects")
    assert status == 401
