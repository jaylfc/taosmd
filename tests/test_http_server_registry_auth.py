"""A2A bus authentication wired into the HTTP server (opt-in).

When the server is built with a registry verifier, POST /a2a/send requires a
valid EdDSA-JWT Bearer token whose ``sub`` matches the message ``from``. With
no verifier (the default / standalone), the bus trusts the handle as before.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

pytest.importorskip("jwt")
pytest.importorskip("cryptography")

import jwt as pyjwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from taosmd import api as taosmd_api
from taosmd import http_server, registry_auth


def _keypair():
    priv = Ed25519PrivateKey.generate()
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return priv_pem, pub_pem


PRIV_PEM, PUB_PEM = _keypair()


def _post_send(base_url, from_, body, token=None):
    payload = json.dumps({"from": from_, "body": body}).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(base_url + "/a2a/send", data=payload,
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


@pytest.fixture
def authed_server(tmp_path, monkeypatch):
    """Live server built with a registry verifier (fake opener, no network)."""
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    def fake_opener(url, token=None):
        if url.endswith(registry_auth.PUBKEY_PATH):
            return json.dumps({"pubkey": PUB_PEM})
        return json.dumps([])  # nothing revoked

    # expected_iss=None opts out of issuer pinning (the default is now the
    # pinned taOS registry iss); this fixture tests the auth mechanics alone.
    verifier = registry_auth.verifier_from_url(
        "http://reg.test", opener=fake_opener, expected_iss=None)
    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir),
                                    verifier=verifier)
    httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        httpd.service_loop.close()


def test_send_without_token_is_rejected(authed_server):
    status, _ = _post_send(authed_server, "agent-1", "hello")
    assert status in (401, 403)


def test_send_with_valid_matching_token_succeeds(authed_server):
    token = pyjwt.encode({"sub": "agent-1"}, PRIV_PEM, algorithm="EdDSA")
    status, body = _post_send(authed_server, "agent-1", "hello", token=token)
    assert status == 200
    assert body.get("id") is not None or body.get("status") == "ok" or body


def test_send_with_mismatched_sub_is_rejected(authed_server):
    token = pyjwt.encode({"sub": "agent-OTHER"}, PRIV_PEM, algorithm="EdDSA")
    status, _ = _post_send(authed_server, "agent-1", "hello", token=token)
    assert status == 403


def test_send_with_token_from_wrong_key_is_rejected(authed_server):
    other_priv, _ = _keypair()
    token = pyjwt.encode({"sub": "agent-1"}, other_priv, algorithm="EdDSA")
    status, _ = _post_send(authed_server, "agent-1", "hello", token=token)
    assert status in (401, 403)


@pytest.fixture
def iss_pinned_server(tmp_path, monkeypatch):
    """Live server whose verifier pins iss to the taOS registry value."""
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    def fake_opener(url, token=None):
        if url.endswith(registry_auth.PUBKEY_PATH):
            return json.dumps({"public_key": PUB_PEM})
        return json.dumps({"revoked": []})

    verifier = registry_auth.verifier_from_url(
        "http://reg.test", opener=fake_opener,
        expected_iss=registry_auth.REGISTRY_ISS)
    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir),
                                    verifier=verifier)
    httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        httpd.service_loop.close()


def test_pinned_issuer_accepts_token_with_correct_iss(iss_pinned_server):
    token = pyjwt.encode({"sub": "agent-1", "iss": registry_auth.REGISTRY_ISS},
                         PRIV_PEM, algorithm="EdDSA")
    status, _ = _post_send(iss_pinned_server, "agent-1", "hello", token=token)
    assert status == 200


def test_pinned_issuer_rejects_token_with_wrong_iss(iss_pinned_server):
    token = pyjwt.encode({"sub": "agent-1", "iss": "someone-else"},
                         PRIV_PEM, algorithm="EdDSA")
    status, _ = _post_send(iss_pinned_server, "agent-1", "hello", token=token)
    assert status == 403


def test_server_builds_verifier_with_token_and_pinned_issuer(tmp_path, monkeypatch):
    # When a registry URL is configured, the server must build its verifier with
    # the configured admin token (for the auth-gated revoked feed) and pin the
    # issuer to the taOS registry value (#710 contract).
    from taosmd import config

    data_dir = tmp_path / "cfg-data"
    data_dir.mkdir()
    monkeypatch.delenv("TAOSMD_REGISTRY_URL", raising=False)
    monkeypatch.delenv("TAOSMD_REGISTRY_TOKEN", raising=False)
    config.set_registry_url("http://reg.test", data_dir=str(data_dir))
    config.set_registry_token("admin-token", data_dir=str(data_dir))

    captured = {}

    def fake_verifier_from_url(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return object()  # dummy; never invoked in this test

    monkeypatch.setattr(registry_auth, "verifier_from_url", fake_verifier_from_url)

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    try:
        assert captured["url"] == "http://reg.test"
        assert captured.get("revoked_token") == "admin-token"
        assert captured.get("expected_iss") == registry_auth.REGISTRY_ISS
    finally:
        httpd.service_loop.close()
        httpd.server_close()


# ---------------------------------------------------------------------------
# Project-scoped grant binding (taOS#744)
# ---------------------------------------------------------------------------

def _make_token(sub, project_id=None, iss=None):
    """Mint an EdDSA JWT with the given sub and optional project_id/iss."""
    claims = {"sub": sub}
    if project_id is not None:
        claims["project_id"] = project_id
    if iss is not None:
        claims["iss"] = iss
    return pyjwt.encode(claims, PRIV_PEM, algorithm="EdDSA")


def _post_json(base_url, path, payload, token=None):
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(base_url + path, data=data,
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def _get_json(base_url, path, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(base_url + path, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


@pytest.fixture
def project_server(tmp_path, monkeypatch):
    """Server with registry verifier + grants verifier for project-scoped tests.

    Grants feed: agent-scoped "proj-a" grants exist for "agent-1"; no grant
    for "proj-b", but a global (no project_id) grant for "agent-global".
    """
    data_dir = tmp_path / "taosmd-proj"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    def fake_opener(url, token=None):
        if url.endswith(registry_auth.PUBKEY_PATH):
            return json.dumps({"pubkey": PUB_PEM})
        return json.dumps([])  # nothing revoked

    # iss=REGISTRY_ISS because tokens will carry it
    verifier = registry_auth.verifier_from_url(
        "http://reg.test", opener=fake_opener,
        expected_iss=registry_auth.REGISTRY_ISS)

    grants = [
        {"canonical_id": "agent-1", "project_id": "proj-a"},
        {"canonical_id": "agent-global"},  # global grant — no project_id
    ]
    gv = registry_auth.GrantsVerifier(grants_loader=lambda: grants)

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir),
                                    verifier=verifier, grants_verifier=gv)
    httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        httpd.service_loop.close()


def test_ingest_token_project_id_overrides_body_project(project_server):
    """A token with project_id=proj-a forces ingest into proj-a even if body says proj-b."""
    token = _make_token("agent-1", project_id="proj-a", iss=registry_auth.REGISTRY_ISS)
    # Body claims proj-b, but the verified token says proj-a.
    status, body = _post_json(project_server, "/ingest",
                              {"text": "hello from proj-a", "agent": "agent-1",
                               "project": "proj-b"},
                              token=token)
    assert status == 200, body


def test_search_token_project_id_scopes_results(project_server):
    """Ingest under token proj-a, then search with same token and find the memory."""
    token = _make_token("agent-1", project_id="proj-a", iss=registry_auth.REGISTRY_ISS)
    # Ingest (token forces project=proj-a regardless of body)
    s, b = _post_json(project_server, "/ingest",
                      {"text": "zebra memory marker", "agent": "agent-1"},
                      token=token)
    assert s == 200, b
    # Search should find it scoped to proj-a
    s2, b2 = _post_json(project_server, "/search",
                        {"query": "zebra", "agent": "agent-1"},
                        token=token)
    assert s2 == 200, b2
    # The result is scoped to the project bound by the token
    hits = b2.get("hits", [])
    assert any("zebra" in h.get("text", "") for h in hits)


def test_invalid_token_on_ingest_is_403(project_server):
    """A presented-but-invalid Bearer token on /ingest yields 403."""
    bad_priv, _ = _keypair()
    token = pyjwt.encode({"sub": "agent-1", "iss": registry_auth.REGISTRY_ISS,
                          "project_id": "proj-a"}, bad_priv, algorithm="EdDSA")
    status, _ = _post_json(project_server, "/ingest",
                           {"text": "spoof", "agent": "agent-1"}, token=token)
    assert status == 403


def test_no_token_ingest_is_unchanged(project_server):
    """Without a token, /ingest works exactly as before (no new requirements)."""
    status, body = _post_json(project_server, "/ingest",
                              {"text": "no token ingest", "agent": "agent-notoken"})
    assert status == 200, body


def test_grant_present_for_sub_and_project_passes(project_server):
    """Grant for (agent-1, proj-a) allows ingest when token claims proj-a."""
    token = _make_token("agent-1", project_id="proj-a", iss=registry_auth.REGISTRY_ISS)
    status, body = _post_json(project_server, "/ingest",
                              {"text": "granted", "agent": "agent-1"}, token=token)
    assert status == 200, body


def test_grant_only_for_other_project_yields_403(project_server):
    """agent-1 only has a grant for proj-a; token claiming proj-b yields 403."""
    token = _make_token("agent-1", project_id="proj-b", iss=registry_auth.REGISTRY_ISS)
    status, _ = _post_json(project_server, "/ingest",
                           {"text": "denied", "agent": "agent-1"}, token=token)
    assert status == 403


def test_global_grant_holder_passes_any_project(project_server):
    """agent-global has a global grant; token with any project_id should pass."""
    token = _make_token("agent-global", project_id="proj-whatever",
                        iss=registry_auth.REGISTRY_ISS)
    status, body = _post_json(project_server, "/ingest",
                              {"text": "global agent", "agent": "agent-global"},
                              token=token)
    assert status == 200, body


def test_token_binding_allows_delegated_shelf_writes(project_server):
    """The agent field names a target shelf, not the caller: a verified token
    whose sub differs from the agent field must still bind and pass (the taOS
    proxy writes the user-memory shelf under its own controller token)."""
    token = pyjwt.encode(
        {"sub": "agent-1", "iss": registry_auth.REGISTRY_ISS, "project_id": "proj-a"},
        PRIV_PEM, algorithm="EdDSA",
    )
    status, body = _post_json(
        project_server, "/ingest",
        {"text": "delegated shelf write", "agent": "user-memory"},
        token=token,
    )
    assert status == 200
    assert body["project"] == "proj-a"
