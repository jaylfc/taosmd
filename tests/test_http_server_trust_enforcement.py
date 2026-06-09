"""Trust & Comms enforcement: grant check on A2A bus + dashboard gating.

Tests the two new enforcement layers added on top of the existing registry
verifier (test_http_server_registry_auth.py covers token verification alone):

1. Grant check: a valid token + an active grant is required; a valid token
   without a grant is rejected with 403.
2. Dashboard gating: when managed_by=taos and serve_dashboard is not overridden,
   GET / and GET /ui return 404; API routes stay up.
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
from taosmd import config as cfg
from taosmd import http_server, registry_auth


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _mint(canonical_id: str) -> str:
    return pyjwt.encode(
        {"sub": canonical_id, "iss": registry_auth.REGISTRY_ISS},
        PRIV_PEM, algorithm="EdDSA",
    )


def _post_send(base_url, from_, body, token=None):
    payload = json.dumps({"from": from_, "body": body}).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        base_url + "/a2a/send", data=payload, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def _get(base_url, path):
    req = urllib.request.Request(base_url + path)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def _make_verifiers(grants: list[dict]):
    """Build (registry_verifier, grants_verifier) pair with a fake network."""
    def fake_opener(url, timeout=5.0, token=None):
        if url.endswith(registry_auth.PUBKEY_PATH):
            return json.dumps({"pubkey": PUB_PEM})
        if url.endswith(registry_auth.REVOKED_PATH):
            return json.dumps([])
        if url.endswith(registry_auth.GRANTS_PATH):
            return json.dumps({"grants": grants})
        raise ValueError(f"unexpected url: {url}")

    verifier = registry_auth.verifier_from_url(
        "http://reg.test", opener=fake_opener,
        expected_iss=registry_auth.REGISTRY_ISS,
    )
    gv = registry_auth.grants_verifier_from_url(
        "http://reg.test", opener=fake_opener,
    )
    return verifier, gv


@pytest.fixture
def enforced_server(tmp_path, monkeypatch):
    """Server with both token verifier AND grants verifier wired in."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    verifier, gv = _make_verifiers([{"canonical_id": "agent-allowed"}])
    httpd = http_server.make_server(
        "127.0.0.1", 0, data_dir=str(data_dir),
        verifier=verifier, grants_verifier=gv,
    )
    httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.service_loop.close()


# ---------------------------------------------------------------------------
# Grant check tests
# ---------------------------------------------------------------------------

def test_send_allowed_with_valid_token_and_grant(enforced_server):
    token = _mint("agent-allowed")
    status, body = _post_send(enforced_server, "agent-allowed", "hello", token=token)
    assert status == 200, body


def test_send_rejected_no_token(enforced_server):
    status, body = _post_send(enforced_server, "agent-allowed", "hello")
    assert status == 401


def test_send_rejected_valid_token_no_grant(enforced_server):
    # agent-no-grant has a valid token but is not in the grants feed.
    token = _mint("agent-no-grant")
    status, body = _post_send(enforced_server, "agent-no-grant", "hello", token=token)
    assert status == 403
    assert "grant" in body.get("error", "").lower()


def test_send_rejected_expired_grant(tmp_path, monkeypatch):
    """An expired grant is treated the same as no grant."""
    import time
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    past = time.time() - 10.0
    verifier, gv = _make_verifiers(
        [{"canonical_id": "agent-expired", "expires_at": past}]
    )
    httpd = http_server.make_server(
        "127.0.0.1", 0, data_dir=str(data_dir),
        verifier=verifier, grants_verifier=gv,
    )
    httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        token = _mint("agent-expired")
        status, body = _post_send(
            f"http://{host}:{port}", "agent-expired", "hi", token=token
        )
        assert status == 403
    finally:
        httpd.shutdown()
        httpd.service_loop.close()


# ---------------------------------------------------------------------------
# Dashboard gating tests
# ---------------------------------------------------------------------------

def _server_with_managed_by(tmp_path, monkeypatch, managed_by, serve_dashboard=None):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    cfg.set_managed_by(managed_by, data_dir)
    if serve_dashboard is not None:
        cfg.set_serve_dashboard(serve_dashboard, data_dir)
    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, f"http://{host}:{port}"


def test_dashboard_served_when_standalone(tmp_path, monkeypatch):
    httpd, base = _server_with_managed_by(
        tmp_path, monkeypatch, cfg.MANAGED_BY_STANDALONE
    )
    try:
        status, _ = _get(base, "/")
        assert status == 200
        status2, _ = _get(base, "/ui")
        assert status2 == 200
    finally:
        httpd.shutdown()
        httpd.service_loop.close()


def test_dashboard_hidden_when_taos(tmp_path, monkeypatch):
    httpd, base = _server_with_managed_by(
        tmp_path, monkeypatch, cfg.MANAGED_BY_TAOS
    )
    try:
        status, _ = _get(base, "/")
        assert status == 404
        status2, _ = _get(base, "/ui")
        assert status2 == 404
    finally:
        httpd.shutdown()
        httpd.service_loop.close()


def test_dashboard_override_serves_when_taos(tmp_path, monkeypatch):
    httpd, base = _server_with_managed_by(
        tmp_path, monkeypatch, cfg.MANAGED_BY_TAOS, serve_dashboard=True
    )
    try:
        status, _ = _get(base, "/")
        assert status == 200
    finally:
        httpd.shutdown()
        httpd.service_loop.close()


def test_api_still_up_when_dashboard_hidden(tmp_path, monkeypatch):
    httpd, base = _server_with_managed_by(
        tmp_path, monkeypatch, cfg.MANAGED_BY_TAOS
    )
    try:
        status, body = _get(base, "/health")
        assert status == 200
        assert json.loads(body)["status"] == "ok"
    finally:
        httpd.shutdown()
        httpd.service_loop.close()
