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

    verifier = registry_auth.verifier_from_url("http://reg.test", opener=fake_opener)
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
