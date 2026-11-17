"""A2A bus read auth enforcement (tsk-rn6rjb).

When a2a_auth_enforce is True and a registry verifier is configured, every
GET /a2a read route requires a verified Bearer token with an active
a2a_receive grant. Warn mode and standalone (no verifier) remain unchanged.
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


def _get(base_url, path, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(base_url + path, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def _make_verifiers(grants: list[dict]):
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


def _start_server(tmp_path, monkeypatch, *, enforce, verifier, grants_verifier):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    cfg.set_a2a_auth_enforce(enforce, str(data_dir))
    httpd = http_server.make_server(
        "127.0.0.1", 0, data_dir=str(data_dir),
        verifier=verifier, grants_verifier=grants_verifier,
    )
    httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, f"http://{host}:{port}"


def _stop_server(httpd):
    httpd.shutdown()
    httpd.service_loop.close()


@pytest.fixture
def enforce_read_server(tmp_path, monkeypatch):
    verifier, gv = _make_verifiers([{"canonical_id": "agent-allowed", "scope": "a2a_receive"}])
    httpd, base = _start_server(
        tmp_path, monkeypatch, enforce=True, verifier=verifier, grants_verifier=gv,
    )
    try:
        yield base
    finally:
        _stop_server(httpd)


@pytest.fixture
def warn_read_server(tmp_path, monkeypatch):
    verifier, gv = _make_verifiers([{"canonical_id": "agent-allowed", "scope": "a2a_receive"}])
    httpd, base = _start_server(
        tmp_path, monkeypatch, enforce=False, verifier=verifier, grants_verifier=gv,
    )
    try:
        yield base
    finally:
        _stop_server(httpd)


@pytest.fixture
def standalone_enforce_server(tmp_path, monkeypatch):
    # No verifier configured; enforce flag must not gate reads.
    httpd, base = _start_server(
        tmp_path, monkeypatch, enforce=True, verifier=None, grants_verifier=None,
    )
    try:
        yield base
    finally:
        _stop_server(httpd)


def test_enforce_no_token_messages_401(enforce_read_server):
    status, body = _get(enforce_read_server, "/a2a/messages")
    assert status == 401, body
    assert "missing Bearer token" in body


def test_enforce_no_token_stream_401(enforce_read_server):
    status, body = _get(enforce_read_server, "/a2a/stream")
    assert status == 401, body
    assert "missing Bearer token" in body


def test_enforce_no_token_threads_401(enforce_read_server):
    status, body = _get(enforce_read_server, "/a2a/threads")
    assert status == 401, body
    assert "missing Bearer token" in body


def test_enforce_forged_token_403(enforce_read_server):
    status, body = _get(enforce_read_server, "/a2a/messages", token="forged.not.a.jwt")
    assert status == 403, body


def test_enforce_valid_token_without_a2a_receive_grant_403(enforce_read_server):
    token = _mint("agent-no-grant")
    status, body = _get(enforce_read_server, "/a2a/messages", token=token)
    assert status == 403, body
    assert "a2a_receive" in body


def test_enforce_valid_token_with_grant_200(enforce_read_server):
    token = _mint("agent-allowed")
    status, body = _get(enforce_read_server, "/a2a/messages", token=token)
    assert status == 200, body


def test_warn_mode_no_token_200(warn_read_server):
    status, body = _get(warn_read_server, "/a2a/messages")
    assert status == 200, body


def test_no_verifier_enforce_true_200(standalone_enforce_server):
    status, body = _get(standalone_enforce_server, "/a2a/messages")
    assert status == 200, body
