"""RED tests for A2A per-channel ACL enforcement.

These tests must fail on unpatched code (no ACL enforcement) and pass after
the fix lands.  They follow the fixture and helper patterns from
``tests/test_http_server_trust_enforcement.py``.
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


def _forge(canonical_id: str) -> str:
    """Return a JWT-shaped string with a garbage signature (verification fails)."""
    token = _mint(canonical_id)
    parts = token.split(".")
    return f"{parts[0]}.{parts[1]}.forged_signature"


def _make_verifier(grants=None):
    def fake_opener(url, timeout=5.0, token=None):
        if url.endswith(registry_auth.PUBKEY_PATH):
            return json.dumps({"pubkey": PUB_PEM})
        if url.endswith(registry_auth.REVOKED_PATH):
            return json.dumps([])
        if url.endswith(registry_auth.GRANTS_PATH):
            return json.dumps({"grants": grants or []})
        raise ValueError(f"unexpected url: {url}")

    verifier = registry_auth.verifier_from_url(
        "http://reg.test", opener=fake_opener,
        expected_iss=registry_auth.REGISTRY_ISS,
    )
    gv = registry_auth.grants_verifier_from_url(
        "http://reg.test", opener=fake_opener,
    )
    return verifier, gv


def _post_send(base_url, from_, body, token=None, thread="general"):
    payload = json.dumps({"from": from_, "body": body, "thread": thread}).encode()
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


def _get_messages(base_url, thread="general"):
    req = urllib.request.Request(base_url + f"/a2a/messages?thread={thread}")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def _set_acl(data_dir, channel, read_ids=None, post_ids=None):
    cfg.set_acl(
        str(data_dir), channel=channel,
        read_ids=read_ids, post_ids=post_ids,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def acl_server(tmp_path, monkeypatch):
    """Server with registry verifier in enforce mode."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    cfg.set_a2a_auth_enforce(True, str(data_dir))

    verifier, gv = _make_verifier([{"canonical_id": "agent-allowed"}])
    httpd = http_server.make_server(
        "127.0.0.1", 0, data_dir=str(data_dir),
        verifier=verifier, grants_verifier=gv,
    )
    httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    host, port = httpd.server_address[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.service_loop.close()


@pytest.fixture
def acl_warn_server(tmp_path, monkeypatch):
    """Server with registry verifier in warn mode (a2a_auth_enforce=False)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    verifier, gv = _make_verifier([{"canonical_id": "agent-allowed"}])
    httpd = http_server.make_server(
        "127.0.0.1", 0, data_dir=str(data_dir),
        verifier=verifier, grants_verifier=gv,
    )
    httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    host, port = httpd.server_address[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.service_loop.close()


# ---------------------------------------------------------------------------
# Test (a): forged unsigned JWT whose sub IS in the allowlist must 403
# ---------------------------------------------------------------------------

def test_forged_jwt_with_matching_sub_is_rejected_by_acl(acl_warn_server, tmp_path):
    """A forged token whose sub happens to be in the post allowlist is 403.

    In warn mode, auth failures are accepted but ACL enforcement must still
    reject because the identity cannot be verified.
    """
    data_dir = tmp_path / "data"
    _set_acl(data_dir, "secret-chan", post_ids=["agent-allowed"])

    forged = _forge("agent-allowed")
    status, body = _post_send(acl_warn_server, "agent-allowed", "hello", token=forged, thread="secret-chan")
    assert status == 403, body


# ---------------------------------------------------------------------------
# Test (b): post with spoofed from to an ACLed channel must 403
# ---------------------------------------------------------------------------

def test_spoofed_from_is_rejected_by_acl(acl_warn_server, tmp_path):
    """Body `from` does not satisfy ACL; verified token identity does.

    Valid token for `agent-evil`, body claims `from: agent-allowed`, ACL
    allows only `agent-allowed`.  In warn mode auth accepts the mismatch,
    but ACL enforcement must reject because the verified identity is
    `agent-evil`, not `agent-allowed`.
    """
    data_dir = tmp_path / "data"
    _set_acl(data_dir, "secret-chan", post_ids=["agent-allowed"])

    token = _mint("agent-evil")
    status, body = _post_send(acl_warn_server, "agent-allowed", "hello", token=token, thread="secret-chan")
    assert status == 403, body


# ---------------------------------------------------------------------------
# Test (c): channel with no ACL entry behaves as before
# ---------------------------------------------------------------------------

def test_no_acl_entry_allows_post(acl_server):
    """A channel with no explicit ACL entry allows any verified identity."""
    token = _mint("agent-allowed")
    status, body = _post_send(acl_server, "agent-allowed", "hello", token=token, thread="no-acl-chan")
    assert status == 200, body
