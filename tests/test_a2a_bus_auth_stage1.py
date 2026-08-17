"""A2A bus auth Stage 1: accept-and-annotate every /a2a/send.

Stage 1 observes identity without rejecting anything.  Every send that was
accepted before is still accepted.  The annotation records:
  * auth        -- verified | unsigned | invalid
  * verified_sub-- the token sub, or null when unsigned
  * from_raw    -- literal from string as sent
  * from_normalised-- strip leading @, casefold; mapped from verified_sub
                     when present, otherwise from from_raw
  * from_mismatch-- whether from_raw disagrees with from_normalised
  * reason      -- present only when auth is invalid or a grant check failed

Tests:
  - unsigned send: accepted, auth=unsigned, verified_sub null
  - valid token: accepted, auth=verified, verified_sub set, from_normalised set
  - invalid signature: ACCEPTED (not rejected), auth=invalid, reason recorded
  - from disagrees with token sub: ACCEPTED, from_mismatch true
  - both handle spellings (@taOSmd-dev and taosmd-dev) normalise to ONE identity
  - regression: a request master accepts, this accepts
"""
from __future__ import annotations

import asyncio
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
from taosmd import http_server, registry_auth, service


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


def _patch_embedder(stores: dict) -> None:
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "taosmd-a2a-stage1"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)
    yield data_dir
    for s in list(taosmd_api._stores_cache.values()):
        for store in (s.get("archive"), s.get("vector"), s.get("kg")):
            if store and hasattr(store, "close"):
                try:
                    asyncio.run(store.close())
                except Exception:
                    pass


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    data_dir = tmp_path / "taosmd-a2a-stage1-http"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)

    host, port = httpd.server_address[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
        for s in list(taosmd_api._stores_cache.values()):
            for store in (s.get("archive"), s.get("vector"), s.get("kg")):
                if store and hasattr(store, "close"):
                    try:
                        httpd.service_loop.run(store.close())
                    except Exception:
                        pass
        httpd.service_loop.close()


@pytest.fixture
def warn_server(tmp_path, monkeypatch):
    data_dir = tmp_path / "taosmd-a2a-stage1-warn"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    def fake_opener(url, timeout=5.0, token=None):
        if url.endswith(registry_auth.PUBKEY_PATH):
            return json.dumps({"pubkey": PUB_PEM})
        if url.endswith(registry_auth.REVOKED_PATH):
            return json.dumps([])
        if url.endswith(registry_auth.GRANTS_PATH):
            return json.dumps({"grants": [{"canonical_id": "taosmd-dev"}]})
        raise ValueError(f"unexpected url: {url}")

    verifier = registry_auth.verifier_from_url(
        "http://reg.test", opener=fake_opener,
        expected_iss=registry_auth.REGISTRY_ISS,
    )
    gv = registry_auth.grants_verifier_from_url(
        "http://reg.test", opener=fake_opener,
    )
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
        httpd.server_close()
        t.join(timeout=5)
        httpd.service_loop.close()


# ---------------------------------------------------------------------------
# Service-layer annotation defaults
# ---------------------------------------------------------------------------

def test_unsigned_service_layer_defaults(isolated_data_dir):
    """service.a2a_send without _auth records unsigned annotation."""
    dd = str(isolated_data_dir)
    receipt = asyncio.run(service.a2a_send("alice", "hello", data_dir=dd))
    assert receipt["auth"] == "unsigned"
    assert receipt["verified_sub"] is None
    assert receipt["from_raw"] == "alice"
    assert receipt["from_normalised"] == "alice"
    assert receipt["from_mismatch"] is False
    assert "reason" not in receipt

    msgs = asyncio.run(service.a2a_feed(data_dir=dd))
    assert len(msgs) == 1
    m = msgs[0]
    assert m["auth"] == "unsigned"
    assert m["verified_sub"] is None
    assert m["from_raw"] == "alice"
    assert m["from_normalised"] == "alice"
    assert m["from_mismatch"] is False
    assert "reason" not in m


# ---------------------------------------------------------------------------
# HTTP-layer: no verifier configured (unsigned baseline)
# ---------------------------------------------------------------------------

def test_unsigned_http_accepted(live_server):
    """No verifier: send is accepted with unsigned annotation."""
    status, body = _post_send(live_server, "alice", "hello")
    assert status == 200, body
    assert body["auth"] == "unsigned"
    assert body["verified_sub"] is None
    assert body["from_raw"] == "alice"
    assert body["from_normalised"] == "alice"
    assert body["from_mismatch"] is False
    assert "reason" not in body


# ---------------------------------------------------------------------------
# HTTP-layer: valid token
# ---------------------------------------------------------------------------

def test_valid_token_accepted_and_annotated(warn_server):
    token = _mint("taosmd-dev")
    status, body = _post_send(warn_server, "taosmd-dev", "hello", token=token)
    assert status == 200, body
    assert body["auth"] == "verified"
    assert body["verified_sub"] == "taosmd-dev"
    assert body["from_raw"] == "taosmd-dev"
    assert body["from_normalised"] == "taosmd-dev"
    assert body["from_mismatch"] is False
    assert "reason" not in body


# ---------------------------------------------------------------------------
# HTTP-layer: invalid signature (accepted, not rejected)
# ---------------------------------------------------------------------------

def test_invalid_signature_accepted_and_annotated(warn_server):
    token = "not-a-jwt-at-all"
    status, body = _post_send(warn_server, "alice", "hello", token=token)
    assert status == 200, body
    assert body["auth"] == "invalid"
    assert body["verified_sub"] is None
    assert body["from_raw"] == "alice"
    assert body["from_normalised"] == "alice"
    assert body["from_mismatch"] is False
    assert "reason" in body
    assert "token verification failed" in body["reason"]


# ---------------------------------------------------------------------------
# HTTP-layer: from disagrees with token sub (accepted, from_mismatch true)
# ---------------------------------------------------------------------------

def test_from_mismatch_accepted_and_annotated(warn_server):
    token = _mint("agent-1")
    status, body = _post_send(warn_server, "agent-OTHER", "hello", token=token)
    assert status == 200, body
    assert body["auth"] == "invalid"
    assert body["verified_sub"] == "agent-1"
    assert body["from_raw"] == "agent-OTHER"
    assert body["from_normalised"] == "agent-1"
    assert body["from_mismatch"] is True
    assert "reason" in body


# ---------------------------------------------------------------------------
# Normalisation: handle spellings collapse to one identity
# ---------------------------------------------------------------------------

def test_normalise_handle_at_prefix_and_casefold():
    assert service._normalise_handle("@taOSmd-dev") == "taosmd-dev"
    assert service._normalise_handle("taosmd-dev") == "taosmd-dev"
    assert service._normalise_handle("@TAOSMD-DEV") == "taosmd-dev"


def test_normalised_identity_matches_via_http(warn_server):
    token = _mint("taosmd-dev")
    status, body = _post_send(warn_server, "@taOSmd-dev", "hello", token=token)
    assert status == 200, body
    assert body["from_normalised"] == "taosmd-dev"
    assert body["from_mismatch"] is False


# ---------------------------------------------------------------------------
# Regression: no new 4xx
# ---------------------------------------------------------------------------

def test_no_new_4xx_without_verifier(live_server):
    status, _ = _post_send(live_server, "any-agent", "hello")
    assert status == 200


def test_no_new_4xx_with_verifier_warn_mode(warn_server):
    status, _ = _post_send(warn_server, "any-agent", "hello")
    assert status == 200
