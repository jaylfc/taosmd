"""Tests for POST /a2a/import: issuer-pinned auth, parity with /a2a/send, idempotency, and batch cap.

Covers all acceptance criteria from tsk-bdj4fv:
1. Auth parity: /a2a/import performs the same checks as /a2a/send on every branch.
2. Issuer pinning: a token from the wrong issuer is rejected with 403 and nothing is written.
3. Idempotency: re-importing the same batch yields no duplicates.
4. Batch cap: the number of envelopes per batch is bounded.
"""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.request

import pytest

from taosmd import api as taosmd_api
from taosmd import http_server, service
from taosmd.registry_auth import REGISTRY_ISS

pytest.importorskip("jwt")
pytest.importorskip("cryptography")

import jwt as pyjwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_embedder(stores: dict) -> None:
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "taosmd-import"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    yield data_dir
    for stores in list(taosmd_api._stores_cache.values()):
        for store in (stores.get("archive"), stores.get("vector"), stores.get("kg")):
            if store and hasattr(store, "close"):
                try:
                    asyncio.run(store.close())
                except Exception:
                    pass


def _setup_stores(data_dir):
    stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)
    return stores


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


def _make_token(sub, project_id=None, iss=None):
    claims = {"sub": sub}
    if project_id is not None:
        claims["project_id"] = project_id
    if iss is not None:
        claims["iss"] = iss
    return pyjwt.encode(claims, PRIV_PEM, algorithm="EdDSA")


def _post(url: str, payload, token=None) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        url, data=data, headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def _get(url: str, token=None) -> tuple[int, dict]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def _count_a2a_events_via_http(base: str, thread: str) -> int:
    status, body = _get(f"{base}/a2a/messages?thread={thread}")
    if status != 200:
        return -1
    return len(body.get("messages", []))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def authed_server(tmp_path, monkeypatch):
    """Live server built with a registry verifier in enforce mode, issuer pinned."""
    data_dir = tmp_path / "taosmd-import-http"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    from taosmd import config as cfg
    cfg.set_a2a_auth_enforce(True, str(data_dir))

    revoked: set[str] = set()

    def fake_opener(url, token=None):
        if url.endswith("pubkey"):
            return json.dumps({"pubkey": PUB_PEM})
        if url.endswith("revoked"):
            return json.dumps(list(revoked))
        return json.dumps([])

    from taosmd import registry_auth
    verifier = registry_auth.verifier_from_url(
        "http://reg.test", opener=fake_opener, expected_iss=REGISTRY_ISS,
    )
    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir), verifier=verifier)
    httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    host, port = httpd.server_address[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://{host}:{port}", data_dir, revoked
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
        httpd.service_loop.close()


@pytest.fixture
def authed_server_with_grants(tmp_path, monkeypatch):
    """Live server with verifier, grants verifier, and issuer pinning."""
    data_dir = tmp_path / "taosmd-import-grants-http"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    from taosmd import config as cfg
    cfg.set_a2a_auth_enforce(True, str(data_dir))

    grants: list[dict] = []

    def fake_opener(url, token=None):
        if url.endswith("pubkey"):
            return json.dumps({"pubkey": PUB_PEM})
        if url.endswith("revoked"):
            return json.dumps([])
        if url.endswith("grants"):
            return json.dumps({"grants": grants})
        return json.dumps([])

    def fake_grants_opener(url, token=None):
        if url.endswith("grants"):
            return json.dumps({"grants": grants})
        return json.dumps([])

    from taosmd import registry_auth
    verifier = registry_auth.verifier_from_url(
        "http://reg.test", opener=fake_opener, expected_iss=REGISTRY_ISS,
    )
    grants_verifier = registry_auth.grants_verifier_from_url(
        "http://reg.test", opener=fake_grants_opener,
    )
    httpd = http_server.make_server(
        "127.0.0.1", 0, data_dir=str(data_dir),
        verifier=verifier, grants_verifier=grants_verifier,
    )
    httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    host, port = httpd.server_address[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://{host}:{port}", data_dir, grants
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
        httpd.service_loop.close()


# ---------------------------------------------------------------------------
# Service-layer tests
# ---------------------------------------------------------------------------

def test_a2a_import_service_roundtrip(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    result = asyncio.run(service.a2a_import(
        [
            {"from": "agentA", "body": "hello import", "thread": "t1"},
        ],
        data_dir=dd,
    ))
    assert result == {"imported": 1, "deduped": 0, "total": 1}

    msgs = asyncio.run(service.a2a_feed(thread="t1", data_dir=dd))
    assert len(msgs) == 1
    assert msgs[0]["body"] == "hello import"


def test_a2a_import_service_idempotent(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    envelopes = [{"from": "agentA", "body": "idempotent", "thread": "t1"}]
    r1 = asyncio.run(service.a2a_import(envelopes, data_dir=dd))
    assert r1["imported"] == 1
    r2 = asyncio.run(service.a2a_import(envelopes, data_dir=dd))
    assert r2["imported"] == 0
    assert r2["deduped"] == 1

    msgs = asyncio.run(service.a2a_feed(thread="t1", data_dir=dd))
    assert len(msgs) == 1


def test_a2a_import_service_validates_batch_type(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    with pytest.raises(ValueError, match="list"):
        asyncio.run(service.a2a_import("not-a-list", data_dir=str(isolated_data_dir)))


def test_a2a_import_service_validates_envelope_type(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    with pytest.raises(ValueError, match="object"):
        asyncio.run(service.a2a_import(
            ["not-an-object"], data_dir=str(isolated_data_dir)
        ))


def test_a2a_import_service_rejects_batch_too_large(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)
    envelopes = [
        {"from": f"a{i}", "body": f"msg{i}", "thread": "t1"}
        for i in range(http_server._A2A_MAX_IMPORT_BATCH + 1)
    ]
    with pytest.raises(ValueError, match="at most"):
        asyncio.run(service.a2a_import(envelopes, data_dir=dd))


def test_a2a_import_service_accepts_batch_at_boundary(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)
    envelopes = [
        {"from": f"a{i}", "body": f"msg{i}", "thread": "t1"}
        for i in range(http_server._A2A_MAX_IMPORT_BATCH)
    ]
    result = asyncio.run(service.a2a_import(envelopes, data_dir=dd))
    assert result["imported"] == http_server._A2A_MAX_IMPORT_BATCH
    assert result["total"] == http_server._A2A_MAX_IMPORT_BATCH


def test_a2a_import_service_dedup_gate_has_teeth(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)
    envelope = {"from": "agentA", "body": "teeth", "thread": "t1"}

    asyncio.run(service.a2a_import([envelope], data_dir=dd))
    stores = asyncio.run(taosmd_api._ensure_stores(dd))
    archive = stores["archive"]
    original_get = archive.get_import_dedup
    async def _stub_get(key):
        return None
    archive.get_import_dedup = _stub_get  # type: ignore[assignment]
    try:
        result = asyncio.run(service.a2a_import([envelope], data_dir=dd))
    finally:
        archive.get_import_dedup = original_get  # type: ignore[assignment]
    assert result["imported"] == 1
    assert result["deduped"] == 0


# ---------------------------------------------------------------------------
# HTTP-layer: auth parity with /a2a/send
# ---------------------------------------------------------------------------

def test_http_a2a_import_no_token_refuses_and_nothing_written(authed_server):
    base, data_dir, _ = authed_server
    s, body = _post(f"{base}/a2a/import", {"envelopes": [
        {"from": "agentA", "body": "secret", "thread": "t1"},
    ]})
    assert s == 401, body
    assert _count_a2a_events_via_http(base, "t1") == 0


def test_http_a2a_import_malformed_token_refuses_and_nothing_written(authed_server):
    base, data_dir, _ = authed_server
    s, body = _post(f"{base}/a2a/import", {"envelopes": [
        {"from": "agentA", "body": "secret", "thread": "t1"},
    ]}, token="not-a-jwt-at-all")
    assert s == 403, body
    assert _count_a2a_events_via_http(base, "t1") == 0


def test_http_a2a_import_bad_signature_refuses_and_nothing_written(authed_server):
    base, data_dir, _ = authed_server
    bad_priv = Ed25519PrivateKey.generate()
    bad_token = pyjwt.encode({"sub": "agentA", "iss": REGISTRY_ISS}, bad_priv, algorithm="EdDSA")
    s, body = _post(f"{base}/a2a/import", {"envelopes": [
        {"from": "agentA", "body": "secret", "thread": "t1"},
    ]}, token=bad_token)
    assert s == 403, body
    assert _count_a2a_events_via_http(base, "t1") == 0


def test_http_a2a_import_revoked_token_refuses_and_nothing_written(authed_server):
    base, data_dir, revoked = authed_server
    revoked.add("agentA")
    token = _make_token("agentA", iss=REGISTRY_ISS)
    s, body = _post(f"{base}/a2a/import", {"envelopes": [
        {"from": "agentA", "body": "secret", "thread": "t1"},
    ]}, token=token)
    assert s == 403, body
    assert _count_a2a_events_via_http(base, "t1") == 0


def test_http_a2a_import_wrong_issuer_refuses_and_nothing_written(authed_server):
    base, data_dir, _ = authed_server
    wrong_iss_token = _make_token("agentA", iss="wrong-issuer")
    s, body = _post(f"{base}/a2a/import", {"envelopes": [
        {"from": "agentA", "body": "secret", "thread": "t1"},
    ]}, token=wrong_iss_token)
    assert s == 403, body
    assert "iss" in json.dumps(body).lower() or "registry auth" in json.dumps(body).lower()
    assert _count_a2a_events_via_http(base, "t1") == 0


def test_http_a2a_import_valid_token_accepts_and_writes(authed_server):
    base, data_dir, _ = authed_server
    token = _make_token("agentA", iss=REGISTRY_ISS)
    s, body = _post(f"{base}/a2a/import", {"envelopes": [
        {"from": "agentA", "body": "hello import", "thread": "t1"},
    ]}, token=token)
    assert s == 200, body
    assert body["imported"] == 1
    assert body["total"] == 1
    status, msgs_body = _get(f"{base}/a2a/messages?thread=t1")
    assert status == 200, msgs_body
    assert any(m["body"] == "hello import" for m in msgs_body.get("messages", []))


# ---------------------------------------------------------------------------
# HTTP-layer: parity with /a2a/send for every branch
# ---------------------------------------------------------------------------

def test_http_a2a_send_and_import_share_auth_gate(authed_server):
    base, data_dir, _ = authed_server
    token = _make_token("agentA", iss=REGISTRY_ISS)

    s_send, _ = _post(f"{base}/a2a/send", {"from": "agentA", "body": "send", "thread": "t1"}, token=token)
    s_import, _ = _post(f"{base}/a2a/import", {"envelopes": [
        {"from": "agentA", "body": "import", "thread": "t1"},
    ]}, token=token)
    assert s_send == 200
    assert s_import == 200


def test_http_a2a_import_parity_no_token(authed_server):
    base, data_dir, _ = authed_server
    s_send, _ = _post(f"{base}/a2a/send", {"from": "agentA", "body": "x", "thread": "t1"})
    s_import, _ = _post(f"{base}/a2a/import", {"envelopes": [
        {"from": "agentA", "body": "x", "thread": "t1"},
    ]})
    assert s_send == s_import == 401
    assert _count_a2a_events_via_http(base, "t1") == 0


def test_http_a2a_import_parity_malformed_token(authed_server):
    base, data_dir, _ = authed_server
    s_send, _ = _post(f"{base}/a2a/send", {"from": "agentA", "body": "x", "thread": "t1"}, token="bad")
    s_import, _ = _post(f"{base}/a2a/import", {"envelopes": [
        {"from": "agentA", "body": "x", "thread": "t1"},
    ]}, token="bad")
    assert s_send == s_import == 403
    assert _count_a2a_events_via_http(base, "t1") == 0


def test_http_a2a_import_parity_bad_signature(authed_server):
    base, data_dir, _ = authed_server
    bad_priv = Ed25519PrivateKey.generate()
    bad_token = pyjwt.encode({"sub": "agentA", "iss": REGISTRY_ISS}, bad_priv, algorithm="EdDSA")
    s_send, _ = _post(f"{base}/a2a/send", {"from": "agentA", "body": "x", "thread": "t1"}, token=bad_token)
    s_import, _ = _post(f"{base}/a2a/import", {"envelopes": [
        {"from": "agentA", "body": "x", "thread": "t1"},
    ]}, token=bad_token)
    assert s_send == s_import == 403
    assert _count_a2a_events_via_http(base, "t1") == 0


def test_http_a2a_import_parity_revoked(authed_server):
    base, data_dir, revoked = authed_server
    revoked.add("agentA")
    token = _make_token("agentA", iss=REGISTRY_ISS)
    s_send, _ = _post(f"{base}/a2a/send", {"from": "agentA", "body": "x", "thread": "t1"}, token=token)
    s_import, _ = _post(f"{base}/a2a/import", {"envelopes": [
        {"from": "agentA", "body": "x", "thread": "t1"},
    ]}, token=token)
    assert s_send == s_import == 403
    assert _count_a2a_events_via_http(base, "t1") == 0


def test_http_a2a_import_parity_wrong_issuer(authed_server):
    base, data_dir, _ = authed_server
    token = _make_token("agentA", iss="wrong-issuer")
    s_send, _ = _post(f"{base}/a2a/send", {"from": "agentA", "body": "x", "thread": "t1"}, token=token)
    s_import, _ = _post(f"{base}/a2a/import", {"envelopes": [
        {"from": "agentA", "body": "x", "thread": "t1"},
    ]}, token=token)
    assert s_send == s_import == 403
    assert _count_a2a_events_via_http(base, "t1") == 0


def test_http_a2a_import_parity_valid_token(authed_server):
    base, data_dir, _ = authed_server
    token = _make_token("agentA", iss=REGISTRY_ISS)
    s_send, r_send = _post(f"{base}/a2a/send", {"from": "agentA", "body": "send-ok", "thread": "t1"}, token=token)
    s_import, r_import = _post(f"{base}/a2a/import", {"envelopes": [
        {"from": "agentA", "body": "import-ok", "thread": "t1"},
    ]}, token=token)
    assert s_send == 200
    assert s_import == 200
    assert r_send["id"] > 0
    assert r_import["imported"] == 1


# ---------------------------------------------------------------------------
# HTTP-layer: batch cap
# ---------------------------------------------------------------------------

def test_http_a2a_import_batch_too_large_returns_400(authed_server):
    base, data_dir, _ = authed_server
    token = _make_token("agentA", iss=REGISTRY_ISS)
    envelopes = [
        {"from": f"a{i}", "body": f"m{i}", "thread": "t1"}
        for i in range(http_server._A2A_MAX_IMPORT_BATCH + 1)
    ]
    s, body = _post(f"{base}/a2a/import", {"envelopes": envelopes}, token=token)
    assert s == 400, body
    assert _count_a2a_events_via_http(base, "t1") == 0


def test_http_a2a_import_batch_at_boundary_accepted(authed_server):
    base, data_dir, _ = authed_server
    token = _make_token("agentA", iss=REGISTRY_ISS)
    envelopes = [
        {"from": f"a{i}", "body": f"m{i}", "thread": "t1"}
        for i in range(http_server._A2A_MAX_IMPORT_BATCH)
    ]
    s, body = _post(f"{base}/a2a/import", {"envelopes": envelopes}, token=token)
    assert s == 200, body
    assert body["imported"] == http_server._A2A_MAX_IMPORT_BATCH


# ---------------------------------------------------------------------------
# HTTP-layer: idempotency
# ---------------------------------------------------------------------------

def test_http_a2a_import_idempotent(authed_server):
    base, data_dir, _ = authed_server
    token = _make_token("agentA", iss=REGISTRY_ISS)
    envelopes = [{"from": "agentA", "body": "once", "thread": "t1"}]

    s1, b1 = _post(f"{base}/a2a/import", {"envelopes": envelopes}, token=token)
    assert s1 == 200
    assert b1["imported"] == 1

    s2, b2 = _post(f"{base}/a2a/import", {"envelopes": envelopes}, token=token)
    assert s2 == 200
    assert b2["imported"] == 0
    assert b2["deduped"] == 1

    assert _count_a2a_events_via_http(base, "t1") == 1


def test_http_a2a_import_partial_idempotent(authed_server):
    base, data_dir, _ = authed_server
    token = _make_token("agentA", iss=REGISTRY_ISS)
    envelopes = [
        {"from": "agentA", "body": "first", "thread": "t1"},
        {"from": "agentA", "body": "second", "thread": "t1"},
    ]

    s1, b1 = _post(f"{base}/a2a/import", {"envelopes": envelopes}, token=token)
    assert s1 == 200
    assert b1["imported"] == 2

    s2, b2 = _post(f"{base}/a2a/import", {"envelopes": envelopes}, token=token)
    assert s2 == 200
    assert b2["deduped"] == 2
    assert b2["imported"] == 0

    assert _count_a2a_events_via_http(base, "t1") == 2


# ---------------------------------------------------------------------------
# HTTP-layer: envelope validation
# ---------------------------------------------------------------------------

def test_http_a2a_import_missing_envelopes_returns_400(authed_server):
    base, data_dir, _ = authed_server
    token = _make_token("agentA", iss=REGISTRY_ISS)
    s, body = _post(f"{base}/a2a/import", {}, token=token)
    assert s == 400, body


def test_http_a2a_import_envelope_missing_from_returns_400(authed_server):
    base, data_dir, _ = authed_server
    token = _make_token("agentA", iss=REGISTRY_ISS)
    s, body = _post(f"{base}/a2a/import", {"envelopes": [
        {"body": "no-from", "thread": "t1"},
    ]}, token=token)
    assert s == 400, body
    assert _count_a2a_events_via_http(base, "t1") == 0


def test_http_a2a_import_envelope_missing_body_returns_400(authed_server):
    base, data_dir, _ = authed_server
    token = _make_token("agentA", iss=REGISTRY_ISS)
    s, body = _post(f"{base}/a2a/import", {"envelopes": [
        {"from": "agentA", "thread": "t1"},
    ]}, token=token)
    assert s == 400, body
    assert _count_a2a_events_via_http(base, "t1") == 0


def test_http_a2a_import_bad_kind_returns_400(authed_server):
    base, data_dir, _ = authed_server
    token = _make_token("agentA", iss=REGISTRY_ISS)
    s, body = _post(f"{base}/a2a/import", {"envelopes": [
        {"from": "agentA", "body": "x", "thread": "t1", "kind": "invalid"},
    ]}, token=token)
    assert s == 400, body
    assert _count_a2a_events_via_http(base, "t1") == 0


def test_http_a2a_import_refs_too_many_returns_400(authed_server):
    base, data_dir, _ = authed_server
    token = _make_token("agentA", iss=REGISTRY_ISS)
    s, body = _post(f"{base}/a2a/import", {"envelopes": [
        {"from": "agentA", "body": "x", "thread": "t1", "refs": [{"kind": "doc"}] * 9},
    ]}, token=token)
    assert s == 400, body
    assert _count_a2a_events_via_http(base, "t1") == 0


def test_http_a2a_import_blocks_not_list_returns_400(authed_server):
    base, data_dir, _ = authed_server
    token = _make_token("agentA", iss=REGISTRY_ISS)
    s, body = _post(f"{base}/a2a/import", {"envelopes": [
        {"from": "agentA", "body": "x", "thread": "t1", "blocks": "not-a-list"},
    ]}, token=token)
    assert s == 400, body
    assert _count_a2a_events_via_http(base, "t1") == 0


def test_http_a2a_import_message_too_large_returns_400(authed_server):
    base, data_dir, _ = authed_server
    token = _make_token("agentA", iss=REGISTRY_ISS)
    big_body = "x" * (64 * 1024)
    s, body = _post(f"{base}/a2a/import", {"envelopes": [
        {"from": "agentA", "body": big_body, "thread": "t1"},
    ]}, token=token)
    assert s == 400, body
    assert _count_a2a_events_via_http(base, "t1") == 0
