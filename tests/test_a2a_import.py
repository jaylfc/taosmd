"""Tests for POST /a2a/import: idempotent batch import of external chat envelopes.

All tests drive a real HTTP server object over the real route; no mocked
clients. POST /a2a/send is included as a positive control in the same run so
a red result unambiguously means the new route is missing, not that the
harness is broken.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from taosmd import api as taosmd_api
from taosmd import http_server, service

pytest.importorskip("jwt")
import jwt as pyjwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization


def _patch_embedder(stores: dict) -> None:
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "taosmd-a2a-import"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    yield data_dir
    for stores in list(taosmd_api._stores_cache.values()):
        for store in (stores.get("archive"), stores.get("vector"), stores.get("kg")):
            if store and hasattr(store, "close"):
                try:
                    import asyncio
                    asyncio.run(store.close())
                except Exception:
                    pass


def _setup_stores(data_dir):
    stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)
    return stores


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    data_dir = tmp_path / "taosmd-a2a-import-http"
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


def _post(url: str, payload, headers: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(
        url, data=data, headers=hdrs, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _get(url: str) -> tuple[int, dict]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


# ---------------------------------------------------------------------------
# 1. Route existence: positive control with /a2a/send, new route /a2a/import
# ---------------------------------------------------------------------------

def test_http_a2a_import_route_exists(live_server):
    status, body = _post(
        f"{live_server}/a2a/send",
        {"from": "ctrl", "body": "positive-control", "thread": "ctrl-t"},
    )
    assert status == 200, body

    status, body = _post(
        f"{live_server}/a2a/import",
        [
            {
                "source": "ctrl-t",
                "source_id": "msg-1",
                "from": "ctrl",
                "body": "imported hello",
                "ts": 1709452800.0,
            }
        ],
    )
    assert status == 200, body
    assert body["imported"] == 1
    assert body["skipped"] == 0
    assert len(body["messages"]) == 1


# ---------------------------------------------------------------------------
# 2. Whole-batch rejection on bad envelope
# ---------------------------------------------------------------------------

def test_http_a2a_import_rejects_whole_batch_on_bad_envelope(live_server):
    status, body = _post(
        f"{live_server}/a2a/import",
        [
            {"source": "ch", "source_id": "1", "from": "a", "body": "ok", "ts": 1709452800.0},
            {"source": "ch", "source_id": "2", "from": "a", "ts": "not-a-float"},
        ],
    )
    assert status == 400, body
    assert "ts" in body["error"]

    status, body = _get(f"{live_server}/a2a/messages?thread=ch")
    assert status == 200
    assert len(body["messages"]) == 0


# ---------------------------------------------------------------------------
# 3. Timestamp preservation
# ---------------------------------------------------------------------------

def test_http_a2a_import_preserves_timestamp(live_server):
    past_ts = 1709452800.0
    status, body = _post(
        f"{live_server}/a2a/import",
        [
            {
                "source": "ts-chan",
                "source_id": "ts-msg-1",
                "from": "agent",
                "body": "old message",
                "ts": past_ts,
            }
        ],
    )
    assert status == 200, body
    assert body["messages"][0]["ts"] == past_ts

    status, body = _get(f"{live_server}/a2a/messages?thread=ts-chan")
    assert status == 200
    assert len(body["messages"]) == 1
    assert body["messages"][0]["ts"] == past_ts
    assert abs(body["messages"][0]["ts"] - time.time()) > 100


# ---------------------------------------------------------------------------
# 4. Idempotence: re-import is a no-op
# ---------------------------------------------------------------------------

def test_http_a2a_import_is_idempotent(live_server):
    batch = [
        {"source": "idem-chan", "source_id": f"msg-{i}", "from": "agent", "body": f"msg {i}", "ts": 1709452800.0 + i}
        for i in range(3)
    ]
    status, body = _post(f"{live_server}/a2a/import", batch)
    assert status == 200, body
    assert body["imported"] == 3
    assert body["skipped"] == 0

    status, body = _get(f"{live_server}/a2a/messages?thread=idem-chan")
    assert status == 200
    assert len(body["messages"]) == 3

    status, body = _post(f"{live_server}/a2a/import", batch)
    assert status == 200, body
    assert body["imported"] == 0
    assert body["skipped"] == 3

    status, body = _get(f"{live_server}/a2a/messages?thread=idem-chan")
    assert status == 200
    assert len(body["messages"]) == 3


def test_http_a2a_import_idempotence_fails_when_dedup_is_broken(isolated_data_dir, monkeypatch):
    """Break the dedup predicate by clearing the dedup table between imports."""
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    batch = [
        {"source": "break-chan", "source_id": "b1", "from": "a", "body": "m1", "ts": 1709452800.0},
    ]
    result = asyncio.run(service.a2a_import(batch, data_dir=dd))
    assert result["imported"] == 1

    msgs = asyncio.run(service.a2a_feed(thread="break-chan", data_dir=dd))
    assert len(msgs) == 1

    stores = asyncio.run(taosmd_api._ensure_stores(dd))
    archive = stores["archive"]
    archive._conn.execute("DELETE FROM a2a_import_dedup")
    archive._conn.commit()

    result = asyncio.run(service.a2a_import(batch, data_dir=dd))
    assert result["imported"] == 1
    assert result["skipped"] == 0

    msgs = asyncio.run(service.a2a_feed(thread="break-chan", data_dir=dd))
    assert len(msgs) == 2


# ---------------------------------------------------------------------------
# 5. Auth parity: caller who cannot send cannot import
# ---------------------------------------------------------------------------

@pytest.fixture
def authed_server(tmp_path, monkeypatch):
    from taosmd import config as cfg, registry_auth
    import jwt as pyjwt
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization

    data_dir = tmp_path / "taosmd-auth"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    cfg.set_a2a_auth_enforce(True, str(data_dir))

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

    def fake_opener(url, token=None):
        if url.endswith(registry_auth.PUBKEY_PATH):
            return json.dumps({"pubkey": pub_pem})
        return json.dumps([])

    verifier = registry_auth.verifier_from_url(
        "http://reg.test", opener=fake_opener, expected_iss=None
    )
    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir), verifier=verifier)
    httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    host, port = httpd.server_address[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://{host}:{port}", priv_pem
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
        httpd.service_loop.close()


def test_http_a2a_import_rejects_unauthenticated(authed_server):
    base_url, _ = authed_server
    status, body = _post(
        f"{base_url}/a2a/import",
        [{"source": "ch", "source_id": "1", "from": "a", "body": "hi", "ts": 1709452800.0}],
    )
    assert status in (401, 403), body


def test_http_a2a_import_accepts_valid_token(authed_server):
    base_url, priv_pem = authed_server
    token = pyjwt.encode({"sub": "agent-1"}, priv_pem, algorithm="EdDSA")
    status, body = _post(
        f"{base_url}/a2a/import",
        [{"source": "ch", "source_id": "1", "from": "agent-1", "body": "hi", "ts": 1709452800.0}],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert status == 200, body


def test_http_a2a_send_and_import_share_auth_gate(authed_server):
    base_url, priv_pem = authed_server
    bad_status_send, _ = _post(
        f"{base_url}/a2a/send",
        {"from": "agent-1", "body": "hi"},
    )
    bad_status_import, _ = _post(
        f"{base_url}/a2a/import",
        [{"source": "ch", "source_id": "1", "from": "agent-1", "body": "hi", "ts": 1709452800.0}],
    )
    assert bad_status_send in (401, 403)
    assert bad_status_import in (401, 403)
    assert bad_status_send == bad_status_import
