"""Tests for A2A v2 slice 3a: alarm_key convergence, dedup, clear, and store-backed state.

Requirements under test:
1. A deduped alarm is NOT stored (no archive record, no bus row).
2. Dedup is enforced atomically at the storage layer (unique index on
   alarm_key+fingerprint within the window), not by a Python scan over the
   archive.
3. POST /a2a/alarms/{key}/clear re-arms the key ONCE: the next same-key alarm
   stores, then cooldown re-applies.
4. Dedup/cleared state lives in the store, not module-global memory.
5. kind="alarm" uses the existing _A2A_KINDS taxonomy (slice 1).
6. RemoteClient.a2a_send forwards alarm_key and alarm_fingerprint in the HTTP
   payload.
7. The clear-refire test calls the clear endpoint and uses an identical
   body/fingerprint so that only the clear explains the refire.
8. Alarm keys containing characters that require URL quoting are handled
   correctly on both the client and server sides.
"""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.parse

import pytest

from taosmd import api as taosmd_api
from taosmd import http_server, service


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
    data_dir = tmp_path / "taosmd-a2a-alarm"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    yield data_dir
    for stores in list(taosmd_api._stores_cache.values()):
        for store in (
            stores.get("archive"),
            stores.get("vector"),
            stores.get("kg"),
        ):
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
    data_dir = tmp_path / "taosmd-a2a-alarm-http"
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
            for store in (
                s.get("archive"),
                s.get("vector"),
                s.get("kg"),
            ):
                if store and hasattr(store, "close"):
                    try:
                        httpd.service_loop.run(store.close())
                    except Exception:
                        pass
        httpd.service_loop.close()


def _post(url: str, payload) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    return _send(req)


def _get(url: str) -> tuple[int, dict]:
    return _send(urllib.request.Request(url, method="GET"))


def _send(req) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


# ---------------------------------------------------------------------------
# Req 5: kind="alarm" accepted via the existing _A2A_KINDS taxonomy
# ---------------------------------------------------------------------------

def test_alarm_kind_accepted_and_roundtrips(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)
    receipt = asyncio.run(service.a2a_send(
        sender="agentA", body="disk full", thread="ops",
        kind="alarm", data_dir=dd,
    ))
    assert receipt["kind"] == "alarm"
    msgs = asyncio.run(service.a2a_feed(thread="ops", data_dir=dd))
    assert len(msgs) == 1
    assert msgs[0]["kind"] == "alarm"


# ---------------------------------------------------------------------------
# Req 1 + Req 2 + Req 3 + Req 4: deduped alarm is NOT stored
# ---------------------------------------------------------------------------

_A2A_ALARM_MIN_INTERVAL = 5.0  # seconds; short so tests don't sleep

_BODY = "disk full on /dev/sda1"
_KEY = "dead-session:@taOSmd-dev"
_FP = "fp:disk-full:sda1"


def test_deduped_alarm_not_stored(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    r1 = asyncio.run(service.a2a_send(
        sender="watcher", body=_BODY, thread="ops",
        kind="alarm", alarm_key=_KEY, alarm_fingerprint=_FP,
        data_dir=dd,
    ))
    assert "deduped" not in r1

    r2 = asyncio.run(service.a2a_send(
        sender="watcher", body=_BODY, thread="ops",
        kind="alarm", alarm_key=_KEY, alarm_fingerprint=_FP,
        data_dir=dd,
    ))
    assert r2.get("deduped") is True

    msgs = asyncio.run(service.a2a_feed(thread="ops", data_dir=dd))
    assert len(msgs) == 1, f"expected 1 stored alarm, got {len(msgs)}"


def test_clear_rearms_key_once(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    asyncio.run(service.a2a_send(
        sender="w", body=_BODY, thread="ops",
        kind="alarm", alarm_key=_KEY, alarm_fingerprint=_FP,
        data_dir=dd,
    ))
    asyncio.run(service.a2a_send(
        sender="w", body=_BODY, thread="ops",
        kind="alarm", alarm_key=_KEY, alarm_fingerprint=_FP,
        data_dir=dd,
    ))
    msgs = asyncio.run(service.a2a_feed(thread="ops", data_dir=dd))
    assert len(msgs) == 1

    asyncio.run(service.a2a_alarms_clear(_KEY, data_dir=dd))
    r3 = asyncio.run(service.a2a_send(
        sender="w", body=_BODY, thread="ops",
        kind="alarm", alarm_key=_KEY, alarm_fingerprint=_FP,
        data_dir=dd,
    ))
    assert "deduped" not in r3

    r4 = asyncio.run(service.a2a_send(
        sender="w", body=_BODY, thread="ops",
        kind="alarm", alarm_key=_KEY, alarm_fingerprint=_FP,
        data_dir=dd,
    ))
    assert r4.get("deduped") is True

    msgs = asyncio.run(service.a2a_feed(thread="ops", data_dir=dd))
    assert len(msgs) == 2, f"expected 2 stored alarms after clear, got {len(msgs)}"


def test_dedup_state_survives_store_reopen(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    asyncio.run(service.a2a_send(
        sender="w", body=_BODY, thread="ops",
        kind="alarm", alarm_key=_KEY, alarm_fingerprint=_FP,
        data_dir=dd,
    ))

    # Close all stores and re-open
    for stores in list(taosmd_api._stores_cache.values()):
        for store in (
            stores.get("archive"),
            stores.get("vector"),
            stores.get("kg"),
        ):
            if store and hasattr(store, "close"):
                try:
                    asyncio.run(store.close())
                except Exception:
                    pass
    taosmd_api._stores_cache.clear()

    _setup_stores(isolated_data_dir)
    r2 = asyncio.run(service.a2a_send(
        sender="w", body=_BODY, thread="ops",
        kind="alarm", alarm_key=_KEY, alarm_fingerprint=_FP,
        data_dir=dd,
    ))
    assert r2.get("deduped") is True, "dedup must persist across store re-open"


# ---------------------------------------------------------------------------
# Req 7: clear-refire test via the HTTP endpoint with identical body/fingerprint
# ---------------------------------------------------------------------------

def test_clear_refire_via_http_endpoint(live_server):
    payload = {
        "from": "watcher",
        "body": _BODY,
        "thread": "ops",
        "kind": "alarm",
        "alarm_key": _KEY,
        "alarm_fingerprint": _FP,
    }
    status, body = _post(f"{live_server}/a2a/send", payload)
    assert status == 200, body
    assert "deduped" not in body

    status, body = _post(f"{live_server}/a2a/send", payload)
    assert status == 200, body
    assert body.get("deduped") is True

    status, body = _post(f"{live_server}/a2a/alarms/{urllib.parse.quote(_KEY)}/clear", {})
    assert status == 200, body

    status, body = _post(f"{live_server}/a2a/send", payload)
    assert status == 200, body
    assert "deduped" not in body, "clear must re-arm so the identical alarm stores"

    status, body = _post(f"{live_server}/a2a/send", payload)
    assert status == 200, body
    assert body.get("deduped") is True, "cooldown must re-apply after the refire"

    status, msgs = _get(f"{live_server}/a2a/messages?thread=ops")
    assert status == 200, msgs
    assert len(msgs["messages"]) == 2, f"expected 2 messages, got {len(msgs['messages'])}"


# ---------------------------------------------------------------------------
# Req 2: dedup uses store-level state, not archive scan
# ---------------------------------------------------------------------------

def test_alarm_dedup_table_exists(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    stores = asyncio.run(taosmd_api._ensure_stores(str(isolated_data_dir)))
    archive = stores["archive"]
    conn = archive._conn
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='a2a_alarm_state'"
    ).fetchone()
    assert row is not None, "a2a_alarm_state table must exist in the store"

    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_a2a_alarm_state_key_fp'"
    ).fetchone()
    assert row is not None, "unique index on (alarm_key, fingerprint) must exist"


# ---------------------------------------------------------------------------
# Req 6: RemoteClient forwards alarm_key and alarm_fingerprint
# ---------------------------------------------------------------------------

def test_remote_a2a_send_forwards_alarm_fields(monkeypatch):
    captured: dict = {}

    async def _fake_run(self, method, path, payload=None, params=None):
        captured["method"] = method
        captured["path"] = path
        captured["payload"] = payload
        return {"id": 1, "from": "x", "thread": "t"}

    from taosmd.remote import RemoteClient
    monkeypatch.setattr(RemoteClient, "_run", _fake_run, raising=False)

    client = RemoteClient.__new__(RemoteClient)
    client._base_url = "http://localhost:9999"
    client._token = None

    asyncio.run(client.a2a_send(
        sender="watcher", body=_BODY, thread="ops",
        kind="alarm", alarm_key=_KEY, alarm_fingerprint=_FP,
    ))
    assert captured["payload"].get("alarm_key") == _KEY
    assert captured["payload"].get("alarm_fingerprint") == _FP


# ---------------------------------------------------------------------------
# Req 4: state is not module-global
# ---------------------------------------------------------------------------

def test_dedup_not_module_global(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    dd = str(isolated_data_dir)

    asyncio.run(service.a2a_send(
        sender="w", body=_BODY, thread="ops",
        kind="alarm", alarm_key=_KEY, alarm_fingerprint=_FP,
        data_dir=dd,
    ))

    # Clear the module-level remote cache (if anything were stored there it would
    # be lost).  The dedup must still work because it lives in the store.
    from taosmd import service as svc
    svc._remote_cache.clear()

    r2 = asyncio.run(service.a2a_send(
        sender="w", body=_BODY, thread="ops",
        kind="alarm", alarm_key=_KEY, alarm_fingerprint=_FP,
        data_dir=dd,
    ))
    assert r2.get("deduped") is True


# ---------------------------------------------------------------------------
# Defect 3: alarm_key with special characters is URL-quoted client-side
# ---------------------------------------------------------------------------

def test_remote_a2a_alarms_clear_quotes_key(monkeypatch):
    captured_path = None

    async def _fake_run(self, method, path, payload=None, params=None):
        nonlocal captured_path
        captured_path = path
        return {"cleared": True, "key": "any"}

    from taosmd.remote import RemoteClient
    monkeypatch.setattr(RemoteClient, "_run", _fake_run, raising=False)

    client = RemoteClient.__new__(RemoteClient)
    client._base_url = "http://localhost:9999"
    client._token = None

    asyncio.run(client.a2a_alarms_clear("path/with spaces and %",))
    assert captured_path == "/a2a/alarms/path%2Fwith%20spaces%20and%20%25/clear"


def test_http_alarms_clear_unquotes_key(tmp_path, monkeypatch):
    from taosmd import api as taosmd_api
    from taosmd import http_server as hs

    data_dir = tmp_path / "taosmd-quote-test"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    httpd = hs.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)

    host, port = httpd.server_address[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        key = "path/with spaces and %"
        status, body = _post(
            f"http://{host}:{port}/a2a/alarms/{urllib.parse.quote(key, safe='')}/clear",
            {},
        )
        assert status == 200, body
        assert body["cleared"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
