"""Tests for admin/data-plane token separation (#154).

The admin surface can be gated by a dedicated ``admin_token`` that is distinct
from the data-plane ``server_token``. This lets an operator authorize admin
operations WITHOUT setting a server token, so data and A2A endpoints stay open
while admin runs (no lockout). Migration: when no ``admin_token`` is configured,
``_check_admin_token`` falls back to the ``server_token`` so existing
token-secured installs keep working unchanged.

Live-server style mirrors tests/test_admin_surface.py.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

from taosmd import api as taosmd_api
from taosmd import http_server


# ---------------------------------------------------------------------------
# HTTP helpers (same shape as test_admin_surface.py)
# ---------------------------------------------------------------------------

def _patch_embedder(stores: dict) -> None:
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


def _post(url: str, payload, token: str | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    return _send(req)


def _send(req) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _boot_server(tmp_path, monkeypatch, *, server_token=None, admin_token=None):
    """Start a live server with the given token env, return (url, httpd, thread)."""
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    if server_token is None:
        monkeypatch.delenv("TAOSMD_TOKEN", raising=False)
    else:
        monkeypatch.setenv("TAOSMD_TOKEN", server_token)
    if admin_token is None:
        monkeypatch.delenv("TAOSMD_ADMIN_TOKEN", raising=False)
    else:
        monkeypatch.setenv("TAOSMD_ADMIN_TOKEN", admin_token)

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)

    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return f"http://{host}:{port}", httpd, thread


def _teardown(httpd, thread):
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


_SERVER_TOKEN = "data-plane-token-xyz"
_ADMIN_TOKEN = "admin-only-token-abc"
_ADMIN_ROUTE = "/shelves"
_ADMIN_PAYLOAD = {"shelf_id": "septest"}


# ---------------------------------------------------------------------------
# admin_token gates the admin surface independently of the data plane
# ---------------------------------------------------------------------------

def test_admin_token_accepted_when_only_admin_token_set(tmp_path, monkeypatch):
    """With only admin_token set, the admin token authorizes admin ops."""
    base, httpd, thread = _boot_server(
        tmp_path, monkeypatch, server_token=None, admin_token=_ADMIN_TOKEN
    )
    try:
        status, body = _post(f"{base}{_ADMIN_ROUTE}", _ADMIN_PAYLOAD, token=_ADMIN_TOKEN)
        assert status == 200, body
        assert body["shelf"]["name"] == "septest"
    finally:
        _teardown(httpd, thread)


def test_data_plane_open_when_only_admin_token_set(tmp_path, monkeypatch):
    """The core #154 fix: only admin_token set => data plane needs no token.

    A tokenless data request (ingest) must still succeed, proving admin gating
    does not lock the data plane.
    """
    base, httpd, thread = _boot_server(
        tmp_path, monkeypatch, server_token=None, admin_token=_ADMIN_TOKEN
    )
    try:
        status, body = _post(
            f"{base}/ingest", {"text": "open data plane", "agent": "someagent"}, token=None
        )
        assert status == 200, body
    finally:
        _teardown(httpd, thread)


def test_admin_rejects_missing_token_when_only_admin_token_set(tmp_path, monkeypatch):
    """Admin still fails closed (401) without the admin token even if data is open."""
    base, httpd, thread = _boot_server(
        tmp_path, monkeypatch, server_token=None, admin_token=_ADMIN_TOKEN
    )
    try:
        status, body = _post(f"{base}{_ADMIN_ROUTE}", _ADMIN_PAYLOAD, token=None)
        assert status == 401, body
        status, body = _post(f"{base}{_ADMIN_ROUTE}", _ADMIN_PAYLOAD, token="wrong")
        assert status == 401, body
    finally:
        _teardown(httpd, thread)


def test_data_plane_token_rejected_on_admin_when_admin_token_set(tmp_path, monkeypatch):
    """With both tokens set, a data-plane-only caller cannot run admin ops.

    Holding only the server_token must be rejected (401) on the admin surface;
    the admin token is required.
    """
    base, httpd, thread = _boot_server(
        tmp_path, monkeypatch, server_token=_SERVER_TOKEN, admin_token=_ADMIN_TOKEN
    )
    try:
        # server_token authorizes the data plane...
        status, _ = _post(
            f"{base}/ingest",
            {"text": "hi", "agent": "a"},
            token=_SERVER_TOKEN,
        )
        assert status == 200
        # ...but NOT the admin surface.
        status, body = _post(f"{base}{_ADMIN_ROUTE}", _ADMIN_PAYLOAD, token=_SERVER_TOKEN)
        assert status == 401, body
        # the admin token works.
        status, body = _post(f"{base}{_ADMIN_ROUTE}", _ADMIN_PAYLOAD, token=_ADMIN_TOKEN)
        assert status == 200, body
    finally:
        _teardown(httpd, thread)


def test_server_token_still_gates_admin_when_no_admin_token(tmp_path, monkeypatch):
    """Back-compat: with only server_token set, it still authorizes admin ops."""
    base, httpd, thread = _boot_server(
        tmp_path, monkeypatch, server_token=_SERVER_TOKEN, admin_token=None
    )
    try:
        status, body = _post(f"{base}{_ADMIN_ROUTE}", _ADMIN_PAYLOAD, token=_SERVER_TOKEN)
        assert status == 200, body
        # wrong token rejected
        status, body = _post(f"{base}{_ADMIN_ROUTE}", _ADMIN_PAYLOAD, token="nope")
        assert status == 401, body
    finally:
        _teardown(httpd, thread)


def test_admin_fails_closed_when_no_token_configured(tmp_path, monkeypatch):
    """With neither token set, the admin surface fails closed (403)."""
    base, httpd, thread = _boot_server(
        tmp_path, monkeypatch, server_token=None, admin_token=None
    )
    try:
        status, body = _post(f"{base}{_ADMIN_ROUTE}", _ADMIN_PAYLOAD, token=None)
        assert status == 403, body
        assert "admin" in body.get("error", "").lower()
    finally:
        _teardown(httpd, thread)
