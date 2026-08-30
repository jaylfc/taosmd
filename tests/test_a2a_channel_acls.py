"""RED tests for A2A per-channel ACL enforcement.

These tests must fail on unpatched code (no ACL enforcement) and pass after
the fix lands.  They follow the fixture and helper patterns from
``tests/test_http_server_trust_enforcement.py``.
"""
from __future__ import annotations

import json
import socket
import threading
import time
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


def _get_messages(base_url, thread=None, token=None):
    path = "/a2a/messages"
    if thread is not None:
        path += f"?thread={thread}"
    req = urllib.request.Request(base_url + path)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def _get_threads(base_url, token=None):
    req = urllib.request.Request(base_url + "/a2a/threads")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def _get_admin_channel_acl(base_url, channel, token=None):
    path = f"/a2a/admin/channel-acl?channel={channel}"
    req = urllib.request.Request(base_url + path)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def _stream_status_code(base_url, path, timeout=3.0):
    """Read only the HTTP status line from the SSE stream at path."""
    parsed = urllib.parse.urlsplit(base_url)
    host = parsed.hostname
    port = parsed.port
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(
            f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n".encode()
        )
        sock.settimeout(timeout)
        line = b""
        while b"\r\n" not in line:
            try:
                chunk = sock.recv(128)
            except (socket.timeout, TimeoutError) as exc:
                raise AssertionError(f"timed out reading stream status line: {exc}") from exc
            if not chunk:
                break
            line += chunk
    status_line = line.decode("utf-8", "replace").splitlines()[0]
    return int(status_line.split()[1])


def _read_sse_frames(base_url, path, timeout=8.0):
    """Open a raw TCP connection, send a minimal HTTP GET, read SSE frames."""
    parsed = urllib.parse.urlsplit(base_url)
    host = parsed.hostname
    port = parsed.port
    frames = []
    deadline = time.monotonic() + timeout
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(
            f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n".encode()
        )
        buf = b""
        header_done = False
        while time.monotonic() < deadline and not frames:
            sock.settimeout(max(0.1, deadline - time.monotonic()))
            try:
                chunk = sock.recv(4096)
            except (socket.timeout, TimeoutError):
                break
            if not chunk:
                break
            buf += chunk
            if not header_done:
                sep = buf.find(b"\r\n\r\n")
                if sep != -1:
                    buf = buf[sep + 4:]
                    header_done = True
            if header_done:
                text = buf.decode("utf-8", errors="replace")
                for line in text.splitlines():
                    if line.startswith("data: "):
                        frames.append(line[6:])
    return frames


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


# ---------------------------------------------------------------------------
# Positive control: ?thread=secret returns 403
# ---------------------------------------------------------------------------

def test_thread_secret_returns_403_positive_control(acl_server, tmp_path):
    """Positive control: ?thread=secret correctly returns 403 for unverified caller."""
    data_dir = tmp_path / "data"
    _set_acl(data_dir, "secret", read_ids=["agent-allowed"])

    status, body = _get_messages(acl_server, thread="secret")
    assert status == 403, body


# ---------------------------------------------------------------------------
# Leak 1: GET /a2a/messages with no thread must not leak restricted bodies
# ---------------------------------------------------------------------------

def test_messages_no_thread_filters_restricted(acl_server, tmp_path):
    """GET /a2a/messages without thread must drop messages from ACL-restricted channels."""
    data_dir = tmp_path / "data"
    _set_acl(data_dir, "secret", read_ids=["agent-allowed"])

    token_allowed = _mint("agent-allowed")
    _post_send(acl_server, "agent-allowed", "canary-body-secret", token=token_allowed, thread="secret")
    _post_send(acl_server, "agent-allowed", "public-body", token=token_allowed, thread="public")

    status, body = _get_messages(acl_server)
    assert status == 200, body
    bodies = [m.get("body", "") for m in body.get("messages", [])]
    assert "canary-body-secret" not in bodies, "restricted canary leaked in unfiltered feed"
    assert "public-body" in bodies, "public message should still be visible"


# ---------------------------------------------------------------------------
# Leak 2: GET /a2a/admin/channel-acl must require admin token
# ---------------------------------------------------------------------------

def test_admin_channel_acl_requires_admin_token(acl_server, tmp_path):
    """GET /a2a/admin/channel-acl without admin token must return 401/403."""
    data_dir = tmp_path / "data"
    _set_acl(data_dir, "secret", read_ids=["agent-allowed"])

    status, body = _get_admin_channel_acl(acl_server, "secret")
    assert status in (401, 403), body


# ---------------------------------------------------------------------------
# Leak 3: GET /a2a/threads must filter restricted channels
# ---------------------------------------------------------------------------

def test_threads_filters_restricted(acl_server, tmp_path):
    """GET /a2a/threads must not expose channels the caller cannot read."""
    data_dir = tmp_path / "data"
    _set_acl(data_dir, "secret", read_ids=["agent-allowed"])

    token_allowed = _mint("agent-allowed")
    _post_send(acl_server, "agent-allowed", "thread-canary", token=token_allowed, thread="secret")
    _post_send(acl_server, "agent-allowed", "public-msg", token=token_allowed, thread="public")

    status, body = _get_threads(acl_server)
    assert status == 200, body
    thread_names = [t.get("thread") for t in body.get("threads", [])]
    assert "secret" not in thread_names, "restricted thread name leaked"
    assert "public" in thread_names, "public thread should still be visible"


# ---------------------------------------------------------------------------
# Leak 4: GET /a2a/stream?thread=secret must deny when ACL denies
# ---------------------------------------------------------------------------

def test_stream_thread_denied_returns_403(acl_server, tmp_path):
    """GET /a2a/stream?thread=secret must 403 before SSE headers when ACL denies."""
    data_dir = tmp_path / "data"
    _set_acl(data_dir, "secret", read_ids=["agent-allowed"])

    status = _stream_status_code(acl_server, "/a2a/stream?thread=secret")
    assert status == 403, "restricted thread stream should 403 before sending SSE headers"


# ---------------------------------------------------------------------------
# Leak 4b: GET /a2a/stream (no thread) must filter restricted messages
# ---------------------------------------------------------------------------

def test_stream_no_thread_filters_restricted(acl_server, tmp_path):
    """GET /a2a/stream without thread must not deliver restricted messages."""
    data_dir = tmp_path / "data"
    _set_acl(data_dir, "secret", read_ids=["agent-allowed"])

    # parsed = urllib.parse.urlsplit(acl_server)
    # host = parsed.hostname
    # port = parsed.port

    frames_received = []
    error_holder = []

    def _stream_reader():
        try:
            result = _read_sse_frames(
                acl_server, "/a2a/stream", timeout=8.0,
            )
            frames_received.extend(result)
        except Exception as exc:
            error_holder.append(exc)

    reader = threading.Thread(target=_stream_reader, daemon=True)
    reader.start()

    time.sleep(1.5)

    token_allowed = _mint("agent-allowed")
    _post_send(acl_server, "agent-allowed", "stream-secret-canary", token=token_allowed, thread="secret")
    _post_send(acl_server, "agent-allowed", "stream-public", token=token_allowed, thread="public")

    reader.join(timeout=10)
    assert not error_holder, f"SSE reader raised: {error_holder[0]}"

    bodies = []
    for frame in frames_received:
        try:
            payload = json.loads(frame)
            bodies.append(payload.get("body", ""))
        except json.JSONDecodeError:
            pass
    assert "stream-secret-canary" not in bodies, "restricted message leaked in stream"
    assert "stream-public" in bodies, "public message should be delivered in stream"
