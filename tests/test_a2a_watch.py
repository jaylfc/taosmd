"""Tests for taosmd a2a-watch and a2a-bridge - realtime A2A wake.

Hermetic: each test starts a real HTTP server on an ephemeral port, seeds
messages over loopback, and runs the streaming command bounded by --count so
it terminates. No network beyond loopback; no ~/.taosmd touched; no port 7900.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
from pathlib import Path

import pytest

from taosmd import api as taosmd_api
from taosmd import http_server
from taosmd.cli import _a2a_watch_cmd, _a2a_bridge_cmd
from taosmd.remote import RemoteClient


def _patch_embedder(stores: dict) -> None:
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def watch_server(tmp_path, monkeypatch):
    """Start a real HTTP server on an ephemeral port. Yields (base_url, rc)."""
    data_dir = tmp_path / "watch-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))
    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)

    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://{host}:{port}"
    rc = RemoteClient(base_url)
    try:
        yield base_url, rc
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        httpd.service_loop.close()


def _seed(rc: RemoteClient, channel: str, messages: list[tuple[str, str]]) -> None:
    for sender, body in messages:
        asyncio.run(rc.a2a_send(sender, body, thread=channel))


def _watch_args(channel, server, exclude=None, count=0):
    return argparse.Namespace(
        cmd="a2a-watch", channel=channel, server=server, exclude=exclude, count=count
    )


def _bridge_args(channel, server, trigger, exclude=None, debounce=0.0,
                 max_concurrency=1, count=0):
    return argparse.Namespace(
        cmd="a2a-bridge", channel=channel, server=server, trigger=trigger,
        exclude=exclude, debounce=debounce, max_concurrency=max_concurrency,
        count=count,
    )


def test_watch_prints_new_messages(watch_server, capsys):
    """a2a-watch streams seeded messages and exits after --count."""
    base_url, rc = watch_server
    _seed(rc, "obs", [("@taOS", "hello one"), ("@taOS", "hello two")])

    rc_code = _a2a_watch_cmd(_watch_args("obs", base_url, count=2))

    assert rc_code == 0
    out = capsys.readouterr().out
    assert "hello one" in out
    assert "hello two" in out
    assert "<@taOS>" in out


def test_watch_excludes_own_sender(watch_server, capsys):
    """--exclude skips our own messages but still reaches later ones."""
    base_url, rc = watch_server
    _seed(rc, "obs", [
        ("@taOSmd", "my own post"),
        ("@taOS", "their reply"),
    ])

    rc_code = _a2a_watch_cmd(
        _watch_args("obs", base_url, exclude="@taOSmd", count=1)
    )

    assert rc_code == 0
    out = capsys.readouterr().out
    assert "their reply" in out
    assert "my own post" not in out


def test_watch_dedups_by_id(watch_server, capsys):
    """Each message is emitted exactly once (id-dedup), no repeats."""
    base_url, rc = watch_server
    _seed(rc, "obs", [("@taOS", "unique-body-xyz")])

    rc_code = _a2a_watch_cmd(_watch_args("obs", base_url, count=1))

    assert rc_code == 0
    out = capsys.readouterr().out
    assert out.count("unique-body-xyz") == 1


def test_bridge_execs_trigger_with_json_stdin(watch_server, tmp_path):
    """a2a-bridge runs the trigger once per message, piping JSON to stdin."""
    base_url, rc = watch_server
    sink = tmp_path / "sink.txt"
    # Trigger reads stdin and appends it to the sink file.
    trigger = f'{sys.executable} -c "import sys,pathlib; ' \
              f"pathlib.Path(r'{sink}').write_text(sys.stdin.read())\""
    _seed(rc, "obs", [("@taOS", "wake-payload")])

    rc_code = _a2a_bridge_cmd(_bridge_args("obs", base_url, trigger, count=1))

    assert rc_code == 0
    # The trigger may finish just after the bridge loop returns; allow a beat.
    import time
    for _ in range(50):
        if sink.exists() and sink.read_text():
            break
        time.sleep(0.1)
    payload = json.loads(sink.read_text())
    assert payload["from"] == "@taOS"
    assert payload["body"] == "wake-payload"
    assert "id" in payload


def test_watch_all_channels(watch_server, capsys):
    """--channel omitted (all) streams every channel; lines show the thread."""
    base_url, rc = watch_server
    _seed(rc, "alpha", [("@taOS", "from alpha chan")])
    _seed(rc, "beta", [("@taOS", "from beta chan")])

    # channel=None => all-channels mode
    rc_code = _a2a_watch_cmd(_watch_args(None, base_url, count=2))

    assert rc_code == 0
    out = capsys.readouterr().out
    assert "from alpha chan" in out
    assert "from beta chan" in out
    # all-mode prefixes each line with its (thread)
    assert "(alpha)" in out
    assert "(beta)" in out


def test_bridge_all_channels(watch_server, tmp_path):
    """a2a-bridge with channel omitted fires on messages from any channel."""
    base_url, rc = watch_server
    sink = tmp_path / "bridge_all.txt"
    trigger = f'{sys.executable} -c "import sys,pathlib; ' \
              f"pathlib.Path(r'{sink}').write_text(sys.stdin.read())\""
    _seed(rc, "gamma", [("@taOS", "wake-from-gamma")])

    rc_code = _a2a_bridge_cmd(_bridge_args(None, base_url, trigger, count=1))

    assert rc_code == 0
    import time
    for _ in range(50):
        if sink.exists() and sink.read_text():
            break
        time.sleep(0.1)
    payload = json.loads(sink.read_text())
    assert payload["thread"] == "gamma"
    assert payload["body"] == "wake-from-gamma"
