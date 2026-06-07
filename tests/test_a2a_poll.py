"""Tests for taosmd a2a-poll — durable bus monitoring.

Hermetic: uses an ephemeral port server and isolated tmp dirs.  No network
calls beyond loopback; no ~/.taosmd touched; no port 7900 used.
"""

from __future__ import annotations

import asyncio
import argparse
import json
import threading
from pathlib import Path

import pytest

from taosmd import api as taosmd_api
from taosmd import http_server
from taosmd.cli import _a2a_poll_cmd
from taosmd.remote import RemoteClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _patch_embedder(stores: dict) -> None:
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def poll_server(tmp_path, monkeypatch):
    """Start a real HTTP server on an ephemeral port for a2a-poll tests.

    Yields (base_url, data_dir_str, RemoteClient).
    """
    data_dir = tmp_path / "poll-data"
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
        yield base_url, str(data_dir), rc
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        httpd.service_loop.close()


def _seed_messages(rc: RemoteClient, channel: str, messages: list[tuple[str, str]]) -> list[dict]:
    """Post ``messages`` (list of (sender, body)) and return receipts."""
    receipts = []
    for sender, body in messages:
        receipts.append(asyncio.run(rc.a2a_send(sender, body, thread=channel)))
    return receipts


# ---------------------------------------------------------------------------
# a2a-poll: print only-new, update state, --exclude filter
# ---------------------------------------------------------------------------

def test_a2a_poll_prints_new_messages(poll_server, tmp_path, capsys):
    """a2a-poll outputs messages not yet seen and updates state file."""
    base_url, _, rc = poll_server
    state_file = tmp_path / "poll-state.json"
    channel = "poll-test-chan"

    _seed_messages(rc, channel, [
        ("agent-a", "First message"),
        ("agent-b", "Second message"),
    ])

    args = argparse.Namespace(
        channel=channel,
        server=base_url,
        state_file=str(state_file),
        exclude=None,
    )
    rc_exit = _a2a_poll_cmd(args)
    assert rc_exit == 0

    captured = capsys.readouterr()
    assert "First message" in captured.out
    assert "Second message" in captured.out

    # State file should exist and record the last-seen ID.
    state = json.loads(state_file.read_text())
    assert channel in state
    assert state[channel] >= 0


def test_a2a_poll_second_run_prints_nothing(poll_server, tmp_path, capsys):
    """After the first poll, a second poll with no new messages prints nothing."""
    base_url, _, rc = poll_server
    state_file = tmp_path / "poll-state2.json"
    channel = "poll-test-empty"

    _seed_messages(rc, channel, [("agent-a", "Only message")])

    args = argparse.Namespace(
        channel=channel,
        server=base_url,
        state_file=str(state_file),
        exclude=None,
    )
    _a2a_poll_cmd(args)
    capsys.readouterr()  # discard first run output

    # Second run: no new messages.
    rc_exit = _a2a_poll_cmd(args)
    assert rc_exit == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == ""


def test_a2a_poll_only_new_since_last_seen(poll_server, tmp_path, capsys):
    """Only messages posted after the last poll are printed on the next run."""
    base_url, _, rc = poll_server
    state_file = tmp_path / "poll-state3.json"
    channel = "poll-test-incremental"

    # Seed two messages and poll them.
    _seed_messages(rc, channel, [
        ("agent-a", "Old message 1"),
        ("agent-a", "Old message 2"),
    ])

    args = argparse.Namespace(
        channel=channel,
        server=base_url,
        state_file=str(state_file),
        exclude=None,
    )
    _a2a_poll_cmd(args)
    capsys.readouterr()

    # Seed one more and poll again.
    _seed_messages(rc, channel, [("agent-b", "New message after poll")])
    _a2a_poll_cmd(args)
    captured = capsys.readouterr()

    assert "New message after poll" in captured.out
    assert "Old message 1" not in captured.out
    assert "Old message 2" not in captured.out


def test_a2a_poll_exclude_filters_sender(poll_server, tmp_path, capsys):
    """Messages from the excluded sender are not printed."""
    base_url, _, rc = poll_server
    state_file = tmp_path / "poll-state-exclude.json"
    channel = "poll-test-exclude"

    _seed_messages(rc, channel, [
        ("myself", "My own message"),
        ("other-agent", "Other agent message"),
    ])

    args = argparse.Namespace(
        channel=channel,
        server=base_url,
        state_file=str(state_file),
        exclude="myself",
    )
    rc_exit = _a2a_poll_cmd(args)
    assert rc_exit == 0

    captured = capsys.readouterr()
    assert "Other agent message" in captured.out
    assert "My own message" not in captured.out


def test_a2a_poll_exclude_advances_state(poll_server, tmp_path, capsys):
    """Excluded messages still advance the last-seen ID in the state file."""
    base_url, _, rc = poll_server
    state_file = tmp_path / "poll-state-excl-adv.json"
    channel = "poll-test-excl-adv"

    receipts = _seed_messages(rc, channel, [("myself", "Excluded")])
    last_id = receipts[-1]["id"]

    args = argparse.Namespace(
        channel=channel,
        server=base_url,
        state_file=str(state_file),
        exclude="myself",
    )
    _a2a_poll_cmd(args)

    state = json.loads(state_file.read_text())
    # The state should be at least as high as the excluded message's ID.
    assert state.get(channel, -1) >= int(last_id)


def test_a2a_poll_state_file_persists_across_channels(poll_server, tmp_path, capsys):
    """State file tracks multiple channels independently."""
    base_url, _, rc = poll_server
    state_file = tmp_path / "poll-state-multi.json"
    chan_a = "poll-multi-a"
    chan_b = "poll-multi-b"

    _seed_messages(rc, chan_a, [("agent-a", "Chan A message")])
    _seed_messages(rc, chan_b, [("agent-b", "Chan B message")])

    for chan in (chan_a, chan_b):
        args = argparse.Namespace(
            channel=chan,
            server=base_url,
            state_file=str(state_file),
            exclude=None,
        )
        _a2a_poll_cmd(args)

    state = json.loads(state_file.read_text())
    assert chan_a in state
    assert chan_b in state
    assert state[chan_a] != state[chan_b] or (state[chan_a] >= 0 and state[chan_b] >= 0)
