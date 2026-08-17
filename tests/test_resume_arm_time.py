"""Tests for scripts/resume_arm_time.py.

Red-first contract:
- test_do_fire_posts_resume_due_to_bus: the property the card asked for --
  firing the durable cron must post a [RESUME DUE] message to the A2A bus
  so a live sibling agent or Jay sees it. This fails against the old
  implementation that only appends to a log and self-deletes.
- test_do_fire_removes_only_own_crontab_entry: exact-marker dedup and
  self-removal must not touch unrelated entries.
- test_do_fire_message_contains_armed_timestamp: the log/bus entry must
  carry the armed-at value, not just the fire time.
"""

from __future__ import annotations

import asyncio
import datetime
import subprocess
import sys
import types

import pytest

sys.path.insert(0, "scripts")
import resume_arm_time  # noqa: E402

from taosmd import service  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_stores(data_dir):
    from taosmd import api as taosmd_api
    stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))

    async def _fake_embed(text, task="search_document"):
        return [0.0] * 8

    stores["vector"].embed = _fake_embed  # type: ignore[assignment]
    return stores


# ---------------------------------------------------------------------------
# Property: firing posts [RESUME DUE] to the bus
# ---------------------------------------------------------------------------

def test_do_fire_posts_resume_due_to_bus(tmp_path, monkeypatch):
    """Firing the resume cron posts [RESUME DUE] to agent-rules."""
    from taosmd import api as taosmd_api

    data_dir = tmp_path / "taosmd-resume"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    _setup_stores(data_dir)
    dd = str(data_dir)

    resume_arm_time.do_fire(
        fire_type="primary",
        marker="TAOSMD-RESUME-PRIMARY-00000001",
        armed_at="2026-08-17T14:00:00+00:00",
        data_dir=dd,
    )

    msgs = asyncio.run(service.a2a_feed(thread="agent-rules", data_dir=dd))
    bodies = [m["body"] for m in msgs]
    assert any("[RESUME DUE]" in b for b in bodies), (
        "expected [RESUME DUE] on agent-rules, got: " + repr(bodies)
    )


# ---------------------------------------------------------------------------
# Property: self-deletion removes only the firing entry
# ---------------------------------------------------------------------------

def test_do_fire_removes_only_own_crontab_entry(monkeypatch):
    """Only the matching marker-prefixed line is removed."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.setdefault("calls", []).append((cmd, kwargs))
        if cmd == ["crontab", "-l"]:
            return types.SimpleNamespace(
                stdout="existing entry\n"
                       "TAOSMD-RESUME-PRIMARY-00000001 do_fire primary\n"
                       "unrelated entry\n",
                stderr="",
                returncode=0,
            )
        if cmd == ["crontab", "-"]:
            written = kwargs.get("input", "")
            captured["written"] = written
            return types.SimpleNamespace(stdout="", stderr="", returncode=0)
        return subprocess.run(cmd, **kwargs)

    monkeypatch.setattr(resume_arm_time.subprocess, "run", fake_run)

    resume_arm_time.do_fire(
        fire_type="primary",
        marker="TAOSMD-RESUME-PRIMARY-00000001",
        armed_at="2026-08-17T14:00:00+00:00",
    )

    written = captured.get("written", "")
    assert "existing entry" in written
    assert "unrelated entry" in written
    assert "TAOSMD-RESUME-PRIMARY-00000001" not in written


# ---------------------------------------------------------------------------
# Property: message carries the armed-at timestamp
# ---------------------------------------------------------------------------

def test_do_fire_message_contains_armed_timestamp(tmp_path, monkeypatch):
    """The posted [RESUME DUE] message must name the armed window."""
    from taosmd import api as taosmd_api

    data_dir = tmp_path / "taosmd-resume-ts"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    _setup_stores(data_dir)
    dd = str(data_dir)

    armed_at = "2026-08-17T14:00:00+00:00"
    resume_arm_time.do_fire(
        fire_type="primary",
        marker="TAOSMD-RESUME-PRIMARY-00000002",
        armed_at=armed_at,
        data_dir=dd,
    )

    msgs = asyncio.run(service.a2a_feed(thread="agent-rules", data_dir=dd))
    bodies = [m["body"] for m in msgs]
    assert any(armed_at in b for b in bodies), (
        "armed_at timestamp missing from bus message: " + repr(bodies)
    )


# ---------------------------------------------------------------------------
# Derivation: primary is strictly after the watcher tick + margin
# ---------------------------------------------------------------------------

def test_derive_primary_after_watcher_tick():
    """Primary fire must be strictly after the next watcher tick."""
    resets_at = datetime.datetime(2026, 8, 17, 14, 2, 0)
    primary_cron, _ = resume_arm_time.derive(resets_at)
    minute, hour = map(int, primary_cron.split()[:2])
    primary_dt = datetime.datetime(2026, 8, 17, hour, minute)
    assert primary_dt > resets_at + datetime.timedelta(minutes=10)


def test_derive_retry_after_primary():
    """Retry fire must be strictly after the primary fire."""
    resets_at = datetime.datetime(2026, 8, 17, 14, 2, 0)
    _, retry_cron = resume_arm_time.derive(resets_at)
    minute, hour = map(int, retry_cron.split()[:2])
    retry_dt = datetime.datetime(2026, 8, 17, hour, minute)
    primary_cron, _ = resume_arm_time.derive(resets_at)
    pmin, phour = map(int, primary_cron.split()[:2])
    primary_dt = datetime.datetime(2026, 8, 17, phour, pmin)
    assert retry_dt > primary_dt


# ---------------------------------------------------------------------------
# Output: main prints both lines under the correct heading
# ---------------------------------------------------------------------------

def test_main_prints_both_lines_under_heading(tmp_path, monkeypatch, capsys):
    """python3 resume_arm_time.py <resets_at> prints primary and retry."""
    monkeypatch.setattr(
        sys, "argv", ["resume_arm_time.py", "2026-08-17T14:00:00+00:00"]
    )
    resume_arm_time.main()
    out = capsys.readouterr().out
    assert "CRONTAB" in out
    lines = [l for l in out.splitlines() if l.strip() and not l.startswith("#") and "CRONTAB" not in l]
    assert len(lines) == 2
