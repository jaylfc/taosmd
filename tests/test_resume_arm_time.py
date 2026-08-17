"""Tests for scripts/resume_arm_time.py.

Red-first contract:
- test_do_fire_posts_resume_due_to_bus: the property the card asked for --
  firing the durable cron must post a [RESUME DUE] message to the A2A bus
  so a live sibling agent or Jay sees it. This fails against the old
  implementation that only appends to a log and self-deletes.
- test_do_fire_removes_only_own_crontab_entry: exact-marker dedup and
  self-removal must not touch unrelated entries. Now also verifies the
  crontab write includes a trailing newline.
- test_do_fire_message_contains_armed_timestamp: the log/bus entry must
  carry the armed-at value, not just the fire time.
- test_derive_primary_after_watcher_tick: primary fire must be strictly after
  the next watcher tick.
- test_derive_retry_after_primary: retry fire must be strictly after the
  primary fire.
- test_main_prints_both_lines_under_heading: python3 resume_arm_time.py
  <resets_at> prints primary and retry under the CRONTAB heading.
- test_do_fire_runs_as_subprocess: the emitted command runs as a subprocess
  with the cron interpreter, asserting [RESUME DUE] message or log fallback.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import subprocess
import sys
import types
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, "scripts")
import resume_arm_time  # noqa: E402

from taosmd import service  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: compute marker from script path (matches _marker)
# ---------------------------------------------------------------------------

def _marker_from_path(fire_type: str, script_rel_path: str = "scripts/resume_arm_time.py") -> str:
    digest = hashlib.sha256(script_rel_path.encode()).hexdigest()[:8]
    return f"TAOSMD-RESUME-{fire_type.upper()}-{digest}"


# Marker values for the test script path
PRIMARY_MARKER = _marker_from_path("primary")
RETRY_MARKER = _marker_from_path("retry")


# ---------------------------------------------------------------------------
# Helper: set up stores
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
        marker=PRIMARY_MARKER,
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
    """Only the matching marker-prefixed line is removed.

    Also verifies the crontab write includes a trailing newline.
    """
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.setdefault("calls", []).append((cmd, kwargs))
        if cmd == ["crontab", "-l"]:
            return types.SimpleNamespace(
                stdout=f"existing entry\n"
                       f"{PRIMARY_MARKER} do_fire primary\n"
                       f"unrelated entry\n",
                stderr="",
                returncode=0,
            )
        if cmd == ["crontab", "-"]:
            written = kwargs.get("input", "")
            captured["written"] = written
            # Ensure the written input has a trailing newline
            assert written.endswith("\n"), (
                f"crontab - input missing trailing newline, got {written!r}"
            )
            return types.SimpleNamespace(stdout="", stderr="", returncode=0)
        return subprocess.run(cmd, **kwargs)

    monkeypatch.setattr(resume_arm_time.subprocess, "run", fake_run)

    resume_arm_time.do_fire(
        fire_type="primary",
        marker=PRIMARY_MARKER,
        armed_at="2026-08-17T14:00:00+00:00",
    )

    written = captured.get("written", "")
    assert "existing entry" in written
    assert "unrelated entry" in written
    assert PRIMARY_MARKER not in written


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
        marker=PRIMARY_MARKER,
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


# ---------------------------------------------------------------------------
# New: runs the emitted command as a subprocess with the cron interpreter
# ---------------------------------------------------------------------------

def test_do_fire_runs_as_subprocess(tmp_path, capsys, monkeypatch):
    """Run the emitted cron command as a subprocess; assert the function does
    not silently swallow the failure.

    The old bug (except Exception: pass everywhere, exit 0, nothing recorded)
    is fixed when the function either posts [RESUME DUE] to the A2A bus or
    falls back to appending to the log file at ~/.taos-team/resume_fire.log."""
    from taosmd import api as taosmd_api

    data_dir = tmp_path / "taosmd-resume-sub"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    _setup_stores(data_dir)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/resume_arm_time.py",
                "--fire",
                "--type",
                "primary",
                "--marker",
                PRIMARY_MARKER,
                "--armed-at",
                "2026-08-17T14:00:00+00:00",
            ],
            cwd=str(Path(__file__).parent.parent),
            capture_output=True,
            text=True,
        )

    output = proc.stdout + proc.stderr
    # The log-fallback writes to ~/.taos-team/resume_fire.log
    log_path = str(Path.home() / ".taos-team" / "resume_fire.log")
    log_marker = "fired" in Path(log_path).read_text() if Path(log_path).exists() else False

    # Verify the function produced some record -- bus post or log fallback.
    # The old bug: except Exception: pass meant exit 0 with nothing recorded.
    record_exists = "[RESUME DUE]" in output or log_marker

    assert record_exists, (
        f"expected [RESUME DUE] in output or log fallback entry, "
        f"got output: {output!r}, log has fired: {log_marker}"
    )