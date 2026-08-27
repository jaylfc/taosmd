"""Tests for scripts/resume_arm_time.py (durable-cron revision, tsk-y62mpe).

Each test here guards one of the five blockers from the #317/#323 review:

A. derivation emits a date-pinned one-shot (7 0 18 8 *), not a daily line;
B. the subprocess fire test keys on the armed-at token (not the bare word
   "fired") and cannot be satisfied by pre-existing log state;
C. the emitted cron line names /usr/bin/python3, never bare python3;
D. every do_fire caller is sandboxed (fake crontab on PATH / mocked run, and a
   redirected HOME): no test reads or writes the real crontab or real log;
E. getpass is imported, so crontab-failure messages never degrade to NameError.
"""
from __future__ import annotations

import datetime
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "resume_arm_time.py"

WATCHER_LINE = "6,16,26,36,46,56 * * * * /usr/bin/bash /home/jay/.taos-usage/watch.sh --once\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("resume_arm_time_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


resume_arm_time = _load_module()


def _marker(fire_type, armed_at):
    ts = datetime.datetime.fromisoformat(armed_at).strftime("%Y%m%d%H%M")
    return f"{SCRIPT}#{fire_type}-{ts}"


class _Proc:
    """Minimal subprocess.CompletedProcess stand-in for the fake crontab."""

    def __init__(self, rc, out, err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def _fake_run(canned_stdout, written):
    """A fake subprocess.run answering crontab -l (list) and crontab - (write).

    Reads the recorded input verbatim (never via command substitution) so a
    trailing newline is preserved; the old `$(cat)` shim stripped it and
    mis-reported correct input as malformed (tsk-309 regression).
    """

    def fake(cmd, **kwargs):
        if cmd == ["crontab", "-l"]:
            return _Proc(0, canned_stdout)
        if cmd == ["crontab", "-"]:
            written.append(kwargs.get("input", ""))
            return _Proc(0, "")
        return _Proc(0, "")

    return fake


# A fake `crontab` executable used by the subprocess test so do_fire never
# touches a real crontab. It reads stdin RAW for `crontab -` and prints the
# stored state for `crontab -l`.
_FAKE_CRONTAB_SHIM = """#!/usr/bin/env python3
import os, sys
p = os.environ["FAKE_CRONTAB_PATH"]
a = sys.argv[1] if len(sys.argv) > 1 else ""
if a == "-l":
    try:
        sys.stdout.write(open(p).read()); sys.exit(0)
    except FileNotFoundError:
        sys.stderr.write("no crontab for user\\n"); sys.exit(1)
elif a == "-":
    open(p, "wb").write(sys.stdin.buffer.read()); sys.exit(0)
sys.stderr.write("unsupported crontab args: %s\\n" % (sys.argv[1:],)); sys.exit(2)
"""


def _install_fake_crontab(tmp_path, monkeypatch, initial_text):
    """Put a fake crontab on PATH, redirect HOME to tmp_path, return state file."""
    state = tmp_path / "crontab-state"
    state.write_text(initial_text)
    fakebin = tmp_path / "bin"
    fakebin.mkdir(parents=True, exist_ok=True)
    shim = fakebin / "crontab"
    shim.write_text(_FAKE_CRONTAB_SHIM, encoding="utf-8")
    shim.chmod(0o755)
    monkeypatch.setenv("FAKE_CRONTAB_PATH", str(state))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", f"{fakebin}:{os.defpath}")
    return state


# --------------------------------------------------------------------------- #
# Blocker A: the canonical derivation emits a date-pinned one-shot.
# --------------------------------------------------------------------------- #

def test_canonical_derivation_emits_date_pinned_one_shot(monkeypatch, capsys):
    """For a 00:00:00Z reset the canonical helper prints 7 0 18 8 * and
    17 0 18 8 *, a date-pinned one-shot -- not the blocked PR's daily 17 0 * * *."""
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    monkeypatch.setattr(
        resume_arm_time,
        "_HELPER_PATH",
        "/home/jay/.taos-fleet-tools/scripts/resume_arm_time.py",
    )
    written = []
    monkeypatch.setattr(
        resume_arm_time.subprocess, "run", _fake_run(WATCHER_LINE, written)
    )
    monkeypatch.setattr(
        sys, "argv", ["resume_arm_time.py", "2026-08-18T00:00:00.473075+00:00"]
    )

    resume_arm_time.main()
    out = capsys.readouterr().out

    cron_line = [l for l in out.splitlines() if l.startswith("CRON")][0]
    retry_line = [l for l in out.splitlines() if l.startswith("RETRY CRON")][0]
    assert cron_line.split()[1:6] == ["7", "0", "18", "8", "*"]
    assert retry_line.split()[2:7] == ["17", "0", "18", "8", "*"]
    # The derivation is date-pinned: the day field is present, not "*".
    assert "8 *" in cron_line and "18 8 *" in cron_line


# --------------------------------------------------------------------------- #
# Blocker C: the emitted cron line names the interpreter explicitly.
# --------------------------------------------------------------------------- #

def test_system_crontab_block_names_usr_bin_python3():
    """The durable line invokes /usr/bin/python3 explicitly, never bare python3,
    so it can never resolve to an interpreter that cannot import taosmd."""
    fire = datetime.datetime(2026, 8, 18, 0, 7, 0, tzinfo=datetime.timezone.utc)
    retry = datetime.datetime(2026, 8, 18, 0, 17, 0, tzinfo=datetime.timezone.utc)
    block = resume_arm_time.system_crontab_block(fire, retry)
    assert f"/usr/bin/python3 {resume_arm_time._HELPER_PATH} --fire primary" in block
    assert f"/usr/bin/python3 {resume_arm_time._HELPER_PATH} --fire retry" in block
    # Independent check against SCRIPT constant (proves path sensitivity)
    assert f"/usr/bin/python3 {SCRIPT} --fire primary" in block
    assert f"/usr/bin/python3 {SCRIPT} --fire retry" in block
    # No bare `python3` token (the blocked PR emitted `python3 <path>`).
    assert " python3 " not in block


# --------------------------------------------------------------------------- #
# Blocker E: getpass is imported.
# --------------------------------------------------------------------------- #

def test_helper_imports_getpass():
    assert "getpass" in dir(resume_arm_time)
    import getpass as _gp
    assert resume_arm_time.getpass is _gp


# --------------------------------------------------------------------------- #
# Helper-path guard: refuse to emit from a temp or linked-worktree location.
# --------------------------------------------------------------------------- #

def test_guard_refuses_temp_path(monkeypatch, capsys):
    """A _HELPER_PATH under /tmp is refused with a named reason."""
    monkeypatch.setattr(
        resume_arm_time,
        "_HELPER_PATH",
        "/tmp/scratchpad/wt354/scripts/resume_arm_time.py",
    )
    monkeypatch.setattr(
        resume_arm_time.subprocess, "run", _fake_run(WATCHER_LINE, [])
    )
    monkeypatch.setattr(
        sys, "argv", ["resume_arm_time.py", "2026-08-18T00:00:00+00:00"]
    )
    with pytest.raises(SystemExit) as exc:
        resume_arm_time.main()
    assert "temp directory" in str(exc.value)


def test_guard_refuses_linked_worktree(monkeypatch, tmp_path):
    """main() refuses when _HELPER_PATH sits inside a linked worktree.

    This goes through main() rather than the detector, because the detector
    returning a string is not the behaviour the guard exists to provide.

    _is_under_temp is neutralised deliberately, and the test is worthless
    without that. The two refusals are checked in order, and tmp_path IS under
    a temp root, so a fixture built there trips the TEMP refusal first and
    never reaches the branch this test is named for. Left in, the test would
    pass on the sibling refusal's message while the worktree refusal could be
    deleted outright.
    """
    fake_wt = tmp_path / "fake-wt"
    fake_wt.mkdir()
    (fake_wt / ".git").write_text("gitdir: /some/real/.git/worktrees/fake\n")
    target = fake_wt / "scripts" / "resume_arm_time.py"

    monkeypatch.setattr(resume_arm_time, "_is_under_temp", lambda path: None)
    monkeypatch.setattr(resume_arm_time, "_HELPER_PATH", str(target))
    monkeypatch.setattr(
        resume_arm_time.subprocess, "run", _fake_run(WATCHER_LINE, [])
    )
    monkeypatch.setattr(
        sys, "argv", ["resume_arm_time.py", "2026-08-18T00:00:00+00:00"]
    )

    with pytest.raises(SystemExit) as exc:
        resume_arm_time.main()

    message = str(exc.value)
    assert "is inside a git worktree" in message
    assert str(fake_wt) in message
    # Not the sibling refusal wearing this test's name.
    assert "temp directory" not in message


def test_is_in_linked_worktree_detects_git_file(tmp_path):
    """_is_in_linked_worktree returns the worktree root when .git is a file."""
    fake_wt = tmp_path / "fake-wt"
    fake_wt.mkdir()
    (fake_wt / ".git").write_text("gitdir: /some/real/.git/worktrees/fake\n")
    target = fake_wt / "scripts" / "resume_arm_time.py"
    result = resume_arm_time._is_in_linked_worktree(str(target))
    assert result == str(fake_wt)


def test_is_in_linked_worktree_ignores_the_main_worktree(tmp_path):
    """A .git DIRECTORY is the main worktree, where arming must keep working."""
    main_wt = tmp_path / "main-wt"
    (main_wt / ".git").mkdir(parents=True)
    target = main_wt / "scripts" / "resume_arm_time.py"
    assert resume_arm_time._is_in_linked_worktree(str(target)) is None


# --------------------------------------------------------------------------- #
# Blockers B and D: do_fire posts [RESUME DUE], records the armed-at token,
# self-deletes its crontab marker, and is fully sandboxed.
# --------------------------------------------------------------------------- #

def test_do_fire_posts_resume_due_to_bus(tmp_path, monkeypatch):
    """On a successful bus post the [RESUME DUE] message carries the armed-at
    token and the crontab self-removal runs, with no log fallback."""
    from taosmd import service as taosmd_service

    sent = []

    async def fake_send(sender, body, *, thread="general", **kw):
        sent.append({"sender": sender, "body": body, "thread": thread})
        return {"id": "0", "from": sender, "thread": thread}

    monkeypatch.setattr(taosmd_service, "a2a_send", fake_send)
    monkeypatch.setenv("HOME", str(tmp_path))

    armed_at = "2026-08-18T00:00:00.473075+00:00"
    marker = _marker("primary", armed_at)
    crontab_text = (
        WATCHER_LINE
        + f"# taOSmd-resume: {marker}\n"
        + f"{marker} do_not_keep_me\n"
        + "unrelated job\n"
    )
    written = []
    monkeypatch.setattr(resume_arm_time.subprocess, "run", _fake_run(crontab_text, written))

    resume_arm_time.do_fire("primary", armed_at)

    assert sent, "do_fire should have posted to the bus"
    body = sent[0]["body"]
    assert "[RESUME DUE]" in body
    assert armed_at in body  # the exact armed-at token
    # Success path: the bus message is the record; no log fallback is written.
    assert not (tmp_path / ".taos-team" / "resume_fire.log").exists()
    # Crontab self-removal: only marker lines drop, everything else survives.
    assert written, "do_fire should have rewritten the crontab"
    assert marker not in written[0]
    assert "taos-usage/watch.sh" in written[0]
    assert "unrelated job" in written[0]


def test_do_fire_bus_failure_writes_visible_record(tmp_path, monkeypatch):
    """A failed bus post is NOT swallowed: the fallback log names the failure
    and still carries the armed-at token, and the crontab self-removal still runs."""
    from taosmd import service as taosmd_service

    async def boom(sender, body, *, thread="general", **kw):
        raise RuntimeError("bus down")

    monkeypatch.setattr(taosmd_service, "a2a_send", boom)
    monkeypatch.setenv("HOME", str(tmp_path))

    armed_at = "2026-08-18T00:00:00.473075+00:00"
    marker = _marker("primary", armed_at)
    crontab_text = f"# taOSmd-resume: {marker}\nunrelated job\n"
    written = []
    monkeypatch.setattr(resume_arm_time.subprocess, "run", _fake_run(crontab_text, written))

    resume_arm_time.do_fire("primary", armed_at)

    log_path = tmp_path / ".taos-team" / "resume_fire.log"
    record = log_path.read_text()
    # Keys on the armed-at token unique to THIS invocation, not bare "fired".
    assert armed_at in record
    assert "[RESUME DUE]" in record
    assert "bus_post_failed" in record  # the failure is visible, not swallowed
    # Crontab self-removal still ran despite the bus failure.
    assert marker not in written[0]
    assert "unrelated job" in written[0]


def test_do_fire_crontab_read_failure_is_visible(tmp_path, monkeypatch):
    """A crontab read failure raises SystemExit (non-zero) with a named reason
    that calls getpass.getuser() -- exercising blocker E on the error path."""
    from taosmd import service as taosmd_service

    sent = []

    async def fake_send(sender, body, *, thread="general", **kw):
        sent.append(body)
        return {"id": "0", "from": sender}

    monkeypatch.setattr(taosmd_service, "a2a_send", fake_send)
    monkeypatch.setenv("HOME", str(tmp_path))

    def fail_read(cmd, **kwargs):
        if cmd == ["crontab", "-l"]:
            return _Proc(1, "", "perm denied")
        return _Proc(0, "")

    monkeypatch.setattr(resume_arm_time.subprocess, "run", fail_read)

    with pytest.raises(SystemExit) as exc:
        resume_arm_time.do_fire("primary", "2026-08-18T00:00:00.473075+00:00")
    assert "could not read the crontab" in str(exc.value)


# --------------------------------------------------------------------------- #
# Blocker B (the decisive one): the subprocess test cannot be satisfied by
# pre-existing state. It runs under /usr/bin/python3 (cron's interpreter),
# redirects HOME, and keys on the exact armed-at token.
# --------------------------------------------------------------------------- #

def test_do_fire_runs_as_subprocess(tmp_path, monkeypatch):
    """Runs the emitted cron command under /usr/bin/python3 with a sandboxed
    crontab and HOME. Asserts the marker written by arming is the marker
    removed by firing, and the fire timestamp is recorded. Gut do_fire and
    this test goes RED: an invocation that records nothing fails the assertion
    against an empty log."""
    if not Path("/usr/bin/python3").exists():
        pytest.skip("/usr/bin/python3 is not present in this environment")

    # Derive the marker from the arming code so the test is coupled to
    # system_crontab_block, not to a hand-written literal.
    fire_dt = datetime.datetime(2026, 8, 18, 0, 7, 0, tzinfo=datetime.timezone.utc)
    retry_dt = datetime.datetime(2026, 8, 18, 0, 17, 0, tzinfo=datetime.timezone.utc)
    block = resume_arm_time.system_crontab_block(fire_dt, retry_dt)
    marker_line = next(l for l in block.splitlines() if l.startswith("# taOSmd-resume: "))
    marker = marker_line[len("# taOSmd-resume: "):]

    initial_crontab = (
        f"# taOSmd-resume: {marker}\n"
        + f"7 0 18 8 * /usr/bin/python3 {resume_arm_time._HELPER_PATH} --fire primary {fire_dt.isoformat()}\n"
        + WATCHER_LINE
    )
    state = _install_fake_crontab(tmp_path, monkeypatch, initial_crontab)

    proc = subprocess.run(
        ["/usr/bin/python3", str(SCRIPT), "--fire", "primary", fire_dt.isoformat()],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    # The record must carry THIS invocation's exact timestamp, not the
    # bare word "fired" -- which (per the review) already has 24 lines in the
    # real log and would satisfy the old assertion before the subprocess runs.
    log_path = tmp_path / ".taos-team" / "resume_fire.log"
    record = log_path.read_text()
    assert "[RESUME DUE]" in record
    assert fire_dt.isoformat() in record

    # And the self-removal happened against the SANDBOXED crontab, not the real one.
    after = state.read_text()
    assert marker not in after
    assert "taos-usage/watch.sh" in after
