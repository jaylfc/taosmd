"""Mutation harness for the _db.connect WAL-pragma fix (acceptance 4).

For each BRANCH of the fix, produce a mutant source (one textual change),
write it into taosmd/_db.py, run the deterministic tests, capture the FAILED
and ERROR counts, then restore the committed fix. Reports both counts together
per mutant.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "taosmd" / "_db.py"
BACKUP = str(DB_PATH) + ".mutbak"
TEST = "tests/test_db_connect_wal_retry.py"


def read_fixed() -> str:
    return DB_PATH.read_text()


def restore_fixed():
    DB_PATH.write_text(read_fixed() if DB_PATH.exists() else "")


MUTANTS = {}

FIXED = None


def make_master(source: str) -> str:
    """M1 -- ordering branch: drop busy_timeout-before-WAL.

    Deletes the single ``busy_timeout`` line so the connection is never armed
    before the WAL pragma. Kills the ordering test.
    """
    return source.replace(
        '    conn.execute(f"PRAGMA busy_timeout={int(BUSY_TIMEOUT_MS)}")\n',
        ""
    )


def make_no_retry(source: str) -> str:
    """M2 -- retry branch: collapse the retry loop to a single WAL pragma call.

    Reverts the WAL pragma to the pre-fix single-shot call (master behaviour).
    Kills every retry-exhaustion/retry-then-succeed test.
    """
    old = '''    mode = ""
    for attempt in range(WAL_RETRY_ATTEMPTS):
        try:
            row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
        except sqlite3.OperationalError as exc:
            lowered = str(exc).lower()
            if "locked" not in lowered and "busy" not in lowered:
                raise
            if attempt == WAL_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(0.05 * (attempt + 1))
            continue
        mode = (row[0] if row else "") or ""
        break
'''
    new = '''    row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
    mode = (row[0] if row else "") or ""
'''
    assert old in source, "retry loop block not found"
    return source.replace(old, new)


def make_no_exhaustion_raise(source: str) -> str:
    """M3 -- exhaustion branch: delete the ``on last attempt, raise`` line.

    A persistent lock error would then fall through instead of surfacing, so the
    exhaustion test stops asserting-raising.
    """
    block = '''            if attempt == WAL_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(0.05 * (attempt + 1))
            continue
'''
    replacement = '''            time.sleep(0.05 * (attempt + 1))
            continue
'''
    assert block in source, "exhaustion block not found"
    return source.replace(block, replacement)


def make_no_fallback_warn(source: str) -> str:
    """M4 -- fallback branch: drop the warning-on-fallback block entirely.

    A WAL fallback must still warn (not raise, not silently pass); deleting the
    branch lets the fallback test observe zero warnings.
    """
    block = '''    if mode.lower() not in ("wal", "memory"):
        warnings.warn(
            f"SQLite WAL mode not enabled for {db_path!r} "
            f"(journal_mode={mode!r}); concurrent access may hit "
            "'database is locked'.",
            RuntimeWarning,
            stacklevel=2,
        )
'''
    assert block in source, "fallback-warn block not found"
    return source.replace(block, "")


MUTANTS = [
    ("M1 ordering: drop busy_timeout-before-WAL", make_master),
    ("M2 retry: collapse to single WAL pragma call", make_no_retry),
    ("M3 exhaustion: drop on-last-attempt raise", make_no_exhaustion_raise),
    ("M4 fallback: drop warn-on-fallback block", make_no_fallback_warn),
]


def run_tests():
    # NOTE: repo rc is permanently 1 from pre-existing mcp_server import errors,
    # so read the verdict off the FAILED/ERROR counts in the pytest summary line,
    # not the exit status.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / TEST), "-q", "-p", "no:cacheprovider"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
    )
    out = proc.stdout + proc.stderr
    # Parse pytest's summary line, e.g. "8 passed in 1.2s" or
    # "2 failed, 6 passed, 1 error in 1.3s".
    passed = failed = errors = 0
    import re
    m = re.search(r"(\d+) passed", out)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+) failed", out)
    if m:
        failed = int(m.group(1))
    m = re.search(r"(\d+) error", out)
    if m:
        errors = int(m.group(1))
    verdict = "RED" if (failed or errors) else "GREEN"
    return passed, failed, errors, verdict, out


def main():
    global FIXED
    FIXED = read_fixed()
    shutil.copyfile(DB_PATH, BACKUP)
    try:
        # Baseline: the real fix should be fully green.
        p, f, e, v, _ = run_tests()
        print(f"[mut] BASELINE fix: passed={p} failed={f} errors={e} -> {v}")
        print()
        for label, mutator in MUTANTS:
            DB_PATH.write_text(mutator(FIXED))
            p, f, e, v, out = run_tests()
            print(f"[mut] {label}")
            print(f"       passed={p} FAILED={f} ERROR={e} -> {v}")
            if v == "RED":
                # Which test names went red?
                for ln in out.splitlines():
                    if "FAILED" in ln or "ERROR" in ln:
                        print(f"         {ln.strip()}")
            print()
    finally:
        shutil.copyfile(BACKUP, DB_PATH)
        Path(BACKUP).unlink()


if __name__ == "__main__":
    main()
