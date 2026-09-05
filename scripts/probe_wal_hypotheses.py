"""Test the two fix hypotheses in isolation before touching _db.py.

Hypothesis A: setting busy_timeout BEFORE journal_mode=WAL makes SQLite's
internal block-and-retry cover the WAL pragma, eliminating the race.

Hypothesis B: a small retry loop around the WAL pragma on SQLITE_BUSY,
re-raising other errors, matching run_schema semantics.

Runs the probe against: master ordering, A only, A+B, reporting failure rate
and failure frames for each.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sqlite3
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from taosmd import _db

BUSY_TIMEOUT_MS = 5000
SCHEMA_RETRY_ATTEMPTS = 5


def connect_master(db_path):
    """Master ordering: WAL pragma BEFORE busy_timeout (the bug)."""
    conn = sqlite3.connect(db_path, check_same_thread=True)
    row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
    mode = (row[0] if row else "") or ""
    conn.execute(f"PRAGMA busy_timeout={int(BUSY_TIMEOUT_MS)}")
    return conn


def connect_A(db_path):
    """Hypothesis A: busy_timeout BEFORE WAL pragma."""
    conn = sqlite3.connect(db_path, check_same_thread=True)
    conn.execute(f"PRAGMA busy_timeout={int(BUSY_TIMEOUT_MS)}")
    row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
    return conn


def connect_AplusB(db_path):
    """Hypothesis A + B: busy_timeout first, then retry WAL pragma on busy."""
    conn = sqlite3.connect(db_path, check_same_thread=True)
    conn.execute(f"PRAGMA busy_timeout={int(BUSY_TIMEOUT_MS)}")
    last_err = None
    for attempt in range(SCHEMA_RETRY_ATTEMPTS):
        try:
            row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
            return conn
        except sqlite3.OperationalError as exc:
            msg = str(exc)
            last_err = exc
            if "locked" not in msg and "busy" not in msg:
                raise
            time.sleep(0.005 * (attempt + 1))
    raise last_err


WORKERS = 8


def _worker(connect_fn, db_path, out_q):
    try:
        conn = connect_fn(db_path)
        conn.close()
        out_q.put(("ok", None))
    except Exception as exc:
        out_q.put(("err", {"repr": repr(exc), "tb": traceback.format_exc()}))


def run_round(connect_fn, root, round_no):
    db_path = str(root / f"r{round_no}" / "test.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.unlink(db_path)
    for ext in ("-wal", "-shm", "-journal"):
        p = Path(db_path + ext)
        if p.exists():
            p.unlink()
    q: mp.Queue = mp.Queue()
    procs = [mp.Process(target=_worker, args=(connect_fn, db_path, q)) for _ in range(WORKERS)]
    for p in procs:
        p.start()
    results = [q.get() for _ in procs]
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    errs = [r[1] for r in results if r[0] == "err"]
    oks = [r for r in results if r[0] == "ok"]
    return len(oks), errs


def probe(connect_fn, label, rounds, root):
    total_fail = 0
    frames = []
    for i in range(rounds):
        _, errs = run_round(connect_fn, root, i)
        total_fail += len(errs)
        for e in errs:
            frames.append(e["tb"])
    rate = total_fail / (rounds * WORKERS) * 100
    print(f"[probe] {label}: rounds={rounds} workers={WORKERS} "
          f"failed={total_fail} rate={rate:.2f}%")
    # frame signatures
    from collections import Counter
    sigs = Counter()
    for tb in frames:
        lines = tb.strip().splitlines()
        # the frame line inside _db or the connect function
        for ln in lines:
            if "PRAGMA journal_mode=WAL" in ln or "busy_timeout" in ln or "database is locked" in ln or "_db.py" in ln:
                sigs[ln.strip()] += 1
    for sig, cnt in sigs.most_common():
        print(f"        {cnt}x  {sig}")
    if frames:
        print(f"        first failure:")
        print("        " + "\n        ".join(frames[0].strip().splitlines()[-3:]))
    return total_fail, frames


def main(rounds=200):
    loadavg = os.getloadavg()[0]
    print(f"[probe] loadavg1={loadavg:.2f}")
    tmp = Path(tempfile.mkdtemp(prefix="wal_hypo_"))
    print("=== Hypothesis A: busy_timeout BEFORE journal_mode (no retry) ===")
    probe(connect_A, "A: busy_timeout-first", rounds, tmp / "a")
    print("\n=== Hypothesis A+b: busy_timeout-first + WAL pragma retry ===")
    probe(connect_AplusB, "A+B: timeout-first+retry", rounds, tmp / "b")
    print("\n=== Control: master ordering (WAL before timeout) ===")
    probe(connect_master, "master: WAL-first", rounds, tmp / "c")


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    main(rounds)
