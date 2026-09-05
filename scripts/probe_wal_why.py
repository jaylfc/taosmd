"""Why doesn't busy_timeout-first alone fix the WAL pragma race?

Tight 2-process contention: time each failure to see whether busy_timeout is
being honoured (slow, ~timeout) or bypassed (fast, immediate).
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time
import traceback
from pathlib import Path

BUSY_TIMEOUT_MS = 5000


def connect_timeout_first(db_path):
    conn = sqlite3.connect(db_path, check_same_thread=True)
    conn.execute(f"PRAGMA busy_timeout={int(BUSY_TIMEOUT_MS)}")
    conn.execute("PRAGMA journal_mode=WAL").fetchone()
    return conn


def connect_master(db_path):
    conn = sqlite3.connect(db_path, check_same_thread=True)
    conn.execute("PRAGMA journal_mode=WAL").fetchone()
    conn.execute(f"PRAGMA busy_timeout={int(BUSY_TIMEOUT_MS)}")
    return conn


def worker(fn, db_path, result):
    t0 = time.perf_counter()
    try:
        conn = fn(db_path)
        conn.close()
        result["ok"] = True
    except Exception as exc:
        result["ok"] = False
        result["err"] = repr(exc)
        result["tb"] = traceback.format_exc()
    result["elapsed"] = time.perf_counter() - t0


def run_round(fn, root, i):
    db_path = str(root / f"r{i}" / "t.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.unlink(db_path)
    for ext in ("-wal", "-shm", "-journal"):
        p = Path(db_path + ext)
        if p.exists():
            p.unlink()
    import multiprocessing as mp
    r1 = mp.Manager().dict()
    r2 = mp.Manager().dict()
    p1 = mp.Process(target=worker, args=(fn, db_path, r1))
    p2 = mp.Process(target=worker, args=(fn, db_path, r2))
    p1.start()
    p2.start()
    p1.join(timeout=30)
    p2.join(timeout=30)
    return r1, r2


def main():
    tmp = Path(tempfile.mkdtemp())
    for label, fn in (("timeout-first", connect_timeout_first), ("master", connect_master)):
        fails = []
        max_elapsed = 0.0
        for i in range(50):
            r1, r2 = run_round(fn, tmp, i)
            for r in (r1, r2):
                if not r.get("ok", True):
                    fails.append((r.get("err"), r.get("elapsed")))
            max_elapsed = max(max_elapsed, r1.get("elapsed", 0), r2.get("elapsed", 0))
        print(f"[{label}] fails={len(fails)}/100 max_elapsed={max_elapsed:.3f}s")
        for err, el in fails[:5]:
            print(f"   err={err} elapsed={el:.3f}s")


if __name__ == "__main__":
    main()
