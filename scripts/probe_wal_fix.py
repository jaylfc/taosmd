"""Stress the winning strategy: busy_timeout-first + WAL pragma retry-on-busy."""
from __future__ import annotations

import multiprocessing as mp
import os
import sqlite3
import sys
import tempfile
import time
import traceback
from pathlib import Path

BUSY_TIMEOUT_MS = 5000
SCHEMA_RETRY_ATTEMPTS = 5


def connect_fixed(db_path):
    conn = sqlite3.connect(db_path, check_same_thread=True)
    conn.execute(f"PRAGMA busy_timeout={int(BUSY_TIMEOUT_MS)}")
    last_err = None
    for attempt in range(SCHEMA_RETRY_ATTEMPTS):
        try:
            conn.execute("PRAGMA journal_mode=WAL").fetchone()
            return conn
        except sqlite3.OperationalError as exc:
            msg = str(exc)
            last_err = exc
            if "locked" not in msg and "busy" not in msg:
                raise
            time.sleep(0.005 * (attempt + 1))
    raise last_err


WORKERS = 16


def _worker(db_path, out_q):
    try:
        conn = connect_fixed(db_path)
        conn.close()
        out_q.put(("ok", None))
    except Exception as exc:
        out_q.put(("err", {"repr": repr(exc), "tb": traceback.format_exc()}))


def run_round(root, i):
    db_path = str(root / f"r{i}" / "t.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.unlink(db_path)
    for ext in ("-wal", "-shm", "-journal"):
        p = Path(db_path + ext)
        if p.exists():
            p.unlink()
    q = mp.Queue()
    procs = [mp.Process(target=_worker, args=(db_path, q)) for _ in range(WORKERS)]
    for p in procs:
        p.start()
    results = [q.get() for _ in procs]
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
    return [r for r in results if r[0] == "err"]


def main(rounds):
    loadavg = os.getloadavg()[0]
    tmp = Path(tempfile.mkdtemp(prefix="wal_fix_"))
    total = 0
    fail_frames = []
    for i in range(rounds):
        errs = run_round(tmp, i)
        total += len(errs)
        for e in errs:
            fail_frames.append(e["tb"])
    print(f"[probe] loadavg1={loadavg:.2f} workers={WORKERS} rounds={rounds} "
          f"failed={total} rate={total/(rounds*WORKERS)*100:.4f}%")
    for tb in fail_frames[:3]:
        print("---- failure ----")
        print(tb)


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    main(rounds)
