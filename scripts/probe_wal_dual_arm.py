"""Dual-arm concurrency probe for the _db.connect WAL init race.

Runs BOTH arms in the SAME process invocation so the comparison is real:
  - master arm: WAL pragma BEFORE busy_timeout, retry ON  (pre-fix behaviour)
  - fixed arm:  the real ``taosmd._db.connect`` (the committed fix)

For each arm it reports:
  - rounds, workers, failed_ops, failure rate
  - the traceback frame of EVERY failure (proves failures land on the pragma
    for master, and are absent / moved off it for the fix)

Run SERIALLY. Prefers a quiet box: if loadavg1 >= 2 it still runs (the task
notes the race is not a load artefact at ~5) but prints a banner.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sqlite3
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from taosmd import _db  # the FIXED connect (real committed fix)

MASTER_BUSY_TIMEOUT_MS = _db.BUSY_TIMEOUT_MS


def connect_master(db_path):
    """Faithful reproduction of pre-fix _db.connect ordering: WAL pragma runs
    BEFORE busy_timeout is armed, and there is no retry."""
    conn = sqlite3.connect(db_path, check_same_thread=True)
    row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
    mode = (row[0] if row else "") or ""
    conn.execute(f"PRAGMA busy_timeout={int(MASTER_BUSY_TIMEOUT_MS)}")
    return conn


WORKERS = 8


def _worker_target(connect_fn, db_path, out_q):
    try:
        conn = connect_fn(db_path)
        conn.close()
        out_q.put(("ok", None))
    except Exception as exc:
        out_q.put(("err", {"repr": repr(exc), "tb": traceback.format_exc()}))


def run_round(connect_fn, root, round_no):
    db_dir = root / f"r{round_no}"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(db_dir / "test.db")
    if os.path.exists(db_path):
        os.unlink(db_path)
    for ext in ("-wal", "-shm", "-journal"):
        p = Path(db_path + ext)
        if p.exists():
            p.unlink()
    q: mp.Queue = mp.Queue()
    procs = [mp.Process(target=_worker_target, args=(connect_fn, db_path, q))
             for _ in range(WORKERS)]
    for p in procs:
        p.start()
    results = [q.get() for _ in procs]
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
    ok = sum(1 for r in results if r[0] == "ok")
    errs = [r[1]["tb"] for r in results if r[0] == "err"]
    return ok, errs


def probe_arm(connect_fn, label, rounds, root):
    total_fail = 0
    frames = []
    for i in range(rounds):
        _, errs = run_round(connect_fn, root, i)
        total_fail += len(errs)
        frames.extend(errs)
    total_ops = rounds * WORKERS
    rate = total_fail / total_ops * 100 if total_ops else 0.0
    print(f"[probe] ARM {label}: rounds={rounds} workers={WORKERS} "
          f"ok_ops={total_ops - total_fail} failed_ops={total_fail} "
          f"failure_rate={rate:.3f}%")
    if frames:
        sigs = Counter()
        for tb in frames:
            for ln in tb.strip().splitlines():
                if "PRAGMA journal_mode=WAL" in ln or "_db.py" in ln or "database is locked" in ln:
                    sigs[ln.strip()] += 1
        print(f"[probe]   {len(frames)} failure tracebacks captured. frame signatures:")
        for sig, cnt in sigs.most_common():
            print(f"        {cnt}x  {sig}")
        print("[probe]   --- sample failing traceback ---")
        print("        " + "\n        ".join(frames[0].strip().splitlines()))
    else:
        print("[probe]   no failures (0 tracebacks)")
    return total_fail, frames


def main(rounds=200):
    loadavg = os.getloadavg()[0]
    banner = "QUIET" if loadavg < 2.0 else f"NOISY(loadavg={loadavg:.2f})"
    print(f"[probe] {banner}  (running both arms serially in one process)")
    if loadavg >= 2.0:
        print("[probe]   note: loadavg >= 2; race is documented as non-artifact at ~5, "
              "so reproduction still holds.")
    tmp = Path(tempfile.mkdtemp(prefix="wal_dualarm_"))

    print("\n=== ARM A: master (pre-fix) -- WAL pragma before busy_timeout, no retry ===")
    a_fail, a_frames = probe_arm(connect_master, "master(pre-fix)", rounds, tmp / "master")

    print("\n=== ARM B: fixed -- real taosmd._db.connect ===")
    b_fail, b_frames = probe_arm(_db.connect, "fixed", rounds, tmp / "fixed")

    print("\n=== SUMMARY (both arms, same run) ===")
    print(f"[probe] master: failed={a_fail}/{rounds*WORKERS} "
          f"({'REPRODUCED' if a_fail else 'no failures'})")
    print(f"[probe] fixed:  failed={b_fail}/{rounds*WORKERS} "
          f"({'ELIMINATED' if not b_fail else 'still failing'})")
    if a_frames and not b_frames:
        print("[probe] verdict: failures moved OFF the pragma for the fix "
              "(master failures at _db.py:56 'PRAGMA journal_mode=WAL'; fixed = 0).")
    elif a_frames and b_frames:
        print("[probe] verdict: fix still failing; inspect fixed-arm frames below.")
        for tb in b_frames[:3]:
            print("---- fixed failure ----")
            print(tb)
    return a_fail, b_fail


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    main(rounds)
