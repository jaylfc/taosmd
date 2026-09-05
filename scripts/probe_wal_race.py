"""Standalone reproduction probe for the _db.connect WAL init race.

Spawns N worker processes that all call _db.connect on the SAME fresh database
file simultaneously, repeats for `rounds` rounds, and reports:
- total failures, failures-per-round distribution
- the traceback frame of every failure (to prove they land on the pragma)
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from taosmd import _db


WORKERS = 8


def _worker_connect(db_path: str, out_q: mp.Queue) -> None:
    """One worker: open the shared fresh db via _db.connect, report result."""
    try:
        conn = _db.connect(db_path)
        conn.close()
        out_q.put(("ok", None))
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        # Extract the innermost sqlite-related frame file:line + the line of code
        out_q.put(("err", {
            "repr": repr(exc),
            "tb": tb,
        }))


def run_round(root: Path, round_no: int):
    db_path = str(root / f"round-{round_no}" / "test.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.unlink(db_path)
    # Also remove wal/shm/lock sidecars from any prior run.
    for ext in ("-wal", "-shm", "-journal", "-lk"):
        p = Path(db_path + ext)
        if p.exists():
            p.unlink()

    q: mp.Queue = mp.Queue()
    procs = [
        mp.Process(target=_worker_connect, args=(db_path, q))
        for _ in range(WORKERS)
    ]
    for p in procs:
        p.start()
    results = []
    for _ in procs:
        results.append(q.get())
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    errs = [r[1] for r in results if r[0] == "err"]
    oks = [r for r in results if r[0] == "ok"]
    return len(oks), errs


def main(rounds: int = 200):
    loadavg = os.getloadavg()[0]
    print(f"[probe] loadavg1={loadavg:.2f} workers={WORKERS} rounds={rounds}")
    if loadavg >= 2.0:
        print("[probe] WARNING: loadavg >= 2, results may be a load artefact")
    tmp = Path(tempfile.mkdtemp(prefix="wal_race_probe_"))
    total_fail = 0
    failure_frames: list[str] = []
    per_round_counts = []
    for i in range(rounds):
        ok, errs = run_round(tmp, i)
        if errs:
            total_fail += len(errs)
            per_round_counts.append((i, len(errs)))
            for e in errs:
                failure_frames.append(e["tb"])
    print(f"[probe] rounds={rounds} workers_per_round={WORKERS}")
    print(f"[probe] ok_ops={(rounds*WORKERS - total_fail)} failed_ops={total_fail}")
    print(f"[probe] failure_rate={total_fail/(rounds*WORKERS)*100:.2f}%")
    print(f"[probe] rounds_with_failures={len(per_round_counts)}")
    if per_round_counts:
        print(f"[probe] per_round_failure_counts (round_idx, fail_count):")
        for ridx, fc in per_round_counts:
            print(f"  round {ridx}: {fc} failures")
    if failure_frames:
        print(f"[probe] total_failed_tracebacks={len(failure_frames)}")
        # Print unique frame signatures
        from collections import Counter
        frames = Counter()
        for tb in failure_frames:
            last_line = tb.strip().splitlines()[-2] if len(tb.strip().splitlines()) >= 2 else "?"
            frames[last_line] += 1
        print("[probe] failure frame signatures (last frame before exception):")
        for frame, count in frames.most_common():
            print(f"  {count}x  {frame}")
        print("[probe] ---- first failing traceback ----")
        print(failure_frames[0])
        if len(failure_frames) > 1:
            print("[probe] ---- last failing traceback ----")
            print(failure_frames[-1])


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    main(rounds)
