"""Concurrency probe for the busy_timeout / journal_mode=WAL ordering in
:func:`taosmd._db.connect`.

On master, ``PRAGMA journal_mode=WAL`` runs before ``busy_timeout`` is armed,
so concurrent first-time opens of the same fresh database raise
``OperationalError: database is locked`` at the pragma.  The race fires at
roughly 2-3 percent under normal load; this probe runs 200 rounds per arm
with both arms in the same test invocation so the comparison is real.

After the fix (``busy_timeout`` before ``journal_mode``), all rounds succeed
because SQLite's block-and-retry is armed for the pragma itself.
"""

from __future__ import annotations

import threading
import traceback

from taosmd import _db

ROUNDS_PER_ARM = 200
WORKERS_PER_ROUND = 4


def _open(db_path: str, results: list, idx: int) -> None:
    try:
        conn = _db.connect(db_path)
        conn.close()
        results[idx] = None
    except BaseException as exc:  # noqa: BLE001
        results[idx] = exc


def test_wal_pragma_ordering_no_lock_errors(tmp_path):
    """200 rounds per arm, both arms in the same run, zero failures expected.

    Each round creates a fresh database and has four threads open it
    concurrently.  On master the unretried ``PRAGMA journal_mode=WAL`` races
    without ``busy_timeout`` and some threads raise ``database is locked``;
    after the fix every round succeeds.
    """
    all_failures = []
    for arm in range(2):
        for r in range(ROUNDS_PER_ARM):
            db_path = str(tmp_path / f"a{arm}_r{r}.db")
            results = [None] * WORKERS_PER_ROUND
            workers = []
            for i in range(WORKERS_PER_ROUND):
                t = threading.Thread(
                    target=_open, args=(db_path, results, i)
                )
                workers.append(t)
            for t in workers:
                t.start()
            for t in workers:
                t.join(timeout=30)
            for i, exc in enumerate(results):
                if exc is not None:
                    tb = "".join(
                        traceback.format_exception(
                            type(exc), exc, exc.__traceback__
                        )
                    )
                    all_failures.append((arm, r, i, exc, tb))

    if all_failures:
        parts = [
            f"{len(all_failures)} failures in "
            f"{ROUNDS_PER_ARM * 2} rounds "
            f"({WORKERS_PER_ROUND} threads each):"
        ]
        for arm, r, i, exc, tb in all_failures[:20]:
            parts.append(
                f"  arm {arm} round {r} thread {i}: "
                f"{type(exc).__name__}: {exc}"
            )
            parts.append(tb)
        if len(all_failures) > 20:
            parts.append(f"  ... and {len(all_failures) - 20} more")
        raise AssertionError("\n".join(parts))
