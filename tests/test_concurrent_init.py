"""Reproduction for tsk-wwhnhv: concurrent first-time store init races.

Spawns real OS processes that all init the same fresh store and then do
writes, then asserts the expected row count.  On the unfixed tree the
init-time DDL races, some writers die with OperationalError, and rows
are lost silently.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import sqlite3
import sys
import time

import pytest

from taosmd.mentions import MentionStore


WORKER_COUNT = 4
INSERTS_PER_WORKER = 200
EXPECTED_TOTAL = WORKER_COUNT * INSERTS_PER_WORKER


def _worker_init_and_insert(db_path: str, worker_id: int, inserts: int) -> None:
    async def _run() -> None:
        store = MentionStore(db_path)
        try:
            await store.init()
            for i in range(inserts):
                await store.record_mentions(
                    message_id=worker_id * 10_000 + i,
                    body="",
                    thread="t",
                    ts=float(i),
                    recipient=f"user{worker_id}",
                )
        except sqlite3.OperationalError as exc:
            print(f"worker {worker_id}: OperationalError: {exc}", flush=True)
            raise
        finally:
            await store.close()

    try:
        asyncio.run(_run())
    except BaseException as exc:
        print(f"worker {worker_id}: {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)


def test_concurrent_first_init_no_row_loss(tmp_path):
    """Four processes racing init on a fresh DB must not lose rows.

    Uses fork so the race window is realistic (CLI process racing a running
    server).  On the unfixed tree some writers die with OperationalError,
    losing 200-400 rows per failure.
    """
    db_path = str(tmp_path / "mentions.db")

    ctx = multiprocessing.get_context("fork")
    procs = [
        ctx.Process(target=_worker_init_and_insert, args=(db_path, i, INSERTS_PER_WORKER))
        for i in range(WORKER_COUNT)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)

    failed = [p for p in procs if p.exitcode != 0]
    assert not failed, f"{len(failed)} workers failed: {[(p.name, p.exitcode) for p in failed]}"

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM mentions").fetchone()[0]
    conn.close()

    assert count == EXPECTED_TOTAL, (
        f"row loss: expected {EXPECTED_TOTAL}, got {count}"
    )
