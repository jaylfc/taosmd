"""Thread-affinity regression: stores opened on one thread must be usable from
another in the Python-API-then-serve deployment shape.

The failing shape: create the store connection on the main thread, then issue
a write and a read from a spawned ``threading.Thread``.  ``sqlite3`` raises
``ProgrammingError`` when a connection created with the default
``check_same_thread=True`` is touched from any thread other than the one that
opened it.
"""
from __future__ import annotations

import sqlite3
import threading

import pytest

from taosmd.archive import ArchiveStore
from taosmd.knowledge_graph import TemporalKnowledgeGraph


def _use_conn_from_other_thread(conn, exc_holder):
    try:
        conn.execute("SELECT 1")
        exc_holder["error"] = None
    except Exception as exc:  # noqa: BLE001
        exc_holder["error"] = exc


def _use_store_from_other_thread(store, exc_holder):
    try:
        store._conn.execute("SELECT 1")
        exc_holder["error"] = None
    except Exception as exc:  # noqa: BLE001
        exc_holder["error"] = exc


@pytest.mark.asyncio
async def test_archive_store_conn_usable_from_worker_thread(tmp_path):
    index = str(tmp_path / "idx.db")
    arc = ArchiveStore(archive_dir=str(tmp_path / "a"), index_path=index)
    await arc.init()
    try:
        exc_holder: dict = {}
        worker = threading.Thread(
            target=_use_store_from_other_thread,
            args=(arc, exc_holder),
        )
        worker.start()
        worker.join()
        assert exc_holder.get("error") is None, (
            f"ArchiveStore connection raised from worker thread: "
            f"{exc_holder['error']!r}"
        )
    finally:
        await arc.close()


@pytest.mark.asyncio
async def test_knowledge_graph_conn_usable_from_worker_thread(tmp_path):
    db = str(tmp_path / "kg.db")
    kg = TemporalKnowledgeGraph(db_path=db)
    await kg.init()
    try:
        exc_holder: dict = {}
        worker = threading.Thread(
            target=_use_store_from_other_thread,
            args=(kg, exc_holder),
        )
        worker.start()
        worker.join()
        assert exc_holder.get("error") is None, (
            f"TemporalKnowledgeGraph connection raised from worker thread: "
            f"{exc_holder['error']!r}"
        )
    finally:
        await kg.close()
