import asyncio
import json
import multiprocessing
import os
import sqlite3
from pathlib import Path

import pytest

from taosmd.archive import ArchiveStore


@pytest.mark.asyncio
async def test_two_instances_alternating_distinct_seq(tmp_path):
    archive_dir = tmp_path / "archive"
    index_path = tmp_path / "archive-index.db"

    a = ArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
    b = ArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
    await a.init()
    await b.init()

    ids_a: list[int] = []
    ids_b: list[int] = []
    all_ids: list[int] = []
    for i in range(50):
        if i % 2 == 0:
            rid = await a.record("conversation", {"text": f"a{i}"}, agent_name="a")
            ids_a.append(rid)
        else:
            rid = await b.record("conversation", {"text": f"b{i}"}, agent_name="b")
            ids_b.append(rid)
        all_ids.append(rid)

    assert len(all_ids) == 50
    assert len(set(all_ids)) == 50, "ids must be distinct across instances"
    assert ids_a == sorted(ids_a), "instance a ids must be strictly increasing"
    assert ids_b == sorted(ids_b), "instance b ids must be strictly increasing"
    assert len(ids_a) == len(set(ids_a))
    assert len(ids_b) == len(set(ids_b))

    # every raw line's seq must equal some index id (seq is the id)
    raw_seqs = []
    for path in sorted(archive_dir.rglob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    raw_seqs.append(json.loads(line)["seq"])
    index_ids = [r["id"] for r in a._conn.execute("SELECT id FROM archive_index")]
    assert sorted(raw_seqs) == sorted(index_ids)
    assert len(raw_seqs) == len(set(raw_seqs)) == 50

    await a.close()
    await b.close()


def _mp_worker(db_dir: str, q, label: str, n: int) -> None:
    async def main() -> None:
        store = ArchiveStore(
            archive_dir=os.path.join(db_dir, "archive"),
            index_path=os.path.join(db_dir, "archive-index.db"),
        )
        await store.init()
        ids = []
        for i in range(n):
            rid = await store.record("conversation", {"text": f"{label}{i}"}, agent_name=label)
            ids.append(rid)
        await store.close()
        q.put(ids)

    asyncio.run(main())


def test_two_processes_distinct_seq(tmp_path):
    db_dir = str(tmp_path)
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_mp_worker, args=(db_dir, q, "p0", 20)),
        ctx.Process(target=_mp_worker, args=(db_dir, q, "p1", 20)),
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker failed with {p.exitcode}"

    ids0 = q.get(timeout=10)
    ids1 = q.get(timeout=10)
    all_ids = ids0 + ids1
    assert len(all_ids) == 40
    assert len(set(all_ids)) == 40, "ids must be distinct across processes"
    assert ids0 == sorted(set(ids0)) and ids0 == sorted(ids0)
    assert ids1 == sorted(set(ids1)) and ids1 == sorted(ids1)

    # every raw line's seq must equal some index id
    conn = sqlite3.connect(os.path.join(db_dir, "archive-index.db"))
    try:
        index_ids = [r[0] for r in conn.execute("SELECT id FROM archive_index")]
    finally:
        conn.close()

    raw_seqs = []
    archive_dir = Path(db_dir) / "archive"
    for path in sorted(archive_dir.rglob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    raw_seqs.append(json.loads(line)["seq"])
    assert sorted(raw_seqs) == sorted(index_ids)
    assert len(raw_seqs) == 40
