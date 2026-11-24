"""Stable record ids: record() allocates a seq, stores it in the raw line and
uses it as the index id; rebuild() restores the same ids from the raw files.

These tests cover the write path (one transaction per record, never wedged),
older-code interop, the automatic legacy id map, an atomic rebuild, and
cross-process allocation.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import multiprocessing
import os
import sqlite3
from pathlib import Path

import pytest

from taosmd import archive as archive_mod
from taosmd.archive import ArchiveStore


def _store(tmp_path: Path) -> ArchiveStore:
    return ArchiveStore(
        archive_dir=str(tmp_path / "archive"),
        index_path=str(tmp_path / "archive-index.db"),
    )


def _raw_hashes(archive_dir: Path) -> dict[str, str]:
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(archive_dir.rglob("*"))
        if p.is_file() and p.name.endswith((".jsonl", ".jsonl.gz"))
    }


def _raw_seqs(archive_dir: Path) -> list[int]:
    seqs = []
    for p in sorted(archive_dir.rglob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seqs.append(json.loads(line)["seq"])
    return seqs


def _index_ids(conn: sqlite3.Connection) -> list[int]:
    return [r[0] for r in conn.execute("SELECT id FROM archive_index ORDER BY id")]


def _legacy_line(i: int) -> str:
    """A raw line as written before seq existed (no seq key, with sha256)."""
    event = {
        "timestamp": 1000.0 + i,
        "event_type": "conversation",
        "agent_name": "a",
        "app_id": None,
        "project": None,
        "summary": f"legacy {i}",
        "data": {"text": f"legacy text {i}"},
    }
    event["sha256"] = hashlib.sha256(json.dumps(event, default=str).encode()).hexdigest()
    return json.dumps(event, default=str)


def _write_legacy_day(archive_dir: Path, rel: str, indexes: list[int]) -> Path:
    path = archive_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_legacy_line(i) + "\n" for i in indexes), encoding="utf-8")
    return path


def _insert_old_row(conn, row_id: int, path: Path, line_number: int) -> None:
    """What older code did: an index row with the next autoincrement id."""
    conn.execute(
        """INSERT INTO archive_index
           (id, timestamp, event_type, agent_name, app_id, project, summary, file_path, line_number, data_json)
           VALUES (?, ?, 'conversation', 'a', NULL, NULL, ?, ?, ?, '{}')""",
        (row_id, float(row_id), f"old {row_id}", str(path), line_number),
    )
    conn.commit()


def _add_reject_trigger(conn: sqlite3.Connection) -> None:
    """Make the index INSERT fail for summary 'boom' with a real IntegrityError."""
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS reject_boom BEFORE INSERT ON archive_index
           WHEN NEW.summary = 'boom'
           BEGIN SELECT RAISE(ABORT, 'rejected by test trigger'); END"""
    )
    conn.commit()


# ----------------------------------------------------------------------
# 1. Wedged writer
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_index_insert_does_not_wedge_writer(tmp_path):
    arc = _store(tmp_path)
    await arc.init()
    first = await arc.record("conversation", {"text": "ok"}, agent_name="a")
    _add_reject_trigger(arc._conn)

    with pytest.raises(sqlite3.IntegrityError):
        await arc.record("conversation", {"text": "x"}, agent_name="a", summary="boom")
    assert not arc._conn.in_transaction

    ids = [
        await arc.record("conversation", {"text": f"after {i}"}, agent_name="a")
        for i in range(5)
    ]
    assert len(set(ids)) == 5
    assert all(i > first for i in ids)
    assert not arc._conn.in_transaction
    # Every committed id is in the index, in order, and the failed one is not.
    assert _index_ids(arc._conn) == [first] + ids
    await arc.close()


@pytest.mark.asyncio
async def test_failed_insert_leaves_raw_line_and_rebuild_repairs_it(tmp_path):
    arc = _store(tmp_path)
    await arc.init()
    a = await arc.record("conversation", {"text": "a"}, agent_name="a")
    _add_reject_trigger(arc._conn)
    with pytest.raises(sqlite3.IntegrityError):
        await arc.record("conversation", {"text": "orphan"}, agent_name="a", summary="boom")
    b = await arc.record("conversation", {"text": "b"}, agent_name="a")

    # The raw line stays (append-only), carrying a seq the index does not have.
    seqs = _raw_seqs(tmp_path / "archive")
    assert len(seqs) == 3 and len(set(seqs)) == 3
    orphan_seq = (set(seqs) - {a, b}).pop()
    assert orphan_seq not in _index_ids(arc._conn)
    # ...and the id was not handed out again.
    assert b > orphan_seq

    before = _raw_hashes(tmp_path / "archive")
    arc._conn.execute("DROP TRIGGER reject_boom")
    arc._conn.commit()
    await arc.rebuild()
    assert _index_ids(arc._conn) == sorted([a, orphan_seq, b])
    assert _raw_hashes(tmp_path / "archive") == before
    await arc.close()


# ----------------------------------------------------------------------
# 2. Crash injection at each step of record()
# ----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["after_alloc", "after_raw_write", "before_commit"])
async def test_crash_at_each_step(tmp_path, monkeypatch, step):
    arc = _store(tmp_path)
    await arc.init()
    first = await arc.record("conversation", {"text": "one"}, agent_name="a")
    archive_dir = tmp_path / "archive"

    if step == "after_alloc":
        def boom(ts):
            raise RuntimeError("crash after allocation")
        monkeypatch.setattr(arc, "_ensure_file", boom)
    elif step == "after_raw_write":
        monkeypatch.setattr(
            archive_mod, "_redact_leaves",
            lambda v: (_ for _ in ()).throw(RuntimeError("crash after raw write")),
        )
    else:
        def boom(key, value):
            raise RuntimeError("crash before commit")
        monkeypatch.setattr(arc, "_raise_setting_to", boom)

    lines_before = len(_raw_seqs(archive_dir))
    with pytest.raises(RuntimeError, match="crash"):
        await arc.record("conversation", {"text": "doomed"}, agent_name="a")
    monkeypatch.undo()

    assert not arc._conn.in_transaction
    # Nothing was committed to the index for the failed record.
    assert _index_ids(arc._conn) == [first]
    raw_after_crash = len(_raw_seqs(archive_dir))
    assert raw_after_crash == lines_before + (0 if step == "after_alloc" else 1)

    hashes = _raw_hashes(archive_dir)
    # The next record works and never reuses a seq already on disk.
    nxt = await arc.record("conversation", {"text": "next"}, agent_name="a")
    seqs = _raw_seqs(archive_dir)
    assert len(set(seqs)) == len(seqs)
    assert nxt in seqs
    # Rebuild works, keeps every committed id, and indexes any orphan line.
    new_hashes = _raw_hashes(archive_dir)
    await arc.rebuild()
    ids = _index_ids(arc._conn)
    assert first in ids and nxt in ids
    assert sorted(ids) == sorted(seqs)
    assert _raw_hashes(archive_dir) == new_hashes
    # earlier raw files were only ever appended to
    for path, digest in hashes.items():
        assert path in new_hashes
    await arc.close()


@pytest.mark.asyncio
async def test_orphan_seq_not_reused_by_another_instance(tmp_path):
    a = _store(tmp_path)
    b = _store(tmp_path)
    await a.init()
    await b.init()
    _add_reject_trigger(a._conn)
    with pytest.raises(sqlite3.IntegrityError):
        await a.record("conversation", {"text": "x"}, summary="boom")
    await b.record("conversation", {"text": "y"})
    await a.record("conversation", {"text": "z"})
    seqs = _raw_seqs(tmp_path / "archive")
    assert len(seqs) == 3 and len(set(seqs)) == 3
    await a.close()
    await b.close()


@pytest.mark.asyncio
async def test_repeated_seq_on_disk_rebuild_keeps_the_indexed_line(tmp_path):
    """If an orphan's seq did get reused (burn failed, other process), rebuild
    must not raise and must not renumber the record the index already holds."""
    a = _store(tmp_path)
    b = _store(tmp_path)
    await a.init()
    await b.init()
    _add_reject_trigger(a._conn)
    a._burn_seq = lambda seq: None  # the orphan's seq is not protected
    with pytest.raises(sqlite3.IntegrityError):
        await a.record("conversation", {"text": "orphan"}, summary="boom")
    kept = await b.record("conversation", {"text": "kept"})  # reuses the seq
    seqs = _raw_seqs(tmp_path / "archive")
    assert seqs == [kept, kept]

    a._conn.execute("DROP TRIGGER reject_boom")
    a._conn.commit()
    report = await a.rebuild()
    assert report["indexed"] == 2
    row = a._conn.execute(
        "SELECT summary, line_number FROM archive_index WHERE id = ?", (kept,)
    ).fetchone()
    assert row["line_number"] == 2, "the indexed (second) line keeps the id"
    assert len(_index_ids(a._conn)) == 2
    await a.close()
    await b.close()


@pytest.mark.asyncio
async def test_open_sees_orphan_line_at_tail(tmp_path):
    arc = _store(tmp_path)
    await arc.init()
    await arc.record("conversation", {"text": "a"})
    await arc.close()
    day = next((tmp_path / "archive").rglob("*.jsonl"))
    with open(day, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"seq": 50, "timestamp": 1.0, "event_type": "conversation",
                             "summary": "", "data": {}}) + "\n")
    arc2 = _store(tmp_path)
    await arc2.init()
    assert await arc2.record("conversation", {"text": "b"}) == 51
    await arc2.close()


# ----------------------------------------------------------------------
# 3. Older code wrote to the same directory
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_allocation_is_above_max_index_id(tmp_path):
    arc = _store(tmp_path)
    await arc.init()
    for _ in range(3):
        await arc.record("conversation", {"text": "new"})
    # Older code adds ids 4..6 without touching the counter.
    legacy = _write_legacy_day(tmp_path / "archive", "2020/01/01.jsonl", [1, 2, 3])
    for n in range(3):
        _insert_old_row(arc._conn, 4 + n, legacy, n + 1)
    await arc.close()

    arc2 = _store(tmp_path)
    await arc2.init()
    ids = [await arc2.record("conversation", {"text": f"n{i}"}) for i in range(3)]
    assert ids == [7, 8, 9]
    await arc2.close()

    # Same, with a counter that is lower than the index max while open.
    arc3 = _store(tmp_path)
    await arc3.init()
    arc3._conn.execute("UPDATE archive_settings SET value = '2' WHERE key = 'archive_seq_counter'")
    arc3._conn.commit()
    arc3._seq_counter = 0
    assert await arc3.record("conversation", {"text": "x"}) == 10
    await arc3.close()


# ----------------------------------------------------------------------
# 4. Legacy id map is automatic
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rebuild_of_legacy_archive_keeps_ids_without_any_freeze(tmp_path):
    archive_dir = tmp_path / "archive"
    day1 = _write_legacy_day(archive_dir, "2020/01/01.jsonl", [1, 2, 3])
    day2 = _write_legacy_day(archive_dir, "2020/01/02.jsonl", [4, 5])
    # Older code, ids with gaps and not in file order.
    layout = [(day1, 1, 30), (day1, 2, 10), (day1, 3, 20), (day2, 1, 55), (day2, 2, 40)]
    arc = _store(tmp_path)
    await arc.init()
    for path, ln, rid in layout:
        _insert_old_row(arc._conn, rid, path, ln)
    await arc.close()
    before = _raw_hashes(archive_dir)

    # First open by the new code builds the map by itself.
    arc2 = _store(tmp_path)
    await arc2.init()
    assert (archive_dir / "legacy-id-map").exists()
    want = {(str(p), ln): rid for p, ln, rid in layout}

    def snapshot(conn):
        return {
            (r["file_path"], r["line_number"]): r["id"]
            for r in conn.execute("SELECT id, file_path, line_number FROM archive_index")
        }

    assert snapshot(arc2._conn) == want
    report = await arc2.rebuild()  # index present, nothing frozen by hand
    assert report == {"indexed": 5, "legacy": 5}
    assert snapshot(arc2._conn) == want
    await arc2.close()

    # And with the index database deleted entirely.
    os.remove(tmp_path / "archive-index.db")
    arc3 = _store(tmp_path)
    await arc3.init()
    await arc3.rebuild()
    assert snapshot(arc3._conn) == want
    # new records land above everything
    assert await arc3.record("conversation", {"text": "n"}) == 56
    await arc3.close()
    after = _raw_hashes(archive_dir)
    assert {k: after[k] for k in before} == before


@pytest.mark.asyncio
async def test_map_is_idempotent_and_append_only(tmp_path):
    archive_dir = tmp_path / "archive"
    day = _write_legacy_day(archive_dir, "2020/01/01.jsonl", [1, 2])
    arc = _store(tmp_path)
    await arc.init()
    _insert_old_row(arc._conn, 1, day, 1)
    _insert_old_row(arc._conn, 2, day, 2)
    await arc.close()
    sizes = []
    for _ in range(3):
        a = _store(tmp_path)
        await a.init()
        await a.rebuild()
        await a.close()
        sizes.append((archive_dir / "legacy-id-map").read_bytes())
    assert sizes[0] == sizes[1] == sizes[2]
    assert len(sizes[0].splitlines()) == 2


@pytest.mark.asyncio
async def test_unmapped_legacy_line_sorting_first_gets_fresh_id(tmp_path):
    archive_dir = tmp_path / "archive"
    late = _write_legacy_day(archive_dir, "2020/02/01.jsonl", [1, 2, 3])
    arc = _store(tmp_path)
    await arc.init()
    for n in range(3):
        _insert_old_row(arc._conn, n + 1, late, n + 1)
    await arc.close()
    # An extra legacy file the index never saw, sorting BEFORE the mapped ones.
    early = _write_legacy_day(archive_dir, "2020/01/01.jsonl", [7, 8])
    arc2 = _store(tmp_path)
    await arc2.init()
    report = await arc2.rebuild()
    assert report["indexed"] == 5
    rows = {
        (r["file_path"], r["line_number"]): r["id"]
        for r in arc2._conn.execute("SELECT id, file_path, line_number FROM archive_index")
    }
    assert [rows[(str(late), n)] for n in (1, 2, 3)] == [1, 2, 3]
    assert sorted(rows[(str(early), n)] for n in (1, 2)) == [4, 5]
    await arc2.close()


# ----------------------------------------------------------------------
# 5. Rebuild: lock and atomicity
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crash_mid_rebuild_leaves_old_index_intact(tmp_path, monkeypatch):
    arc = _store(tmp_path)
    await arc.init()
    ids = [await arc.record("conversation", {"text": f"r{i}"}) for i in range(6)]
    rows_before = [tuple(r) for r in arc._conn.execute("SELECT * FROM archive_index ORDER BY id")]
    calls = {"n": 0}
    real = archive_mod._redact_leaves

    def flaky(value):
        calls["n"] += 1
        if calls["n"] == 4:
            raise RuntimeError("crash inside rebuild")
        return real(value)

    monkeypatch.setattr(archive_mod, "_redact_leaves", flaky)
    with pytest.raises(RuntimeError, match="crash inside rebuild"):
        await arc.rebuild()
    monkeypatch.undo()
    assert not arc._conn.in_transaction
    rows_after = [tuple(r) for r in arc._conn.execute("SELECT * FROM archive_index ORDER BY id")]
    assert rows_after == rows_before
    assert arc._conn.execute("SELECT COUNT(*) FROM archive_fts").fetchone()[0] == 6
    # And both record() and a fresh rebuild work afterwards.
    assert await arc.record("conversation", {"text": "more"}) == ids[-1] + 1
    assert (await arc.rebuild())["indexed"] == 7
    await arc.close()


@pytest.mark.asyncio
async def test_rebuild_waits_for_the_write_lock(tmp_path):
    arc = _store(tmp_path)
    await arc.init()
    await arc.record("conversation", {"text": "a"})
    async with arc._write_lock:
        task = asyncio.ensure_future(arc.rebuild())
        await asyncio.sleep(0.2)
        assert not task.done(), "rebuild must hold the same lock as record()"
    await asyncio.wait_for(task, 30)
    await arc.close()


# ----------------------------------------------------------------------
# 6. Cross-process allocation (fails without BEGIN IMMEDIATE)
# ----------------------------------------------------------------------


def _mp_worker(db_dir: str, barrier, label: str, n: int, out_path: str) -> None:
    async def main() -> None:
        store = ArchiveStore(
            archive_dir=os.path.join(db_dir, "archive"),
            index_path=os.path.join(db_dir, "archive-index.db"),
        )
        await store.init()
        barrier.wait(60)
        ids = [
            await store.record("conversation", {"text": f"{label}{i}"}, agent_name=label)
            for i in range(n)
        ]
        await store.close()
        with open(out_path, "w") as fh:
            json.dump(ids, fh)

    asyncio.run(main())


@pytest.mark.parametrize("round_no", [0, 1, 2])
def test_two_processes_behind_a_barrier_never_collide(tmp_path, round_no):
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    outs = [str(tmp_path / f"out{i}.json") for i in range(2)]
    procs = [
        ctx.Process(target=_mp_worker, args=(str(tmp_path), barrier, f"p{i}", 300, outs[i]))
        for i in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
    assert [p.exitcode for p in procs] == [0, 0]

    all_ids = []
    for o in outs:
        with open(o) as fh:
            all_ids += json.load(fh)
    assert len(all_ids) == 600 and len(set(all_ids)) == 600
    conn = sqlite3.connect(tmp_path / "archive-index.db")
    try:
        assert sorted(all_ids) == _index_ids(conn)
    finally:
        conn.close()
    assert sorted(_raw_seqs(tmp_path / "archive")) == sorted(all_ids)


# ----------------------------------------------------------------------
# 7. Open cost and read-only guarantees
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reopen_does_not_read_raw_files(tmp_path, monkeypatch):
    arc = _store(tmp_path)
    await arc.init()
    for i in range(5):
        await arc.record("conversation", {"text": f"r{i}"})
    await arc.close()

    reads = []
    real = ArchiveStore._read_archive_lines

    def counting(self, path):
        reads.append(str(path))
        return real(self, path)

    monkeypatch.setattr(ArchiveStore, "_read_archive_lines", counting)
    monkeypatch.setattr(
        ArchiveStore, "_max_seq_in_files",
        lambda self: (_ for _ in ()).throw(AssertionError("full scan on open")),
    )
    for _ in range(3):
        a = _store(tmp_path)
        await a.init()
        await a.close()
    assert reads == []
    # lines that carry a seq never need a map entry
    assert not (tmp_path / "archive" / "legacy-id-map").exists()


@pytest.mark.asyncio
async def test_legacy_dir_is_scanned_once_then_never(tmp_path, monkeypatch):
    archive_dir = tmp_path / "archive"
    day1 = _write_legacy_day(archive_dir, "2020/01/01.jsonl", [1, 2, 3])
    day2 = _write_legacy_day(archive_dir, "2020/01/02.jsonl", [4, 5, 6])
    arc = _store(tmp_path)
    await arc.init()
    n = 0
    for day in (day1, day2):
        for ln in (1, 2, 3):
            n += 1
            _insert_old_row(arc._conn, n, day, ln)
    await arc.close()

    reads = []
    real = ArchiveStore._read_archive_lines

    def counting(self, path):
        reads.append(str(path))
        return real(self, path)

    monkeypatch.setattr(ArchiveStore, "_read_archive_lines", counting)
    a = _store(tmp_path)
    await a.init()
    await a.close()
    # each raw file is read once for the map (not once per index row)
    assert reads == [str(day1), str(day2)]
    for _ in range(2):
        b = _store(tmp_path)
        await b.init()
        await b.close()
    assert reads == [str(day1), str(day2)]


@pytest.mark.asyncio
async def test_deleted_index_scans_once_and_persists_counter(tmp_path, monkeypatch):
    arc = _store(tmp_path)
    await arc.init()
    for i in range(4):
        await arc.record("conversation", {"text": f"r{i}"})
    await arc.close()
    os.remove(tmp_path / "archive-index.db")

    calls = []
    real = ArchiveStore._max_seq_in_files
    monkeypatch.setattr(
        ArchiveStore, "_max_seq_in_files", lambda self: calls.append(1) or real(self)
    )
    a = _store(tmp_path)
    await a.init()
    assert a._seq_counter == 4
    await a.rebuild()
    assert not (tmp_path / "archive" / "legacy-id-map").exists()
    assert await a.record("conversation", {"text": "x"}) == 5
    await a.close()
    b = _store(tmp_path)
    await b.init()
    await b.close()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_raw_files_byte_identical_through_every_operation(tmp_path):
    archive_dir = tmp_path / "archive"
    legacy = _write_legacy_day(archive_dir, "2020/01/01.jsonl", [1, 2])
    gz = archive_dir / "2020/01/02.jsonl.gz"
    with gzip.open(gz, "wb") as fh:
        fh.write((_legacy_line(3) + "\n").encode())
    arc = _store(tmp_path)
    await arc.init()
    _insert_old_row(arc._conn, 1, legacy, 1)
    _insert_old_row(arc._conn, 2, legacy, 2)
    before = _raw_hashes(archive_dir)
    await arc.close()

    a = _store(tmp_path)
    await a.init()
    assert _raw_hashes(archive_dir) == before
    await a.rebuild()
    assert _raw_hashes(archive_dir) == before
    await a.freeze_legacy_id_map()
    assert _raw_hashes(archive_dir) == before
    await a.record("conversation", {"text": "new"})
    after = _raw_hashes(archive_dir)
    for path, digest in before.items():
        assert after[path] == digest
    await a.rebuild()
    assert {k: v for k, v in _raw_hashes(archive_dir).items() if k in before} == before
    await a.close()


@pytest.mark.asyncio
async def test_same_instance_never_reuses_orphan_seq_even_if_burn_fails(tmp_path):
    arc = _store(tmp_path)
    await arc.init()
    _add_reject_trigger(arc._conn)
    arc._burn_seq = lambda seq: None
    with pytest.raises(sqlite3.IntegrityError):
        await arc.record("conversation", {"text": "x"}, summary="boom")
    await arc.record("conversation", {"text": "y"})
    seqs = _raw_seqs(tmp_path / "archive")
    assert len(seqs) == 2 and len(set(seqs)) == 2
    await arc.close()


@pytest.mark.asyncio
async def test_pending_transaction_does_not_block_record(tmp_path):
    arc = _store(tmp_path)
    await arc.init()
    # Something left an implicit transaction open on the connection.
    arc._conn.execute(
        "INSERT INTO archive_settings (key, value) VALUES ('left_open', '1')"
    )
    assert arc._conn.in_transaction
    assert await arc.record("conversation", {"text": "x"}) == 1
    assert not arc._conn.in_transaction
    await arc.close()


@pytest.mark.asyncio
async def test_counter_settings_never_move_backwards(tmp_path):
    arc = _store(tmp_path)
    await arc.init()
    arc._raise_setting_to("archive_seq_counter", 9)
    arc._raise_setting_to("archive_seq_counter", 3)
    arc._conn.commit()
    assert arc._read_int_setting("archive_seq_counter") == 9
    await arc.close()


@pytest.mark.asyncio
async def test_rebuild_maps_rows_added_after_open(tmp_path):
    """Older code wrote rows while this store was already open: rebuild must
    still map them before it throws the index away."""
    archive_dir = tmp_path / "archive"
    arc = _store(tmp_path)
    await arc.init()
    day = _write_legacy_day(archive_dir, "2020/01/01.jsonl", [1, 2, 3])
    for ln, rid in ((1, 7), (2, 3), (3, 5)):
        _insert_old_row(arc._conn, rid, day, ln)
    await arc.rebuild()
    got = {
        r["line_number"]: r["id"]
        for r in arc._conn.execute("SELECT id, line_number FROM archive_index")
    }
    assert got == {1: 7, 2: 3, 3: 5}
    await arc.close()


@pytest.mark.asyncio
async def test_legacy_ids_survive_the_nightly_compress(tmp_path):
    """compress_old_files() turns DD.jsonl into DD.jsonl.gz but leaves the
    index pointing at the .jsonl name; the map must still match."""
    archive_dir = tmp_path / "archive"
    day = _write_legacy_day(archive_dir, "2020/01/01.jsonl", [1, 2, 3])
    arc = _store(tmp_path)
    await arc.init()
    for ln, rid in ((1, 9), (2, 4), (3, 6)):
        _insert_old_row(arc._conn, rid, day, ln)
    await arc.close()

    arc2 = _store(tmp_path)
    await arc2.init()  # builds the map against the .jsonl name
    await arc2.close()
    with open(day, "rb") as src, gzip.open(str(day) + ".gz", "wb") as dst:
        dst.write(src.read())
    day.unlink()

    os.remove(tmp_path / "archive-index.db")
    arc3 = _store(tmp_path)
    await arc3.init()
    await arc3.rebuild()
    got = {
        r["line_number"]: r["id"]
        for r in arc3._conn.execute("SELECT id, line_number FROM archive_index")
    }
    assert got == {1: 9, 2: 4, 3: 6}
    await arc3.close()


@pytest.mark.asyncio
async def test_open_survives_an_unwritable_map(tmp_path, monkeypatch):
    archive_dir = tmp_path / "archive"
    day = _write_legacy_day(archive_dir, "2020/01/01.jsonl", [1])
    arc = _store(tmp_path)
    await arc.init()
    _insert_old_row(arc._conn, 1, day, 1)
    await arc.close()

    def refuse(self, only_above_watermark):
        raise PermissionError("read-only directory")

    monkeypatch.setattr(ArchiveStore, "_extend_legacy_map", refuse)
    a = _store(tmp_path)
    await a.init()  # must not raise
    await a.close()


@pytest.mark.asyncio
async def test_rebuild_tolerates_odd_lines(tmp_path):
    """A seq of 0, JSON that is not an object, and a non-object data field must
    not stop a rebuild or produce an id of 0."""
    archive_dir = tmp_path / "archive"
    day = archive_dir / "2020" / "01" / "01.jsonl"
    day.parent.mkdir(parents=True)
    lines = [
        json.dumps({"seq": 5, "timestamp": 1.0, "event_type": "conversation", "summary": "ok", "data": {}}),
        json.dumps({"seq": 0, "timestamp": 2.0, "event_type": "conversation", "summary": "zero", "data": {}}),
        "123",
        "[1, 2]",
        json.dumps({"seq": 7, "timestamp": 3.0, "event_type": "conversation", "summary": "list", "data": [1, 2]}),
        "not json at all",
    ]
    day.write_text("\n".join(lines) + "\n", encoding="utf-8")
    arc = _store(tmp_path)
    await arc.init()
    report = await arc.rebuild()
    assert report["indexed"] == 3
    ids = _index_ids(arc._conn)
    assert 0 not in ids and 5 in ids and 7 in ids and len(ids) == 3
    await arc.close()


@pytest.mark.asyncio
async def test_orphan_seq_does_not_take_a_legacy_rows_id(tmp_path):
    """Failed write leaves a seq-N orphan, then older code indexes a legacy row
    as N. Rebuild must leave the indexed legacy row on N."""
    archive_dir = tmp_path / "archive"
    arc = _store(tmp_path)
    await arc.init()
    a = await arc.record("conversation", {"text": "first"})
    _add_reject_trigger(arc._conn)
    arc._burn_seq = lambda seq: None
    with pytest.raises(sqlite3.IntegrityError):
        await arc.record("conversation", {"text": "orphan"}, summary="boom")
    orphan = a + 1
    arc._conn.execute("DROP TRIGGER reject_boom")
    arc._conn.commit()
    legacy_day = _write_legacy_day(archive_dir, "2019/01/01.jsonl", [1])
    _insert_old_row(arc._conn, orphan, legacy_day, 1)  # older code reused the id
    await arc.close()

    arc2 = _store(tmp_path)
    await arc2.init()  # maps the legacy row at id `orphan`
    await arc2.rebuild()
    row = arc2._conn.execute(
        "SELECT file_path FROM archive_index WHERE id = ?", (orphan,)
    ).fetchone()
    assert row["file_path"] == str(legacy_day)
    assert len(_index_ids(arc2._conn)) == 3
    await arc2.close()


@pytest.mark.asyncio
async def test_day_with_both_jsonl_and_gz_is_indexed_once(tmp_path):
    arc = _store(tmp_path)
    await arc.init()
    for i in range(3):
        await arc.record("conversation", {"text": f"r{i}"})
    day = next((tmp_path / "archive").rglob("*.jsonl"))
    with open(day, "rb") as src, gzip.open(str(day) + ".gz", "wb") as dst:
        dst.write(src.read())  # compress interrupted before the source was removed
    before = _raw_hashes(tmp_path / "archive")
    report = await arc.rebuild()
    assert report["indexed"] == 3
    assert _index_ids(arc._conn) == [1, 2, 3]
    assert _raw_hashes(tmp_path / "archive") == before
    await arc.close()
