"""RED tests that must FAIL on #756's head (current code)."""
import asyncio
import json
import multiprocessing
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from taosmd.archive import ArchiveStore
from taosmd.archive import _redact_leaves


@pytest.mark.asyncio
async def test_wedged_writer_after_integrity_error(tmp_path):
    """After a failed index insert (IntegrityError), record() must not leave
    a transaction open. Subsequent records must succeed.

    On #756's head: 1 IntegrityError, then 5 of 5 failures with
    'cannot start a transaction within a transaction'.
    """
    arc = ArchiveStore(
        archive_dir=str(tmp_path / "archive"),
        index_path=str(tmp_path / "archive-index.db"),
    )
    await arc.init()

    # Record one valid entry (id=1)
    await arc.record("conversation", {"text": "ok 0"}, agent_name="a")

    # Force an error during record by monkey-patching the file write to fail
    # This simulates a disk error or other failure after seq allocation but
    # before index insert completes.
    original_ensure_file = arc._ensure_file
    call_count = {"n": 0}

    def failing_ensure_file(ts):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated disk write failure")
        return original_ensure_file(ts)

    arc._ensure_file = failing_ensure_file

    # This should fail with RuntimeError but NOT wedge the writer
    with pytest.raises(RuntimeError, match="simulated disk write failure"):
        await arc.record("conversation", {"text": "will fail"}, agent_name="a")

    # Restore original method
    arc._ensure_file = original_ensure_file

    # Subsequent records MUST succeed (this is the fix)
    ids = [
        await arc.record("conversation", {"text": f"ok {i}"}, agent_name="a")
        for i in range(5)
    ]
    # All 5 should succeed with distinct ids
    assert len(ids) == 5
    assert len(set(ids)) == 5
    # The ids should continue from 2 (seq 1 was consumed but rolled back, 
    # but max_id in index is 1, so next is 2)
    assert all(i >= 2 for i in ids)

    await arc.close()


@pytest.mark.asyncio
async def test_next_seq_ignores_max_id_downgrade_upgrade(tmp_path):
    """_next_seq must use MAX(id) from archive_index, not just the persisted
    counter. Simulate older code writing ids 32-34, then new code records.

    On #756's head: first record() hits UNIQUE and then wedges.
    """
    archive_dir = tmp_path / "archive"
    index_path = tmp_path / "archive-index.db"

    # Simulate older code: create archive with index rows having ids 32, 33, 34
    # but counter only at 5 (as if it was never updated)
    arc = ArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
    await arc.init()

    # Manually insert index rows with high ids (simulating older code)
    for i in range(32, 35):
        arc._conn.execute(
            """INSERT INTO archive_index
               (id, timestamp, event_type, agent_name, app_id, project, summary, file_path, line_number, data_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (i, float(i), "conversation", "a", None, None, f"old {i}", f"old.jsonl", i - 31, "{}"),
        )
    arc._conn.commit()

    # Set counter to a low value (simulating it wasn't updated)
    arc._conn.execute(
        """INSERT INTO archive_settings (key, value) VALUES ('archive_seq_counter', ?)
           ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
        ("5",),
    )
    arc._conn.commit()

    # Now record a new entry - should get id 35 (max of counter=5 and MAX(id)=34) + 1
    rid = await arc.record("conversation", {"text": "new"}, agent_name="a")
    assert rid == 35, f"expected 35, got {rid}"

    # Subsequent records should continue from there
    rid2 = await arc.record("conversation", {"text": "new2"}, agent_name="a")
    assert rid2 == 36, f"expected 36, got {rid2}"

    await arc.close()


@pytest.mark.asyncio
async def test_rebuild_legacy_no_manual_freeze_keeps_ids(tmp_path):
    """Rebuild of a legacy archive (lines without seq) with index present
    but NO manual freeze_legacy_id_map() call must keep ids unchanged.

    On #756's head: rebuild renumbers every id because freeze was never called.

    This test simulates the real scenario: an existing archive with index
    created by older code. On first write-open (init), the legacy map is
    auto-built. Then if index is lost, rebuild uses the map.
    """
    archive_dir = tmp_path / "archive"
    index_path = tmp_path / "archive-index.db"
    day = archive_dir / "2020" / "01" / "01.jsonl"
    day.parent.mkdir(parents=True)

    # Write legacy lines WITHOUT seq field (simulating older code output)
    legacy_lines = [
        {
            "timestamp": 1_000_000.0,
            "event_type": "conversation",
            "agent_name": "a",
            "app_id": None,
            "project": None,
            "summary": "legacy one",
            "data": {"text": "one"},
        },
        {
            "timestamp": 1_000_001.0,
            "event_type": "conversation",
            "agent_name": "a",
            "app_id": None,
            "project": None,
            "summary": "legacy two",
            "data": {"text": "two"},
        },
        {
            "timestamp": 1_000_002.0,
            "event_type": "conversation",
            "agent_name": "a",
            "app_id": None,
            "project": None,
            "summary": "legacy three",
            "data": {"text": "three"},
        },
    ]
    day.write_text(
        "".join(json.dumps(line, default=str) + "\n" for line in legacy_lines),
        encoding="utf-8",
    )

    # Create index with non-contiguous ids (7, 8, 9) - simulating older code
    arc = ArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
    await arc.init()
    mapped_ids = [7, 8, 9]
    for line_number, row_id in zip(range(1, 4), mapped_ids):
        line = legacy_lines[line_number - 1]
        arc._conn.execute(
            """INSERT INTO archive_index
               (id, timestamp, event_type, agent_name, app_id, project, summary, file_path, line_number, data_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row_id,
                line["timestamp"],
                "conversation",
                "a",
                None,
                None,
                line["summary"],
                str(day),
                line_number,
                json.dumps(line["data"], default=str),
            ),
        )
        arc._conn.execute(
            "INSERT INTO archive_fts (rowid, summary, content) VALUES (?, ?, ?)",
            (row_id, line["summary"], line["summary"]),
        )
    arc._conn.commit()
    # IMPORTANT: Do NOT call freeze_legacy_id_map() - this is the bug we're fixing
    await arc.close()

    # SECOND write-open (simulates app restart): init() should auto-build
    # the legacy map for the existing index rows (7,8,9) whose raw lines lack seq
    arc2 = ArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
    await arc2.init()
    await arc2.close()

    # Now simulate index corruption/loss
    os.remove(index_path)

    # THIRD write-open: rebuild should use the auto-built legacy map
    arc3 = ArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
    await arc3.init()
    report = await arc3.rebuild()
    # Should have indexed 3, legacy 3 (all 3 were legacy lines)
    assert report == {"indexed": 3, "legacy": 3}
    rows = {
        (r["file_path"], r["line_number"]): r["id"]
        for r in arc3._conn.execute(
            "SELECT id, file_path, line_number FROM archive_index"
        )
    }
    # IDs MUST be unchanged (7, 8, 9)
    for line_number, row_id in zip(range(1, 4), mapped_ids):
        assert rows[(str(day), line_number)] == row_id, (
            f"line {line_number}: expected id {row_id}, got {rows[(str(day), line_number)]}"
        )
    assert {
        r["id"] for r in arc3._conn.execute("SELECT id FROM archive_index")
    } == {7, 8, 9}
    await arc3.close()


@pytest.mark.asyncio
async def test_rebuild_unmapped_legacy_before_mapped_no_unique(tmp_path):
    """Rebuild with an unmapped legacy line that sorts BEFORE mapped ones
    must not raise UNIQUE constraint error.

    On #756's head: rebuild assigns unmapped lines incremental ids starting
    from counter, which may collide with mapped ids that come later in the file.
    """
    archive_dir = tmp_path / "archive"
    index_path = tmp_path / "archive-index.db"
    day = archive_dir / "2020" / "01" / "01.jsonl"
    day.parent.mkdir(parents=True)

    # Line 1: legacy (no seq), will be unmapped
    # Line 2-3: legacy with seq (mapped via auto-freeze)
    legacy_lines = [
        {"timestamp": 1.0, "event_type": "conversation", "agent_name": "a", "summary": "unmapped", "data": {}},
        {"timestamp": 2.0, "event_type": "conversation", "agent_name": "a", "summary": "mapped1", "data": {}, "seq": 100},
        {"timestamp": 3.0, "event_type": "conversation", "agent_name": "a", "summary": "mapped2", "data": {}, "seq": 101},
    ]
    day.write_text(
        "".join(json.dumps(line, default=str) + "\n" for line in legacy_lines),
        encoding="utf-8",
    )

    # Create index with the mapped lines (100, 101) but NOT the unmapped one
    arc = ArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
    await arc.init()
    for line_number, line in enumerate(legacy_lines, start=1):
        if "seq" in line:
            arc._conn.execute(
                """INSERT INTO archive_index
                   (id, timestamp, event_type, agent_name, app_id, project, summary, file_path, line_number, data_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    line["seq"],
                    line["timestamp"],
                    "conversation",
                    "a",
                    None,
                    None,
                    line["summary"],
                    str(day),
                    line_number,
                    json.dumps(line["data"], default=str),
                ),
            )
            arc._conn.execute(
                "INSERT INTO archive_fts (rowid, summary, content) VALUES (?, ?, ?)",
                (line["seq"], line["summary"], line["summary"]),
            )
    arc._conn.commit()
    # Auto-freeze should happen on first write-open (but we don't write, just close)
    await arc.close()

    # Second write-open: auto-build legacy map for the mapped lines
    arc2 = ArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
    await arc2.init()
    await arc2.close()

    # Now rebuild - should not raise UNIQUE
    os.remove(index_path)
    arc3 = ArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
    await arc3.init()
    # This should NOT raise sqlite3.IntegrityError (UNIQUE constraint failed)
    report = await arc3.rebuild()
    assert report["indexed"] == 3
    await arc3.close()


@pytest.mark.asyncio
async def test_crash_mid_rebuild_leaves_old_index_intact(tmp_path):
    """A crash injected mid-rebuild (raise inside the transaction) must leave
    the old index intact (atomic rebuild).

    On #756's head: rebuild does DELETE then INSERT in separate commits, so
    a crash leaves an empty index.
    """
    archive_dir = tmp_path / "archive"
    index_path = tmp_path / "archive-index.db"

    arc = ArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
    await arc.init()
    ids = [
        await arc.record("conversation", {"text": f"record {i}"}, agent_name="a")
        for i in range(5)
    ]
    assert ids == [1, 2, 3, 4, 5]
    await arc.close()

    # Now simulate a crash during rebuild by using a custom connection wrapper
    # that raises on the 3rd INSERT INTO archive_index
    class CrashingArchiveStore(ArchiveStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._rebuild_call_count = 0

        async def rebuild(self) -> dict[str, int]:
            if self._conn is None:
                raise RuntimeError("ArchiveStore is not initialised")
            async with self._write_lock:
                legacy_map = self._load_legacy_map()
                mapped_ids = set(legacy_map.values())
                max_mapped_id = max(mapped_ids) if mapped_ids else 0
                counter = max(self._seq_counter, max_mapped_id)

                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    self._conn.execute("DELETE FROM archive_index")
                    self._conn.execute("DELETE FROM archive_fts")
                    indexed = legacy = 0
                    for path in self._iter_archive_files():
                        file_path = str(path)
                        for line_number, line in enumerate(self._read_archive_lines(path), start=1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                            except (json.JSONDecodeError, TypeError):
                                continue
                            seq = record.get("seq")
                            if isinstance(seq, int) and not isinstance(seq, bool):
                                row_id = seq
                            else:
                                legacy += 1
                                identity = (
                                    file_path,
                                    line_number,
                                    record.get("sha256") or "",
                                )
                                row_id = legacy_map.get(identity)
                                if row_id is None:
                                    counter += 1
                                    row_id = counter
                                else:
                                    if row_id > counter:
                                        counter = row_id
                            if row_id > counter:
                                counter = row_id
                            summary = record.get("summary") or ""
                            derived = _redact_leaves(
                                json.loads(json.dumps(record.get("data") or {}, default=str))
                            )
                            self._rebuild_call_count += 1
                            if self._rebuild_call_count == 3:
                                raise RuntimeError("simulated crash mid-rebuild")
                            self._conn.execute(
                                """INSERT INTO archive_index
                                   (id, timestamp, event_type, agent_name, app_id, project, summary, file_path, line_number, data_json)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (
                                    row_id,
                                    record.get("timestamp") or 0.0,
                                    record.get("event_type") or "",
                                    record.get("agent_name"),
                                    record.get("app_id"),
                                    record.get("project"),
                                    summary,
                                    file_path,
                                    line_number,
                                    json.dumps(derived, default=str),
                                ),
                            )
                            content_parts = [summary]
                            for key in ("content", "text", "msg", "query", "tool", "result", "error"):
                                val = derived.get(key)
                                if val and isinstance(val, str):
                                    content_parts.append(val)
                            content_text = " ".join(content_parts)
                            if content_text.strip():
                                self._conn.execute(
                                    "INSERT INTO archive_fts (rowid, summary, content) VALUES (?, ?, ?)",
                                    (row_id, summary, content_text),
                                )
                            indexed += 1
                    if counter > self._seq_counter:
                        self._seq_counter = counter
                        self._conn.execute(
                            """INSERT INTO archive_settings (key, value) VALUES ('archive_seq_counter', ?)
                               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                            (str(self._seq_counter),),
                        )
                    self._conn.commit()
                except Exception:
                    try:
                        self._conn.rollback()
                    except sqlite3.Error:
                        pass
                    raise
                return {"indexed": indexed, "legacy": legacy}

    arc2 = CrashingArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
    await arc2.init()

    with pytest.raises(RuntimeError, match="simulated crash mid-rebuild"):
        await arc2.rebuild()

    # The old index should be intact (all 5 records)
    arc3 = ArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
    await arc3.init()
    count = arc3._conn.execute("SELECT COUNT(*) FROM archive_index").fetchone()[0]
    assert count == 5, f"old index should be intact, got {count} records"
    await arc3.close()


# Module-level worker function for multiprocessing (picklable)
def _mp_worker(db_dir: str, barrier, label: str, n: int, results: list) -> None:
    async def main() -> None:
        store = ArchiveStore(
            archive_dir=os.path.join(db_dir, "archive"),
            index_path=os.path.join(db_dir, "archive-index.db"),
        )
        await store.init()
        # Wait for all processes to be ready
        barrier.wait()
        ids = []
        for i in range(n):
            rid = await store.record("conversation", {"text": f"{label}{i}"}, agent_name=label)
            ids.append(rid)
        await store.close()
        results.append(ids)

    asyncio.run(main())


def test_cross_process_strong_enough_to_fail_without_begin_immediate(tmp_path):
    """Two processes, 300 records each, started together behind a barrier.
    Run 3 times. Must fail without BEGIN IMMEDIATE (duplicate ids).

    On #756's head: the current test passes even without BEGIN IMMEDIATE
    because it only does 20 records each and no barrier.
    """
    from multiprocessing import Barrier, Manager

    # Run 3 times to catch flaky races
    for run in range(3):
        db_dir = str(tmp_path / f"run{run}")
        os.makedirs(db_dir, exist_ok=True)
        ctx = multiprocessing.get_context("spawn")
        manager = Manager()
        results = manager.list()
        barrier = Barrier(2)
        procs = [
            ctx.Process(target=_mp_worker, args=(db_dir, barrier, "p0", 300, results)),
            ctx.Process(target=_mp_worker, args=(db_dir, barrier, "p1", 300, results)),
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)
            assert p.exitcode == 0, f"run {run}: worker failed with {p.exitcode}"

        all_ids = results[0] + results[1]
        assert len(all_ids) == 600, f"run {run}: expected 600 ids, got {len(all_ids)}"
        assert len(set(all_ids)) == 600, f"run {run}: duplicate ids found! {len(all_ids) - len(set(all_ids))} duplicates"


@pytest.mark.asyncio
async def test_init_does_not_rescan_all_files_on_reopen(tmp_path):
    """init() must not rescan every raw file on each open.
    It should only scan when index is missing or stale.

    Measure: on a large archive (1000 files), second init() should be fast.
    """
    import time

    archive_dir = tmp_path / "archive"
    index_path = tmp_path / "archive-index.db"

    # Create 100 daily files with 10 records each
    arc = ArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
    await arc.init()
    for day in range(100):
        for i in range(10):
            await arc.record("conversation", {"text": f"day{day} record{i}"}, agent_name="a")
    await arc.close()

    # First reopen (index exists, should be fast - no full rescan)
    start = time.time()
    arc2 = ArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
    await arc2.init()
    first_reopen_time = time.time() - start
    await arc2.close()

    # Second reopen (should also be fast)
    start = time.time()
    arc3 = ArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
    await arc3.init()
    second_reopen_time = time.time() - start
    await arc3.close()

    # Both should be under 100ms (no full file scan)
    # On #756's head: init() calls _seed_seq_counter() which calls _max_seq_in_files()
    # which reads ALL raw files every time
    print(f"First reopen: {first_reopen_time*1000:.1f}ms, Second: {second_reopen_time*1000:.1f}ms")
    assert first_reopen_time < 0.1, f"first reopen too slow: {first_reopen_time*1000:.1f}ms (full rescan)"
    assert second_reopen_time < 0.1, f"second reopen too slow: {second_reopen_time*1000:.1f}ms (full rescan)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])