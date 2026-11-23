"""Tests for the archive aggregation helpers used by the dashboard /stats."""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import stat

import pytest

from taosmd.archive import ArchiveStore


@pytest.mark.asyncio
async def test_daily_counts_and_recent(tmp_path):
    arc = ArchiveStore(archive_dir=str(tmp_path / "a"), index_path=str(tmp_path / "idx.db"))
    await arc.init()
    rid = await arc.record(event_type="conversation", data={"text": "hello"}, agent_name="a")
    assert rid >= 0, "conversation events should be recorded"

    counts = await arc.daily_counts(days=30)
    assert isinstance(counts, list)
    assert sum(d["count"] for d in counts) >= 1
    assert all(set(d) == {"date", "count"} for d in counts)

    recent = await arc.recent(limit=5)
    assert recent, "expected at least one recent event"
    assert set(recent[0]) == {"kind", "label", "ts"}
    assert recent[0]["kind"] == "conversation"
    assert "a" in recent[0]["label"]

    assert await arc.distinct_agents() == 1
    top = await arc.top_by("agent_name", limit=5)
    assert top and top[0]["name"] == "a" and top[0]["count"] >= 1
    with pytest.raises(ValueError):
        await arc.top_by("evil; DROP TABLE", limit=5)


@pytest.mark.asyncio
async def test_scope_filtering_and_list_memories(tmp_path):
    arc = ArchiveStore(archive_dir=str(tmp_path / "a"), index_path=str(tmp_path / "idx.db"))
    await arc.init()
    await arc.record(event_type="conversation", data={"text": "user note one"}, agent_name="user")
    await arc.record(event_type="conversation", data={"text": "agent note"}, agent_name="bot")

    assert await arc.scoped_total() == 2
    assert await arc.scoped_total("user") == 1
    g_all = await arc.daily_counts(30)
    g_user = await arc.daily_counts(30, agent="user")
    assert sum(d["count"] for d in g_all) == 2
    assert sum(d["count"] for d in g_user) == 1

    mems = await arc.list_memories(agent="user", limit=10)
    assert len(mems) == 1
    assert mems[0]["agent"] == "user"
    assert "user note" in mems[0]["text"]
    assert set(mems[0]) == {"text", "agent", "kind", "ts"}
    assert len(await arc.list_memories(limit=10)) == 2


@pytest.mark.asyncio
async def test_compress_keeps_source_when_gz_is_corrupt(tmp_path, monkeypatch):
    """A truncated/corrupt .gz must NOT cause the raw .jsonl to be deleted.

    compress_old_files is the only code path that deletes raw archive data,
    so a short write or power cut after unlink would lose a day of transcripts.
    """
    import gzip
    import os
    import time

    arc = ArchiveStore(archive_dir=str(tmp_path / "a"), index_path=str(tmp_path / "idx.db"))
    await arc.init()
    await arc.record(event_type="conversation", data={"text": "keep me"}, agent_name="a")

    jsonl = arc._archive_dir / "2026" / "11" / "08.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text('{"text": "keep me"}\n', encoding="utf-8")
    original_bytes = jsonl.read_bytes()
    os.utime(jsonl, (time.time() - 3 * 86400, time.time() - 3 * 86400))

    real_gzip_open = gzip.open

    class _CorruptGzipFile:
        """Wraps a real gzip file but drops the second half of writes."""

        def __init__(self, *args, **kwargs):
            self._fh = real_gzip_open(*args, **kwargs)
            self._dropped = False

        def __enter__(self):
            self._fh.__enter__()
            return self

        def __exit__(self, *exc):
            return self._fh.__exit__(*exc)

        def write(self, data):
            if not self._dropped:
                self._dropped = True
                self._fh.write(data[: len(data) // 2])
            # silently drop the rest, simulating a truncated write

        def __getattr__(self, name):
            return getattr(self._fh, name)

    monkeypatch.setattr("taosmd.archive.gzip.open", _CorruptGzipFile)

    result = await arc.compress_old_files(days_old=1)
    assert result == 0, "corrupt compression must not be counted"
    assert jsonl.exists(), "raw .jsonl must survive a corrupt .gz"
    assert jsonl.read_bytes() == original_bytes, "raw .jsonl bytes must be unchanged"
    assert not jsonl.with_suffix(".jsonl.gz").exists(), "no .gz must be left behind"


@pytest.mark.asyncio
async def test_compress_round_trip_ok(tmp_path):
    """A clean .gz round-trips byte-equal and the source .jsonl is removed."""
    import gzip
    import os
    import time

    arc = ArchiveStore(archive_dir=str(tmp_path / "a"), index_path=str(tmp_path / "idx.db"))
    await arc.init()

    jsonl = arc._archive_dir / "2026" / "11" / "08.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text('{"text": "round trip"}\n', encoding="utf-8")
    original_bytes = jsonl.read_bytes()
    os.utime(jsonl, (time.time() - 3 * 86400, time.time() - 3 * 86400))

    result = await arc.compress_old_files(days_old=1)
    assert result == 1, "one file compressed"
    assert not jsonl.exists(), "raw .jsonl is removed after successful compression"
    gz_path = jsonl.with_suffix(".jsonl.gz")
    assert gz_path.exists(), "the .gz must exist"
    with gzip.open(gz_path, "rb") as fh:
        assert fh.read() == original_bytes, ".gz must decompress to the original bytes"


@pytest.mark.asyncio
async def test_compress_fsyncs_complete_gz(tmp_path, monkeypatch):
    """The fsync must happen AFTER the gzip trailer is written.

    On the base code the fsync runs inside the gzip context manager, before
    GzipFile.close() emits the 8-byte CRC/size trailer. The largest recorded
    regular-file fsync size will be 10 bytes short of the final .gz size.
    """
    import os
    import stat
    import time

    arc = ArchiveStore(archive_dir=str(tmp_path / "a"), index_path=str(tmp_path / "idx.db"))
    await arc.init()

    jsonl = arc._archive_dir / "2026" / "11" / "08.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    # Write enough data to produce a visible gzip trailer difference
    payload = '{"text": "x" * 15000}\n'
    jsonl.write_text(payload, encoding="utf-8")
    os.utime(jsonl, (time.time() - 3 * 86400, time.time() - 3 * 86400))

    recorded_sizes: list[int] = []
    recorded_is_dir: list[bool] = []

    real_fsync = os.fsync

    def recording_fsync(fd):
        try:
            st = os.fstat(fd)
            recorded_sizes.append(st.st_size)
            recorded_is_dir.append(stat.S_ISDIR(st.st_mode))
        except OSError:
            pass
        return real_fsync(fd)

    monkeypatch.setattr("taosmd.archive.os.fsync", recording_fsync)

    result = await arc.compress_old_files(days_old=1)
    assert result == 1

    gz_path = jsonl.with_suffix(".jsonl.gz")
    assert gz_path.exists()
    final_size = gz_path.stat().st_size

    # Find the largest regular-file fsync size
    regular_sizes = [s for s, is_dir in zip(recorded_sizes, recorded_is_dir) if not is_dir]
    assert regular_sizes, "expected at least one regular-file fsync"
    max_regular_sync = max(regular_sizes)

    # On the base this fails: max_regular_sync == final_size - 10 (trailer missing)
    assert max_regular_sync == final_size, (
        f"largest regular-file fsync size {max_regular_sync} != final .gz size {final_size}; "
        "fsync ran before gzip trailer was written"
    )


@pytest.mark.asyncio
async def test_compress_dir_fsync_refused_still_compresses(tmp_path, monkeypatch, caplog):
    """A refused directory fsync must not cause a failure.

    On the base code the directory fsync sits in the same try block as the
    replace/unlink, so an OSError jumps to the failure branch AFTER os.replace
    has already created the final .gz. The .jsonl is kept, 0 is counted, a
    misleading 'verification failed' warning is logged, and every later run
    skips that day forever because the .gz exists.
    """
    import os
    import time

    arc = ArchiveStore(archive_dir=str(tmp_path / "a"), index_path=str(tmp_path / "idx.db"))
    await arc.init()

    jsonl = arc._archive_dir / "2026" / "11" / "08.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    payload = '{"text": "dir fsync refused"}\n'
    jsonl.write_text(payload, encoding="utf-8")
    original_bytes = jsonl.read_bytes()
    os.utime(jsonl, (time.time() - 3 * 86400, time.time() - 3 * 86400))

    real_fsync = os.fsync

    def refusing_fsync(fd):
        st = os.fstat(fd)
        if stat.S_ISDIR(st.st_mode):
            raise OSError("directory fsync not supported")
        return real_fsync(fd)

    monkeypatch.setattr("taosmd.archive.os.fsync", refusing_fsync)

    with caplog.at_level(logging.WARNING):
        result = await arc.compress_old_files(days_old=1)

    assert result == 1, "compression should succeed even when dir fsync is refused"
    assert not jsonl.exists(), "raw .jsonl must be removed after successful compression"
    gz_path = jsonl.with_suffix(".jsonl.gz")
    assert gz_path.exists(), "the .gz must exist"
    with gzip.open(gz_path, "rb") as fh:
        assert fh.read() == original_bytes, ".gz must decompress to the original bytes"

    # No 'verification failed' warning should be logged
    verification_warnings = [r for r in caplog.records if "verification failed" in r.getMessage().lower()]
    assert not verification_warnings, f"unexpected verification failed warning: {verification_warnings}"


@pytest.mark.asyncio
async def test_compress_dir_os_open_raises_on_windows(tmp_path, monkeypatch):
    """On Windows opening a directory with os.open raises PermissionError.

    The bug: dir_fd = os.open(dir, os.O_RDONLY) sits OUTSIDE the try/except
    OSError around the directory fsync, so the exception escapes and the
    .jsonl is kept beside the .gz forever (since os.replace already happened).

    Fix: move the os.open inside the try (keep the close in a finally only
    when the open succeeded).

    Test: monkeypatch os.open in taosmd.archive to raise PermissionError for
    a directory path (os.path.isdir) and call through otherwise. compress_old_files
    returns 1, the .jsonl is gone, the .gz decompresses to the source bytes.
    """
    import os
    import time

    THREE_DAYS = 3 * 86400

    arc = ArchiveStore(archive_dir=str(tmp_path / "a"), index_path=str(tmp_path / "idx.db"))
    await arc.init()

    jsonl = arc._archive_dir / "2026" / "11" / "08.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    payload = '{"text": "windows dir open fails"}\n'
    jsonl.write_text(payload, encoding="utf-8")
    original_bytes = jsonl.read_bytes()
    os.utime(jsonl, (time.time() - THREE_DAYS, time.time() - THREE_DAYS))

    real_open = os.open

    def raising_open(path, flags, *args, **kwargs):
        # Raise PermissionError when trying to open a directory
        if os.path.isdir(path):
            raise PermissionError("Windows does not allow opening directories")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr("taosmd.archive.os.open", raising_open)

    result = await arc.compress_old_files(days_old=1)

    assert result == 1, "compression should succeed even when dir os.open raises"
    assert not jsonl.exists(), "raw .jsonl must be removed after successful compression"
    gz_path = jsonl.with_suffix(".jsonl.gz")
    assert gz_path.exists(), "the .gz must exist"
    with gzip.open(gz_path, "rb") as fh:
        assert fh.read() == original_bytes, ".gz must decompress to the original bytes"


@pytest.mark.asyncio
async def test_rebuild_uses_seq_ids_never_renumbers(tmp_path):
    """(a) Record N entries, delete the index db, rebuild: every id survives
    and every line's seq equals its id."""
    archive_dir = tmp_path / "archive"
    index_path = tmp_path / "archive-index.db"

    async def open_store():
        arc = ArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
        await arc.init()
        return arc

    arc = await open_store()
    ids = [
        await arc.record("conversation", {"text": f"record {i}"}, agent_name="a")
        for i in range(5)
    ]
    assert ids == [1, 2, 3, 4, 5]
    await arc.close()

    def raw_hashes():
        return {
            str(path): path.read_bytes()
            for path in sorted(archive_dir.rglob("*"))
            if path.is_file()
        }

    before = raw_hashes()
    os.remove(index_path)

    arc2 = await open_store()
    report = await arc2.rebuild()
    assert report == {"indexed": 5, "legacy": 0}
    rows = {
        (r["file_path"], r["line_number"]): r["id"]
        for r in arc2._conn.execute(
            "SELECT id, file_path, line_number FROM archive_index"
        )
    }
    assert len(rows) == 5
    for path in sorted(archive_dir.rglob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                assert record["seq"] == rows[(str(path), line_number)]
    assert {
        r["id"] for r in arc2._conn.execute("SELECT id FROM archive_index")
    } == {1, 2, 3, 4, 5}
    await arc2.close()

    after = raw_hashes()
    assert before == after, "rebuild must not rewrite raw .jsonl files"


@pytest.mark.asyncio
async def test_rebuild_legacy_lines_keep_mapped_ids(tmp_path):
    """(b) A legacy archive (lines without seq) plus an index: build the
    legacy-id-map, delete the index, rebuild, ids unchanged."""
    archive_dir = tmp_path / "archive"
    index_path = tmp_path / "archive-index.db"
    day = archive_dir / "2020" / "01" / "01.jsonl"
    day.parent.mkdir(parents=True)

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

    arc = ArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
    await arc.init()
    # Index the legacy lines with non-contiguous ids to prove the frozen map
    # is honoured instead of renumbering from 1.
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
    frozen = await arc.freeze_legacy_id_map()
    assert frozen == 3
    await arc.close()

    os.remove(index_path)

    arc2 = ArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
    await arc2.init()
    report = await arc2.rebuild()
    assert report == {"indexed": 3, "legacy": 3}
    rows = {
        (r["file_path"], r["line_number"]): r["id"]
        for r in arc2._conn.execute(
            "SELECT id, file_path, line_number FROM archive_index"
        )
    }
    for line_number, row_id in zip(range(1, 4), mapped_ids):
        assert rows[(str(day), line_number)] == row_id
    assert {
        r["id"] for r in arc2._conn.execute("SELECT id FROM archive_index")
    } == {7, 8, 9}
    await arc2.close()


@pytest.mark.asyncio
async def test_crash_between_seq_and_raw_write_leaves_gap_never_duplicate(
    tmp_path, monkeypatch
):
    """(c) A crash after the seq is assigned but before the raw write
    consumes the id (a gap) but never reuses it."""
    arc = ArchiveStore(
        archive_dir=str(tmp_path / "archive"),
        index_path=str(tmp_path / "archive-index.db"),
    )
    await arc.init()

    real_ensure_file = arc._ensure_file
    state = {"crashed": False}

    def boom(*args, **kwargs):
        if not state["crashed"]:
            state["crashed"] = True
            raise RuntimeError("crash before the raw write")
        return real_ensure_file(*args, **kwargs)

    monkeypatch.setattr(arc, "_ensure_file", boom)
    with pytest.raises(RuntimeError, match="crash before the raw write"):
        await arc.record("conversation", {"text": "lost"}, agent_name="a")

    # The seq was durably persisted before the raw write: seq 1 is consumed.
    counter = arc._conn.execute(
        "SELECT value FROM archive_settings WHERE key = 'archive_seq_counter'"
    ).fetchone()["value"]
    assert counter == "1"
    # Nothing was written to the raw files for the lost record.
    assert not list(arc._archive_dir.rglob("*.jsonl"))

    # Subsequent records continue above the consumed id: a gap at 1, no duplicates.
    ids = [
        await arc.record("conversation", {"text": f"ok {i}"}, agent_name="a")
        for i in range(3)
    ]
    assert ids == [2, 3, 4]

    seqs = []
    for path in arc._archive_dir.rglob("*.jsonl"):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    seqs.append(json.loads(line)["seq"])
    assert seqs == [2, 3, 4]
    assert len(seqs) == len(set(seqs))
    await arc.close()


@pytest.mark.asyncio
async def test_rebuild_keeps_raw_files_byte_identical(tmp_path):
    """(d) Existing .jsonl and .jsonl.gz files are byte-identical before and
    after a rebuild (hash-compare); the rebuild only reads raw files."""
    archive_dir = tmp_path / "archive"
    index_path = tmp_path / "archive-index.db"

    arc = ArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
    await arc.init()
    await arc.record("conversation", {"text": "plain day"}, agent_name="a")
    # A second day stored only as a .gz, as the nightly compress produces.
    gz_day = archive_dir / "2020" / "02" / "02.jsonl.gz"
    gz_day.parent.mkdir(parents=True)
    payload = (
        '{"timestamp": 2.0, "event_type": "conversation", "summary": "gz day", "data": {}}\n'
    ).encode()
    with gzip.open(gz_day, "wb") as fh:
        fh.write(payload)
    await arc.close()

    def raw_hashes():
        return {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(archive_dir.rglob("*"))
            if path.is_file()
        }

    before = raw_hashes()
    os.remove(index_path)

    arc2 = ArchiveStore(archive_dir=str(archive_dir), index_path=str(index_path))
    await arc2.init()
    report = await arc2.rebuild()
    assert report["indexed"] == 2
    await arc2.close()

    after = raw_hashes()
    assert before == after, "rebuild must not rewrite raw .jsonl or .jsonl.gz files"


@pytest.mark.asyncio
async def test_rebuild_ids_unchanged_after_partial_index(tmp_path):
    """Write 5 records, drop index rows 2 and 4 only (simulating a partial
    index), rebuild: every id must be unchanged.

    Fails on origin-master: there is no rebuild at all there, and a naive
    rebuild that renumbers from 1 would also fail this assertion."""
    arc = ArchiveStore(
        archive_dir=str(tmp_path / "archive"),
        index_path=str(tmp_path / "archive-index.db"),
    )
    await arc.init()
    ids = [
        await arc.record("conversation", {"text": f"record {i}"}, agent_name="a")
        for i in range(5)
    ]
    assert ids == [1, 2, 3, 4, 5]
    arc._conn.execute("DELETE FROM archive_index WHERE id IN (2, 4)")
    arc._conn.execute("DELETE FROM archive_fts WHERE rowid IN (2, 4)")
    arc._conn.commit()
    await arc.close()

    arc2 = ArchiveStore(
        archive_dir=str(tmp_path / "archive"),
        index_path=str(tmp_path / "archive-index.db"),
    )
    await arc2.init()
    report = await arc2.rebuild()
    assert report == {"indexed": 5, "legacy": 0}
    remaining = {
        r["id"] for r in arc2._conn.execute("SELECT id FROM archive_index ORDER BY id")
    }
    assert remaining == {1, 2, 3, 4, 5}
    rows = {
        (r["file_path"], r["line_number"]): r["id"]
        for r in arc2._conn.execute(
            "SELECT id, file_path, line_number FROM archive_index"
        )
    }
    for path in sorted((tmp_path / "archive").rglob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if line.strip():
                    assert rows[(str(path), line_number)] == json.loads(line)["seq"]
    await arc2.close()
