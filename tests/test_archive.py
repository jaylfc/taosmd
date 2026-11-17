"""Tests for the archive aggregation helpers used by the dashboard /stats."""
from __future__ import annotations

import gzip
import logging
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
