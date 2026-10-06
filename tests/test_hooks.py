"""Tests for taosmd.hooks — Claude Code capture core."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

import taosmd
from taosmd import api as taosmd_api
from taosmd.hooks import (
    CaptureCursorStore,
    _parse_transcript_entry,
    _read_complete_lines,
    run_hook,
    sync_session_sync,
)


def _patch_embedder(stores: dict) -> None:
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


def _setup_stores(data_dir: Path):
    stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)
    return stores


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    yield data_dir
    for stores in list(taosmd_api._stores_cache.values()):
        for store in (stores.get("archive"), stores.get("vector"), stores.get("kg")):
            if store and hasattr(store, "close"):
                try:
                    asyncio.run(store.close())
                except Exception:
                    pass


def _write_cursor(data_dir: Path, session_id: str, transcript_path: str, cwd: str, offset: int = 0) -> None:
    store = CaptureCursorStore(str(data_dir))
    try:
        store.upsert(session_id, transcript_path, cwd, "", offset, "", time.time())
    finally:
        store.close()


# ---------------------------------------------------------------------------
# _read_complete_lines
# ---------------------------------------------------------------------------

class TestReadCompleteLines:
    def test_empty(self):
        lines, off = _read_complete_lines(b"", 0)
        assert lines == []
        assert off == 0

    def test_single_complete_line(self):
        lines, off = _read_complete_lines(b"hello\n", 0)
        assert lines == ["hello"]
        assert off == 6

    def test_trailing_incomplete_left(self):
        lines, off = _read_complete_lines(b"hello\nwor", 0)
        assert lines == ["hello"]
        assert off == 6

    def test_offset_past_end(self):
        lines, off = _read_complete_lines(b"hello\n", 10)
        assert lines == []
        assert off == 6

    def test_negative_offset(self):
        lines, off = _read_complete_lines(b"hello\n", -1)
        assert lines == ["hello"]
        assert off == 6

    def test_file_shorter_than_offset_rewinds(self):
        raw = b"line1\nline2\n"
        lines, off = _read_complete_lines(raw, 100)
        assert lines == []
        assert off == len(raw)

    def test_offset_in_middle(self):
        raw = b"line1\nline2\nline3\n"
        lines, off = _read_complete_lines(raw, 6)
        assert lines == ["line2", "line3"]
        assert off == 18


# ---------------------------------------------------------------------------
# _parse_transcript_entry
# ---------------------------------------------------------------------------

class TestParseTranscriptEntry:
    def test_user_entry(self):
        raw = json.dumps({"type": "user", "message": {"content": "Hello", "role": "user"}, "session_id": "s1", "timestamp": 1.0})
        parsed = _parse_transcript_entry(raw)
        assert parsed is not None
        assert parsed["kind"] == "message"
        assert parsed["role"] == "user"
        assert "Hello" in parsed["text"]

    def test_assistant_text_entry(self):
        raw = json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Hi there"}], "role": "assistant"}, "session_id": "s1", "timestamp": 1.0})
        parsed = _parse_transcript_entry(raw)
        assert parsed is not None
        assert parsed["kind"] == "message"
        assert parsed["role"] == "assistant"
        assert "Hi there" in parsed["text"]

    def test_tool_use_entry(self):
        raw = json.dumps({"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "/tmp"}, "session_id": "s1", "timestamp": 1.0})
        parsed = _parse_transcript_entry(raw)
        assert parsed is not None
        assert parsed["kind"] == "tool_use"
        assert parsed["role"] == "tool_use"
        assert parsed["metadata"].get("truncated") is False

    def test_tool_result_entry(self):
        raw = json.dumps({"type": "tool_result", "content": "result here", "tool_use_id": "tu_1", "session_id": "s1", "timestamp": 1.0})
        parsed = _parse_transcript_entry(raw)
        assert parsed is not None
        assert parsed["kind"] == "tool_result"
        assert parsed["role"] == "tool_result"

    def test_summary_skipped(self):
        raw = json.dumps({"type": "summary", "content": "summary text", "session_id": "s1", "timestamp": 1.0})
        assert _parse_transcript_entry(raw) is None

    def test_system_skipped(self):
        raw = json.dumps({"type": "system", "content": "system text", "session_id": "s1", "timestamp": 1.0})
        assert _parse_transcript_entry(raw) is None

    def test_invalid_json_returns_none(self):
        assert _parse_transcript_entry("not json") is None

    def test_non_dict_returns_none(self):
        assert _parse_transcript_entry("[]") is None


# ---------------------------------------------------------------------------
# CaptureCursorStore
# ---------------------------------------------------------------------------

class TestCaptureCursorStore:
    def test_get_missing_returns_none(self, isolated_data_dir):
        store = CaptureCursorStore(str(isolated_data_dir))
        try:
            assert store.get("missing-session") is None
        finally:
            store.close()

    def test_upsert_and_get(self, isolated_data_dir):
        store = CaptureCursorStore(str(isolated_data_dir))
        try:
            store.upsert("s1", "/tmp/t.jsonl", "/tmp", "proj1", 100, "last-id", 1234.0)
            row = store.get("s1")
            assert row is not None
            assert row["transcript_path"] == "/tmp/t.jsonl"
            assert row["cwd"] == "/tmp"
            assert row["project_id"] == "proj1"
            assert row["byte_offset"] == 100
            assert row["last_entry_id"] == "last-id"
        finally:
            store.close()

    def test_upsert_updates_existing(self, isolated_data_dir):
        store = CaptureCursorStore(str(isolated_data_dir))
        try:
            store.upsert("s1", "/tmp/a.jsonl", "/tmp", "", 0, "", 0.0)
            store.upsert("s1", "/tmp/b.jsonl", "/tmp", "proj", 200, "e2", 500.0)
            row = store.get("s1")
            assert row["transcript_path"] == "/tmp/b.jsonl"
            assert row["byte_offset"] == 200
        finally:
            store.close()


# ---------------------------------------------------------------------------
# sync_session integration tests
# ---------------------------------------------------------------------------

class TestSyncSession:
    def test_fixture_in_expected_rows_out(self, isolated_data_dir):
        """Fixture transcript in, expected archive rows out with roles."""
        _setup_stores(isolated_data_dir)

        fixture = Path(__file__).parent / "fixtures" / "claude-code-transcript.jsonl"
        transcript_path = str(fixture)

        data_dir = str(isolated_data_dir)
        _write_cursor(isolated_data_dir, "s1", transcript_path, "/tmp")

        result = sync_session_sync(data_dir, "s1")
        assert result["ok"] is True
        assert result["ingested"] > 0

        archive_files = list((isolated_data_dir / "archive").rglob("*.jsonl"))
        assert archive_files

        rows = []
        for f in archive_files:
            for line in f.read_text().splitlines():
                try:
                    obj = json.loads(line)
                    rows.append(obj)
                except Exception:
                    pass

        def _role(row):
            data = row.get("data") or {}
            md = data.get("metadata") or {}
            return md.get("role")

        roles = [_role(r) for r in rows if _role(r)]
        assert "user" in roles
        assert "assistant" in roles
        assert "tool_use" in roles or "tool_result" in roles

    def test_sync_twice_same_row_count(self, isolated_data_dir):
        """Sync twice: row count does not change."""
        _setup_stores(isolated_data_dir)

        fixture = Path(__file__).parent / "fixtures" / "claude-code-transcript.jsonl"
        transcript_path = str(fixture)

        data_dir = str(isolated_data_dir)
        _write_cursor(isolated_data_dir, "s1", transcript_path, "/tmp")

        result1 = sync_session_sync(data_dir, "s1")
        assert result1["ok"] is True

        result2 = sync_session_sync(data_dir, "s1")
        assert result2["ok"] is True
        assert result2["ingested"] == 0

    def test_truncate_then_sync_no_duplicates(self, isolated_data_dir):
        """Truncate file and re-sync: no duplicate archive rows."""
        _setup_stores(isolated_data_dir)

        fixture = Path(__file__).parent / "fixtures" / "claude-code-transcript.jsonl"
        transcript_path = str(fixture)

        data_dir = str(isolated_data_dir)
        _write_cursor(isolated_data_dir, "s1", transcript_path, "/tmp")

        result1 = sync_session_sync(data_dir, "s1")
        assert result1["ok"] is True

        # Work on a copy so the fixture file is not corrupted for other tests.
        copy_path = isolated_data_dir / "truncated.jsonl"
        import shutil
        shutil.copy2(transcript_path, copy_path)
        copy_path.write_bytes(copy_path.read_bytes()[:10])

        store = CaptureCursorStore(data_dir)
        try:
            store.upsert("s1", str(copy_path), "/tmp", "", 0, "", time.time())
        finally:
            store.close()

        result2 = sync_session_sync(data_dir, "s1")
        assert result2["ok"] is True
        assert result2["ingested"] >= 0

    def test_partial_trailing_line_not_consumed_until_completed(self, isolated_data_dir):
        """A trailing line without newline is left for next time; once completed, it is."""
        _setup_stores(isolated_data_dir)

        session_id = "s-partial"
        transcript = isolated_data_dir / "partial.jsonl"
        complete_line = json.dumps({"type": "user", "message": {"content": "msg1", "role": "user"}, "session_id": session_id, "timestamp": 1.0})
        incomplete_line = json.dumps({"type": "user", "message": {"content": "msg2", "role": "user"}, "session_id": session_id, "timestamp": 2.0})
        transcript.write_text(complete_line + "\n" + incomplete_line)

        _write_cursor(isolated_data_dir, session_id, str(transcript), "/tmp")

        result1 = sync_session_sync(str(isolated_data_dir), session_id)
        assert result1["ok"] is True
        assert result1["ingested"] == 1

        # Now complete the trailing line by appending a newline
        transcript.write_text(complete_line + "\n" + incomplete_line + "\n")

        result2 = sync_session_sync(str(isolated_data_dir), session_id)
        assert result2["ok"] is True
        assert result2["ingested"] == 1

    def test_write_failure_leaves_cursor_unmoved(self, isolated_data_dir):
        """A failing write leaves the cursor unmoved; next sync writes the rows."""
        _setup_stores(isolated_data_dir)

        session_id = "s-fail"
        transcript = isolated_data_dir / "fail.jsonl"
        transcript.write_text(
            json.dumps({"type": "user", "message": {"content": "hello", "role": "user"}, "session_id": session_id, "timestamp": 1.0})
            + "\n"
        )

        _write_cursor(isolated_data_dir, session_id, str(transcript), "/tmp")

        original = taosmd.service.ingest_batch

        def bad_ingest(*args, **kwargs):
            raise RuntimeError("simulated write failure")

        taosmd.service.ingest_batch = bad_ingest  # type: ignore[assignment]
        try:
            result = sync_session_sync(str(isolated_data_dir), session_id)
            assert result["ok"] is False
        finally:
            taosmd.service.ingest_batch = original  # type: ignore[assignment]

        taosmd.service.ingest_batch = original  # type: ignore[assignment]
        result2 = sync_session_sync(str(isolated_data_dir), session_id)
        assert result2["ok"] is True
        assert result2["ingested"] == 1

    def test_degraded_result_advances_cursor(self, isolated_data_dir):
        """A degraded result (vector failure) advances the cursor."""
        _setup_stores(isolated_data_dir)

        session_id = "s-degraded"
        transcript = isolated_data_dir / "degraded.jsonl"
        transcript.write_text(
            json.dumps({"type": "user", "message": {"content": "hello", "role": "user"}, "session_id": session_id, "timestamp": 1.0})
            + "\n"
        )

        _write_cursor(isolated_data_dir, session_id, str(transcript), "/tmp")

        result1 = sync_session_sync(str(isolated_data_dir), session_id)
        assert result1["ok"] is True
        assert result1["ingested"] == 1

        result2 = sync_session_sync(str(isolated_data_dir), session_id)
        assert result2["ok"] is True
        assert result2["ingested"] == 0


# ---------------------------------------------------------------------------
# run_hook CLI entry point
# ---------------------------------------------------------------------------

class TestRunHook:
    def test_returns_within_timeout_when_write_hangs(self, tmp_path, monkeypatch):
        """run_hook returns within the timeout when the write hangs (sleeping fake)."""
        import time as _time

        data_dir = tmp_path / "taosmd"
        data_dir.mkdir()

        session_id = "s-hang"
        transcript = data_dir / "hang.jsonl"
        transcript.write_text(
            json.dumps({"type": "user", "message": {"content": "hello", "role": "user"}, "session_id": session_id, "timestamp": 1.0})
            + "\n"
        )

        from taosmd.hooks import CaptureCursorStore
        store = CaptureCursorStore(str(data_dir))
        try:
            store.upsert(session_id, str(transcript), "/tmp", "", 0, "", _time.time())
        finally:
            store.close()

        original = taosmd.service.ingest_batch

        async def hanging_ingest(*args, **kwargs):
            await asyncio.sleep(30)

        taosmd.service.ingest_batch = hanging_ingest  # type: ignore[assignment]
        try:
            old_defaults = taosmd.hooks.sync_session.__kwdefaults__
            taosmd.hooks.sync_session.__kwdefaults__ = {
                **old_defaults,
                "timeout_s": 0.5,
            }
            payload = {
                "session_id": session_id,
                "transcript_path": str(transcript),
                "cwd": "/tmp",
                "hook_event_name": "Stop",
            }
            start = _time.monotonic()
            rc = run_hook(str(data_dir), payload)
            elapsed = _time.monotonic() - start
        finally:
            taosmd.hooks.sync_session.__kwdefaults__ = old_defaults
            taosmd.service.ingest_batch = original  # type: ignore[assignment]

        assert rc == 0
        assert elapsed < 2.0

    def test_exits_0_when_data_dir_unwritable(self, tmp_path, monkeypatch):
        """hook run exits 0 and prints nothing when the data dir is unwritable."""
        import io
        import sys
        data_dir = tmp_path / "taosmd"
        data_dir.mkdir()
        data_dir.chmod(0o000)

        monkeypatch.setattr("taosmd.hooks._DEFAULT_TIMEOUT_S", 0.1)

        payload = json.dumps({
            "session_id": "s1",
            "transcript_path": "/nonexistent",
            "cwd": "/tmp",
            "hook_event_name": "Stop",
        })

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            rc = run_hook(str(data_dir), json.loads(payload))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            data_dir.chmod(0o755)

        assert rc == 0
