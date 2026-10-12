"""Tests for the per-session Claude Code capture cursor store."""
from __future__ import annotations

import pytest

from taosmd.hooks.cursors import CursorStore


class TestCursorStore:
    @staticmethod
    async def _make(tmp_path):
        store = CursorStore(str(tmp_path / "capture-cursors.db"))
        await store.init()
        return store

    @pytest.mark.asyncio
    async def test_update_get_roundtrip_and_unknown(self, tmp_path):
        store = await self._make(tmp_path)
        await store.update(
            "sess-1",
            transcript_path="/transcripts/sess-1.jsonl",
            cwd="/home/user/repo",
            byte_offset=128,
            last_entry_id="entry-abc",
        )

        row = await store.get("sess-1")
        assert row is not None
        assert row["session_id"] == "sess-1"
        assert row["transcript_path"] == "/transcripts/sess-1.jsonl"
        assert row["cwd"] == "/home/user/repo"
        assert row["byte_offset"] == 128
        assert row["last_entry_id"] == "entry-abc"
        assert isinstance(row["updated_at"], float)

        assert await store.get("does-not-exist") is None
        await store.close()

    @pytest.mark.asyncio
    async def test_update_overwrites_existing_row(self, tmp_path):
        store = await self._make(tmp_path)
        await store.update(
            "sess-1", transcript_path="/a/1.jsonl", cwd="/x", byte_offset=10
        )
        await store.update(
            "sess-1",
            transcript_path="/b/2.jsonl",
            cwd="/y",
            byte_offset=4096,
            last_entry_id="e2",
        )

        sessions = await store.all_sessions()
        assert len(sessions) == 1

        row = sessions[0]
        assert row["byte_offset"] == 4096
        assert row["cwd"] == "/y"
        assert row["transcript_path"] == "/b/2.jsonl"
        assert row["last_entry_id"] == "e2"
        await store.close()

    @pytest.mark.asyncio
    async def test_all_sessions_returns_every_row_with_cwd(self, tmp_path):
        store = await self._make(tmp_path)
        await store.update(
            "sess-1", transcript_path="/t/1.jsonl", cwd="/alpha", byte_offset=0
        )
        await store.update(
            "sess-2", transcript_path="/t/2.jsonl", cwd="/beta", byte_offset=5
        )

        sessions = await store.all_sessions()
        assert len(sessions) == 2
        by_id = {s["session_id"]: s for s in sessions}
        assert set(by_id) == {"sess-1", "sess-2"}
        assert by_id["sess-1"]["cwd"] == "/alpha"
        assert by_id["sess-2"]["cwd"] == "/beta"
        await store.close()
