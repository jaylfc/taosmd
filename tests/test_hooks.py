"""Tests for Claude Code capture hooks.

All tests are hermetic: tmp_path data dir, no network, no models.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest

import taosmd
from taosmd import api as taosmd_api
from taosmd.hooks import parser, sync as hooks_sync


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "claude_code_transcript.jsonl"
_FIXTURE_LINES = _FIXTURE_PATH.read_text().splitlines()


def _patch_embedder(stores: dict) -> None:
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


def _setup(data_dir: Path) -> dict:
    stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)
    return stores


def _write_transcript(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def _archive_rows(data_dir: Path, agent: str = "claude-code") -> list[dict]:
    stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
    archive = stores["archive"]
    return asyncio.run(
        archive.query(event_type="conversation", agent_name=agent, limit=1000)
    )


def _vector_metadata(data_dir: Path) -> list[dict]:
    db = data_dir / "vector-memory.db"
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT metadata_json FROM vector_memory").fetchall()
    return [json.loads(r[0]) for r in rows]


def _init_git_repo(path: Path) -> str:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/test/test.git"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def isolated(tmp_path, monkeypatch):
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


@pytest.fixture
def git_cwd(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    return repo


# ---------------------------------------------------------------------------
# Parser unit tests
# ---------------------------------------------------------------------------

class TestParser:
    def test_fixture_item_counts(self):
        kinds: dict[str, int] = {}
        roles: dict[str, int] = {}
        for line in _FIXTURE_LINES:
            items, _ = parser.parse_entry(0, line, "s1")
            if items is None:
                continue
            for item in items:
                kinds[item["metadata"]["kind"]] = kinds.get(item["metadata"]["kind"], 0) + 1
                roles[item["metadata"]["role"]] = roles.get(item["metadata"]["role"], 0) + 1

        assert kinds.get("message", 0) == 4
        assert kinds.get("tool_use", 0) == 2
        assert kinds.get("tool_result", 0) == 2
        assert roles.get("user", 0) == 4
        assert roles.get("assistant", 0) == 4

    def test_user_plain_string(self):
        line = '{"type":"user","message":{"role":"user","content":"hello"},"uuid":"u1","timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}'
        items, base_id = parser.parse_entry(0, line, "s1")
        assert items is not None
        assert len(items) == 1
        assert items[0]["metadata"]["kind"] == "message"
        assert items[0]["metadata"]["role"] == "user"
        assert items[0]["text"] == "hello"
        assert base_id == "claude-code:s1:u1"

    def test_user_blocks_text(self):
        line = '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"hi"}]},"uuid":"u2","timestamp":"2026-10-07T22:53:26.123Z","sessionId":"s1"}'
        items, _ = parser.parse_entry(0, line, "s1")
        assert items is not None
        assert len(items) == 1
        assert items[0]["text"] == "hi"

    def test_assistant_thinking_skipped(self):
        line = '{"type":"assistant","message":{"role":"assistant","content":[{"type":"thinking","thinking":"hmm","signature":"sig"}]},"uuid":"a2","timestamp":"2026-10-07T22:53:28.123Z","sessionId":"s1"}'
        items, _ = parser.parse_entry(0, line, "s1")
        assert items is None

    def test_assistant_tool_use(self):
        line = '{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"ls"}}]},"uuid":"a3","timestamp":"2026-10-07T22:53:29.123Z","sessionId":"s1"}'
        items, _ = parser.parse_entry(0, line, "s1")
        assert items is not None
        assert len(items) == 1
        assert items[0]["metadata"]["kind"] == "tool_use"
        assert items[0]["metadata"]["role"] == "assistant"
        assert items[0]["metadata"]["tool_name"] == "Bash"
        assert '"command":"ls"' in items[0]["text"]

    def test_user_tool_result_string_content(self):
        line = '{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"output text","is_error":false}]},"toolUseResult":{"stdout":"ok","stderr":""},"uuid":"u3","timestamp":"2026-10-07T22:53:30.123Z","sessionId":"s1"}'
        items, _ = parser.parse_entry(0, line, "s1")
        assert items is not None
        assert len(items) == 1
        assert items[0]["metadata"]["kind"] == "tool_result"
        assert items[0]["metadata"]["role"] == "user"
        assert items[0]["text"] == "output text"

    def test_user_tool_result_list_content(self):
        line = '{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_2","content":[{"type":"text","text":"result text"}],"is_error":false}]},"toolUseResult":{"stdout":"ok","stderr":""},"uuid":"u4","timestamp":"2026-10-07T22:53:39.123Z","sessionId":"s1"}'
        items, _ = parser.parse_entry(0, line, "s1")
        assert items is not None
        assert len(items) == 1
        assert items[0]["text"] == "result text"

    def test_noise_types_skipped(self):
        for noise_type in ("queue-operation", "attachment", "system", "file-history-snapshot",
                           "last-prompt", "ai-title", "cost-state"):
            line = json.dumps({"type": noise_type, "sessionId": "s1", "timestamp": "2026-10-07T22:53:00.000Z"})
            items, _ = parser.parse_entry(0, line, "s1")
            assert items is None, f"{noise_type} should be skipped"

    def test_no_type_skipped(self):
        line = '{"sessionId":"s1"}'
        items, _ = parser.parse_entry(0, line, "s1")
        assert items is None

    def test_timestamp_converted(self):
        line = '{"type":"user","message":{"role":"user","content":"hi"},"uuid":"u1","timestamp":"2026-10-07T22:53:25.123Z","sessionId":"s1"}'
        items, _ = parser.parse_entry(0, line, "s1")
        assert items is not None
        assert items[0]["metadata"]["timestamp"] == pytest.approx(
            parser._iso_to_epoch("2026-10-07T22:53:25.123Z"), abs=1
        )

    def test_8kb_cap(self):
        big_text = "a" * (8 * 1024 + 10)
        line = json.dumps({
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": big_text}]},
            "uuid": "a_big",
            "timestamp": "2026-10-07T22:53:00.000Z",
            "sessionId": "s1",
        })
        items, _ = parser.parse_entry(0, line, "s1")
        assert items is not None
        assert items[0]["metadata"]["truncated"] is True
        assert len(items[0]["text"].encode("utf-8")) <= 8 * 1024

    def test_multi_block_assistant_text_and_tool_use(self):
        line = '{"type":"assistant","message":{"role":"assistant","model":"claude","id":"msg4","content":[{"type":"text","text":"let me run that"},{"type":"tool_use","id":"toolu_2","name":"Read","input":{"path":"/tmp/test.txt"}}]},"uuid":"a4","timestamp":"2026-10-07T22:53:38.123Z","sessionId":"s1"}'
        items, base_id = parser.parse_entry(0, line, "s1")
        assert items is not None
        assert len(items) == 2
        kinds = {i["metadata"]["kind"] for i in items}
        assert kinds == {"message", "tool_use"}
        assert items[0]["id"] == base_id
        assert items[1]["id"] == f"{base_id}:tool_use:0"


# ---------------------------------------------------------------------------
# Integration tests: sync + ingest
# ---------------------------------------------------------------------------

class TestSync:
    def test_fixture_ingested_rows(self, isolated: Path, git_cwd: Path):
        _setup(isolated)
        transcript = isolated / "transcript.jsonl"
        _write_transcript(transcript, _FIXTURE_LINES)

        cwd = str(git_cwd)
        result = asyncio.run(
            hooks_sync.sync_session(
                str(isolated), "s1", str(transcript), cwd, timeout=5
            )
        )
        assert result["ok"]
        assert result["ingested"] == 8

        rows = _archive_rows(isolated)
        assert len(rows) == 8

        expected_project = taosmd.project.get_project_id(cwd=cwd)
        for row in rows:
            assert row.get("project") == expected_project

        vector_meta = _vector_metadata(isolated)
        assert len(vector_meta) == 8
        for meta in vector_meta:
            assert meta.get("project") == expected_project

    def test_sync_twice_no_duplicates(self, isolated: Path, git_cwd: Path):
        _setup(isolated)
        transcript = isolated / "transcript.jsonl"
        _write_transcript(transcript, _FIXTURE_LINES)

        cwd = str(git_cwd)
        result1 = asyncio.run(
            hooks_sync.sync_session(
                str(isolated), "s1", str(transcript), cwd, timeout=5
            )
        )
        assert result1["ok"]
        assert result1["ingested"] == 8

        result2 = asyncio.run(
            hooks_sync.sync_session(
                str(isolated), "s1", str(transcript), cwd, timeout=5
            )
        )
        assert result2["ok"]
        assert result2["ingested"] == 0

        rows = _archive_rows(isolated)
        assert len(rows) == 8

    def test_truncate_then_sync_no_duplicates(self, isolated: Path, git_cwd: Path):
        _setup(isolated)
        transcript = isolated / "transcript.jsonl"
        _write_transcript(transcript, _FIXTURE_LINES)

        cwd = str(git_cwd)
        result1 = asyncio.run(
            hooks_sync.sync_session(
                str(isolated), "s1", str(transcript), cwd, timeout=5
            )
        )
        assert result1["ok"]
        assert result1["ingested"] == 8

        rows_after_1 = _archive_rows(isolated)
        source_ids_1 = {r.get("id") for r in rows_after_1}

        # Truncate below the stored offset
        current_size = transcript.stat().st_size
        transcript.write_bytes(transcript.read_bytes()[: current_size // 2])

        result2 = asyncio.run(
            hooks_sync.sync_session(
                str(isolated), "s1", str(transcript), cwd, timeout=5
            )
        )
        assert result2["ok"]
        assert result2["ingested"] == 0

        rows_after_2 = _archive_rows(isolated)
        source_ids_2 = {r.get("id") for r in rows_after_2}
        assert source_ids_1 == source_ids_2

    def test_partial_trailing_line_not_consumed(self, isolated: Path, git_cwd: Path):
        _setup(isolated)
        transcript = isolated / "transcript.jsonl"
        content = "\n".join(_FIXTURE_LINES[:-1]) + "\n" + "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"partial\"},\"uuid\":\"u_partial\",\"timestamp\":\"2026-10-07T22:53:40.000Z\",\"sessionId\":\"s1\"}"
        transcript.write_bytes(content.encode("utf-8"))

        cwd = str(git_cwd)
        result = asyncio.run(
            hooks_sync.sync_session(
                str(isolated), "s1", str(transcript), cwd, timeout=5
            )
        )
        assert result["ok"]
        assert result["ingested"] == 8

        transcript.write_bytes(content.encode("utf-8") + b"\n")

        result2 = asyncio.run(
            hooks_sync.sync_session(
                str(isolated), "s1", str(transcript), cwd, timeout=5
            )
        )
        assert result2["ok"]
        assert result2["ingested"] == 1
        assert len(_archive_rows(isolated)) == 9

    def test_write_failure_leaves_cursor(self, isolated: Path, git_cwd: Path):
        _setup(isolated)
        transcript = isolated / "transcript.jsonl"
        _write_transcript(transcript, _FIXTURE_LINES)

        cwd = str(git_cwd)
        call_count = 0

        async def _failing_ingest(*a, **kw):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("write failed")

        original = taosmd.service.ingest_batch
        taosmd.service.ingest_batch = _failing_ingest  # type: ignore[assignment]
        try:
            result = asyncio.run(
                hooks_sync.sync_session(
                    str(isolated), "s1", str(transcript), cwd, timeout=5
                )
            )
            assert not result["ok"]
            assert call_count == 1
        finally:
            taosmd.service.ingest_batch = original  # type: ignore[assignment]

        assert _archive_rows(isolated) == []

        result2 = asyncio.run(
            hooks_sync.sync_session(
                str(isolated), "s1", str(transcript), cwd, timeout=5
            )
        )
        assert result2["ok"]
        assert result2["ingested"] == 8
        assert len(_archive_rows(isolated)) == 8

    def test_degraded_advances_cursor(self, isolated: Path, git_cwd: Path):
        _setup(isolated)
        transcript = isolated / "transcript.jsonl"
        _write_transcript(transcript, _FIXTURE_LINES)

        cwd = str(git_cwd)
        call_count = 0

        async def _degraded_ingest(*a, **kw):
            nonlocal call_count
            call_count += 1
            return {"ingested": 8, "skipped": 0, "degraded": True, "vector_failures": 8}

        original = taosmd.service.ingest_batch
        taosmd.service.ingest_batch = _degraded_ingest  # type: ignore[assignment]
        try:
            result = asyncio.run(
                hooks_sync.sync_session(
                    str(isolated), "s1", str(transcript), cwd, timeout=5
                )
            )
            assert result["ok"]
            assert call_count == 1
        finally:
            taosmd.service.ingest_batch = original  # type: ignore[assignment]

        assert _archive_rows(isolated) == []

        result2 = asyncio.run(
            hooks_sync.sync_session(
                str(isolated), "s1", str(transcript), cwd, timeout=5
            )
        )
        assert result2["ok"]
        assert result2["ingested"] == 0

    def test_empty_batch_advances_cursor(self, isolated: Path, git_cwd: Path):
        _setup(isolated)
        transcript = isolated / "transcript.jsonl"
        # Empty file
        transcript.write_bytes(b"")

        cwd = str(git_cwd)
        # Pre-populate cursor
        store = hooks_sync.cursors.CursorStore(
            str(isolated / "capture-cursors.db")
        )
        asyncio.run(store.init())
        asyncio.run(store.update("s1", str(transcript), 0, None))
        asyncio.run(store.close())

        result = asyncio.run(
            hooks_sync.sync_session(
                str(isolated), "s1", str(transcript), cwd, timeout=5
            )
        )
        assert result["ok"]
        assert result["ingested"] == 0

        store = hooks_sync.cursors.CursorStore(
            str(isolated / "capture-cursors.db")
        )
        asyncio.run(store.init())
        cursor = asyncio.run(store.get("s1"))
        asyncio.run(store.close())
        assert cursor is not None
        assert cursor["byte_offset"] == 0

    def test_byte_offset_not_char_index(self, isolated: Path, git_cwd: Path):
        _setup(isolated)
        transcript = isolated / "transcript.jsonl"
        japanese = "日本語"
        jp_line = json.dumps({
            "type": "user",
            "message": {"role": "user", "content": japanese},
            "uuid": "u_jp",
            "timestamp": "2026-10-07T22:53:00.000Z",
            "sessionId": "s1",
        })
        partial = "partial"
        content = jp_line + "\n" + partial
        transcript.write_bytes(content.encode("utf-8"))

        cwd = str(git_cwd)
        result = asyncio.run(
            hooks_sync.sync_session(
                str(isolated), "s1", str(transcript), cwd, timeout=5
            )
        )
        assert result["ok"]
        assert result["ingested"] == 1

        store = hooks_sync.cursors.CursorStore(
            str(isolated / "capture-cursors.db")
        )
        asyncio.run(store.init())
        cursor = asyncio.run(store.get("s1"))
        asyncio.run(store.close())
        assert cursor is not None
        jp_bytes = len(jp_line.encode("utf-8")) + 1
        assert cursor["byte_offset"] == jp_bytes

    def test_rewind_on_truncated_file(self, isolated: Path, git_cwd: Path):
        _setup(isolated)
        transcript = isolated / "transcript.jsonl"
        _write_transcript(transcript, _FIXTURE_LINES)

        cwd = str(git_cwd)
        result1 = asyncio.run(
            hooks_sync.sync_session(
                str(isolated), "s1", str(transcript), cwd, timeout=5
            )
        )
        assert result1["ok"]
        assert result1["ingested"] == 8

        # Truncate below stored offset
        current_size = transcript.stat().st_size
        transcript.write_bytes(transcript.read_bytes()[: current_size // 2])

        result2 = asyncio.run(
            hooks_sync.sync_session(
                str(isolated), "s1", str(transcript), cwd, timeout=5
            )
        )
        assert result2["ok"]
        assert result2["ingested"] == 0

        rows = _archive_rows(isolated)
        assert len(rows) == 8

    def test_project_id_stored_in_archive_and_vector(
        self, isolated: Path, git_cwd: Path
    ):
        _setup(isolated)
        transcript = isolated / "transcript.jsonl"
        _write_transcript(transcript, _FIXTURE_LINES)

        cwd = str(git_cwd)
        result = asyncio.run(
            hooks_sync.sync_session(
                str(isolated), "s1", str(transcript), cwd, timeout=5
            )
        )
        assert result["ok"]

        expected_project = taosmd.project.get_project_id(cwd=cwd)

        rows = _archive_rows(isolated)
        assert len(rows) == 8
        for row in rows:
            assert row.get("project") == expected_project

        vector_meta = _vector_metadata(isolated)
        assert len(vector_meta) == 8
        for meta in vector_meta:
            assert meta.get("project") == expected_project

    def test_timeout_triggers_reconcile_once(self, isolated: Path, git_cwd: Path):
        _setup(isolated)
        transcript = isolated / "transcript.jsonl"
        _write_transcript(transcript, _FIXTURE_LINES)

        cwd = str(git_cwd)
        reconcile_calls = 0

        async def _slow_ingest(*a, **kw):
            await asyncio.sleep(10)
            return {"ingested": 0, "skipped": 0}

        original_ingest = taosmd.service.ingest_batch
        original_reconcile = taosmd.service.reconcile

        async def _tracked_reconcile(*a, **kw):
            nonlocal reconcile_calls
            reconcile_calls += 1
            return await original_reconcile(*a, **kw)

        taosmd.service.ingest_batch = _slow_ingest  # type: ignore[assignment]
        taosmd.service.reconcile = _tracked_reconcile  # type: ignore[assignment]
        try:
            result = asyncio.run(
                hooks_sync.sync_session(
                    str(isolated), "s1", str(transcript), cwd, timeout=2
                )
            )
            assert not result["ok"]
            assert result.get("timed_out")
            assert reconcile_calls == 1
        finally:
            taosmd.service.ingest_batch = original_ingest  # type: ignore[assignment]
            taosmd.service.reconcile = original_reconcile  # type: ignore[assignment]

    def test_no_reconcile_on_success(self, isolated: Path, git_cwd: Path):
        _setup(isolated)
        transcript = isolated / "transcript.jsonl"
        _write_transcript(transcript, _FIXTURE_LINES)

        cwd = str(git_cwd)
        reconcile_calls = 0

        async def _tracked_reconcile(*a, **kw):
            nonlocal reconcile_calls
            reconcile_calls += 1
            return await taosmd.service.reconcile(*a, **kw)

        original_reconcile = taosmd.service.reconcile
        taosmd.service.reconcile = _tracked_reconcile  # type: ignore[assignment]
        try:
            result = asyncio.run(
                hooks_sync.sync_session(
                    str(isolated), "s1", str(transcript), cwd, timeout=5
                )
            )
            assert result["ok"]
            assert reconcile_calls == 0
        finally:
            taosmd.service.reconcile = original_reconcile  # type: ignore[assignment]

    def test_hooks_sync_session_returns_1_on_failure(self, isolated: Path):
        _setup(isolated)
        result = asyncio.run(
            hooks_sync.sync_session(
                str(isolated), "nonexistent", "/nonexistent/path", "", timeout=5
            )
        )
        assert result["ok"]
        assert result["ingested"] == 0


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

class TestHooksCLI:
    def test_hooks_run_bad_stdin_exits_0(self, isolated: Path, capsys):
        from taosmd.cli import main

        rc = main(["--data-dir", str(isolated), "hooks", "run", "Stop"])
        assert rc == 0
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_hooks_run_non_dict_stdin_exits_0(self, isolated: Path, capsys, monkeypatch):
        from taosmd.cli import main

        monkeypatch.setattr("sys.stdin.read", lambda: "[]")
        rc = main(["--data-dir", str(isolated), "hooks", "run", "Stop"])
        assert rc == 0
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_hooks_run_prints_nothing(self, isolated: Path, capsys, git_cwd: Path, monkeypatch):
        from taosmd.cli import main

        _setup(isolated)
        transcript = isolated / "transcript.jsonl"
        _write_transcript(transcript, _FIXTURE_LINES)

        payload = json.dumps({
            "session_id": "s1",
            "transcript_path": str(transcript),
            "cwd": str(git_cwd),
            "hook_event_name": "Stop",
        })
        monkeypatch.setattr("sys.stdin.read", lambda: payload)

        rc = main(["--data-dir", str(isolated), "hooks", "run", "Stop"])
        assert rc == 0
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_hooks_sync_session_returns_1_when_ok_false(self, isolated: Path, monkeypatch):
        from taosmd.cli import main

        async def _failing_sync(*a, **kw):
            return {"ok": False}

        original = hooks_sync.sync_session
        hooks_sync.sync_session = _failing_sync  # type: ignore[assignment]
        try:
            rc = main(["--data-dir", str(isolated), "hooks", "sync", "--session", "s1"])
            assert rc == 1
        finally:
            hooks_sync.sync_session = original  # type: ignore[assignment]

    def test_hooks_run_unwritable_data_dir_exits_0(self, tmp_path, capsys, monkeypatch):
        from taosmd.cli import main

        data_dir = tmp_path / "taosmd-ro"
        data_dir.mkdir()
        os.chmod(str(data_dir), 0o444)

        try:
            monkeypatch.setattr("sys.stdin.read", lambda: json.dumps({
                "session_id": "s1",
                "transcript_path": "/nonexistent",
                "cwd": "/",
            }))
            rc = main(["--data-dir", str(data_dir), "hooks", "run", "Stop"])
            assert rc == 0
            captured = capsys.readouterr()
            assert captured.out == ""
        finally:
            os.chmod(str(data_dir), 0o755)

    def test_hooks_run_hanging_write_returns_within_timeout(
        self, isolated: Path, git_cwd: Path, monkeypatch
    ):
        from taosmd.cli import main

        _setup(isolated)
        transcript = isolated / "transcript.jsonl"
        _write_transcript(transcript, _FIXTURE_LINES)

        async def _slow_ingest(*a, **kw):
            await asyncio.sleep(10)
            return {"ingested": 0, "skipped": 0}

        original = taosmd.service.ingest_batch
        taosmd.service.ingest_batch = _slow_ingest  # type: ignore[assignment]
        try:
            payload = json.dumps({
                "session_id": "s1",
                "transcript_path": str(transcript),
                "cwd": str(git_cwd),
                "hook_event_name": "Stop",
            })
            monkeypatch.setattr("sys.stdin.read", lambda: payload)

            start = time.time()
            rc = main(["--data-dir", str(isolated), "hooks", "run", "Stop"])
            elapsed = time.time() - start
            assert rc == 0
            assert elapsed < 6
        finally:
            taosmd.service.ingest_batch = original  # type: ignore[assignment]
