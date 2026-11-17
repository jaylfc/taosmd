"""Tests for archive.record redaction: caller dict must not be mutated, and
ALL string leaves in data (including nested, tool, result, error) must be
redacted in archive_index.data_json and archive_fts.
"""

from __future__ import annotations

import copy
import json
import threading

import pytest

from taosmd.archive import ArchiveStore


# GitHub PAT pattern: ghp_ followed by 36 chars
TOKEN = "ghp_" + "A" * 36


@pytest.mark.asyncio
async def test_record_does_not_mutate_caller_dict(tmp_path):
    """The caller's data dict must not be mutated by record()."""
    store = ArchiveStore(
        archive_dir=tmp_path / "archive",
        index_path=tmp_path / "archive-index.db",
    )
    await store.init()

    data = {"content": "key " + TOKEN}
    original = copy.deepcopy(data)

    await store.record("test_event", data)

    # Caller's dict must be unchanged
    assert data == original, "Caller's dict was mutated"

    # But the stored version should be redacted
    row = store._conn.execute(
        "SELECT data_json FROM archive_index WHERE id = (SELECT MAX(id) FROM archive_index)"
    ).fetchone()
    assert row is not None
    assert TOKEN not in row["data_json"], "Token found in stored data_json"
    assert "[REDACTED:github_pat]" in row["data_json"], "Redaction placeholder missing"

    await store.close()


@pytest.mark.asyncio
async def test_result_and_error_redacted_in_index(tmp_path):
    """Secrets in tool, result, error keys must be redacted in index and FTS."""
    store = ArchiveStore(
        archive_dir=tmp_path / "archive",
        index_path=tmp_path / "archive-index.db",
    )
    await store.init()

    data = {"result": "out " + TOKEN, "error": TOKEN, "tool": TOKEN}
    await store.record("tool_call", data)

    row = store._conn.execute(
        "SELECT data_json FROM archive_index WHERE id = (SELECT MAX(id) FROM archive_index)"
    ).fetchone()
    assert row is not None
    data_json = row["data_json"]
    assert TOKEN not in data_json, f"Token found in data_json: {data_json}"
    assert "[REDACTED:github_pat]" in data_json, "Redaction placeholder missing from data_json"

    # Also check FTS content
    fts_row = store._conn.execute(
        "SELECT content FROM archive_fts WHERE rowid = (SELECT MAX(rowid) FROM archive_fts)"
    ).fetchone()
    assert fts_row is not None
    fts_content = fts_row["content"]
    assert TOKEN not in fts_content, f"Token found in FTS content: {fts_content}"
    assert "[REDACTED:github_pat]" in fts_content, "Redaction placeholder missing from FTS"

    await store.close()


@pytest.mark.asyncio
async def test_nested_leaf_redacted_in_index(tmp_path):
    """Secrets in nested structures (lists of dicts) must be redacted."""
    store = ArchiveStore(
        archive_dir=tmp_path / "archive",
        index_path=tmp_path / "archive-index.db",
    )
    await store.init()

    data = {"messages": [{"content": TOKEN}]}
    await store.record("test_event", data)

    row = store._conn.execute(
        "SELECT data_json FROM archive_index WHERE id = (SELECT MAX(id) FROM archive_index)"
    ).fetchone()
    assert row is not None
    data_json = row["data_json"]
    assert TOKEN not in data_json, f"Token found in nested data_json: {data_json}"
    assert "[REDACTED:github_pat]" in data_json, "Redaction placeholder missing from nested data_json"

    await store.close()


@pytest.mark.asyncio
async def test_uncopyable_value_still_archived(tmp_path):
    """A value deepcopy cannot copy must still be archived (raw write must not fail).

    Regression: record() used to run copy.deepcopy(data) before the JSONL write.
    A value like threading.Lock() cannot be deep-copied, so the raw record was
    lost. The caller's dict must still never change, and the JSONL line must be
    byte-identical to master's for the same input (json.dumps(event, default=str)).
    """
    store = ArchiveStore(
        archive_dir=tmp_path / "archive",
        index_path=tmp_path / "archive-index.db",
    )
    await store.init()

    lock = threading.Lock()
    data = {"content": "hello", "handle": lock}

    row_id = await store.record("test_event", data)

    assert row_id > 0

    # Caller's dict must still hold the same Lock object (identity preserved).
    assert data["handle"] is lock

    # Read the day's JSONL file (the store's archive_dir, the only *.jsonl file).
    jsonl_files = list(tmp_path.glob("**/*.jsonl"))
    assert len(jsonl_files) == 1, f"expected exactly one .jsonl file, got {jsonl_files}"
    lines = jsonl_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, f"expected exactly one line, got {len(lines)}"
    record = json.loads(lines[0])
    assert record["data"]["content"] == "hello"
    assert record["data"]["handle"].startswith("<unlocked _thread.lock")

    await store.close()


@pytest.mark.asyncio
async def test_non_str_leaf_str_is_redacted(tmp_path):
    """A non-str leaf whose str() contains a token must be redacted in data_json.

    _redact_leaves leaves non-str, non-container values unchanged, and the
    index INSERT serialised them with json.dumps(derived, default=str), so
    str(obj) landed in archive_index.data_json UNSCANNED. Build the derived
    copy from a JSON-plain snapshot so every leaf that becomes a string is
    scanned.
    """
    store = ArchiveStore(
        archive_dir=tmp_path / "archive",
        index_path=tmp_path / "archive-index.db",
    )
    await store.init()

    class Tok:
        def __str__(self) -> str:
            return "tok " + TOKEN

    data = {"obj": Tok()}

    await store.record("test_event", data)

    row = store._conn.execute(
        "SELECT data_json FROM archive_index WHERE id = (SELECT MAX(id) FROM archive_index)"
    ).fetchone()
    assert row is not None
    data_json = row["data_json"]
    assert TOKEN not in data_json, f"Token leaked in data_json: {data_json}"
    assert "[REDACTED:github_pat]" in data_json, "Redaction placeholder missing from data_json"

    await store.close()