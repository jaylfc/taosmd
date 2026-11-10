"""Tests for archive.record redaction: caller dict must not be mutated, and
ALL string leaves in data (including nested, tool, result, error) must be
redacted in archive_index.data_json and archive_fts.
"""

from __future__ import annotations

import copy

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