"""Tests for taosmd.api — top-level ingest()/search() entry points."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

import taosmd
from taosmd import api as taosmd_api


def _patch_embedder(stores: dict) -> None:
    """Override the vector store's embed() so tests don't need ONNX/QMD.

    Returns a deterministic 8-dim hash-based vector — same input → same vector,
    different inputs → different vectors. Good enough for the integration tests
    here which only need search to find a row whose text matches the query.
    """
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Each test gets its own data dir + a clean stores cache."""
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    yield data_dir
    # Force-close cached stores so SQLite handles release before tmp cleanup.
    for stores in list(taosmd_api._stores_cache.values()):
        for store in (stores.get("archive"), stores.get("vector"), stores.get("kg")):
            if store and hasattr(store, "close"):
                try:
                    asyncio.run(store.close())
                except Exception:
                    pass


def _setup_stores(data_dir: Path):
    """Init the cached stores and patch the embedder."""
    stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)
    return stores


def test_top_level_exports_exist():
    """Regression: agent-rules.md calls taosmd.ingest / taosmd.search verbatim."""
    assert hasattr(taosmd, "ingest"), "agent-rules.md depends on this attribute"
    assert hasattr(taosmd, "search"), "agent-rules.md depends on this attribute"
    assert callable(taosmd.ingest)
    assert callable(taosmd.search)


def test_ingest_string_archives_and_embeds(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    result = asyncio.run(taosmd.ingest(
        "Jay decided to ship the adjacent_neighbors port today.",
        agent="test-agent",
        data_dir=str(isolated_data_dir),
    ))
    assert result["archived"] == 1
    assert result["agent"] == "test-agent"
    assert Path(result["data_dir"]).resolve() == isolated_data_dir.resolve()

    # Verify a JSONL file ended up under archive/
    archive_files = list((isolated_data_dir / "archive").rglob("*.jsonl"))
    assert archive_files, "expected at least one archive jsonl file"
    contents = archive_files[0].read_text()
    assert "adjacent_neighbors" in contents


def test_ingest_skips_empty_content(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    result = asyncio.run(taosmd.ingest(
        [{"role": "user", "content": "real message"}, {"role": "user", "content": "   "}, {"role": "user", "content": ""}],
        agent="test-agent",
        data_dir=str(isolated_data_dir),
    ))
    assert result["archived"] == 1


def test_ingest_accepts_dict_and_iterable(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    one = asyncio.run(taosmd.ingest(
        {"role": "user", "content": "single dict"},
        agent="test-agent",
        data_dir=str(isolated_data_dir),
    ))
    assert one["archived"] == 1

    many = asyncio.run(taosmd.ingest(
        [
            {"role": "user", "content": "turn one"},
            {"role": "assistant", "content": "turn two"},
            "bare string turn",
        ],
        agent="test-agent",
        data_dir=str(isolated_data_dir),
    ))
    assert many["archived"] == 3


def test_ingest_requires_agent(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    with pytest.raises(ValueError, match="agent name is required"):
        asyncio.run(taosmd.ingest("hi", agent="", data_dir=str(isolated_data_dir)))


def test_search_returns_agent_contract_shape(isolated_data_dir):
    """Hit shape must include source, timestamp, confidence per agent-rules.md."""
    _setup_stores(isolated_data_dir)
    asyncio.run(taosmd.ingest(
        "The benchmark leader is rrf_full_stack at 0.557.",
        agent="test-agent",
        data_dir=str(isolated_data_dir),
    ))
    hits = asyncio.run(taosmd.search(
        "The benchmark leader is rrf_full_stack at 0.557.",
        agent="test-agent",
        data_dir=str(isolated_data_dir),
    ))
    assert hits, "expected at least one hit for an exact-content query"
    hit = hits[0]
    assert set(hit.keys()) >= {"text", "source", "timestamp", "confidence", "metadata"}
    assert hit["source"] in {"vector", "kg", "archive", "catalog", "crystals"}
    assert isinstance(hit["confidence"], float)
    # The fake embedder is deterministic, so identical query and document yield
    # max similarity (cosine ≈ 1.0). Confidence should land high.
    assert hit["confidence"] > 0.6


def test_search_empty_query_returns_empty(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    hits = asyncio.run(taosmd.search("", agent="test-agent", data_dir=str(isolated_data_dir)))
    assert hits == []


def test_search_requires_agent(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    with pytest.raises(ValueError, match="agent name is required"):
        asyncio.run(taosmd.search("anything", agent="", data_dir=str(isolated_data_dir)))


def test_data_dir_resolution_env_var(monkeypatch, tmp_path):
    """TAOSMD_DATA_DIR is honoured when no explicit data_dir is passed."""
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(tmp_path))
    assert taosmd_api._resolve_data_dir(None) == str(tmp_path)


def test_data_dir_resolution_default(monkeypatch):
    monkeypatch.delenv("TAOSMD_DATA_DIR", raising=False)
    assert taosmd_api._resolve_data_dir(None) == os.path.expanduser("~/.taosmd")


def test_config_json_overrides_embed_mode(isolated_data_dir, monkeypatch):
    """A config.json written by auto_setup steers embed_mode."""
    config = {"vector_memory": {"embed_mode": "qmd", "hybrid_search": True}}
    (isolated_data_dir / "config.json").write_text(json.dumps(config))
    loaded = taosmd_api._load_config(str(isolated_data_dir))
    assert loaded["vector_memory"]["embed_mode"] == "qmd"


def test_format_hit_prefers_similarity_over_source_score():
    """Vector hits should report similarity (cosine) as confidence, not source_score (which is the same here but the helper should pick from metadata first)."""
    hit = {
        "text": "x",
        "source": "vector",
        "source_id": "1",
        "rank": 0,
        "source_score": 0.9,
        "metadata": {
            "id": 1,
            "similarity": 0.85,
            "metadata": {"position": 7, "timestamp": 1700000000},
            "created_at": 1234.0,
        },
    }
    formatted = taosmd_api._format_hit(hit)
    assert formatted["confidence"] == 0.85
    assert formatted["source"] == "vector"
    assert formatted["timestamp"] == 1700000000
    assert formatted["metadata"] == {"position": 7, "timestamp": 1700000000}


def test_format_hit_falls_back_to_source_score():
    """Non-vector hits without similarity should use source_score as confidence."""
    hit = {
        "text": "kg fact",
        "source": "kg",
        "source_id": "person:alice",
        "rank": 0,
        "source_score": 0.95,
        "metadata": {"confidence": 0.95},
    }
    formatted = taosmd_api._format_hit(hit)
    assert formatted["confidence"] == 0.95
    assert formatted["source"] == "kg"


# ---------------------------------------------------------------------------
# ingest_batch + mode="bm25" search (#25 user-memory contract)
# ---------------------------------------------------------------------------

def _batch_items():
    return [
        {"text": "Reverse a list in Python with list.reverse() or slicing.",
         "id": "hash-py-reverse",
         "metadata": {"collection": "snippets", "title": "Python list reverse"}},
        {"text": "The quarterly planning meeting moved to Thursday afternoon.",
         "id": "hash-meeting",
         "metadata": {"collection": "notes", "title": "Planning meeting"}},
    ]


def test_ingest_batch_ingests_and_dedups(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    first = asyncio.run(taosmd.ingest_batch(
        _batch_items(), agent="user-memory", data_dir=str(isolated_data_dir),
    ))
    assert first["ingested"] == 2
    assert first["skipped"] == 0

    # Re-importing the same batch is idempotent: everything skips on id.
    again = asyncio.run(taosmd.ingest_batch(
        _batch_items(), agent="user-memory", data_dir=str(isolated_data_dir),
    ))
    assert again["ingested"] == 0
    assert again["skipped"] == 2

    # A mixed batch only ingests the novel item; in-batch repeats also skip.
    mixed = _batch_items() + [
        {"text": "Fresh chunk with no prior hash.", "id": "hash-fresh"},
        {"text": "Fresh chunk with no prior hash.", "id": "hash-fresh"},
    ]
    third = asyncio.run(taosmd.ingest_batch(
        mixed, agent="user-memory", data_dir=str(isolated_data_dir),
    ))
    assert third["ingested"] == 1
    assert third["skipped"] == 3


def test_ingest_batch_skips_empty_text_counts(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    result = asyncio.run(taosmd.ingest_batch(
        [{"text": "   "}, {"text": "real content", "id": "h1"}],
        agent="user-memory",
        data_dir=str(isolated_data_dir),
    ))
    assert result["ingested"] == 1
    assert result["skipped"] == 1


def test_ingest_batch_validates_before_writing(isolated_data_dir):
    stores = _setup_stores(isolated_data_dir)
    with pytest.raises(ValueError, match="items"):
        asyncio.run(taosmd.ingest_batch(
            "not-a-list", agent="user-memory", data_dir=str(isolated_data_dir),
        ))
    with pytest.raises(ValueError, match=r"items\[1\]\.text"):
        asyncio.run(taosmd.ingest_batch(
            [{"text": "ok", "id": "h1"}, {"id": "h2"}],
            agent="user-memory",
            data_dir=str(isolated_data_dir),
        ))
    # Fail-fast validation: the valid first item must NOT have been written.
    assert stores["vector"].existing_source_ids() == set()
    with pytest.raises(ValueError, match="agent name is required"):
        asyncio.run(taosmd.ingest_batch([], agent="", data_dir=str(isolated_data_dir)))


def test_search_mode_bm25_skips_embedding(isolated_data_dir):
    """mode="bm25" must never call embed() and must return the contract shape."""
    stores = _setup_stores(isolated_data_dir)
    asyncio.run(taosmd.ingest_batch(
        _batch_items(), agent="user-memory", data_dir=str(isolated_data_dir),
    ))

    async def _explode(text: str, task: str = "search_query") -> list[float]:
        raise AssertionError("embed() must not be called on the bm25 path")

    stores["vector"].embed = _explode  # type: ignore[assignment]

    hits = asyncio.run(taosmd.search(
        "planning meeting Thursday",
        agent="user-memory",
        mode="bm25",
        data_dir=str(isolated_data_dir),
    ))
    assert hits, "expected a BM25 hit for overlapping keywords"
    hit = hits[0]
    assert set(hit.keys()) >= {"text", "source", "timestamp", "confidence", "metadata"}
    assert "meeting" in hit["text"]
    assert hit["source"] == "vector"
    assert 0.0 < hit["confidence"] <= 1.0
    # User metadata (collection/title/source_id) survives the round trip.
    assert hit["metadata"].get("collection") == "notes"
    assert hit["metadata"].get("source_id") == "hash-meeting"

    # Zero term overlap -> no hits, not arbitrary padding.
    misses = asyncio.run(taosmd.search(
        "zzqx unrelated",
        agent="user-memory",
        mode="bm25",
        data_dir=str(isolated_data_dir),
    ))
    assert misses == []


def test_search_mode_bm25_python_fallback(isolated_data_dir, monkeypatch):
    """With bm25s unavailable the pure-Python BM25 must serve the same path."""
    import sys

    _setup_stores(isolated_data_dir)
    asyncio.run(taosmd.ingest_batch(
        _batch_items(), agent="user-memory", data_dir=str(isolated_data_dir),
    ))
    monkeypatch.setitem(sys.modules, "bm25s", None)  # forces ImportError

    hits = asyncio.run(taosmd.search(
        "reverse a Python list",
        agent="user-memory",
        mode="bm25",
        data_dir=str(isolated_data_dir),
    ))
    assert hits, "pure-Python BM25 fallback returned no hits"
    assert "reverse" in hits[0]["text"].lower()
    assert 0.0 < hits[0]["confidence"] <= 1.0


def test_search_rejects_unknown_mode(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    with pytest.raises(ValueError, match="unsupported search mode"):
        asyncio.run(taosmd.search(
            "anything", agent="a", mode="vector9000", data_dir=str(isolated_data_dir),
        ))


def test_bm25_python_rank_orders_by_relevance():
    from taosmd.vector_memory import _bm25_python_rank

    texts = [
        "the cat sat on the mat",
        "dogs chase cats around the garden",
        "a completely unrelated sentence about tax law",
    ]
    ranked = _bm25_python_rank("cat mat", texts)
    assert ranked[0][0] == 0, "exact-term doc should rank first"
    assert ranked[0][1] > 0.0
    assert ranked[-1][1] == 0.0, "no-overlap doc should score zero"
