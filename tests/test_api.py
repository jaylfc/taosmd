"""Tests for taosmd.api — top-level ingest()/search() entry points."""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import logging
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
    assert formatted["metadata"] == {"position": 7, "timestamp": 1700000000, "as_of": 1700000000.0, "is_current": True}


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
# Collection doc-currency metadata on the hit envelope
# ---------------------------------------------------------------------------

def _collection_hit(doc_id=None, version=None, review_by=None,
                    indexed_at=1700000000.0, hidden_by=None, archive_span_id=42):
    """Build a hit matching the real ``ingest_folder`` envelope shape.

    ingest_batch() wraps user metadata under an outer envelope that also
    carries ``archive_span_id`` (and, for superseded rows, ``hidden_by``).
    The doc-currency keys (``doc_id``, ``version``, ``review_by``,
    ``indexed_at``) live in the *inner* user-metadata dict, exactly as
    _parse_front_matter + ingest_folder store them. This fixture mirrors
    that shape so tests exercise the real data layout, not a hand-invented
    one.
    """
    user_md: dict = {
        "collection_id": "col-docs",
        "file_path": "docs/intro.md",
        "source": "collection",
        "chunk_index": 0,
        "file_hash": "abc123",
        "indexed_at": indexed_at,
    }
    if doc_id is not None:
        user_md["doc_id"] = doc_id
    if version is not None:
        user_md["version"] = version
    if review_by is not None:
        user_md["review_by"] = review_by
    outer: dict = {
        "archive_span_id": archive_span_id,
        "metadata": user_md,
        "created_at": indexed_at,
    }
    if hidden_by is not None:
        outer["hidden_by"] = hidden_by
    return {
        "text": "test content",
        "source": "vector",
        "source_score": 0.5,
        "metadata": outer,
    }


def test_format_hit_includes_front_matter_metadata():
    """_format_hit passes through doc_id, version, review_by from user metadata.

    The doc keys live in the *inner* user-metadata dict (that is how
    ingest_folder writes them), so _format_hit must read them after the
    unwrap, not from the outer envelope before it.
    """
    hit = _collection_hit(doc_id="doc-abc123", version=5, review_by="2020-01-01")
    formatted = taosmd_api._format_hit(hit)
    assert formatted["metadata"]["doc_id"] == "doc-abc123"
    assert formatted["metadata"]["version"] == 5
    assert formatted["metadata"]["review_by"] == "2020-01-01"
    assert formatted["metadata"]["is_current"] is True
    assert isinstance(formatted["metadata"]["as_of"], float)
    assert formatted["metadata"]["as_of"] == 1700000000.0
    assert formatted["metadata"]["is_past_review"] is True


def test_format_hit_no_front_matter_no_doc_id():
    """_format_hit without doc_id/version in user metadata leaves those keys absent."""
    hit = _collection_hit()
    formatted = taosmd_api._format_hit(hit)
    assert "doc_id" not in formatted["metadata"]
    assert "version" not in formatted["metadata"]
    assert "is_past_review" not in formatted["metadata"]
    assert "review_by" not in formatted["metadata"]
    # is_current and as_of should always be present
    assert formatted["metadata"]["is_current"] is True
    assert isinstance(formatted["metadata"]["as_of"], float)


def test_format_hit_review_by_in_future_is_not_past_review():
    """_format_hit is_past_review is false when review_by is after today."""
    hit = _collection_hit(doc_id="doc-x", version=1, review_by="2099-12-31")
    formatted = taosmd_api._format_hit(hit)
    assert formatted["metadata"]["is_past_review"] is False
    assert formatted["metadata"]["review_by"] == "2099-12-31"


def test_format_hit_superseded_row_is_not_current():
    """A row carrying ``hidden_by`` is not current and exposes superseded_by."""
    hit = _collection_hit(
        doc_id="doc-x", version=1, review_by="2020-01-01",
        hidden_by="collection-reindex:12345",
    )
    formatted = taosmd_api._format_hit(hit)
    assert formatted["metadata"]["is_current"] is False
    assert formatted["metadata"]["superseded_by"] == "collection-reindex:12345"


def test_format_hit_non_dict_user_metadata_does_not_crash():
    """Non-dict user metadata degrades, never raises."""
    hit = {
        "text": "ok",
        "source": "vector",
        "metadata": "not-a-dict",
    }
    formatted = taosmd_api._format_hit(hit)
    assert formatted["metadata"] == {}


def test_format_hit_no_assert_crash_under_optimization():
    """No bare ``assert`` in _format_hit so -O doesn't gut it."""
    tree = ast.parse(inspect.getsource(taosmd_api._format_hit))
    assert not any(isinstance(n, ast.Assert) for n in ast.walk(tree)), (
        "_format_hit must not use bare assert (stripped under -O)"
    )


def test_format_hit_as_of_is_float_without_indexed_at():
    """as_of is always a float even without indexed_at."""
    hit = {
        "text": "ok",
        "source": "vector",
        "metadata": {
            "agent": "test",
            "metadata": {"file_path": "docs/intro.md"},
            "created_at": 1700000000.0,
        },
    }
    formatted = taosmd_api._format_hit(hit)
    assert isinstance(formatted["metadata"]["as_of"], float)


# ---------------------------------------------------------------------------
# Contract test: public search() output shape carries doc-currency fields
# ---------------------------------------------------------------------------

def test_search_hit_metadata_always_has_doc_currency_fields(isolated_data_dir):
    """Every hit returned by search() must carry is_current and as_of in
    its metadata, regardless of the source path (semantic or BM25).

    This is the deliberate contract test for the output-shape change:
    is_current and as_of are now attached on every path.
    """
    _setup_stores(isolated_data_dir)
    asyncio.run(taosmd.ingest_batch(
        [{"text": "The quarterly review moved to Friday morning.",
          "id": "hash-contract",
          "metadata": {"file_path": "notes/review.md"}}],
        agent="contract-agent", data_dir=str(isolated_data_dir),
    ))
    # Semantic path
    hits = asyncio.run(taosmd.search(
        "The quarterly review moved to Friday morning.",
        agent="contract-agent", data_dir=str(isolated_data_dir),
    ))
    assert hits, "expected a semantic hit"
    for h in hits:
        assert "is_current" in h["metadata"]
        assert "as_of" in h["metadata"]
        assert isinstance(h["metadata"]["as_of"], float)
    # BM25 path
    bm25_hits = asyncio.run(taosmd.search(
        "quarterly review", agent="contract-agent", mode="bm25",
        data_dir=str(isolated_data_dir),
    ))
    assert bm25_hits
    for h in bm25_hits:
        assert "is_current" in h["metadata"]
        assert "as_of" in h["metadata"]
        assert isinstance(h["metadata"]["as_of"], float)


# ---------------------------------------------------------------------------
# Coercion regression: ingest_batch -> search with malformed doc-currency
# metadata.  These are RED against the unguarded tree (exec/tsk-x6ph7n,
# which crashes on ISO-string timestamps and non-string review_by) and GREEN
# on this branch where as_of/review_by are coerced defensively.
# ---------------------------------------------------------------------------

def _coerce_and_search(data_dir, metadata, text, agent="coerce-agent",
                       query=None):
    """Ingest one item with *metadata* via ingest_batch, then search for it.

    Returns the search hits.  On the unguarded tree the search() call itself
    raises (ValueError for a non-coercible as_of, TypeError for a non-string
    review_by), so callers that expect hits on the fixed tree will see a
    hard failure here on the unfixed tree.
    """
    _setup_stores(data_dir)
    asyncio.run(taosmd.ingest_batch(
        [{"text": text, "id": f"coerce-{hash(text) & 0xFFFF:04x}",
          "metadata": dict(metadata)}],
        agent=agent, data_dir=str(data_dir),
    ))
    q = query or text
    return asyncio.run(taosmd.search(
        q, agent=agent, mode="bm25", data_dir=str(data_dir),
    ))


def test_coerce_iso_string_timestamp_returns_hits(isolated_data_dir):
    """An ISO-8601 string timestamp must degrade to as_of=0.0, not raise.

    On the unfixed tree float("2020-01-01T00:00:00Z") raises ValueError at
    api.py and search() propagates it as an error instead of returning hits.
    """
    hits = _coerce_and_search(
        isolated_data_dir, {"timestamp": "2020-01-01T00:00:00Z"},
        "The unique token zqxklbm-coerce-iso-timestamp was ingested.",
        query="zqxklbm",
    )
    assert hits, "expected at least one hit for an ISO-string timestamp"
    assert hits[0]["metadata"]["as_of"] == 0.0


def test_coerce_int_review_by_returns_hits(isolated_data_dir):
    """An int review_by must degrade to is_past_review=False, not raise.

    On the unfixed tree 2020 < "2026-08-18" raises TypeError at api.py.
    """
    hits = _coerce_and_search(
        isolated_data_dir, {"review_by": 2020},
        "The unique token zqxklbm-coerce-int-review was ingested.",
        query="zqxklbm",
    )
    assert hits, "expected at least one hit for an int review_by"
    assert hits[0]["metadata"]["is_past_review"] is False
    assert hits[0]["metadata"]["review_by"] == 2020


def test_coerce_list_review_by_returns_hits(isolated_data_dir):
    """A list review_by must degrade to is_past_review=False, not raise.

    On the unfixed tree [2020, 1, 1] < "..." raises TypeError at api.py.
    """
    hits = _coerce_and_search(
        isolated_data_dir, {"review_by": [2020, 1, 1]},
        "The unique token zqxklbm-coerce-list-review was ingested.",
        query="zqxklbm",
    )
    assert hits, "expected at least one hit for a list review_by"
    assert hits[0]["metadata"]["is_past_review"] is False


def test_coerce_str_review_by_overdue_is_past_review(isolated_data_dir):
    """A string review_by in the past yields is_past_review=True on both trees."""
    hits = _coerce_and_search(
        isolated_data_dir, {"review_by": "2020-01-01"},
        "The unique token zqxklbm-coerce-str-review was ingested.",
        query="zqxklbm",
    )
    assert hits, "expected at least one hit"
    assert hits[0]["metadata"]["is_past_review"] is True


def test_coerce_plain_metadata_returns_hits(isolated_data_dir):
    """Plain metadata without doc-currency keys simply returns hits."""
    hits = _coerce_and_search(
        isolated_data_dir, {"category": "notes"},
        "The unique token zqxklbm-coerce-plain was ingested.",
        query="zqxklbm",
    )
    assert hits, "expected at least one hit for plain metadata"
    assert "is_past_review" not in hits[0]["metadata"]
    assert hits[0]["metadata"]["is_current"] is True


def test_coerce_non_string_review_by_warns(isolated_data_dir, caplog):
    """Non-string review_by must log a warning (symmetric with as_of coercion).

    A non-coercible as_of logs; a non-string review_by must also log so
    malformed input is observable, not silently reported as "reviewed and
    current".
    """
    with caplog.at_level(logging.WARNING, logger="taosmd.api"):
        hits = _coerce_and_search(
            isolated_data_dir, {"review_by": 2020},
            "The unique token zqxklbm-coerce-warns was ingested.",
            query="zqxklbm",
        )
    assert hits, "expected at least one hit for an int review_by"
    assert hits[0]["metadata"]["is_past_review"] is False
    assert any("review_by" in r.message and "not a string" in r.message
               for r in caplog.records), (
        f"expected a review_by warning, got: {[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Runtime controls overlay the recipe (dashboard / PUT /controls levers)
# ---------------------------------------------------------------------------

def test_apply_runtime_overrides_noop_keeps_recipe():
    rc = {"reranker": "bge-v2-m3", "fusion": "rrf", "adjacent_neighbors": 2}
    out = taosmd_api._apply_runtime_overrides(rc, {})
    assert out == rc
    assert out is not rc  # never mutate the recipe object


def test_apply_runtime_overrides_reranker_off_maps_to_none():
    rc = {"reranker": "bge-v2-m3", "fusion": "rrf", "adjacent_neighbors": 2}
    out = taosmd_api._apply_runtime_overrides(rc, {"reranker": "off"})
    assert out["reranker"] == "none"
    assert rc["reranker"] == "bge-v2-m3"  # original untouched


def test_apply_runtime_overrides_reranker_on_over_recipe_none():
    rc = {"reranker": "none", "fusion": "rrf", "adjacent_neighbors": 2}
    out = taosmd_api._apply_runtime_overrides(rc, {"reranker": "bge-v2-m3"})
    assert out["reranker"] == "bge-v2-m3"


def test_apply_runtime_overrides_fusion_and_adjacency():
    rc = {"reranker": "none", "fusion": "boost", "adjacent_neighbors": 0}
    out = taosmd_api._apply_runtime_overrides(
        rc, {"fusion": "mem0_additive", "adjacent_turns": 4})
    assert out["fusion"] == "mem0_additive"
    assert out["adjacent_neighbors"] == 4


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


# ---------------------------------------------------------------------------
# Claims-gate provenance must survive _format_hit's metadata unwrap
# ---------------------------------------------------------------------------

def _batch_row_span(stores, text: str) -> int:
    """Read a batch row's archive_span_id straight off the raw vector row."""
    rows = stores["vector"]._conn.execute(
        "SELECT text, metadata_json FROM vector_memory"
    ).fetchall()
    for r in rows:
        if r["text"] == text:
            meta = json.loads(r["metadata_json"])
            span = meta.get("archive_span_id")
            assert isinstance(span, int), "batch row lost its archive_span_id at write time"
            return span
    raise AssertionError(f"no vector row stored for {text!r}")


def test_semantic_search_batch_row_keeps_archive_span_id(isolated_data_dir):
    """Regression: _format_hit's unwrap-to-innermost loop must not strip the
    provenance envelope. Batch rows are double-wrapped on the semantic path
    (retrieval envelope -> row meta -> user metadata); descending all the way
    down dropped archive_span_id, blinding the prefer_verified claims gate.
    The formatted hit must expose BOTH the user metadata (e.g. file_path for
    collection hits) and the archive span the gate reads."""
    stores = _setup_stores(isolated_data_dir)
    asyncio.run(taosmd.ingest_batch(
        [{"text": "The rack in bay four is painted teal.",
          "id": "hash-rack",
          "metadata": {"file_path": "docs/rack.md"}}],
        agent="batch-agent", data_dir=str(isolated_data_dir),
    ))
    span = _batch_row_span(stores, "The rack in bay four is painted teal.")

    hits = asyncio.run(taosmd.search(
        "The rack in bay four is painted teal.",
        agent="batch-agent",
        prefer_verified="off",
        data_dir=str(isolated_data_dir),
    ))
    assert hits, "expected a semantic hit for an exact-content query"
    top = hits[0]
    assert top["metadata"].get("file_path") == "docs/rack.md"   # client-facing win stays
    assert top["metadata"].get("archive_span_id") == span       # gate input preserved


def test_bm25_search_batch_row_keeps_archive_span_id(isolated_data_dir):
    """Same contract on the BM25 path: row meta is one level shallower there,
    but the formatted hit must still carry the provenance span."""
    stores = _setup_stores(isolated_data_dir)
    asyncio.run(taosmd.ingest_batch(
        [{"text": "The quarterly review moved to Friday morning.",
          "id": "hash-review",
          "metadata": {"file_path": "notes/review.md"}}],
        agent="batch-agent", data_dir=str(isolated_data_dir),
    ))
    span = _batch_row_span(stores, "The quarterly review moved to Friday morning.")
    hits = asyncio.run(taosmd.search(
        "quarterly review Friday",
        agent="batch-agent",
        mode="bm25",
        prefer_verified="off",
        data_dir=str(isolated_data_dir),
    ))
    assert hits
    assert hits[0]["metadata"].get("file_path") == "notes/review.md"
    assert hits[0]["metadata"].get("archive_span_id") == span


def test_prefer_verified_drops_contradicted_batch_row(isolated_data_dir):
    """Regression: with a batch row's backing claim contradicted, the
    prefer_verified gate must drop the row from semantic recall. This is the
    master behaviour the metadata unwrap regressed (gate went blind because
    the formatted hit no longer carried archive_span_id)."""
    stores = _setup_stores(isolated_data_dir)
    asyncio.run(taosmd.ingest_batch(
        [{"text": "The rack in bay four is painted teal.",
          "id": "hash-rack",
          "metadata": {"file_path": "docs/rack.md"}}],
        agent="batch-agent", data_dir=str(isolated_data_dir),
    ))
    span = _batch_row_span(stores, "The rack in bay four is painted teal.")

    # Sanity: without the gate the row is recalled.
    ungated = asyncio.run(taosmd.search(
        "The rack in bay four is painted teal.",
        agent="batch-agent", prefer_verified="off",
        data_dir=str(isolated_data_dir),
    ))
    assert any("teal" in h["text"] for h in ungated)

    # Contradict the claim backing that span; the gate must now drop the row.
    cs = stores["claims"]
    cid = asyncio.run(cs.add_claim("rack colour", [span], source_extractor="test"))
    asyncio.run(cs.set_status(cid, "contradicted", verifier_model="m", now=1.0))
    gated = asyncio.run(taosmd.search(
        "The rack in bay four is painted teal.",
        agent="batch-agent", prefer_verified="prefer_verified",
        data_dir=str(isolated_data_dir),
    ))
    assert not any("teal" in h["text"] for h in gated), (
        "contradicted-claim row survived: the claims gate is blind to batch rows"
    )


def test_search_prefer_verified_resolves_from_config():
    """Provable memory ships on by default: search()'s prefer_verified param is a
    sentinel (None) that resolves from the persisted controls at call time, and
    the resolved default is "prefer_verified" (E-018 tri-judge evidenced, a safe
    no-op until claims are verified). A per-call argument always overrides."""
    import inspect
    from taosmd import config as _cfg, controls as _ctrl
    assert inspect.signature(taosmd_api.search).parameters["prefer_verified"].default is None
    with tempfile.TemporaryDirectory() as d:
        assert _cfg.get_controls(data_dir=d)["prefer_verified"] == "prefer_verified"
    assert _ctrl.default_controls()["prefer_verified"] == "prefer_verified"


# ---------------------------------------------------------------------------
# Dashboard stats aggregator
# ---------------------------------------------------------------------------

def test_stats_shape(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    asyncio.run(taosmd.ingest(
        "Stats overview test memory.", agent="s", data_dir=str(isolated_data_dir)))
    out = asyncio.run(taosmd_api.dashboard_stats(data_dir=str(isolated_data_dir)))
    assert set(out) >= {
        "memories", "agents", "projects", "growth",
        "verification", "top_agents", "top_projects", "recent_activity",
    }
    assert out["memories"]["total"] >= 1
    assert out["agents"] >= 1
    assert isinstance(out["growth"], list)
    assert set(out["verification"]) >= {"supported", "unverified", "flagged", "hallucination_rate"}
    assert any(a["name"] == "s" for a in out["top_agents"])


def test_stats_empty_install_returns_zeros(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    out = asyncio.run(taosmd_api.dashboard_stats(data_dir=str(isolated_data_dir)))
    assert out["memories"]["total"] == 0
    assert out["agents"] == 0
    assert out["growth"] == []
    assert out["top_agents"] == []
    assert out["recent_activity"] == []


def test_list_memories_and_scoped_stats(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    asyncio.run(taosmd.ingest("User scoped memory.", agent="user", data_dir=str(isolated_data_dir)))
    asyncio.run(taosmd.ingest("Bot scoped memory.", agent="bot", data_dir=str(isolated_data_dir)))
    mems = asyncio.run(taosmd_api.list_memories(scope="user", data_dir=str(isolated_data_dir)))
    assert mems and all(m["agent"] == "user" for m in mems)
    all_stats = asyncio.run(taosmd_api.dashboard_stats(data_dir=str(isolated_data_dir)))
    user_stats = asyncio.run(taosmd_api.dashboard_stats(scope="user", data_dir=str(isolated_data_dir)))
    assert user_stats["memories"]["total"] >= 1
    assert all_stats["memories"]["total"] > user_stats["memories"]["total"]
    assert user_stats["scope"] == "user" and all_stats["scope"] == "all"


def test_stats_includes_categories(isolated_data_dir):
    _setup_stores(isolated_data_dir)
    asyncio.run(taosmd.ingest("I prefer metric units and dark mode.", agent="user", data_dir=str(isolated_data_dir)))
    asyncio.run(taosmd.ingest("Deployed the benchmark feature for the project.", agent="bot", data_dir=str(isolated_data_dir)))
    out = asyncio.run(taosmd_api.dashboard_stats(data_dir=str(isolated_data_dir)))
    assert "categories" in out and isinstance(out["categories"], list)
    names = {c["name"] for c in out["categories"]}
    assert names & {"Identity & Preferences", "Work & Learning"}
