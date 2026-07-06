"""Zero-loss integrity: ingest must surface vector-write failures, and
ingest_batch must carry the same archive provenance as single ingest.

When the embedder is down, VectorMemory.add() returns -1 (the internal
backends catch embed errors and return an empty vector). Archive-first means
the turn is still safely recorded, so ingest must NOT raise, but the caller
has to be able to see the degradation in the result so the gap can be
repaired later via reconcile().

All tests are hermetic: each uses its own tmp_path data dir and a
deterministic fake embedder so ONNX/QMD are never required.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

import taosmd
from taosmd import api as taosmd_api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_embedder(stores: dict) -> None:
    """Replace embed() with a stable hash-based 8-dim vector (no ONNX/QMD)."""
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


def _break_embedder(stores: dict) -> None:
    """Simulate an embedder outage.

    The real backends (_embed_onnx / _embed_local / _embed_qmd) catch their
    own exceptions and return [] — that empty vector is exactly what add()
    sees when the embedder is down, so the injection point is embed()
    returning [].
    """
    vmem = stores["vector"]

    async def _dead_embed(text: str, task: str = "search_document") -> list[float]:
        return []

    vmem.embed = _dead_embed  # type: ignore[assignment]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Isolated data dir + clean stores cache."""
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    yield data_dir
    # Release SQLite handles before pytest removes tmp_path.
    for stores in list(taosmd_api._stores_cache.values()):
        for store in (stores.get("archive"), stores.get("vector"),
                      stores.get("kg"), stores.get("claims")):
            if store and hasattr(store, "close"):
                try:
                    asyncio.run(store.close())
                except Exception:
                    pass


def _vector_rows(stores: dict) -> list[tuple[str, dict]]:
    """All (text, metadata) rows in the vector store, oldest first."""
    rows = stores["vector"]._conn.execute(
        "SELECT text, metadata_json FROM vector_memory ORDER BY rowid"
    ).fetchall()
    return [(r[0], json.loads(r[1])) for r in rows]


# ---------------------------------------------------------------------------
# C1: embedder-down ingest must be visible, not silent
# ---------------------------------------------------------------------------

def test_ingest_surfaces_vector_failures_when_embedder_down(isolated, caplog):
    stores = asyncio.run(taosmd_api._ensure_stores(str(isolated)))
    _break_embedder(stores)

    with caplog.at_level(logging.WARNING, logger="taosmd.api"):
        result = asyncio.run(taosmd.ingest(
            ["turn one survives", "turn two survives"],
            agent="degraded-agent",
            data_dir=str(isolated),
        ))

    # Archive-first: the turns are still archived, ingest does not raise.
    assert result["archived"] == 2
    from taosmd.archive import EVENT_CONVERSATION
    rows = asyncio.run(stores["archive"].query(
        event_type=EVENT_CONVERSATION, agent_name="degraded-agent"))
    assert len(rows) == 2

    # But the degradation must be visible in the result...
    assert result["vector_failures"] == 2
    assert result["degraded"] is True

    # ...and in the log, naming the embed backend.
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "expected a WARNING when vector writes fail"
    backend = stores["vector"]._embedder_identity()
    assert any(backend in r.getMessage() for r in warnings), (
        f"warning should name the embed backend {backend!r}")

    # Nothing landed in the vector store: the gap reconcile() must report.
    assert _vector_rows(stores) == []
    _patch_embedder(stores)  # embedder comes back
    check = asyncio.run(taosmd_api.reconcile(
        agent="degraded-agent", data_dir=str(isolated), repair=False))
    assert check["missing"] == 2


def test_ingest_healthy_path_reports_no_degradation(isolated):
    stores = asyncio.run(taosmd_api._ensure_stores(str(isolated)))
    _patch_embedder(stores)

    result = asyncio.run(taosmd.ingest(
        "all systems nominal",
        agent="healthy-agent",
        data_dir=str(isolated),
    ))
    assert result["archived"] == 1
    assert "vector_failures" not in result
    assert "degraded" not in result


def test_ingest_batch_surfaces_vector_failures_when_embedder_down(isolated, caplog):
    stores = asyncio.run(taosmd_api._ensure_stores(str(isolated)))
    _break_embedder(stores)

    with caplog.at_level(logging.WARNING, logger="taosmd.api"):
        result = asyncio.run(taosmd.ingest_batch(
            [{"text": "batch item one"}, {"text": "batch item two"}],
            agent="degraded-batch-agent",
            data_dir=str(isolated),
        ))

    assert result["ingested"] == 2
    assert result["vector_failures"] == 2
    assert result["degraded"] is True
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "expected a WARNING when batch vector writes fail"

    # Batch items are archived (zero-loss) even though embedding failed.
    from taosmd.archive import EVENT_CONVERSATION
    rows = asyncio.run(stores["archive"].query(
        event_type=EVENT_CONVERSATION, agent_name="degraded-batch-agent"))
    assert len(rows) == 2
    assert _vector_rows(stores) == []


def test_ingest_batch_healthy_path_reports_no_degradation(isolated):
    stores = asyncio.run(taosmd_api._ensure_stores(str(isolated)))
    _patch_embedder(stores)

    result = asyncio.run(taosmd.ingest_batch(
        [{"text": "fine item"}],
        agent="healthy-batch-agent",
        data_dir=str(isolated),
    ))
    assert result["ingested"] == 1
    assert "vector_failures" not in result
    assert "degraded" not in result


# ---------------------------------------------------------------------------
# M1: ingest_batch must carry archive provenance like single ingest
# ---------------------------------------------------------------------------

def test_ingest_batch_links_vector_rows_to_archive_spans(isolated):
    stores = asyncio.run(taosmd_api._ensure_stores(str(isolated)))
    _patch_embedder(stores)
    agent = "linkage-agent"

    result = asyncio.run(taosmd.ingest_batch(
        [
            {"text": "My name is Jay", "id": "h1"},
            {"text": "My favourite editor is vim", "id": "h2"},
        ],
        agent=agent,
        data_dir=str(isolated),
    ))
    assert result["ingested"] == 2

    from taosmd.archive import EVENT_CONVERSATION
    archive_rows = asyncio.run(stores["archive"].query(
        event_type=EVENT_CONVERSATION, agent_name=agent))
    span_ids = {r["id"] for r in archive_rows}
    assert len(span_ids) == 2

    vec = _vector_rows(stores)
    assert len(vec) == 2
    for text, meta in vec:
        assert meta.get("archive_span_id") in span_ids, (
            f"vector row {text!r} must carry the archive span it came from")


def test_ingest_batch_extracts_claims_like_single_ingest(isolated):
    stores = asyncio.run(taosmd_api._ensure_stores(str(isolated)))
    _patch_embedder(stores)
    agent = "claims-parity-agent"

    # "My name is ..." matches the regex fact extractor, so single ingest
    # produces a claim for it; batch ingest of the same text must too.
    asyncio.run(taosmd.ingest_batch(
        [{"text": "My name is Jay", "id": "c1"}],
        agent=agent,
        data_dir=str(isolated),
    ))

    claims = asyncio.run(stores["claims"].pull_unverified())
    assert claims, "batch ingest must extract claims like single ingest does"

    from taosmd.archive import EVENT_CONVERSATION
    archive_rows = asyncio.run(stores["archive"].query(
        event_type=EVENT_CONVERSATION, agent_name=agent))
    span_ids = {r["id"] for r in archive_rows}
    for claim in claims:
        backing = set(claim["archive_span_ids"])
        assert backing and backing <= span_ids, (
            "claim must be backed by the batch item's archive span")


def test_ingest_batch_prevalidation_still_rejects_before_writing(isolated):
    """Regression guard: the linkage fix must not change batch pre-validation."""
    stores = asyncio.run(taosmd_api._ensure_stores(str(isolated)))
    _patch_embedder(stores)

    with pytest.raises(ValueError):
        asyncio.run(taosmd.ingest_batch(
            [{"text": "good"}, {"text": 42}],
            agent="prevalidate-agent",
            data_dir=str(isolated),
        ))
    # Nothing written: validation runs before any write.
    from taosmd.archive import EVENT_CONVERSATION
    rows = asyncio.run(stores["archive"].query(
        event_type=EVENT_CONVERSATION, agent_name="prevalidate-agent"))
    assert rows == []
    assert _vector_rows(stores) == []
