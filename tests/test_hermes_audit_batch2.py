"""Tests for Hermes Agent Audit batch-2 fixes.

Covers issues #94, #93, #103, #105, #108, #111.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# #94 — temporal_boost writes to rrf_score when present
# ---------------------------------------------------------------------------


def test_temporal_boost_affects_rrf_score_field():
    """Boost must land on rrf_score (not only similarity) for hybrid results."""
    from taosmd.temporal_boost import temporal_rerank

    # Simulated results from vmem.search(fusion="rrf") — both fields present.
    # Result A has higher similarity but lower rrf_score; after temporal boost
    # it should be promoted when it contains strong temporal signals.
    results = [
        {
            "id": 1,
            "text": "2024-01-15 meeting happened on Monday morning",
            "similarity": 0.90,
            "rrf_score": 0.010,
        },
        {
            "id": 2,
            "text": "general discussion about the project",
            "similarity": 0.92,
            "rrf_score": 0.050,
        },
    ]

    query = "when did the first meeting happen"
    boosted = temporal_rerank(results, query, boost_factor=0.5)

    # The temporal entry (id=1) should have received a temporal_boost marker.
    temporal_result = next(r for r in boosted if r["id"] == 1)
    assert "temporal_boost" in temporal_result
    assert temporal_result["temporal_boost"] > 0

    # rrf_score on the boosted entry must be larger than the original 0.010.
    assert temporal_result["rrf_score"] > 0.010


def test_temporal_boost_falls_back_to_similarity_when_no_rrf_score():
    """For pure-semantic results (no rrf_score), boost goes to similarity."""
    from taosmd.temporal_boost import temporal_rerank

    results = [
        {"id": 1, "text": "2023-11-14 event occurred on Tuesday afternoon", "similarity": 0.70},
        {"id": 2, "text": "unrelated text", "similarity": 0.95},
    ]
    query = "when did that event happen last month"
    boosted = temporal_rerank(results, query, boost_factor=0.4)

    temporal_result = next(r for r in boosted if r["id"] == 1)
    assert "temporal_boost" in temporal_result
    # similarity must be > 0.70 after boost
    assert temporal_result["similarity"] > 0.70
    # no rrf_score key should be injected
    assert "rrf_score" not in temporal_result


def test_temporal_boost_reorders_by_rrf_score_not_similarity():
    """When rrf_score is present, the sort key must be rrf_score."""
    from taosmd.temporal_boost import temporal_rerank

    # id=1: lower similarity, lower rrf_score — but rich temporal content.
    # id=2: higher similarity, higher rrf_score — no temporal content.
    # The boost should push id=1's rrf_score above id=2's.
    results = [
        {
            "id": 1,
            "text": "January 2024 the project launched on Friday morning 15 days after",
            "similarity": 0.60,
            "rrf_score": 0.010,
        },
        {
            "id": 2,
            "text": "no date info here at all",
            "similarity": 0.99,
            "rrf_score": 0.020,
        },
    ]
    query = "when did the project launch in January"
    boosted = temporal_rerank(results, query, boost_factor=1.0)

    # id=1 must be ranked first after a large boost
    assert boosted[0]["id"] == 1, "temporally rich result should be ranked first"


# ---------------------------------------------------------------------------
# #93 — _adapt_kg direction: source_id and text for incoming edges
# ---------------------------------------------------------------------------


def test_adapt_kg_outgoing_source_id_is_subject():
    from taosmd.retrieval import _adapt_kg

    row = {
        "subject_id": "person:alice",
        "subject_name": "Alice",
        "predicate": "works_at",
        "object_id": "org:jan_labs",
        "object_name": "JAN Labs",
        "direction": "outgoing",
        "confidence": 1.0,
    }
    adapted = _adapt_kg([row])
    assert adapted[0]["source_id"] == "person:alice"
    assert "Alice works_at JAN Labs" == adapted[0]["text"]


def test_adapt_kg_incoming_source_id_is_object():
    from taosmd.retrieval import _adapt_kg

    # Incoming query on "jan_labs": subject_name = "Alice" (far node),
    # object_id = "org:jan_labs" (the anchor that was queried).
    row = {
        "subject_id": "person:alice",
        "subject_name": "Alice",
        "predicate": "works_at",
        "object_id": "org:jan_labs",
        "direction": "incoming",
        "confidence": 0.9,
        # object_name is absent in real incoming results (no join for it)
    }
    adapted = _adapt_kg([row])
    # source_id must track the anchor (object_id), not the far node
    assert adapted[0]["source_id"] == "org:jan_labs"
    # text must include the far-node subject
    assert "Alice" in adapted[0]["text"]


def test_adapt_kg_incoming_and_outgoing_differ():
    from taosmd.retrieval import _adapt_kg

    outgoing_row = {
        "subject_id": "alice",
        "subject_name": "Alice",
        "predicate": "knows",
        "object_id": "bob",
        "object_name": "Bob",
        "direction": "outgoing",
        "confidence": 1.0,
    }
    incoming_row = {
        "subject_id": "alice",
        "subject_name": "Alice",
        "predicate": "knows",
        "object_id": "charlie",
        "direction": "incoming",
        "confidence": 1.0,
    }
    adapted = _adapt_kg([outgoing_row, incoming_row])
    assert adapted[0]["source_id"] != adapted[1]["source_id"], (
        "outgoing and incoming source_ids must differ"
    )


# ---------------------------------------------------------------------------
# #103 — BM25 cache: index rebuilt once, invalidated on write
# ---------------------------------------------------------------------------


def test_bm25_cache_is_populated_after_search(tmp_path):
    """After a bm25_rrf search the cache entry must be set."""
    from taosmd.vector_memory import VectorMemory

    vm = VectorMemory(db_path=tmp_path / "vm.db", embed_mode="qmd")

    async def go():
        # Patch embed so no real HTTP call is made
        vm._embed_mode = "local"

        with patch.object(vm, "embed", new=AsyncMock(return_value=[0.1] * 8)):
            await vm.init()
            await vm.add("the quick brown fox jumps over the lazy dog")
            await vm.add("a second document about cats and dogs")

            # Cache should be dirty before first search
            assert vm._bm25_dirty is True

            results = await vm.search("quick fox", hybrid=True, fusion="bm25_rrf", limit=2)

        # After a bm25_rrf search the cache must be populated and clean
        assert "bm25_rrf" in vm._bm25_cache
        assert vm._bm25_dirty is False

    asyncio.run(go())


def test_bm25_cache_invalidated_on_add(tmp_path):
    """Adding a new document must mark the cache dirty."""
    from taosmd.vector_memory import VectorMemory

    vm = VectorMemory(db_path=tmp_path / "vm.db", embed_mode="qmd")

    async def go():
        with patch.object(vm, "embed", new=AsyncMock(return_value=[0.1] * 8)):
            await vm.init()
            await vm.add("first document")
            await vm.search("first", hybrid=True, fusion="bm25_rrf", limit=1)

            # Cache populated and clean
            assert vm._bm25_dirty is False

            # Adding a new doc must dirty the cache
            await vm.add("second document added later")
            assert vm._bm25_dirty is True

    asyncio.run(go())


def test_bm25_cache_not_rebuilt_on_second_query(tmp_path):
    """Two consecutive queries on the same corpus must reuse the cached index."""
    from taosmd.vector_memory import VectorMemory

    vm = VectorMemory(db_path=tmp_path / "vm.db", embed_mode="qmd")

    async def go():
        with patch.object(vm, "embed", new=AsyncMock(return_value=[0.2] * 8)):
            await vm.init()
            await vm.add("cats are independent animals")
            await vm.add("dogs are loyal companions")

            await vm.search("cats", hybrid=True, fusion="bm25_rrf", limit=1)
            first_retriever = vm._bm25_cache["bm25_rrf"][0]

            await vm.search("dogs", hybrid=True, fusion="bm25_rrf", limit=1)
            second_retriever = vm._bm25_cache["bm25_rrf"][0]

        # Same object — not rebuilt
        assert first_retriever is second_retriever

    asyncio.run(go())


# ---------------------------------------------------------------------------
# #105 — Contradiction detection uses predicate_vocab.SINGULAR_PREDICATES
# ---------------------------------------------------------------------------


def test_singular_predicates_exported_from_vocab():
    """SINGULAR_PREDICATES must be importable from predicate_vocab."""
    from taosmd.predicate_vocab import SINGULAR_PREDICATES, ALLOWED_PREDICATES

    assert isinstance(SINGULAR_PREDICATES, frozenset)
    assert len(SINGULAR_PREDICATES) > 0
    # Every singular predicate must be in the allowed set
    assert SINGULAR_PREDICATES.issubset(ALLOWED_PREDICATES)


def test_knowledge_graph_singular_predicates_matches_vocab():
    """TemporalKnowledgeGraph.SINGULAR_PREDICATES must equal the vocab set."""
    from taosmd.knowledge_graph import TemporalKnowledgeGraph
    from taosmd.predicate_vocab import SINGULAR_PREDICATES

    assert TemporalKnowledgeGraph.SINGULAR_PREDICATES == SINGULAR_PREDICATES


def test_contradiction_detected_for_vocab_predicate(tmp_path):
    """detect_contradictions fires for predicates now in the vocab-derived set."""
    from taosmd.knowledge_graph import TemporalKnowledgeGraph
    from taosmd.predicate_vocab import SINGULAR_PREDICATES

    # Pick a predicate that IS in vocab's SINGULAR_PREDICATES but was NOT in
    # the old hardcoded set (works_at was added in the vocab extension).
    assert "works_at" in SINGULAR_PREDICATES

    async def go():
        kg = TemporalKnowledgeGraph(db_path=tmp_path / "kg.db")
        await kg.init()

        await kg.add_triple("Alice", "works_at", "CompanyA")
        contradictions = await kg.detect_contradictions("Alice", "works_at", "CompanyB")
        await kg.close()
        return contradictions

    contradictions = asyncio.run(go())
    assert len(contradictions) == 1
    assert contradictions[0]["object_name"].lower() == "companya"


def test_no_contradiction_for_non_singular_predicate(tmp_path):
    """Non-singular predicates must NOT trigger contradiction detection."""
    from taosmd.knowledge_graph import TemporalKnowledgeGraph
    from taosmd.predicate_vocab import SINGULAR_PREDICATES

    # "knows" is many-to-many — should NOT be singular
    assert "knows" not in SINGULAR_PREDICATES

    async def go():
        kg = TemporalKnowledgeGraph(db_path=tmp_path / "kg.db")
        await kg.init()

        await kg.add_triple("Alice", "knows", "Bob")
        contradictions = await kg.detect_contradictions("Alice", "knows", "Charlie")
        await kg.close()
        return contradictions

    contradictions = asyncio.run(go())
    assert contradictions == []


# ---------------------------------------------------------------------------
# #108 — LLM reranker wired into retrieve() as opt-in
# ---------------------------------------------------------------------------


def test_retrieve_accepts_llm_reranker_param():
    """retrieve() must accept llm_reranker kwarg without error when None."""
    import inspect
    from taosmd.retrieval import retrieve

    sig = inspect.signature(retrieve)
    assert "llm_reranker" in sig.parameters


def test_retrieve_calls_llm_reranker_when_set():
    """When llm_reranker dict is supplied, _apply_llm_rerank must be called."""
    from taosmd.retrieval import retrieve

    call_log = []

    async def fake_llm_rerank(query, results, llm_reranker, top_k):
        call_log.append({"query": query, "n": len(results)})
        return list(reversed(results))[:top_k]  # flip order to prove it was called

    class MockVMem:
        async def search(self, q, limit=5, hybrid=True, **_):
            return [
                {"id": 1, "text": "alpha", "similarity": 0.9, "metadata": {}, "created_at": 1.0},
                {"id": 2, "text": "beta", "similarity": 0.8, "metadata": {}, "created_at": 1.0},
                {"id": 3, "text": "gamma", "similarity": 0.7, "metadata": {}, "created_at": 1.0},
            ]

    dummy_reranker_cfg = {
        "client": object(),
        "ollama_url": "http://localhost:11434",
        "model": "test-model",
    }

    async def go():
        with patch("taosmd.retrieval._apply_llm_rerank", side_effect=fake_llm_rerank):
            results = await retrieve(
                query="test query",
                strategy="thorough",
                sources={"vector": MockVMem()},
                limit=2,
                llm_reranker=dummy_reranker_cfg,
            )
        return results

    results = asyncio.run(go())
    assert len(call_log) == 1, "_apply_llm_rerank should have been called once"
    assert call_log[0]["query"] == "test query"


def test_retrieve_default_behavior_unchanged_without_llm_reranker():
    """When llm_reranker=None (default), behavior is unchanged."""
    from taosmd.retrieval import retrieve

    call_log = []

    class MockVMem:
        async def search(self, q, limit=5, hybrid=True, **_):
            return [
                {"id": 1, "text": "result one", "similarity": 0.9, "metadata": {}, "created_at": 1.0},
            ]

    async def go():
        with patch("taosmd.retrieval._apply_llm_rerank", side_effect=lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not be called"))):
            return await retrieve(
                query="test",
                strategy="thorough",
                sources={"vector": MockVMem()},
                limit=1,
                llm_reranker=None,  # default — must not call LLM reranker
            )

    # Must succeed without raising
    results = asyncio.run(go())
    assert len(results) == 1


# ---------------------------------------------------------------------------
# #111 — Archive entry checksums
# ---------------------------------------------------------------------------


def test_archive_record_writes_sha256(tmp_path):
    """Every recorded event must have a sha256 field in the JSONL line."""
    from taosmd.archive import ArchiveStore

    async def go():
        store = ArchiveStore(
            archive_dir=tmp_path / "archive",
            index_path=tmp_path / "idx.db",
        )
        await store.init()
        await store.record("test_event", {"content": "hello world"}, summary="test")
        await store.close()

    asyncio.run(go())

    jsonl_files = list((tmp_path / "archive").rglob("*.jsonl"))
    assert len(jsonl_files) == 1

    with open(jsonl_files[0], "r", encoding="utf-8") as fh:
        line = fh.readline().strip()

    record = json.loads(line)
    assert "sha256" in record, "sha256 key must be present in every JSONL entry"
    assert len(record["sha256"]) == 64, "sha256 must be a 64-char hex string"


def test_verify_entry_passes_for_intact_record(tmp_path):
    """verify_entry() must return True for a freshly written line."""
    from taosmd.archive import ArchiveStore

    async def go():
        store = ArchiveStore(
            archive_dir=tmp_path / "archive",
            index_path=tmp_path / "idx.db",
        )
        await store.init()
        await store.record("test_event", {"content": "integrity check"})
        await store.close()

    asyncio.run(go())

    jsonl_files = list((tmp_path / "archive").rglob("*.jsonl"))
    with open(jsonl_files[0], "r", encoding="utf-8") as fh:
        line = fh.readline()

    assert ArchiveStore.verify_entry(line) is True


def test_verify_entry_fails_for_tampered_record():
    """verify_entry() must return False when the content doesn't match the hash."""
    import hashlib
    import json
    from taosmd.archive import ArchiveStore

    event = {
        "timestamp": 1700000000.0,
        "event_type": "test",
        "agent_name": None,
        "app_id": None,
        "summary": "original",
        "data": {"content": "original"},
    }
    # Build a valid line
    line_for_hash = json.dumps(event, default=str)
    event["sha256"] = hashlib.sha256(line_for_hash.encode()).hexdigest()
    # Now tamper with the content
    event["summary"] = "tampered"
    tampered_line = json.dumps(event, default=str)

    assert ArchiveStore.verify_entry(tampered_line) is False


def test_verify_entry_passes_for_legacy_record_without_sha256():
    """Legacy entries without sha256 must pass verify_entry (backwards compat)."""
    import json
    from taosmd.archive import ArchiveStore

    legacy_event = {
        "timestamp": 1700000000.0,
        "event_type": "legacy",
        "summary": "old entry",
        "data": {},
    }
    line = json.dumps(legacy_event)
    assert ArchiveStore.verify_entry(line) is True


def test_verify_day_reports_correct_counts(tmp_path):
    """verify_day() must count ok / bad / legacy entries correctly."""
    from taosmd.archive import ArchiveStore

    async def go():
        store = ArchiveStore(
            archive_dir=tmp_path / "archive",
            index_path=tmp_path / "idx.db",
        )
        await store.init()
        # Write two clean entries
        import taosmd.archive as archive_mod
        from unittest.mock import patch as _patch
        # Pin timestamp so both go to the same day file
        fixed_ts = 1_700_000_000.0
        with _patch.object(archive_mod.time, "time", return_value=fixed_ts):
            await store.record("ev", {"x": 1})
            await store.record("ev", {"x": 2})
        await store.close()
        return fixed_ts

    fixed_ts = asyncio.run(go())

    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(fixed_ts, tz=timezone.utc)
    date_str = f"{dt.year}-{dt.month:02d}-{dt.day:02d}"

    async def verify():
        store = ArchiveStore(
            archive_dir=tmp_path / "archive",
            index_path=tmp_path / "idx.db",
        )
        await store.init()
        result = await store.verify_day(date_str)
        await store.close()
        return result

    result = asyncio.run(verify())
    assert result["total"] == 2
    assert result["ok"] == 2
    assert result["bad"] == 0
    assert result["legacy"] == 0
