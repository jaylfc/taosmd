"""Tests for taosmd.pending_decisions + KG deferral integration."""

from __future__ import annotations

import asyncio

import pytest

from taosmd.pending_decisions import PendingDecisionsStore
from taosmd.knowledge_graph import TemporalKnowledgeGraph


def _run(coro):
    return asyncio.run(coro)


def test_defer_and_list(tmp_path):
    async def go():
        store = PendingDecisionsStore(db_path=tmp_path / "kg.db")
        await store.init()
        try:
            pid = await store.defer(
                kind="contradiction",
                subject="user",
                predicate="lives_in",
                new_object="paris",
                old_triple_ids=["abc123"],
                suggested_action="invalidate_old_add_new",
                evidence="The user mentioned moving to Paris yesterday.",
                source="archive/2026-05-15.jsonl",
                new_triple_confidence=0.62,
            )
            assert pid
            pending = await store.list_pending()
            return pid, pending
        finally:
            await store.close()

    pid, pending = _run(go())
    assert len(pending) == 1
    assert pending[0]["subject"] == "user"
    assert pending[0]["predicate"] == "lives_in"
    assert pending[0]["new_object"] == "paris"
    assert pending[0]["old_triple_ids"] == ["abc123"]
    assert pending[0]["new_triple_confidence"] == pytest.approx(0.62)


def test_same_day_defer_dedupes(tmp_path):
    """Repeated nightly-pipeline runs on the same conflict don't spam the queue."""
    async def go():
        store = PendingDecisionsStore(db_path=tmp_path / "kg.db")
        await store.init()
        try:
            pid1 = await store.defer(
                kind="contradiction", subject="alice", predicate="works_at",
                new_object="initech", old_triple_ids=["t1"],
                suggested_action="invalidate_old_add_new", evidence="ev1",
            )
            pid2 = await store.defer(
                kind="contradiction", subject="alice", predicate="works_at",
                new_object="initech", old_triple_ids=["t1"],
                suggested_action="invalidate_old_add_new",
                evidence="ev2 — newer evidence",
            )
            pending = await store.list_pending()
            return pid1, pid2, pending
        finally:
            await store.close()

    pid1, pid2, pending = _run(go())
    assert pid1 == pid2
    assert len(pending) == 1
    assert pending[0]["evidence"] == "ev2 — newer evidence"


def test_resolve_marks_done(tmp_path):
    async def go():
        store = PendingDecisionsStore(db_path=tmp_path / "kg.db")
        await store.init()
        try:
            pid = await store.defer(
                kind="contradiction", subject="bob", predicate="prefers",
                new_object="vim", old_triple_ids=["x"],
                suggested_action="invalidate_old_add_new",
            )
            ok = await store.resolve(pid, resolution="accepted",
                                     note="confirmed by user 2026-05-16")
            pending_after = await store.list_pending()
            decision = await store.get(pid)
            return ok, pending_after, decision
        finally:
            await store.close()

    ok, pending_after, decision = _run(go())
    assert ok is True
    assert pending_after == []
    assert decision["resolution"] == "accepted"
    assert decision["resolution_note"] == "confirmed by user 2026-05-16"
    assert decision["resolved_at"] is not None


def test_resolve_invalid_id_returns_false(tmp_path):
    async def go():
        store = PendingDecisionsStore(db_path=tmp_path / "kg.db")
        await store.init()
        try:
            return await store.resolve("nonexistent", resolution="accepted")
        finally:
            await store.close()
    assert _run(go()) is False


def test_resolve_invalid_resolution_raises(tmp_path):
    async def go():
        store = PendingDecisionsStore(db_path=tmp_path / "kg.db")
        await store.init()
        try:
            with pytest.raises(ValueError):
                await store.resolve("x", resolution="maybe")
        finally:
            await store.close()
    _run(go())


def test_defer_invalid_kind_raises(tmp_path):
    async def go():
        store = PendingDecisionsStore(db_path=tmp_path / "kg.db")
        await store.init()
        try:
            with pytest.raises(ValueError):
                await store.defer(
                    kind="not_a_kind", subject="x", predicate="y", new_object="z",
                    old_triple_ids=[], suggested_action="invalidate_old_add_new",
                )
        finally:
            await store.close()
    _run(go())


def test_stats_counts(tmp_path):
    async def go():
        store = PendingDecisionsStore(db_path=tmp_path / "kg.db")
        await store.init()
        try:
            await store.defer(kind="contradiction", subject="s1",
                              predicate="lives_in", new_object="paris",
                              old_triple_ids=[],
                              suggested_action="invalidate_old_add_new")
            await store.defer(kind="contradiction", subject="s2",
                              predicate="lives_in", new_object="tokyo",
                              old_triple_ids=[],
                              suggested_action="invalidate_old_add_new")
            pid = await store.defer(kind="contradiction", subject="s3",
                                    predicate="lives_in", new_object="berlin",
                                    old_triple_ids=[],
                                    suggested_action="invalidate_old_add_new")
            await store.resolve(pid, resolution="accepted")
            return await store.stats()
        finally:
            await store.close()

    stats = _run(go())
    assert stats == {"total": 3, "pending": 2, "resolved": 1}


def test_kg_defers_low_confidence_contradictions(tmp_path):
    """Existing high-confidence fact + low-confidence conflict -> deferred."""
    async def go():
        kg = TemporalKnowledgeGraph(db_path=tmp_path / "kg.db")
        pending = PendingDecisionsStore(db_path=tmp_path / "kg.db")
        await kg.init()
        await pending.init()
        try:
            await kg.add_triple(
                subject="alice", predicate="lives_in", obj="london",
                confidence=1.0, source="user-direct-statement",
            )
            result = await kg.add_triple_with_contradiction_check(
                subject="alice", predicate="lives_in", obj="paris",
                confidence=0.4, source="nightly-catalog",
                evidence="Maybe Alice mentioned Paris last week?",
                pending_store=pending,
                defer_below_confidence=0.8,
            )
            active = await kg.query_entity("alice")
            queue = await pending.list_pending()
            return result, active, queue
        finally:
            await pending.close()
            await kg.close()

    result, active, queue = _run(go())
    assert result["status"] == "deferred"
    assert result["pending_id"]
    assert result["triple_id"] == ""

    objs = {t["object_name"] for t in active}
    assert "london" in objs
    assert "paris" not in objs

    assert len(queue) == 1
    assert queue[0]["subject"] == "alice"
    assert queue[0]["new_object"] == "paris"


def test_kg_auto_resolves_high_confidence_even_with_pending_store(tmp_path):
    """A direct user statement (confidence=1.0) auto-resolves."""
    async def go():
        kg = TemporalKnowledgeGraph(db_path=tmp_path / "kg.db")
        pending = PendingDecisionsStore(db_path=tmp_path / "kg.db")
        await kg.init()
        await pending.init()
        try:
            await kg.add_triple(
                subject="alice", predicate="lives_in", obj="london",
                confidence=1.0, source="archive/jan",
            )
            result = await kg.add_triple_with_contradiction_check(
                subject="alice", predicate="lives_in", obj="paris",
                confidence=1.0, source="user-direct-statement-today",
                pending_store=pending,
                defer_below_confidence=0.8,
            )
            queue = await pending.list_pending()
            return result, queue
        finally:
            await pending.close()
            await kg.close()

    result, queue = _run(go())
    assert result["status"] == "written"
    assert result["triple_id"]
    assert result["pending_id"] == ""
    assert result["contradictions_resolved"] == 1
    assert queue == []
