"""Tests for taosmd.emem_event_lift — EMem pass-2 typed event lift.

Tests stub Ollama via a fake httpx-shaped client. We exercise:

  - happy-path: well-formed JSON with vocab-valid predicates round-trips
  - belt-and-braces filter: predicates outside ALLOWED_PREDICATES are
    dropped client-side even if the model emits them
  - synonym normalisation: model emits 'employed_by', we record 'works_at'
  - error fallback: transport / JSON failure returns the no-op shape
  - batch lifter: lift_edus_to_events preserves source_edu + source_turn_ids

Plus an integration with TemporalKnowledgeGraph that exercises the
strict_vocab write path — pass-2 outputs going into a KG must round-trip
through the closed vocab.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from taosmd.emem_event_lift import lift_edu_to_triples, lift_edus_to_events
from taosmd.knowledge_graph import TemporalKnowledgeGraph
from taosmd.predicate_vocab import ALLOWED_PREDICATES


def _run(coro):
    return asyncio.run(coro)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeOllamaClient:
    """httpx.AsyncClient stub that returns a predetermined JSON payload."""

    def __init__(self, content_json: dict[str, Any]):
        self._content = json.dumps(content_json)

    async def post(self, url, json=None, timeout=None):
        # Ignore inputs — return the canned content. Mirrors Ollama's
        # /api/chat shape: {"message": {"content": "<json string>"}}.
        return _FakeResponse({"message": {"content": self._content}})


# ---------------------------------------------------------------------------
# Single-EDU lift
# ---------------------------------------------------------------------------


def test_lift_edu_happy_path():
    """Model emits 3 vocab-valid triples → all 3 survive the filter."""
    client = _FakeOllamaClient({
        "event_type": "travel",
        "triples": [
            {"subject": "bob", "predicate": "moved_to", "object": "tokyo"},
            {"subject": "bob", "predicate": "attended",
             "object": "global ai innovation symposium 2024"},
            {"subject": "global ai innovation symposium 2024",
             "predicate": "located_in", "object": "tokyo university"},
        ],
    })

    result = _run(lift_edu_to_triples(
        "Bob traveled to Tokyo for the Global AI Innovation Symposium.",
        model="llama3.1:8b",
        ollama_url="http://localhost:11434",
        http_client=client,
    ))

    assert result["event_type"] == "travel"
    assert len(result["triples"]) == 3
    assert all(t["predicate"] in ALLOWED_PREDICATES for t in result["triples"])
    subjects = {t["subject"] for t in result["triples"]}
    assert "bob" in subjects


def test_lift_edu_drops_unknown_predicates():
    """Model invents a predicate not in the closed vocab → lifter drops it."""
    client = _FakeOllamaClient({
        "event_type": "misc",
        "triples": [
            {"subject": "alice", "predicate": "works_at", "object": "stripe"},
            {"subject": "alice", "predicate": "frobnicates", "object": "things"},
            {"subject": "alice", "predicate": "dabbles_in", "object": "ml"},
        ],
    })

    result = _run(lift_edu_to_triples(
        "Alice works at Stripe and tinkers with ML.",
        model="llama3.1:8b",
        ollama_url="http://localhost:11434",
        http_client=client,
    ))

    # Only the works_at triple survives.
    assert len(result["triples"]) == 1
    assert result["triples"][0]["predicate"] == "works_at"


def test_lift_edu_resolves_synonyms_via_predicate_vocab():
    """Model emits 'employed_by' (a synonym) → recorded as canonical 'works_at'."""
    client = _FakeOllamaClient({
        "event_type": "employment",
        "triples": [
            {"subject": "alice", "predicate": "employed_by", "object": "stripe"},
        ],
    })

    result = _run(lift_edu_to_triples(
        "Alice is employed by Stripe.",
        model="llama3.1:8b",
        ollama_url="http://localhost:11434",
        http_client=client,
    ))

    assert len(result["triples"]) == 1
    assert result["triples"][0]["predicate"] == "works_at"


def test_lift_edu_drops_empty_fields():
    """Triples missing subject / predicate / object are silently dropped."""
    client = _FakeOllamaClient({
        "event_type": "misc",
        "triples": [
            {"subject": "", "predicate": "works_at", "object": "stripe"},
            {"subject": "alice", "predicate": "", "object": "stripe"},
            {"subject": "alice", "predicate": "works_at", "object": ""},
            {"subject": "alice", "predicate": "works_at", "object": "stripe"},
        ],
    })

    result = _run(lift_edu_to_triples(
        "x", model="m", ollama_url="http://x", http_client=client,
    ))
    assert len(result["triples"]) == 1


class _BrokenClient:
    async def post(self, url, json=None, timeout=None):
        import httpx
        raise httpx.HTTPError("simulated transport failure")


def test_lift_edu_returns_noop_on_transport_error():
    result = _run(lift_edu_to_triples(
        "x", model="m", ollama_url="http://x", http_client=_BrokenClient(),
    ))
    assert result == {"event_type": "none", "triples": []}


class _BadJsonClient:
    async def post(self, url, json=None, timeout=None):
        return _FakeResponse({"message": {"content": "not json at all"}})


def test_lift_edu_returns_noop_on_bad_json():
    result = _run(lift_edu_to_triples(
        "x", model="m", ollama_url="http://x", http_client=_BadJsonClient(),
    ))
    assert result == {"event_type": "none", "triples": []}


# ---------------------------------------------------------------------------
# Batch lift
# ---------------------------------------------------------------------------


def test_lift_edus_preserves_source_metadata():
    """source_edu + source_turn_ids must travel with each lift result."""
    client = _FakeOllamaClient({
        "event_type": "travel",
        "triples": [
            {"subject": "bob", "predicate": "moved_to", "object": "tokyo"},
        ],
    })

    edus = [
        {"edu_text": "Bob moved to Tokyo.", "source_turn_ids": [2, 4]},
        {"edu_text": "Bob attended the conference.", "source_turn_ids": [6]},
    ]
    results = _run(lift_edus_to_events(
        edus, model="m", ollama_url="http://x", http_client=client,
    ))

    assert len(results) == 2
    assert results[0]["source_edu"] == "Bob moved to Tokyo."
    assert results[0]["source_turn_ids"] == [2, 4]
    assert results[1]["source_edu"] == "Bob attended the conference."
    assert results[1]["source_turn_ids"] == [6]


def test_lift_edus_skips_empty_edus():
    """EDU entries with empty edu_text are skipped (no LLM call made)."""
    client = _FakeOllamaClient({
        "event_type": "x",
        "triples": [{"subject": "a", "predicate": "works_at", "object": "b"}],
    })
    edus = [
        {"edu_text": "", "source_turn_ids": [1]},
        {"edu_text": "   ", "source_turn_ids": [2]},
        {"edu_text": "Real EDU.", "source_turn_ids": [3]},
    ]
    results = _run(lift_edus_to_events(
        edus, model="m", ollama_url="http://x", http_client=client,
    ))
    assert len(results) == 1
    assert results[0]["source_edu"] == "Real EDU."


# ---------------------------------------------------------------------------
# KG round-trip with strict_vocab
# ---------------------------------------------------------------------------


def test_lifted_triples_round_trip_through_strict_kg(tmp_path):
    """Pass-2 output → kg.add_triple(strict_vocab=True) succeeds for all triples."""
    client = _FakeOllamaClient({
        "event_type": "employment",
        "triples": [
            {"subject": "alice", "predicate": "works_at", "object": "stripe"},
            {"subject": "alice", "predicate": "lives_in", "object": "san francisco"},
            {"subject": "alice", "predicate": "founded", "object": "side-project"},
        ],
    })

    async def go():
        kg = TemporalKnowledgeGraph(db_path=tmp_path / "kg.db")
        await kg.init()
        try:
            lift = await lift_edu_to_triples(
                "Alice works at Stripe and founded a side project in SF.",
                model="m", ollama_url="http://x", http_client=client,
            )
            # All triples should land via strict_vocab without raising —
            # they came from a lifter that already filtered against the
            # vocab, so the strict KG write is a no-op safety check.
            for t in lift["triples"]:
                await kg.add_triple(
                    subject=t["subject"], predicate=t["predicate"],
                    obj=t["object"], strict_vocab=True,
                    source="test:pass2",
                )
            return await kg.stats()
        finally:
            await kg.close()

    stats = _run(go())
    assert stats["active_triples"] == 3
