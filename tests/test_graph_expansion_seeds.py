"""Tests for seed entity ranking in graph expansion (issue #198).

expand_from_results previously capped seed entities at a hardcoded 10 in
encounter order, so on long hit lists the seeds were whatever proper nouns
appeared first rather than the ones most worth expanding. These tests pin
the new relevance-ranked cap:

  - Numeric scores (score key, then source_score fallback) rank entities
    by their highest-scoring mentioning result.
  - Mention count across ALL results (scored or not) is the first tie-break.
  - Encounter order is the second tie-break (earlier wins).
  - When no result carries a numeric score, order is byte-identical to the
    old encounter behaviour.
"""

from __future__ import annotations

import asyncio

from taosmd.graph_expansion import expand_from_results


class _FakeKG:
    """Records every entity query_entity is called with; returns no triples."""

    def __init__(self):
        self.queries: list[str] = []

    async def query_entity(self, entity: str, as_of=None, track_access=False) -> list:
        self.queries.append(entity)
        return []


# ---------------------------------------------------------------------------
# MUST 1: max_seeds parameter replaces the hardcoded 10
# ---------------------------------------------------------------------------

def test_scored_results_ranked_by_highest_score():
    results = [{"text": f"Entity{i} met someone", "score": i} for i in range(1, 13)]
    kg = _FakeKG()
    asyncio.run(expand_from_results(kg, results, max_seeds=3))
    assert kg.queries == ["Entity12", "Entity11", "Entity10"]


def test_max_seeds_two_queries_exactly_two():
    results = [{"text": f"Entity{i} met someone", "score": i} for i in range(1, 13)]
    kg = _FakeKG()
    asyncio.run(expand_from_results(kg, results, max_seeds=2))
    assert len(kg.queries) == 2


def test_max_seeds_zero_returns_empty():
    results = [{"text": f"Entity{i} met someone", "score": i} for i in range(1, 13)]
    kg = _FakeKG()
    result = asyncio.run(expand_from_results(kg, results, max_seeds=0))
    assert result == []
    assert kg.queries == []


# ---------------------------------------------------------------------------
# MUST 3: when no result carries a numeric score, order is exactly encounter order
# ---------------------------------------------------------------------------

def test_unscored_results_preserve_encounter_order():
    results = [{"text": f"Entity{i} met someone"} for i in range(1, 11)]
    kg = _FakeKG()
    asyncio.run(expand_from_results(kg, results))
    assert kg.queries == [
        "Entity1", "Entity2", "Entity3", "Entity4", "Entity5",
        "Entity6", "Entity7", "Entity8", "Entity9", "Entity10",
    ]


def test_non_numeric_score_falls_back_to_encounter_order():
    results = [
        {"text": "Beta has no score", "score": "abc"},
        {"text": "Alpha is bool score", "score": True},
        {"text": "Gamma is nan score", "score": float("nan")},
    ]
    kg = _FakeKG()
    asyncio.run(expand_from_results(kg, results, max_seeds=10))
    assert kg.queries == ["Beta", "Alpha", "Gamma"]


# ---------------------------------------------------------------------------
# MUST 2 (tie-breaks): higher score first, then more mentions, then encounter
# ---------------------------------------------------------------------------

def test_mention_count_includes_unscoped_results():
    results = [
        {"text": "Alpha is here", "score": 5},
        {"text": "Beta is here", "score": 5},
        {"text": "Beta mentioned again"},
    ]
    kg = _FakeKG()
    asyncio.run(expand_from_results(kg, results, max_seeds=1))
    assert kg.queries == ["Beta"]


def test_case_insensitive_mention_matching_not_substring():
    results = [
        {"text": "Ann went to the store", "score": 1},
        {"text": "Anna went to the park", "score": 9},
        {"text": "Bob went home", "score": 5},
    ]
    kg = _FakeKG()
    asyncio.run(expand_from_results(kg, results, max_seeds=1))
    assert kg.queries == ["Anna"]


def test_max_score_not_min():
    results = [
        {"text": "Alpha is here", "score": 1},
        {"text": "Alpha again", "score": 9},
        {"text": "Beta is here", "score": 5},
    ]
    kg = _FakeKG()
    asyncio.run(expand_from_results(kg, results, max_seeds=1))
    assert kg.queries == ["Alpha"]


# ---------------------------------------------------------------------------
# R1 / R4: score extraction and source_score fallback
# ---------------------------------------------------------------------------

def test_source_score_only_ranking():
    results = [
        {"text": "Alpha is first", "source_score": 3},
        {"text": "Beta is second", "source_score": 7},
        {"text": "Gamma is third", "source_score": 5},
    ]
    kg = _FakeKG()
    asyncio.run(expand_from_results(kg, results, max_seeds=2))
    assert kg.queries == ["Beta", "Gamma"]


def test_score_none_falls_back_to_source_score():
    results = [
        {"text": "Alpha is first", "score": None, "source_score": 3},
        {"text": "Beta is second", "score": None, "source_score": 7},
    ]
    kg = _FakeKG()
    asyncio.run(expand_from_results(kg, results, max_seeds=1))
    assert kg.queries == ["Beta"]


def test_bool_score_treated_as_none():
    results = [
        {"text": "Beta is first", "score": 50},
        {"text": "Alpha is second", "score": True, "source_score": 100},
    ]
    kg = _FakeKG()
    asyncio.run(expand_from_results(kg, results, max_seeds=1))
    assert kg.queries == ["Alpha"]


# ---------------------------------------------------------------------------
# S1 / S2: full tie - earlier-encountered entity must win
# ---------------------------------------------------------------------------

def test_full_tie_earlier_wins_at_max1():
    results = [
        {"text": "Alpha is here", "score": 1},
        {"text": "Beta is here", "score": 1},
    ]
    kg = _FakeKG()
    asyncio.run(expand_from_results(kg, results, max_seeds=1))
    assert kg.queries == ["Alpha"]


def test_full_tie_earlier_wins_at_max2():
    results = [
        {"text": "Alpha is here", "score": 1},
        {"text": "Beta is here", "score": 1},
        {"text": "Gamma is here", "score": 1},
    ]
    kg = _FakeKG()
    asyncio.run(expand_from_results(kg, results, max_seeds=2))
    assert kg.queries == ["Alpha", "Beta"]


# ---------------------------------------------------------------------------
# Optional: NaN guard and negative max_seeds guard
# ---------------------------------------------------------------------------

def test_nan_score_treated_as_none():
    results = [
        {"text": "Beta has no score", "score": None},
        {"text": "Alpha has nan score", "score": float("nan")},
        {"text": "Gamma has no score", "score": None},
    ]
    kg = _FakeKG()
    asyncio.run(expand_from_results(kg, results, max_seeds=3))
    assert kg.queries == ["Beta", "Alpha", "Gamma"]


def test_max_seeds_negative_returns_empty():
    results = [
        {"text": "Alpha is here", "score": 1},
        {"text": "Beta is here", "score": 2},
    ]
    kg = _FakeKG()
    result = asyncio.run(expand_from_results(kg, results, max_seeds=-1))
    assert result == []
    assert kg.queries == []
