"""Tests for graph_expansion seed selection and max_seeds cap.

Red-first: tests a and d MUST fail on unmodified master and pass after fix.
"""

from __future__ import annotations

import asyncio

from taosmd.graph_expansion import expand_from_results


class _FakeKG:
    """Tiny fake KG that records every entity it is asked about."""

    def __init__(self):
        self.queried: list[str] = []

    async def query_entity(self, entity, as_of=None, track_access=False):
        self.queried.append(entity)
        return []


def _call(results, max_seeds=10, max_hops=2, max_expanded=10):
    kg = _FakeKG()
    expanded = asyncio.run(
        expand_from_results(kg, results, max_hops=max_hops, max_expanded=max_expanded, max_seeds=max_seeds)
    )
    return kg.queried, expanded


# ---------------------------------------------------------------------------
# a. 12 scored results, each with a distinct entity: max_seeds=3 picks the
#    3 highest-scored entities, in score order.
# ---------------------------------------------------------------------------

def test_max_seeds_picks_highest_scored_entities():
    results = []
    for i in range(12):
        entity = f"Entity{i:02d}"
        results.append({"text": f"some text about {entity}", "score": i})

    queried, expanded = _call(results, max_seeds=3)

    assert queried == [f"Entity{i:02d}" for i in range(11, 8, -1)]


# ---------------------------------------------------------------------------
# b. max_seeds=2 queries exactly 2 entities; max_seeds=0 queries none and
#    returns [].
# ---------------------------------------------------------------------------

def test_max_seeds_zero_queries_none_and_returns_empty():
    results = [
        {"text": "Alice alpha", "score": 1},
        {"text": "Bob beta", "score": 2},
    ]
    queried, expanded = _call(results, max_seeds=0)

    assert queried == []
    assert expanded == []


def test_max_seeds_two_queries_exactly_two():
    results = [
        {"text": "Alice alpha", "score": 1},
        {"text": "Bob beta", "score": 2},
        {"text": "Carol gamma", "score": 3},
    ]
    queried, expanded = _call(results, max_seeds=2)

    assert queried == ["Carol", "Bob"]  # score 3 then score 2


# ---------------------------------------------------------------------------
# c. Unscored results: order must be byte-identical encounter order.
# ---------------------------------------------------------------------------

def test_unscored_results_preserve_encounter_order():
    results = [
        {"text": "Alice alpha"},
        {"text": "Bob beta"},
        {"text": "Carol gamma"},
    ]
    queried, _ = _call(results, max_seeds=10)

    assert queried == ["Alice", "Bob", "Carol"]


# ---------------------------------------------------------------------------
# d. Tie on score: entity mentioned in more results comes first.
# ---------------------------------------------------------------------------

def test_score_tie_broken_by_mention_count():
    # EntityX appears in 2 results (score 5), EntityY in 1 result (score 5)
    results = [
        {"text": "EntityX and EntityY", "score": 5},
        {"text": "EntityX alone", "score": 5},
    ]
    queried, expanded = _call(results, max_seeds=2)

    assert queried == ["EntityX", "EntityY"]
