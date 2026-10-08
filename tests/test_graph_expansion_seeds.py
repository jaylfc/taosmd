"""Tests for seed entity relevance ranking in graph expansion.

These tests pin the seed selection behaviour introduced to fix the
hardcoded encounter-order slice in expand_from_results().
"""

from __future__ import annotations

import pytest

from taosmd.graph_expansion import expand_from_results


class _RecordingKG:
    """Tiny fake KG that records every entity it is asked about."""

    def __init__(self):
        self.queried: list[str] = []

    async def query_entity(self, entity, as_of=None, track_access=False):
        self.queried.append(entity)
        return []


@pytest.mark.asyncio
async def test_max_seeds_3_returns_top_3_scored_entities():
    kg = _RecordingKG()
    vector_results = [
        {"text": "Apple", "score": 0, "id": 0},
        {"text": "Banana", "score": 1, "id": 1},
        {"text": "Cherry", "score": 2, "id": 2},
        {"text": "Date", "score": 3, "id": 3},
        {"text": "Elderberry", "score": 4, "id": 4},
        {"text": "Fig", "score": 5, "id": 5},
        {"text": "Grape", "score": 6, "id": 6},
        {"text": "Honeydew", "score": 7, "id": 7},
        {"text": "Kiwi", "score": 8, "id": 8},
        {"text": "Lemon", "score": 9, "id": 9},
        {"text": "Mango", "score": 10, "id": 10},
        {"text": "Nectarine", "score": 11, "id": 11},
    ]
    await expand_from_results(kg, vector_results, max_seeds=3)
    assert kg.queried == ["Nectarine", "Mango", "Lemon"]


@pytest.mark.asyncio
async def test_max_seeds_2_queries_two_entities():
    kg = _RecordingKG()
    vector_results = [
        {"text": "Alice", "score": 1, "id": 0},
        {"text": "Bob", "score": 2, "id": 1},
        {"text": "Carol", "score": 3, "id": 2},
    ]
    await expand_from_results(kg, vector_results, max_seeds=2)
    assert kg.queried == ["Carol", "Bob"]


@pytest.mark.asyncio
async def test_max_seeds_0_queries_none_and_returns_empty():
    kg = _RecordingKG()
    vector_results = [{"text": "Alice", "score": 1, "id": 0}]
    result = await expand_from_results(kg, vector_results, max_seeds=0)
    assert kg.queried == []
    assert result == []


@pytest.mark.asyncio
async def test_unscored_results_preserve_encounter_order():
    kg = _RecordingKG()
    vector_results = [
        {"text": "Charlie", "id": 0},
        {"text": "Alice", "id": 1},
        {"text": "Bob", "id": 2},
    ]
    await expand_from_results(kg, vector_results, max_seeds=3)
    assert kg.queried == ["Charlie", "Alice", "Bob"]


@pytest.mark.asyncio
async def test_tie_on_score_more_mentioned_entity_wins():
    kg = _RecordingKG()
    vector_results = [
        {"text": "Beta", "score": 1, "id": 0},
        {"text": "Beta and Alpha", "score": 1, "id": 1},
        {"text": "Alpha", "score": 1, "id": 2},
        {"text": "Alpha", "score": 1, "id": 3},
    ]
    await expand_from_results(kg, vector_results, max_seeds=2)
    assert kg.queried == ["Alpha", "Beta"]


@pytest.mark.asyncio
async def test_highest_scored_entity_wins_even_when_lower_score_in_more_results():
    kg = _RecordingKG()
    vector_results = [
        {"text": "Beta", "score": 5, "id": 0},
        {"text": "Alpha", "score": 1, "id": 1},
        {"text": "Alpha", "score": 9, "id": 2},
    ]
    await expand_from_results(kg, vector_results, max_seeds=2)
    assert kg.queried == ["Alpha", "Beta"]


@pytest.mark.asyncio
async def test_source_score_only_is_ranked():
    kg = _RecordingKG()
    vector_results = [
        {"text": "Delta", "source_score": 2, "id": 0},
        {"text": "Echo", "source_score": 5, "id": 1},
        {"text": "Foxtrot", "source_score": 3, "id": 2},
    ]
    await expand_from_results(kg, vector_results, max_seeds=2)
    assert kg.queried == ["Echo", "Foxtrot"]


@pytest.mark.asyncio
async def test_score_none_falls_back_to_source_score():
    kg = _RecordingKG()
    vector_results = [
        {"text": "Golf", "score": None, "source_score": 1, "id": 0},
        {"text": "Hotel", "score": None, "source_score": 4, "id": 1},
        {"text": "India", "score": None, "source_score": 2, "id": 2},
    ]
    await expand_from_results(kg, vector_results, max_seeds=2)
    assert kg.queried == ["Hotel", "India"]


@pytest.mark.asyncio
async def test_non_numeric_score_does_not_raise_and_falls_back_to_encounter_order():
    kg = _RecordingKG()
    vector_results = [
        {"text": "Juliet", "score": "not-a-number", "id": 0},
        {"text": "Kilo", "score": 1, "id": 1},
        {"text": "Lima", "score": 2, "id": 2},
    ]
    await expand_from_results(kg, vector_results, max_seeds=3)
    assert kg.queried == ["Lima", "Kilo", "Juliet"]


@pytest.mark.asyncio
async def test_nan_score_is_ignored():
    kg = _RecordingKG()
    vector_results = [
        {"text": "Mike", "score": float("nan"), "id": 0},
        {"text": "November", "score": 1, "id": 1},
    ]
    await expand_from_results(kg, vector_results, max_seeds=2)
    assert kg.queried == ["November", "Mike"]
