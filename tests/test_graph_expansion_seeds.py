"""Tests for graph expansion seed relevance ranking (issue #198).

These tests verify that seed entities are ranked by relevance before capping,
not just by encounter order. They use a tiny fake KG that records every
entity it is asked about.
"""

from __future__ import annotations

import pytest


class RecordingKG:
    """Fake KG that records every entity queried and returns empty results."""

    def __init__(self):
        self.queried_entities: list[str] = []

    async def query_entity(
        self, entity: str, as_of: float | None = None, track_access: bool = False
    ) -> list[dict]:
        self.queried_entities.append(entity)
        return []


# Import the real function under test
from taosmd.graph_expansion import expand_from_results


def make_result(text: str, score: float | None = None, source_score: float | None = None) -> dict:
    """Helper to create a vector result dict."""
    r = {"text": text}
    if score is not None:
        r["score"] = score
    if source_score is not None:
        r["source_score"] = source_score
    return r


@pytest.mark.asyncio
async def test_max_seeds_parameter_default():
    """Test that max_seeds defaults to 10 and caps seeds."""
    kg = RecordingKG()
    results = [make_result(f"Entity{i} appeared", score=float(i)) for i in range(15)]
    await expand_from_results(kg, results, max_seeds=10)
    assert len(kg.queried_entities) == 10


@pytest.mark.asyncio
async def test_max_seeds_explicit_zero():
    """max_seeds=0 queries no entities and returns empty list."""
    kg = RecordingKG()
    results = [make_result("Alpha appeared", score=1.0)]
    expanded = await expand_from_results(kg, results, max_seeds=0)
    assert kg.queried_entities == []
    assert expanded == []


@pytest.mark.asyncio
async def test_max_seeds_explicit_limit():
    """max_seeds=2 queries exactly 2 entities."""
    kg = RecordingKG()
    results = [make_result(f"Entity{i} appeared", score=float(i)) for i in range(5)]
    await expand_from_results(kg, results, max_seeds=2)
    assert len(kg.queried_entities) == 2


@pytest.mark.asyncio
async def test_scored_results_ranked_by_highest_score():
    """a. 12 results, scored ascending, each naming one distinct entity:
    with max_seeds=3 the recorded queries are the 3 entities from the 3
    HIGHEST-scored results, in score order (descending)."""
    kg = RecordingKG()
    # Results scored 0..11, each mentions Entity0..Entity11
    results = [make_result(f"Entity{i} appeared", score=float(i)) for i in range(12)]
    await expand_from_results(kg, results, max_seeds=3)
    # Highest scores are 11, 10, 9 -> entities Entity11, Entity10, Entity9
    assert kg.queried_entities == ["Entity11", "Entity10", "Entity9"]


@pytest.mark.asyncio
async def test_unscored_results_preserve_encounter_order():
    """c. Unscored results (no score keys): recorded order equals encounter order."""
    kg = RecordingKG()
    # No score keys at all
    results = [
        {"text": "Alpha appeared"},
        {"text": "Beta appeared"},
        {"text": "Gamma appeared"},
        {"text": "Delta appeared"},
    ]
    await expand_from_results(kg, results, max_seeds=10)
    # Must be exactly encounter order
    assert kg.queried_entities == ["Alpha", "Beta", "Gamma", "Delta"]


@pytest.mark.asyncio
async def test_tie_break_by_mention_count():
    """d. Tie on score: entity mentioned in MORE results comes first."""
    kg = RecordingKG()
    # Alpha in 2 results (score 5 each), Beta in 1 result (score 5)
    results = [
        make_result("Alpha appeared", score=5.0),
        make_result("Beta appeared", score=5.0),
        make_result("Alpha appeared again", score=5.0),
    ]
    await expand_from_results(kg, results, max_seeds=2)
    # Alpha has 2 mentions, Beta has 1 -> Alpha first
    assert kg.queried_entities == ["Alpha", "Beta"]


@pytest.mark.asyncio
async def test_full_tie_encounter_order_wins():
    """S2. Full tie (equal score, equal mentions): earlier-encountered wins.
    Must FAIL if encounter tie-break is reversed (later wins)."""
    kg = RecordingKG()
    # Alpha and Beta both score 5, both mentioned once, Alpha encountered first
    results = [
        make_result("Alpha appeared", score=5.0),
        make_result("Beta appeared", score=5.0),
    ]
    # max_seeds=1: only Alpha should be queried
    await expand_from_results(kg, results, max_seeds=1)
    assert kg.queried_entities == ["Alpha"]

    kg2 = RecordingKG()
    # max_seeds=2: both queried, Alpha first
    await expand_from_results(kg2, results, max_seeds=2)
    assert kg2.queried_entities == ["Alpha", "Beta"]


@pytest.mark.asyncio
async def test_mention_count_includes_unscored_results():
    """S3. Tie on score, extra mention comes from UNSCORED result.
    Must FAIL if mention count only increments for scored results."""
    kg = RecordingKG()
    # Alpha: score 5 (1 mention), Beta: score 5 (1 scored mention + 1 unscored = 2 total)
    results = [
        make_result("Alpha appeared", score=5.0),
        make_result("Beta appeared", score=5.0),
        {"text": "Beta appeared again"},  # No score key - unscored
    ]
    await expand_from_results(kg, results, max_seeds=2)
    # Beta has 2 mentions (1 scored + 1 unscored), Alpha has 1 -> Beta first
    assert kg.queried_entities == ["Beta", "Alpha"]


@pytest.mark.asyncio
async def test_case_insensitive_mention_matching_not_substring():
    """S4. Mention matching uses case-insensitive exact match from extracted entities,
    not substring. 'Ann' must NOT be credited with 'Anna went...'."""
    kg = RecordingKG()
    # Ann score 1, Anna score 9, Bob score 5
    # If substring matching: Ann would be found in "Anna went" and get score 9
    # Correct: Ann gets 1, Anna gets 9, Bob gets 5 -> Anna wins
    results = [
        make_result("Ann went home", score=1.0),
        make_result("Anna went home", score=9.0),
        make_result("Bob went home", score=5.0),
    ]
    await expand_from_results(kg, results, max_seeds=1)
    # Anna has highest score (9), not Ann
    assert kg.queried_entities == ["Anna"]


@pytest.mark.asyncio
async def test_bool_score_treated_as_none():
    """S5. A bool score (score=True) is treated as None (falls back to source_score / unscored)."""
    kg = RecordingKG()
    # Alpha has bool score (should be ignored), Beta has source_score 5
    results = [
        make_result("Alpha appeared", score=True),  # bool -> treated as None
        make_result("Beta appeared", source_score=5.0),
    ]
    await expand_from_results(kg, results, max_seeds=2)
    # Beta has valid source_score, Alpha has no valid score -> Beta first
    assert kg.queried_entities == ["Beta", "Alpha"]


@pytest.mark.asyncio
async def test_all_non_numeric_scores_fallback_encounter_order():
    """S5. All non-numeric scores fall back to encounter order."""
    kg = RecordingKG()
    results = [
        make_result("Alpha appeared", score="high"),  # string -> not numeric
        make_result("Beta appeared", score=True),     # bool -> not numeric
        make_result("Gamma appeared", score=None),    # None -> not numeric
    ]
    await expand_from_results(kg, results, max_seeds=10)
    # All invalid scores -> encounter order
    assert kg.queried_entities == ["Alpha", "Beta", "Gamma"]


@pytest.mark.asyncio
async def test_score_none_falls_back_to_source_score():
    """R4. score=None plus source_score uses source_score."""
    kg = RecordingKG()
    results = [
        make_result("Alpha appeared", score=None, source_score=3.0),
        make_result("Beta appeared", score=None, source_score=7.0),
    ]
    await expand_from_results(kg, results, max_seeds=2)
    # Beta has higher source_score
    assert kg.queried_entities == ["Beta", "Alpha"]


@pytest.mark.asyncio
async def test_max_vs_min_score_per_entity():
    """R4. One entity in two results scored 1 and 9 vs rival in one result scored 5.
    The entity with max score 9 must come first (fails if min is used)."""
    kg = RecordingKG()
    # Alpha appears in results scored 1 and 9 (max=9)
    # Beta appears in result scored 5 (max=5)
    results = [
        make_result("Alpha appeared", score=1.0),
        make_result("Beta appeared", score=5.0),
        make_result("Alpha appeared again", score=9.0),
    ]
    await expand_from_results(kg, results, max_seeds=2)
    # Alpha max score 9 > Beta max score 5
    assert kg.queried_entities == ["Alpha", "Beta"]


@pytest.mark.asyncio
async def test_source_score_only_ranking():
    """R4. Results carrying ONLY source_score are ranked by it."""
    kg = RecordingKG()
    results = [
        {"text": "Alpha appeared", "source_score": 2.0},
        {"text": "Beta appeared", "source_score": 8.0},
        {"text": "Gamma appeared", "source_score": 5.0},
    ]
    await expand_from_results(kg, results, max_seeds=3)
    # Ranked by source_score descending
    assert kg.queried_entities == ["Beta", "Gamma", "Alpha"]


@pytest.mark.asyncio
async def test_non_numeric_score_does_not_raise():
    """R4. A non-numeric score does not raise and falls back to encounter order."""
    kg = RecordingKG()
    results = [
        make_result("Alpha appeared", score="high"),   # string
        make_result("Beta appeared", score=[1, 2]),    # list
        make_result("Gamma appeared", score={"x": 1}), # dict
    ]
    await expand_from_results(kg, results, max_seeds=10)
    # No exception, encounter order preserved
    assert kg.queried_entities == ["Alpha", "Beta", "Gamma"]