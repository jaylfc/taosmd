"""Tests for the timeline intent type in the intent classifier."""

import re

import pytest
from taosmd.intent_classifier import (
    classify_intent,
    get_search_strategy,
    INTENT_PATTERNS,
    INTENT_PRIORITY,
    INTENT_TIMELINE,
    INTENT_FACTUAL,
    INTENT_TECHNICAL,
    INTENT_PREFERENCE,
)


def test_timeline_queries():
    """Queries about activity, sessions, and time-based work should classify as timeline."""
    timeline_queries = [
        "What was I working on Monday morning?",
        "Show me my activity on Monday",
        "What did I do last Tuesday?",
        "Show my activity for this week",
        "Last Friday session recap",
    ]
    for query in timeline_queries:
        result = classify_intent(query)
        assert result == INTENT_TIMELINE, (
            f"Expected 'timeline' for query {query!r}, got {result!r}"
        )


def test_timeline_strategy_has_catalog():
    """Timeline strategy should use catalog as primary with catalog_weight=1.0."""
    strategy = get_search_strategy("What was I working on last Monday?")
    assert strategy["intent"] == INTENT_TIMELINE
    assert strategy["primary"] == "catalog"
    assert strategy["catalog_weight"] == 1.0


def test_non_timeline_queries_unchanged():
    """Non-timeline queries should not be classified as timeline."""
    assert classify_intent("What is taOS?") == INTENT_FACTUAL
    assert classify_intent("How does embedding work?") == INTENT_TECHNICAL
    assert classify_intent("What does Jay prefer?") == INTENT_PREFERENCE


def _scores(query: str) -> dict:
    """Replicate the classifier's scoring for tie inspection in tests."""
    ql = query.lower().strip()
    boost = {"recent": 2, "preference": 2, "timeline": 2}
    out = {}
    for intent, patterns in INTENT_PATTERNS.items():
        s = sum(boost.get(intent, 1) for p in patterns if re.search(p, ql))
        if s:
            out[intent] = s
    return out


def test_tie_resolves_by_explicit_priority():
    """A genuine score tie must resolve to the higher-priority intent.

    "What is the architecture" matches one factual pattern ("what is") and
    one technical pattern ("architecture") — a 1-1 tie. Technical outranks
    factual in INTENT_PRIORITY, so technical must win.
    """
    query = "What is the architecture"
    scores = _scores(query)
    assert scores.get(INTENT_FACTUAL) == scores.get(INTENT_TECHNICAL) == 1
    assert INTENT_PRIORITY.index(INTENT_TECHNICAL) < INTENT_PRIORITY.index(
        INTENT_FACTUAL
    )
    assert classify_intent(query) == INTENT_TECHNICAL


def test_priority_independent_of_pattern_decl_order():
    """Tie-break must follow INTENT_PRIORITY, not dict declaration order."""
    # Every pattern-bearing intent appears in the explicit priority list.
    for intent in INTENT_PATTERNS:
        assert intent in INTENT_PRIORITY
    # Priority is intentionally different from declaration order (factual is
    # declared first but is near-last in priority).
    decl_order = list(INTENT_PATTERNS.keys())
    assert decl_order[0] == INTENT_FACTUAL
    assert INTENT_PRIORITY.index(INTENT_FACTUAL) > INTENT_PRIORITY.index(
        INTENT_TECHNICAL
    )
