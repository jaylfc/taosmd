"""Tests for the timeline intent type in the intent classifier."""

import pytest
from taosmd.intent_classifier import (
    classify_intent,
    get_search_strategy,
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
