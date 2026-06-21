"""Tests for the memory category classifier."""
from __future__ import annotations

from taosmd import categories as C


def test_classify_known_signals():
    assert C.classify("I prefer metric units and dark mode") == "identity"
    assert C.classify("Shipped the benchmark feature for the project") == "work"
    assert C.classify("Had lunch with my colleague and the team") == "people"
    assert C.classify("Wrote a long email and read an article") == "communication"
    assert C.classify("Connected my Yelp account and rotated the api key") == "sources"
    assert C.classify("We booked a hotel and a flight for the trip") == "activities"


def test_classify_empty_and_unknown():
    assert C.classify("") == "uncategorized"
    assert C.classify(None) == "uncategorized"
    assert C.classify("xyzzy plugh frobnicate") == "uncategorized"


def test_classify_is_llm_free_and_deterministic():
    # Same input, same output, no network or model needed.
    text = "My birthday is in March and I prefer the metric system"
    assert C.classify(text) == C.classify(text) == "identity"


def test_category_counts_shape_and_order():
    texts = [
        "I prefer dark mode",          # identity
        "My favorite colour is blue",  # identity
        "Deployed the project today",  # work
        "xyzzy",                       # uncategorized
    ]
    counts = C.category_counts(texts)
    assert all(set(d) == {"name", "count"} for d in counts)
    assert counts[0]["name"] == "Identity & Preferences" and counts[0]["count"] == 2
    # highest first
    assert [d["count"] for d in counts] == sorted((d["count"] for d in counts), reverse=True)
