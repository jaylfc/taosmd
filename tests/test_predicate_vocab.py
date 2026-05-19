"""Tests for taosmd.predicate_vocab — closed-vocab enforcement + regex inference."""

from __future__ import annotations

import asyncio
import logging

import pytest

from taosmd.predicate_vocab import (
    ALLOWED_PREDICATES,
    PREDICATE_CATEGORIES,
    SYNONYMS,
    categories,
    extract_with_vocab,
    is_allowed,
    normalise,
    validate,
)
from taosmd.knowledge_graph import TemporalKnowledgeGraph


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Vocab membership + normalisation
# ---------------------------------------------------------------------------

def test_allowed_predicates_set_is_consistent_with_categories():
    """Every predicate in PREDICATE_CATEGORIES must show up in ALLOWED_PREDICATES."""
    assert ALLOWED_PREDICATES == frozenset(PREDICATE_CATEGORIES)


def test_normalise_lowercases_and_underscores():
    assert normalise("Works At") == "works_at"
    assert normalise("WORKS-AT") == "works_at"
    assert normalise(" works_at ") == "works_at"


def test_normalise_resolves_synonyms():
    assert normalise("employed_by") == "works_at"
    assert normalise("co_founded") == "founded"
    assert normalise("married_to") == "spouse_of"
    assert normalise("born_in") == "birth_place"


def test_normalise_returns_unknown_unchanged():
    assert normalise("frobnicated") == "frobnicated"


def test_normalise_empty_returns_empty():
    assert normalise("") == ""
    assert normalise("   ") == ""


def test_is_allowed_accepts_canonical_and_synonyms():
    assert is_allowed("works_at")
    assert is_allowed("employed_by")    # synonym -> works_at
    assert is_allowed("Works At")        # casing
    assert not is_allowed("dabbles_in")
    assert not is_allowed("")


def test_validate_strict_rejects_unknown():
    with pytest.raises(ValueError, match="not in the closed vocab"):
        validate("frobnicated", strict=True)
    with pytest.raises(ValueError, match="empty predicate"):
        validate("", strict=True)


def test_validate_strict_accepts_synonym_returns_canonical():
    assert validate("employed_by", strict=True) == "works_at"
    assert validate("co_founded", strict=True) == "founded"


def test_validate_nonstrict_warns_but_returns(caplog):
    with caplog.at_level(logging.WARNING, logger="taosmd.predicate_vocab"):
        result = validate("dabbles_in", strict=False)
    assert result == "dabbles_in"
    assert any("not in the closed vocab" in r.message for r in caplog.records)


def test_categories_groups_predicates():
    grouped = categories()
    assert "employment" in grouped
    assert "works_at" in grouped["employment"]
    assert "founded" in grouped["employment"]
    assert "knows" in grouped["social"]
    assert "lives_in" in grouped["location"]
    # Every predicate must appear in exactly one category.
    flat = [p for preds in grouped.values() for p in preds]
    assert sorted(flat) == sorted(ALLOWED_PREDICATES)


# ---------------------------------------------------------------------------
# Regex inference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_pred,expected_obj", [
    ("Alice works at Stripe.", "works_at", "Stripe"),
    ("Bob founded Acme Co.", "founded", "Acme Co"),
    ("Carol invested in OpenAI.", "invested_in", "OpenAI"),
    ("Dave advises three startups.", "advises", "three startups"),
    ("Eve attended YC W24.", "attended", "YC W24"),
    ("Frank lives in Berlin.", "lives_in", "Berlin"),
    ("Greta was born in Stockholm.", "birth_place", "Stockholm"),
    ("Henrietta is married to Igor.", "spouse_of", "Igor"),
    ("Jack prefers emacs.", "prefers", "emacs"),
])
def test_extract_with_vocab_recognises_patterns(text, expected_pred, expected_obj):
    facts = extract_with_vocab(text)
    assert any(
        f["predicate"] == expected_pred and f["object"].lower() == expected_obj.lower()
        for f in facts
    ), f"expected {expected_pred} on {text!r}, got {facts}"


def test_extract_with_vocab_emits_only_vocab_predicates():
    text = (
        "Alice works at Stripe. "
        "Bob founded Acme. "
        "Carol invested in OpenAI. "
        "Dave attended YC. "
        "Eve lives in Berlin."
    )
    facts = extract_with_vocab(text)
    assert facts, "expected at least one fact"
    for f in facts:
        assert f["predicate"] in ALLOWED_PREDICATES, (
            f"predicate {f['predicate']!r} emitted but not in closed vocab"
        )


def test_extract_with_vocab_dedupes():
    text = "Alice works at Stripe. Alice works at Stripe."
    facts = extract_with_vocab(text)
    works_at = [f for f in facts if f["predicate"] == "works_at"]
    assert len(works_at) == 1


# ---------------------------------------------------------------------------
# KG integration — add_triple honours strict_vocab + normalises by default
# ---------------------------------------------------------------------------

def test_add_triple_normalises_synonym_by_default(tmp_path):
    async def go():
        kg = TemporalKnowledgeGraph(db_path=tmp_path / "kg.db")
        await kg.init()
        try:
            # "employed_by" should normalise to "works_at".
            tid = await kg.add_triple(
                subject="alice", predicate="employed_by", obj="stripe",
            )
            assert tid
            # query_entity should find the canonical predicate.
            triples = await kg.query_entity("alice")
            return [t["predicate"] for t in triples]
        finally:
            await kg.close()

    preds = _run(go())
    assert "works_at" in preds
    assert "employed_by" not in preds


def test_add_triple_strict_vocab_rejects_unknown(tmp_path):
    async def go():
        kg = TemporalKnowledgeGraph(db_path=tmp_path / "kg.db")
        await kg.init()
        try:
            with pytest.raises(ValueError, match="not in the closed vocab"):
                await kg.add_triple(
                    subject="alice", predicate="frobnicated", obj="stripe",
                    strict_vocab=True,
                )
        finally:
            await kg.close()
    _run(go())


def test_add_triple_strict_vocab_accepts_canonical(tmp_path):
    async def go():
        kg = TemporalKnowledgeGraph(db_path=tmp_path / "kg.db")
        await kg.init()
        try:
            tid = await kg.add_triple(
                subject="alice", predicate="works_at", obj="stripe",
                strict_vocab=True,
            )
            assert tid
        finally:
            await kg.close()
    _run(go())


def test_add_triple_nonstrict_still_accepts_unknown(tmp_path, caplog):
    """Backwards-compat: existing extractors that emit free-form predicates
    keep working — they just generate a warning."""
    async def go():
        kg = TemporalKnowledgeGraph(db_path=tmp_path / "kg.db")
        await kg.init()
        try:
            with caplog.at_level(logging.WARNING, logger="taosmd.predicate_vocab"):
                tid = await kg.add_triple(
                    subject="alice", predicate="frobnicated", obj="stripe",
                )
            return tid
        finally:
            await kg.close()
    tid = _run(go())
    assert tid
    assert any("not in the closed vocab" in r.message for r in caplog.records)
