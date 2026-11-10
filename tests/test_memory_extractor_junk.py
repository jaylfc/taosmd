"""Regression tests for junk filtering in the regex extractor (#653 item 2).

Fixture corpus is drawn from the shadow-deployment claims.db reported in
#653: real agent meta-text that the pattern extractor turned into bogus
"is_a" claims. Each bad case below currently produces a fact on master.
"""

import pytest

from taosmd.memory_extractor import extract_facts_from_text


def _objects(text: str) -> list[tuple[str, str, str]]:
    return [(f["subject"], f["predicate"], f["object"])
            for f in extract_facts_from_text(text)]


def test_object_starting_with_infinitive_to_is_rejected():
    # shadow corpus: "Your job is to publish the REAL stories"
    facts = extract_facts_from_text("Your job is to publish the REAL stories.")
    assert facts == []


def test_object_starting_with_adverb_is_rejected():
    # shadow corpus: "If there is genuinely nothing new to report"
    facts = extract_facts_from_text("If there is genuinely nothing new to report.")
    assert facts == []


def test_object_starting_with_preposition_is_rejected():
    # clause fragment: object begins with 'at'
    facts = extract_facts_from_text("The Pi is at 100 percent load.")
    assert facts == []


def test_trailing_adverb_object_is_rejected():
    # shadow corpus worst offender: "full skill content is a loaded below" x886
    facts = extract_facts_from_text("The full skill content is loaded below.")
    assert facts == []


def test_clause_fragment_subject_if_there_is_rejected():
    facts = extract_facts_from_text("If there is a server running it answers.")
    subjects = [f["subject"] for f in facts]
    assert all(not s.lower().startswith("if ") for s in subjects)


def test_vague_subject_last_word_rejected():
    # "Your job is a real thing" style fragments -- head noun carries no entity
    facts = extract_facts_from_text("Your job is a published story.")
    assert all(f["subject"].lower() != "your job" for f in facts)


def test_good_facts_still_extracted():
    text = (
        "Jay manages the Stalwart mail server. "
        "Dialtone uses Twilio for calls. "
        "The Pi runs Ubuntu server."
    )
    facts = extract_facts_from_text(text)
    triples = {(f["subject"].lower(), f["predicate"]) for f in facts}
    assert ("jay", "manages") in triples
    assert ("dialtone", "uses") in triples
    assert ("pi", "uses") in triples


# tsk-g6ga5l: trim trailing time words instead of dropping the fact;
# keep real 'X is a Y' facts with capitalized proper-name subjects.
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Bob moved to Berlin today.", [("Bob", "moved_to", "Berlin")]),
        ("Big Data is a company.", [("Big Data", "is_a", "company")]),
    ],
)
def test_trim_trailing_time_and_keep_proper_subject(text, expected):
    assert _objects(text) == expected
