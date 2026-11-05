"""Regression tests for the meta-text skip guard (#653 item 3).

Fixture corpus is drawn from the shadow-deployment claims.db reported in
#653: verbatim agent scaffolding (cron delivery instructions, delivery
tokens, system-prompt bullets) that the regex extractor turned into world
facts. Each bad case below produces a fact on unpatched master; the
positive controls prove the guard does not eat real prose.
"""

from taosmd.memory_extractor import extract_facts_from_text


def test_silent_token_sentence_is_skipped():
    # shadow corpus: cron prompt text -> is_a(genuinely nothing new to report)
    text = (
        "If there is genuinely nothing new to report, "
        "respond with exactly [SILENT] and nothing more."
    )
    assert extract_facts_from_text(text) == []


def test_final_response_instruction_is_skipped():
    # shadow corpus: delivery instruction lifted as an is_a fact
    text = "Your final response is automatically delivered to the user."
    assert extract_facts_from_text(text) == []


def test_skill_content_injection_is_skipped():
    # shadow corpus worst offender x886: "full skill content is a loaded below"
    text = "The full skill content is loaded below."
    assert extract_facts_from_text(text) == []


def test_bullet_artifact_line_is_skipped():
    # bare "- " bullet lines are prompt scaffolding, not conversational prose
    text = "- The Pi runs Ubuntu server."
    assert extract_facts_from_text(text) == []


def test_media_token_sentence_is_skipped():
    # shadow corpus: scaffolding line -> is_a(delivery instruction ...)
    text = "A MEDIA: token is a delivery instruction in your final response."
    assert extract_facts_from_text(text) == []


def test_real_facts_survive_the_guard():
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


def test_mixed_turn_keeps_real_sentences():
    turn = (
        "Your final response is automatically delivered to the user.\n"
        "Jay moved to a Hetzner box."
    )
    facts = extract_facts_from_text(turn)
    triples = {(f["subject"].lower(), f["predicate"]) for f in facts}
    assert ("jay", "moved_to") in triples
    assert all("final response" not in f["subject"].lower() for f in facts)
