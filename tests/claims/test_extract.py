from taosmd.claims.extract import claims_from_text


def test_claims_carry_span_id():
    # "Alice is a doctor." -> {subject: Alice, predicate: is_a, object: doctor}
    out = claims_from_text("Alice is a doctor.", archive_span_id=42)
    assert out, "expected at least one claim"
    c = out[0]
    assert c["archive_span_ids"] == [42]
    assert "Alice" in c["text"] and "doctor" in c["text"]
    assert "_" not in c["text"]              # predicate underscores rendered as spaces
    assert c["source_extractor"] == "regex"


def test_empty_text_no_claims():
    assert claims_from_text("", archive_span_id=1) == []
    assert claims_from_text("   ", archive_span_id=1) == []
