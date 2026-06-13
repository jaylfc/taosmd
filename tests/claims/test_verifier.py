from taosmd.claims.verifier import FakeVerifier, parse_verdict, VERDICTS


def test_fake_scripted():
    v = FakeVerifier({"alice has a dog": "supported"})
    assert v.verify("alice has a dog", ["alice adopted a golden retriever"]) == ("supported", "fake")


def test_fake_default_unverified_on_unknown():
    v = FakeVerifier({}, default="unsupported")
    assert v.verify("x", ["y"]) == ("unsupported", "fake")


def test_parse_verdict_maps_model_text():
    assert parse_verdict("SUPPORTED") == "supported"
    assert parse_verdict("the claim is UNSUPPORTED by the text") == "unsupported"
    assert parse_verdict("PARTIAL") == "partial"


def test_parse_verdict_unknown_is_unverified():
    # An unparseable judge reply must fail closed, never silently "supported".
    assert parse_verdict("i am not sure, maybe?") == "unverified"


def test_verdicts_constant():
    assert set(VERDICTS) >= {"supported", "partial", "unsupported"}
