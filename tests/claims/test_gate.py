from taosmd.claims.gate import apply_claims_gate


def _hit(i, score, status):
    return {"id": i, "score": score, "claim_status": status}


def test_prefer_verified_excludes_unsupported():
    hits = [_hit(1, 0.9, "unsupported"), _hit(2, 0.5, "supported"), _hit(3, 0.4, None)]
    out = apply_claims_gate(hits, mode="prefer_verified")
    ids = [h["id"] for h in out]
    assert 1 not in ids        # unsupported dropped from default
    assert ids[0] == 2         # supported boosted above the unscored raw hit
    assert 3 in ids            # raw (non-claim) hit kept


def test_prefer_verified_keeps_unverified():
    hits = [_hit(1, 0.9, "unverified"), _hit(2, 0.5, "supported")]
    out = apply_claims_gate(hits, mode="prefer_verified")
    assert {h["id"] for h in out} == {1, 2}   # unverified kept in the lenient mode


def test_strict_excludes_unverified_too():
    hits = [_hit(1, 0.9, "unverified"), _hit(2, 0.5, "supported"), _hit(3, 0.4, "unsupported")]
    out = apply_claims_gate(hits, mode="strict")
    assert {h["id"] for h in out} == {2}      # only proven-supported survive strict


def test_off_is_identity():
    hits = [_hit(1, 0.9, "unsupported")]
    assert apply_claims_gate(hits, mode="off") == hits


def test_empty():
    assert apply_claims_gate([], mode="strict") == []
