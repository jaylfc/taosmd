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


def test_prefer_verified_does_not_resort_across_source_scales():
    """Scores arrive in source-native units (BM25 vs cosine); re-sorting the
    pool by score demoted a rank-2 keyword hit to rank 10 in a live shadow
    deployment even though its claim status was fine. The gate must filter
    and front, never re-sort: retrieval order stays intact inside groups."""
    hits = [
        _hit("cos-a", 0.197, None),
        _hit("bm25-answer", 0.099, "unverified"),
        _hit("cos-b", 0.314, "unverified"),
        _hit("cos-c", 0.274, None),
        _hit("bad", 0.263, "contradicted"),
    ]
    out = apply_claims_gate(hits, mode="prefer_verified")
    ids = [h["id"] for h in out]
    assert "bad" not in ids                                     # drop unchanged
    assert ids == ["cos-a", "bm25-answer", "cos-b", "cos-c"]    # order preserved


def test_supported_hits_front_without_resorting_rest():
    hits = [
        _hit("raw-high", 0.9, None),
        _hit("supported-late", 0.1, "supported"),
        _hit("unverified-mid", 0.5, "unverified"),
    ]
    out = apply_claims_gate(hits, mode="prefer_verified")
    ids = [h["id"] for h in out]
    assert ids[0] == "supported-late"                   # supported still fronts
    assert ids[1:] == ["raw-high", "unverified-mid"]    # rest keeps retrieval order
