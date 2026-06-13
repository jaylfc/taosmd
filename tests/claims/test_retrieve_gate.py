"""The claims gate wired into the recall path (default off)."""
import asyncio

from taosmd import api as a
from taosmd.claims.store import ClaimStore


def _cs(tmp_path):
    cs = ClaimStore(db_path=str(tmp_path / "claims.db"))
    asyncio.run(cs.init())
    return cs


def test_gate_demotes_unsupported_hit(tmp_path):
    cs = _cs(tmp_path)
    cid = asyncio.run(cs.add_claim("bad fact", [99], source_extractor="r"))
    asyncio.run(cs.set_status(cid, "unsupported", verifier_model="m", now=1.0))
    hits = [
        {"text": "a", "confidence": 0.9, "metadata": {"archive_span_id": 99}},
        {"text": "b", "confidence": 0.5, "metadata": {"archive_span_id": 7}},
    ]
    gated = asyncio.run(a._attach_and_gate_claims(hits, cs, "prefer_verified"))
    assert [h["text"] for h in gated] == ["b"]      # unsupported-backed hit dropped
    assert "claim_status" not in gated[0]           # transient keys stripped
    assert "score" not in gated[0]


def test_gate_prefers_supported(tmp_path):
    cs = _cs(tmp_path)
    cid = asyncio.run(cs.add_claim("good", [5], source_extractor="r"))
    asyncio.run(cs.set_status(cid, "supported", verifier_model="m", now=1.0))
    hits = [
        {"text": "raw", "confidence": 0.9, "metadata": {"archive_span_id": 1}},
        {"text": "verified", "confidence": 0.5, "metadata": {"archive_span_id": 5}},
    ]
    gated = asyncio.run(a._attach_and_gate_claims(hits, cs, "prefer_verified"))
    assert gated[0]["text"] == "verified"           # supported boosted above higher-confidence raw


def test_gate_off_is_passthrough(tmp_path):
    cs = _cs(tmp_path)
    hits = [{"text": "a", "confidence": 0.9, "metadata": {"archive_span_id": 99}}]
    out = asyncio.run(a._attach_and_gate_claims(hits, cs, "off"))
    assert out == hits
