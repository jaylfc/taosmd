import asyncio
from taosmd.claims.store import ClaimStore
from taosmd.claims.verifier import FakeVerifier
from taosmd.claims.verify_pass import verify_pass


def _store(tmp_path):
    s = ClaimStore(db_path=str(tmp_path / "claims.db"))
    asyncio.run(s.init())
    return s


async def _spans_text(span_ids):
    return [f"span-{i}-text" for i in span_ids]


def test_verify_pass_marks_status(tmp_path):
    s = _store(tmp_path)
    c1 = asyncio.run(s.add_claim("alice has a dog", [1], source_extractor="r"))
    c2 = asyncio.run(s.add_claim("bob flew to mars", [2], source_extractor="r"))
    v = FakeVerifier({"alice has a dog": "supported", "bob flew to mars": "unsupported"})
    n = asyncio.run(verify_pass(s, v, _spans_text, batch=10, now=5.0))
    assert n == 2
    assert asyncio.run(s.get(c1))["status"] == "supported"
    assert asyncio.run(s.get(c2))["status"] == "unsupported"


def test_verify_pass_is_idempotent(tmp_path):
    s = _store(tmp_path)
    asyncio.run(s.add_claim("x", [1], source_extractor="r"))
    v = FakeVerifier({"x": "supported"})
    asyncio.run(verify_pass(s, v, _spans_text, batch=10, now=1.0))
    n2 = asyncio.run(verify_pass(s, v, _spans_text, batch=10, now=2.0))
    assert n2 == 0


def test_verify_pass_fail_closed_terminates(tmp_path):
    s = _store(tmp_path)
    cid = asyncio.run(s.add_claim("x", [1], source_extractor="r"))
    class Boom:
        def verify(self, *a): return ("unverified", "m")
    # must terminate (the seen-guard), and leave the claim unverified
    n = asyncio.run(verify_pass(s, Boom(), _spans_text, batch=10, now=1.0))
    assert n == 0
    assert asyncio.run(s.get(cid))["status"] == "unverified"
