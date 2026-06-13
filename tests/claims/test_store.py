"""ClaimStore: zero-loss sqlite store for verifiable claims."""
import asyncio
from taosmd.claims.store import ClaimStore, VALID_STATUSES


def _store(tmp_path):
    s = ClaimStore(db_path=str(tmp_path / "claims.db"))
    asyncio.run(s.init())
    return s


def test_add_and_get(tmp_path):
    s = _store(tmp_path)
    cid = asyncio.run(s.add_claim("alice adopted a dog", [12, 13], source_extractor="regex"))
    row = asyncio.run(s.get(cid))
    assert row["text"] == "alice adopted a dog"
    assert row["archive_span_ids"] == [12, 13]
    assert row["status"] == "unverified"
    assert row["source_extractor"] == "regex"


def test_set_status_valid(tmp_path):
    s = _store(tmp_path)
    cid = asyncio.run(s.add_claim("x", [1], source_extractor="r"))
    asyncio.run(s.set_status(cid, "supported", verifier_model="qwen3:4b", now=100.0))
    row = asyncio.run(s.get(cid))
    assert row["status"] == "supported"
    assert row["verifier_model"] == "qwen3:4b"
    assert row["last_checked"] == 100.0


def test_set_status_rejects_unknown(tmp_path):
    s = _store(tmp_path)
    cid = asyncio.run(s.add_claim("x", [1], source_extractor="r"))
    try:
        asyncio.run(s.set_status(cid, "bogus", verifier_model="m", now=1.0))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_pull_unverified(tmp_path):
    s = _store(tmp_path)
    a = asyncio.run(s.add_claim("a", [1], source_extractor="r"))
    b = asyncio.run(s.add_claim("b", [2], source_extractor="r"))
    asyncio.run(s.set_status(a, "supported", verifier_model="m", now=1.0))
    pending = asyncio.run(s.pull_unverified(limit=10))
    assert [c["id"] for c in pending] == [b]


def test_rate(tmp_path):
    s = _store(tmp_path)
    ids = [asyncio.run(s.add_claim(t, [1], source_extractor="r")) for t in "abcd"]
    asyncio.run(s.set_status(ids[0], "supported", verifier_model="m", now=1.0))
    asyncio.run(s.set_status(ids[1], "supported", verifier_model="m", now=1.0))
    asyncio.run(s.set_status(ids[2], "unsupported", verifier_model="m", now=1.0))
    r = asyncio.run(s.rate())
    assert r["supported"] == 2 and r["unsupported"] == 1 and r["unverified"] == 1
    assert abs(r["hallucination_rate"] - (1 / 3)) < 1e-9


def test_status_set_constant():
    assert "unverified" in VALID_STATUSES and "supported" in VALID_STATUSES
