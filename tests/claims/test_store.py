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


# --- dedup at insert (#653) ---------------------------------------------


def test_add_claim_dedups_by_normalized_text(tmp_path):
    """The same sentence re-extracted every turn must not pile up rows.

    A multi-day shadow deployment measured 10,063 rows for 3,915 distinct
    texts (61% duplicates) because add_claim inserted unconditionally.
    """
    s = _store(tmp_path)
    a = asyncio.run(s.add_claim("Alice lives in Paris", [1], source_extractor="regex"))
    b = asyncio.run(s.add_claim("  alice lives IN   paris ", [2, 3],
                                source_extractor="regex"))
    assert a == b, "case/whitespace variants must merge into one row"
    row = asyncio.run(s.get(a))
    assert row["archive_span_ids"] == [1, 2, 3], "spans merge, zero-loss"
    r = asyncio.run(s.rate())
    assert r["unverified"] == 1


def test_dedup_merge_keeps_verified_status(tmp_path):
    s = _store(tmp_path)
    a = asyncio.run(s.add_claim("x is a thing", [1], source_extractor="regex"))
    asyncio.run(s.set_status(a, "supported", verifier_model="m", now=1.0))
    b = asyncio.run(s.add_claim("x is a thing", [9], source_extractor="regex"))
    assert a == b
    row = asyncio.run(s.get(a))
    assert row["status"] == "supported", "re-add must not reset verification"
    assert row["archive_span_ids"] == [1, 9]


def test_different_claims_stay_distinct(tmp_path):
    """Dedup is exact-normalized, not fuzzy: paraphrases keep their own rows."""
    s = _store(tmp_path)
    a = asyncio.run(s.add_claim("alice lives in Paris", [1], source_extractor="regex"))
    b = asyncio.run(s.add_claim("alice lives near Paris", [1], source_extractor="regex"))
    assert a != b


def test_concurrent_add_claim_same_text_single_row(tmp_path):
    """Two processes writing the same claim must not both insert (CodeRabbit #654).

    Without the BEGIN IMMEDIATE wrapper the norm lookup runs in autocommit,
    so two connections can both miss the lookup and each insert a row;
    idx_claims_norm is non-unique so nothing rejects the second.
    """
    import threading

    n = 8
    barrier = threading.Barrier(n)
    ids: list[int] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        async def go() -> None:
            s = ClaimStore(db_path=str(tmp_path / "claims.db"))
            await s.init()
            barrier.wait()
            cid = await s.add_claim("alice lives in Paris", [i], source_extractor="regex")
            with lock:
                ids.append(cid)
            await s.close()
        asyncio.run(go())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(ids) == n
    assert len(set(ids)) == 1, f"all writers must converge on one row id, got {sorted(ids)}"
    s = ClaimStore(db_path=str(tmp_path / "claims.db"))
    asyncio.run(s.init())
    row = asyncio.run(s.get(ids[0]))
    assert sorted(row["archive_span_ids"]) == list(range(n)), "every writer's span merged"
    asyncio.run(s.close())


def test_legacy_db_without_norm_backfills_and_dedups(tmp_path):
    """A claims.db predating the norm column still dedups after init."""
    import sqlite3

    db = str(tmp_path / "claims.db")
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            archive_span_ids TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'unverified',
            verifier_model TEXT,
            last_checked REAL,
            source_extractor TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL
        );
        INSERT INTO claims (text, archive_span_ids, source_extractor, created_at)
        VALUES ('bob loves olives', '[7]', 'regex', 1.0);
        """
    )
    conn.commit()
    conn.close()

    s = ClaimStore(db_path=db)
    asyncio.run(s.init())
    cid = asyncio.run(s.add_claim("Bob loves olives", [8], source_extractor="regex"))
    assert cid == 1, "legacy row must be found by its backfilled norm"
    row = asyncio.run(s.get(1))
    assert row["archive_span_ids"] == [7, 8]
