import asyncio
import sqlite3

import pytest

from taosmd.claims.extract import (
    REGEX_EXTRACTOR_VERSION,
    REGEX_PATTERN_HASH,
    claims_from_text,
    llm_extractor_id,
    regex_extractor_id,
)
from taosmd.claims.store import ClaimStore
from taosmd.migrations import table_exists


def _store(tmp_path):
    s = ClaimStore(db_path=str(tmp_path / "claims.db"))
    asyncio.run(s.init())
    return s


def test_regex_extractor_id():
    assert regex_extractor_id() == "regex@1"
    assert REGEX_EXTRACTOR_VERSION == 1


def test_regex_pattern_hash_pinned():
    assert (REGEX_EXTRACTOR_VERSION, REGEX_PATTERN_HASH) == (
        1,
        "5bf6443d5a55bb040b83535868a2009389110bad053c612c7cfb86524bb578ac",
    ), "regex extraction behaviour changed: bump REGEX_EXTRACTOR_VERSION in taosmd/claims/extract.py and update this pinned hash in the same commit"


def test_llm_extractor_id_changes_with_prompt():
    a = llm_extractor_id("qwen3:4b", "prompt A")
    b = llm_extractor_id("qwen3:4b", "prompt B")
    c = llm_extractor_id("qwen3:4b", "prompt A")
    assert a != b
    assert a == c
    assert a.startswith("llm:qwen3:4b:")


def test_claims_from_text_stamps_regex_at_1():
    out = claims_from_text("Alice is a doctor.", archive_span_id=42)
    assert out
    assert all(c["source_extractor"] == "regex@1" for c in out)


def test_migration_creates_reextract_log_fresh(tmp_path):
    s = _store(tmp_path)
    conn = sqlite3.connect(str(tmp_path / "claims.db"))
    assert table_exists(conn, "reextract_log")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(reextract_log)")}
    assert cols == {
        "span_id", "extractor_id", "run_id", "extracted_at",
        "claims_found", "claims_new", "claims_matched", "target",
    }
    pk = conn.execute("PRAGMA table_info(reextract_log)").fetchall()
    # PRIMARY KEY(span_id, extractor_id, target) -> pk positions non-zero
    pk_cols = {r[1] for r in pk if r[5] > 0}
    assert pk_cols == {"span_id", "extractor_id", "target"}
    notnull = {r[1]: r[3] for r in pk}
    for c in cols:
        assert notnull[c] == 1
    conn.close()
    asyncio.run(s.close())


def test_migration_creates_reextract_log_on_master_schema_db(tmp_path):
    """DB built by today's master claims schema (baseline + norm_dedup) upgrades."""
    from taosmd.migrations import migrate
    from taosmd.claims.store import _SCHEMA

    db = tmp_path / "claims.db"
    conn = sqlite3.connect(db)
    conn.executescript(_SCHEMA)
    conn.commit()
    # simulate state after migrations 1,2 (current master)
    migrate(conn, "claims")
    conn.close()

    # reopen via store init (should be no-op upgrade-safe)
    s = ClaimStore(db_path=str(db))
    asyncio.run(s.init())
    conn = sqlite3.connect(db)
    assert table_exists(conn, "reextract_log")
    conn.close()
    asyncio.run(s.close())


def test_log_coverage_twice_raises_integrity_error(tmp_path):
    s = _store(tmp_path)
    row = {
        "span_id": 7, "extractor_id": "regex@1", "run_id": "r1",
        "extracted_at": 1.0, "claims_found": 2, "claims_new": 2,
        "claims_matched": 0, "target": "claims",
    }
    asyncio.run(s.log_coverage(dict(row)))
    with pytest.raises(sqlite3.IntegrityError):
        asyncio.run(s.log_coverage(dict(row)))
    asyncio.run(s.close())


def test_add_claim_ex_reports_created_and_unions_spans(tmp_path):
    s = _store(tmp_path)
    cid1, created1 = asyncio.run(s.add_claim_ex("Alice lives in Paris", [1], "regex@1"))
    cid2, created2 = asyncio.run(s.add_claim_ex("  alice lives IN   paris ", [2, 3], "regex@1"))
    assert cid1 == cid2
    assert created1 is True
    assert created2 is False
    row = asyncio.run(s.get(cid1))
    assert row["archive_span_ids"] == [1, 2, 3]
    asyncio.run(s.close())


def test_rollback_after_commit_false_leaves_both_tables_unchanged(tmp_path):
    s = _store(tmp_path)
    async def run():
        await s.init() if s._conn is None else None
        conn = s._conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            cid, created = await s.add_claim_ex("Bob loves olives", [9], "regex@1", commit=False)
            assert created is True
            await s.log_coverage({
                "span_id": 9, "extractor_id": "regex@1", "run_id": "r2",
                "extracted_at": 2.0, "claims_found": 1, "claims_new": 1,
                "claims_matched": 0, "target": "claims",
            }, commit=False)
            raise RuntimeError("boom")
        except RuntimeError:
            conn.rollback()
        cur = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        cur2 = conn.execute("SELECT COUNT(*) FROM reextract_log").fetchone()[0]
        assert cur == 0
        assert cur2 == 0
        await s.close()
    asyncio.run(run())


def test_covered_span_ids(tmp_path):
    s = _store(tmp_path)
    asyncio.run(s.log_coverage({
        "span_id": 5, "extractor_id": "regex@1", "run_id": "r",
        "extracted_at": 1.0, "claims_found": 0, "claims_new": 0,
        "claims_matched": 0, "target": "claims",
    }))
    covered = asyncio.run(s.covered_span_ids("regex@1", "claims", [1, 5, 9]))
    assert covered == {5}
    assert asyncio.run(s.covered_span_ids("regex@1", "claims", [])) == set()
    asyncio.run(s.close())
