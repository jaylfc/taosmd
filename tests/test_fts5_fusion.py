"""Tests for the opt-in FTS5 lexical-fusion path in VectorMemory.

fts5_rrf is a benchmark research lever: RRF over (semantic ranks, SQLite FTS5
BM25 ranks). It must be OFF by default (opt-in via fusion="fts5_rrf"), source
keyword ranks from the built-in FTS5 index (no new deps, no per-query rebuild),
and survive arbitrary user text without raising FTS5 syntax errors.
"""

from __future__ import annotations

import asyncio

import pytest

from taosmd.vector_memory import VectorMemory, fts5_query_string


def _deterministic_embedder(vmem: VectorMemory) -> None:
    """Patch embed() with a deterministic 16-dim hash vector (no ONNX/QMD)."""

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFFFFFFFFFF
        return [((h >> (i * 3)) & 0xFF) / 255.0 - 0.5 for i in range(16)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


def _make_store(tmp_path) -> VectorMemory:
    vmem = VectorMemory(
        db_path=str(tmp_path / "vec.db"),
        embed_mode="onnx",  # falls through; we patch embed() directly
    )
    asyncio.run(vmem.init())
    _deterministic_embedder(vmem)
    return vmem


def test_fts5_available_in_runtime():
    """The whole lever assumes CPython's sqlite3 ships FTS5; assert it does."""
    import sqlite3

    con = sqlite3.connect(":memory:")
    try:
        con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
    finally:
        con.close()


def test_fts5_query_string_sanitizes_special_chars():
    """Arbitrary text becomes an OR-joined, quote-escaped FTS5 MATCH query."""
    assert fts5_query_string("") == ""
    assert fts5_query_string("(?!)") == ""  # no word chars
    q = fts5_query_string('say "hi" (really?)')
    assert " OR " in q
    # Every bareword token is double-quoted.
    assert '"say"' in q and '"hi"' in q and '"really"' in q


def test_fts5_index_built_lazily_not_on_add(tmp_path):
    """add() must not populate FTS; the index materialises on first fts5_rrf search."""
    vmem = _make_store(tmp_path)

    def _fts_matches(term):
        return vmem._conn.execute(
            "SELECT rowid FROM vector_memory_fts WHERE vector_memory_fts MATCH ?",
            (term,),
        ).fetchall()

    try:
        asyncio.run(vmem.add("the cat sat on the mat"))
        # External-content FTS5 reports a nonzero COUNT(*) from the content table
        # even when its inverted index is empty, so assert searchability: a MATCH
        # returns nothing until the lazy rebuild runs on the first fts5_rrf search.
        assert vmem._fts5_synced_max_id == -1
        assert _fts_matches("cat") == [], "FTS index should not be queryable before first search"
        asyncio.run(vmem.search("cat", limit=1, fusion="fts5_rrf"))
        assert _fts_matches("cat"), "FTS index should be lazily built on first fts5_rrf search"
    finally:
        asyncio.run(vmem.close())


def test_fts5_rrf_surfaces_exact_keyword_match(tmp_path):
    """An exact lexical match should rank first under fts5_rrf."""
    vmem = _make_store(tmp_path)
    try:
        docs = [
            "quantum field theory lecture notes",
            "grocery list eggs milk bread",
            "the pelican flew over the harbour at dawn",
        ]
        for d in docs:
            asyncio.run(vmem.add(d))
        hits = asyncio.run(vmem.search("pelican harbour", limit=3, fusion="fts5_rrf"))
        assert hits, "fts5_rrf returned no hits"
        assert hits[0]["text"] == "the pelican flew over the harbour at dawn"
        # RRF path tags each hit with an rrf_score.
        assert "rrf_score" in hits[0]
    finally:
        asyncio.run(vmem.close())


def test_fts5_rrf_handles_special_chars_without_crashing(tmp_path):
    """FTS5-special chars in the query must not raise a syntax error."""
    vmem = _make_store(tmp_path)
    try:
        for d in ("alpha beta gamma", "delta epsilon zeta"):
            asyncio.run(vmem.add(d))
        # Quotes, parens, AND/OR/NOT, NEAR, wildcard — all FTS5 grammar tokens.
        bad_queries = [
            '"unterminated',
            "alpha (beta",
            "gamma) OR",
            "NEAR/",
            "epsilon*",
            ''.join([")", ")", '"', '"', "(", "("]),
        ]
        for bad in bad_queries:
            hits = asyncio.run(vmem.search(bad, limit=2, fusion="fts5_rrf"))
            assert isinstance(hits, list)
    finally:
        asyncio.run(vmem.close())


def test_fts5_index_resyncs_after_more_adds(tmp_path):
    """Rows added between searches get reindexed on the next fts5_rrf search."""
    vmem = _make_store(tmp_path)
    try:
        asyncio.run(vmem.add("first document about otters"))
        asyncio.run(vmem.search("otters", limit=1, fusion="fts5_rrf"))
        asyncio.run(vmem.add("second document about badgers"))
        hits = asyncio.run(vmem.search("badgers", limit=2, fusion="fts5_rrf"))
        assert any("badgers" in h["text"] for h in hits)
    finally:
        asyncio.run(vmem.close())
