"""Tests for correction-supersede in the vector layer.

A superseded vector row must leave active recall (``search()`` skips it,
mirroring the KG's ``valid_to IS NULL`` filter) while the raw row is *retained*
in the table — zero-loss is the whole point. Supersede is additive/opt-in: with
nothing superseded, behaviour is identical to before.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from taosmd.knowledge_graph import TemporalKnowledgeGraph
from taosmd.vector_memory import SCHEMA, VectorMemory


def _deterministic_embedder(vmem: VectorMemory) -> None:
    """Patch embed() with a deterministic 16-dim hash vector (no ONNX/QMD)."""

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFFFFFFFFFF
        return [((h >> (i * 3)) & 0xFF) / 255.0 - 0.5 for i in range(16)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


def _make_store(tmp_path, *, binary_quant: bool = False) -> VectorMemory:
    vmem = VectorMemory(
        db_path=str(tmp_path / "vec.db"),
        embed_mode="onnx",  # falls through; we patch embed() directly
        binary_quant=binary_quant,
    )
    asyncio.run(vmem.init())
    _deterministic_embedder(vmem)
    return vmem


def _row_count(db_path) -> int:
    con = sqlite3.connect(db_path)
    try:
        return con.execute("SELECT COUNT(*) FROM vector_memory").fetchone()[0]
    finally:
        con.close()


def test_default_nothing_superseded_returns_all(tmp_path):
    """Opt-in: with nothing superseded, every added row is recallable."""
    vmem = _make_store(tmp_path)
    try:
        for t in ("jay works on project alpha", "the cat sat on the mat", "grocery list eggs milk"):
            asyncio.run(vmem.add(t))
        hits = asyncio.run(vmem.search("jay works on project alpha", limit=5, hybrid=False, fusion="none"))
        texts = {h["text"] for h in hits}
        assert "jay works on project alpha" in texts
        assert len(texts) == 3
    finally:
        asyncio.run(vmem.close())


def test_supersede_by_id_excludes_from_search_but_keeps_row(tmp_path):
    """supersede(id) soft-hides from recall; the raw row STILL exists."""
    vmem = _make_store(tmp_path)
    try:
        rid = asyncio.run(vmem.add("jay works on project alpha"))
        asyncio.run(vmem.add("jay lives in berlin"))

        # Present before supersede.
        hits = asyncio.run(vmem.search("jay works on project alpha", limit=5, hybrid=False, fusion="none"))
        assert any(h["id"] == rid for h in hits)

        affected = asyncio.run(vmem.supersede(rid))
        assert affected is True

        # Absent from active recall.
        hits = asyncio.run(vmem.search("jay works on project alpha", limit=5, hybrid=False, fusion="none"))
        assert all(h["id"] != rid for h in hits)
    finally:
        asyncio.run(vmem.close())

    # Raw row retained in the table (zero-loss) — both rows still present.
    assert _row_count(tmp_path / "vec.db") == 2


def test_supersede_idempotent_returns_false_second_time(tmp_path):
    """A second supersede of the same row is a no-op (mirrors KG invalidate)."""
    vmem = _make_store(tmp_path)
    try:
        rid = asyncio.run(vmem.add("stale fact"))
        assert asyncio.run(vmem.supersede(rid)) is True
        assert asyncio.run(vmem.supersede(rid)) is False
    finally:
        asyncio.run(vmem.close())


def test_supersede_matching_by_text(tmp_path):
    """supersede_matching hides every active row containing the substring."""
    vmem = _make_store(tmp_path)
    try:
        asyncio.run(vmem.add("jay works on ProjectAlpha"))
        asyncio.run(vmem.add("we discussed ProjectAlpha at length"))
        keep = asyncio.run(vmem.add("jay works on ProjectBeta"))

        n = asyncio.run(vmem.supersede_matching("ProjectAlpha"))
        assert n == 2

        hits = asyncio.run(vmem.search("project work", limit=10, hybrid=False, fusion="none"))
        texts = {h["text"] for h in hits}
        assert "jay works on ProjectAlpha" not in texts
        assert "we discussed ProjectAlpha at length" not in texts
        # The non-matching row survives recall.
        assert any(h["id"] == keep for h in hits)
    finally:
        asyncio.run(vmem.close())

    # All three raw rows retained.
    assert _row_count(tmp_path / "vec.db") == 3


def test_supersede_matching_blank_is_noop(tmp_path):
    """A blank match never sweeps the store (defensive guard)."""
    vmem = _make_store(tmp_path)
    try:
        asyncio.run(vmem.add("alpha"))
        asyncio.run(vmem.add("beta"))
        assert asyncio.run(vmem.supersede_matching("")) == 0
        assert asyncio.run(vmem.supersede_matching("   ")) == 0
        hits = asyncio.run(vmem.search("alpha", limit=5, hybrid=False, fusion="none"))
        assert len(hits) == 2
    finally:
        asyncio.run(vmem.close())


def test_supersede_excludes_from_binary_quant_path(tmp_path):
    """The active filter applies to the binary-quant scoring path too."""
    vmem = _make_store(tmp_path, binary_quant=True)
    try:
        rid = asyncio.run(vmem.add("alpha alpha alpha"))
        asyncio.run(vmem.add("beta beta beta"))
        asyncio.run(vmem.supersede(rid))
        hits = asyncio.run(vmem.search("alpha alpha alpha", limit=5, hybrid=False, fusion="none"))
        assert all(h["id"] != rid for h in hits)
    finally:
        asyncio.run(vmem.close())


def test_migration_adds_valid_to_to_legacy_db(tmp_path):
    """A DB created WITHOUT valid_to upgrades in init() with no data loss."""
    db_path = tmp_path / "legacy.db"

    # Build a legacy schema (no valid_to column) and seed a row.
    con = sqlite3.connect(db_path)
    con.execute(
        """CREATE TABLE vector_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        )"""
    )
    con.execute(
        "INSERT INTO vector_memory (text, embedding, metadata_json, created_at) VALUES (?, ?, ?, ?)",
        ("legacy fact", "[0.1, 0.2]", "{}", 123.0),
    )
    con.commit()
    con.close()

    # Sanity: column absent before migration.
    con = sqlite3.connect(db_path)
    cols = {r[1] for r in con.execute("PRAGMA table_info(vector_memory)")}
    con.close()
    assert "valid_to" not in cols

    # init() should migrate in place.
    vmem = VectorMemory(db_path=str(db_path))
    asyncio.run(vmem.init())
    try:
        # Column now present, legacy row preserved and active.
        cols = {r["name"] for r in vmem._conn.execute("PRAGMA table_info(vector_memory)")}
        assert "valid_to" in cols
        row = vmem._conn.execute(
            "SELECT text, valid_to FROM vector_memory WHERE text = 'legacy fact'"
        ).fetchone()
        assert row is not None
        assert row["valid_to"] is None  # default NULL = active
    finally:
        asyncio.run(vmem.close())


def test_kg_contradiction_supersedes_matching_vector_chunk(tmp_path):
    """The auto-resolve KG path opportunistically supersedes vector chunks.

    Wires a correction end-to-end: an auto-resolved contradiction invalidates
    the old triple AND soft-hides the vector chunk carrying the old value.
    """
    vmem = _make_store(tmp_path)
    kg = TemporalKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
    asyncio.run(kg.init())
    try:
        old_chunk = asyncio.run(vmem.add("jay works on ProjectAlpha"))
        asyncio.run(vmem.add("jay enjoys hiking"))

        # First fact.
        asyncio.run(kg.add_triple_with_contradiction_check(
            subject="jay", predicate="works_on", obj="ProjectAlpha", vmem=vmem,
        ))
        # Correction — contradicts the singular works_on fact, auto-resolves.
        result = asyncio.run(kg.add_triple_with_contradiction_check(
            subject="jay", predicate="works_on", obj="ProjectBeta", vmem=vmem,
        ))
        assert result["contradictions_resolved"] == 1

        # The old chunk is gone from recall...
        hits = asyncio.run(vmem.search("jay project", limit=10, hybrid=False, fusion="none"))
        assert all(h["id"] != old_chunk for h in hits)
    finally:
        asyncio.run(vmem.close())
        asyncio.run(kg.close())

    # ...but retained in the table (zero-loss).
    assert _row_count(tmp_path / "vec.db") == 2


def test_kg_contradiction_without_vmem_unchanged(tmp_path):
    """vmem=None (default) leaves the vector store completely untouched."""
    vmem = _make_store(tmp_path)
    kg = TemporalKnowledgeGraph(db_path=str(tmp_path / "kg.db"))
    asyncio.run(kg.init())
    try:
        asyncio.run(vmem.add("jay works on ProjectAlpha"))
        asyncio.run(kg.add_triple_with_contradiction_check(
            subject="jay", predicate="works_on", obj="ProjectAlpha",
        ))
        # No vmem passed — KG-only behaviour.
        asyncio.run(kg.add_triple_with_contradiction_check(
            subject="jay", predicate="works_on", obj="ProjectBeta",
        ))
        # Chunk still recallable; nothing superseded.
        hits = asyncio.run(vmem.search("jay works on ProjectAlpha", limit=5, hybrid=False, fusion="none"))
        assert any(h["text"] == "jay works on ProjectAlpha" for h in hits)
    finally:
        asyncio.run(vmem.close())
        asyncio.run(kg.close())


def test_schema_constant_includes_valid_to(tmp_path):
    """Fresh stores carry the column from the SCHEMA, not just migration."""
    assert "valid_to" in SCHEMA
    vmem = _make_store(tmp_path)
    try:
        cols = {r["name"] for r in vmem._conn.execute("PRAGMA table_info(vector_memory)")}
        assert "valid_to" in cols
    finally:
        asyncio.run(vmem.close())
