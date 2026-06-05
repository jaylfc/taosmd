"""Tests for the opt-in binary-quantization scoring path in VectorMemory.

Binary quant is a footprint/speed option (sign-bit Hamming similarity instead
of full-precision cosine). It must be OFF by default, and when enabled it must
still return ranked results scored in [0, 1] without disturbing recall on
trivially separable inputs.
"""

from __future__ import annotations

import asyncio

import pytest

from taosmd.vector_memory import VectorMemory


def _deterministic_embedder(vmem: VectorMemory) -> None:
    """Patch embed() with a deterministic 16-dim hash vector (no ONNX/QMD)."""

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        """Deterministic 16-dim hash vector centred on 0 (so sign bits vary)."""
        h = hash(text) & 0xFFFFFFFFFFFFFFFF
        return [((h >> (i * 3)) & 0xFF) / 255.0 - 0.5 for i in range(16)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


def _make_store(tmp_path, *, binary_quant: bool) -> VectorMemory:
    """Init a VectorMemory with a deterministic embedder and the given mode."""
    vmem = VectorMemory(
        db_path=str(tmp_path / "vec.db"),
        embed_mode="onnx",  # falls through; we patch embed() directly
        binary_quant=binary_quant,
    )
    asyncio.run(vmem.init())
    _deterministic_embedder(vmem)
    return vmem


def test_binary_quant_defaults_off(tmp_path):
    """binary_quant is opt-in: default construction leaves it disabled."""
    vmem = VectorMemory(db_path=str(tmp_path / "vec.db"))
    assert vmem.binary_quant is False


def test_binary_quant_is_constructor_settable(tmp_path):
    """The constructor flag enables binary-quant mode."""
    vmem = VectorMemory(db_path=str(tmp_path / "vec.db"), binary_quant=True)
    assert vmem.binary_quant is True


def test_binary_quant_search_returns_bounded_scores(tmp_path):
    """With binary_quant on, search still ranks rows and scores stay in [0, 1]."""
    vmem = _make_store(tmp_path, binary_quant=True)
    try:
        for text in ("the cat sat on the mat", "quantum field theory lecture", "grocery list: eggs milk"):
            asyncio.run(vmem.add(text))
        hits = asyncio.run(vmem.search("the cat sat on the mat", limit=3, hybrid=False, fusion="none"))
        assert hits, "binary-quant search returned no hits"
        scores = [h["similarity"] for h in hits if "similarity" in h]
        assert scores, "hits carried no similarity field"
        assert all(0.0 <= s <= 1.0 for s in scores), f"scores out of [0,1]: {scores}"
        # The exact-match document should rank first.
        assert hits[0]["text"] == "the cat sat on the mat"
    finally:
        asyncio.run(vmem.close())


def test_binary_quant_stores_packed_bits(tmp_path):
    """The footprint claim must be real: a binq store persists ~1 bit/dim, not floats."""
    import sqlite3

    cos = _make_store(tmp_path / "cos", binary_quant=False)
    binq = _make_store(tmp_path / "binq", binary_quant=True)
    text = "the quick brown fox jumps over the lazy dog"
    try:
        asyncio.run(cos.add(text))
        asyncio.run(binq.add(text))
    finally:
        asyncio.run(cos.close())
        asyncio.run(binq.close())

    def _stored_len(db):
        con = sqlite3.connect(db)
        try:
            return len(con.execute("SELECT embedding FROM vector_memory LIMIT 1").fetchone()[0])
        finally:
            con.close()

    cos_len = _stored_len(tmp_path / "cos" / "vec.db")
    binq_len = _stored_len(tmp_path / "binq" / "vec.db")
    # 16-dim embedder → 2 packed bytes, base64-encoded to 4 chars. The JSON float
    # list is far larger. Assert a real, large reduction (not a no-op).
    assert binq_len < cos_len / 4, f"binq stored {binq_len}B vs cosine {cos_len}B — not packed"


def test_binary_quant_matches_cosine_top_rank(tmp_path):
    """Both paths should agree on the top hit for trivially separable inputs."""
    docs = ["alpha alpha alpha", "beta beta beta", "gamma gamma gamma"]

    cos = _make_store(tmp_path / "a", binary_quant=False)
    binq = _make_store(tmp_path / "b", binary_quant=True)
    try:
        for d in docs:
            asyncio.run(cos.add(d))
            asyncio.run(binq.add(d))
        q = "beta beta beta"
        top_cos = asyncio.run(cos.search(q, limit=1, hybrid=False, fusion="none"))[0]["text"]
        top_binq = asyncio.run(binq.search(q, limit=1, hybrid=False, fusion="none"))[0]["text"]
        assert top_cos == top_binq == q
    finally:
        asyncio.run(cos.close())
        asyncio.run(binq.close())
