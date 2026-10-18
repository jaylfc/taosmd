"""Tests for BM25 fallback when bm25s is not installed (tsk-cap5hm).

Verifies that VectorMemory.search with fusion="bm25_rrf" correctly falls back
to the pure-Python BM25 implementation when bm25s is unavailable, and that
binary_quant/late_interaction do not raise a false "requires numpy" error.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from taosmd.vector_memory import VectorMemory


def _constant_embedder(vmem: VectorMemory, dim: int = 8) -> None:
    """Replace embed with a constant vector so cosine ties; only BM25 separates."""

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        return [0.5] * dim

    vmem.embed = _fake_embed  # type: ignore[assignment]


def _constant_token_embedder(vmem: VectorMemory, dim: int = 8) -> None:
    """Replace embed_tokens with a constant token matrix for late_interaction."""

    async def _fake_embed_tokens(text: str, task: str = "search_document") -> list[list[float]]:
        # Return a 3-token matrix, all same values so MaxSim is constant
        return [[0.5] * dim for _ in range(3)]

    vmem.embed_tokens = _fake_embed_tokens  # type: ignore[assignment]


def _make_vmem_no_bm25s(tmp_path, monkeypatch, binary_quant=False, late_interaction=False):
    """VectorMemory with bm25s blocked and constant embedder."""
    monkeypatch.setitem(sys.modules, "bm25s", None)
    vmem = VectorMemory(
        db_path=str(tmp_path / "vec.db"),
        embed_mode="onnx",
        binary_quant=binary_quant,
        late_interaction=late_interaction,
    )
    asyncio.run(vmem.init())
    _constant_embedder(vmem, dim=8)
    if late_interaction:
        vmem._token_dim = 8  # Match the mock token dimension
        _constant_token_embedder(vmem, dim=8)
    return vmem


async def _add_three_docs(vmem: VectorMemory) -> None:
    """Add three docs, the one containing 'zebra' added LAST."""
    await vmem.add("apple banana cherry")
    await vmem.add("dog elephant fox")
    await vmem.add("zebra stripes")


def test_bm25_rrf_fallback_returns_zebra_first(tmp_path, monkeypatch, caplog):
    """search with fusion=bm25_rrf returns zebra doc first when bm25s missing."""
    vmem = _make_vmem_no_bm25s(tmp_path, monkeypatch)
    asyncio.run(_add_three_docs(vmem))

    results = asyncio.run(vmem.search("zebra stripes", limit=1, fusion="bm25_rrf"))

    assert len(results) == 1
    assert "zebra" in results[0]["text"]
    # Verify the fallback warning was logged
    assert any("bm25s not installed" in record.message for record in caplog.records)


def test_bm25_rrf_binary_quant_no_false_numpy_error(tmp_path, monkeypatch, caplog):
    """binary_quant with bm25_rrf does not raise 'requires numpy' when bm25s missing."""
    vmem = _make_vmem_no_bm25s(tmp_path, monkeypatch, binary_quant=True)
    asyncio.run(_add_three_docs(vmem))

    # Should not raise RuntimeError about numpy
    results = asyncio.run(vmem.search("zebra stripes", limit=1, fusion="bm25_rrf"))

    assert len(results) == 1
    assert "zebra" in results[0]["text"]
    assert any("bm25s not installed" in record.message for record in caplog.records)


def test_bm25_rrf_late_interaction_no_false_numpy_error(tmp_path, monkeypatch, caplog):
    """late_interaction with bm25_rrf does not raise 'requires numpy' when bm25s missing."""
    vmem = _make_vmem_no_bm25s(tmp_path, monkeypatch, late_interaction=True)
    asyncio.run(_add_three_docs(vmem))

    # Should not raise RuntimeError about numpy
    results = asyncio.run(vmem.search("zebra stripes", limit=1, fusion="bm25_rrf"))

    assert len(results) == 1
    assert "zebra" in results[0]["text"]
    assert any("bm25s not installed" in record.message for record in caplog.records)


def test_bm25_lemma_rrf_raises_import_error_when_lemmatization_missing(tmp_path, monkeypatch):
    """fusion=bm25_lemma_rrf raises ImportError when lemmatization module missing."""
    monkeypatch.setitem(sys.modules, "bm25s", None)
    monkeypatch.setitem(sys.modules, "taosmd.utils.lemmatization", None)
    vmem = VectorMemory(db_path=str(tmp_path / "vec.db"), embed_mode="onnx")
    asyncio.run(vmem.init())
    _constant_embedder(vmem, dim=8)
    asyncio.run(_add_three_docs(vmem))

    with pytest.raises(ImportError):
        asyncio.run(vmem.search("zebra stripes", limit=1, fusion="bm25_lemma_rrf"))


def test_bm25_lemma_rrf_raises_import_error_when_lemmatization_missing_and_bm25s_blocked(
    tmp_path, monkeypatch
):
    """fusion=bm25_lemma_rrf raises ImportError when lemmatization missing (bm25s also blocked)."""
    monkeypatch.setitem(sys.modules, "bm25s", None)
    monkeypatch.setitem(sys.modules, "taosmd.utils.lemmatization", None)
    vmem = VectorMemory(db_path=str(tmp_path / "vec.db"), embed_mode="onnx")
    asyncio.run(vmem.init())
    _constant_embedder(vmem, dim=8)
    asyncio.run(_add_three_docs(vmem))

    with pytest.raises(ImportError):
        asyncio.run(vmem.search("zebra stripes", limit=1, fusion="bm25_lemma_rrf"))


def test_bm25_rrf_with_bm25s_returns_zebra_first(tmp_path):
    pytest.importorskip("bm25s")
    """Control: with bm25s installed, bm25_rrf returns zebra doc first."""
    vmem = VectorMemory(db_path=str(tmp_path / "vec.db"), embed_mode="onnx")
    asyncio.run(vmem.init())
    _constant_embedder(vmem, dim=8)
    asyncio.run(_add_three_docs(vmem))

    results = asyncio.run(vmem.search("zebra stripes", limit=1, fusion="bm25_rrf"))

    assert len(results) == 1
    assert "zebra" in results[0]["text"]
