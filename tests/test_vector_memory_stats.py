"""Tests for VectorMemory.stats() (issue #92).

stats() previously did not exist, so taosmd_backend.get_stats() silently
returned an empty dict for the vector layer (wrapped in try/except). Verify
stats() now returns a count.
"""

from __future__ import annotations

import asyncio

from taosmd.vector_memory import VectorMemory


def _deterministic_embedder(vmem: VectorMemory) -> None:
    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        h = hash(text) & 0xFFFFFFFFFFFFFFFF
        return [((h >> (i * 3)) & 0xFF) / 255.0 - 0.5 for i in range(16)]

    vmem.embed = _fake_embed  # type: ignore[assignment]


def test_stats_returns_count(tmp_path):
    async def go():
        vmem = VectorMemory(db_path=str(tmp_path / "vec.db"), embed_mode="onnx")
        await vmem.init()
        _deterministic_embedder(vmem)
        empty = await vmem.stats()
        await vmem.add("the cat sat on the mat")
        await vmem.add("grocery list: eggs milk")
        full = await vmem.stats()
        return empty, full

    empty, full = asyncio.run(go())
    assert isinstance(empty, dict) and empty["count"] == 0
    assert full["count"] == 2
