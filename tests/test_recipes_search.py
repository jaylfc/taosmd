# tests/test_recipes_search.py
from __future__ import annotations
import asyncio
import pytest
from taosmd import retrieval


class _FakeVector:
    """Records the kwargs retrieve() passes to the vector source search."""
    def __init__(self):
        self.calls = []

    async def search(self, query, *, limit=5, hybrid=True, fusion="boost",
                     project=None, search_agents=None):
        self.calls.append({"limit": limit, "fusion": fusion})
        return [{"text": "hit", "score": 1.0, "metadata": {}}]


def test_retrieve_threads_fusion_and_candidate_top_k():
    vec = _FakeVector()
    asyncio.run(retrieval.retrieve(
        "q", sources={"vector": vec}, strategy="custom",
        memory_layers=["vector"], limit=5, fusion="mem0_additive",
        candidate_top_k=50))
    assert vec.calls, "vector source was not queried"
    assert vec.calls[0]["fusion"] == "mem0_additive"
    assert vec.calls[0]["limit"] == 50  # candidate pool, not the final limit
