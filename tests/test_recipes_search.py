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


from pathlib import Path
from taosmd import api as taosmd_api
from taosmd import agents as taosmd_agents
from taosmd import recipes


def _patch_embedder(stores):
    vmem = stores["vector"]

    async def _fake(text, task="search_document"):
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

    vmem.embed = _fake


def test_search_applies_resolved_recipe_reranker_degrades(tmp_path, monkeypatch):
    # The recipes/agents layer isolates via data_dir=, not env vars or singletons.
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    # Keep the degrade path hermetic: the absent reranker must NOT kick off a
    # real network download in the test.
    monkeypatch.setattr(recipes, "ensure_reranker_model",
                        lambda *a, **k: "downloading")

    stores = asyncio.run(taosmd_api._ensure_stores(str(tmp_path)))
    _patch_embedder(stores)
    asyncio.run(taosmd_api.ingest("the sky is blue", agent="dave",
                                  data_dir=str(tmp_path)))

    # ingest()'s ensure_agent() targets the default registry; register the
    # agent in the tmp registry so apply_recipe/resolve_recipe see it.
    taosmd_agents.ensure_agent("dave", data_dir=str(tmp_path))
    # Force the leader recipe (wants bge-v2-m3, which is absent -> degrade).
    recipes.apply_recipe("dave", "maxsim-rerank-9b", data_dir=str(tmp_path))
    hits = asyncio.run(taosmd_api.search("sky", agent="dave",
                                         data_dir=str(tmp_path)))
    assert isinstance(hits, list)
    # The search must not raise and must return results even though the
    # reranker model is missing (degraded path).
    assert hits, "degraded search should still return the embedded turn"
    # The wiring must mark the degraded reranker on the top hit.
    assert hits[0]["metadata"].get("recipe_degraded") == "reranker-downloading"
