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
        # _adapt_vector reads r["id"] and r["similarity"]; without them the
        # adapter raises KeyError which _query_source swallows to [], so the
        # call-kwargs asserts would pass while no results flowed through.
        return [{"id": "v1", "text": "hit", "similarity": 1.0, "metadata": {}}]


def test_retrieve_threads_fusion_and_candidate_top_k():
    vec = _FakeVector()
    results = asyncio.run(retrieval.retrieve(
        "q", sources={"vector": vec}, strategy="custom",
        memory_layers=["vector"], limit=5, fusion="mem0_additive",
        candidate_top_k=50))
    assert vec.calls, "vector source was not queried"
    assert vec.calls[0]["fusion"] == "mem0_additive"
    assert vec.calls[0]["limit"] == 50  # candidate pool, not the final limit
    # Verify the hit actually flowed through the adapter (not swallowed to []).
    assert results, "the fake vector hit did not flow through retrieve()"
    assert results[0]["text"] == "hit"


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


def test_recipe_controls_retrieval_breadth_via_fanout(tmp_path):
    # Recipe-differentiated breadth, end to end: applying a recipe writes its
    # librarian.fanout through to the agent, and effective_fanout reflects it.
    # maxsim-rerank-9b uses fanout "med" (K=10); lite-pi uses "low" (K=3).
    d = str(tmp_path)
    taosmd_agents.ensure_agent("gina", data_dir=d)

    recipes.apply_recipe("gina", "maxsim-rerank-9b", data_dir=d)
    assert (taosmd_agents._registry(d)
            .effective_fanout("gina", worker_capabilities=None)
            == taosmd_agents.FANOUT_LEVELS["med"])

    recipes.apply_recipe("gina", "lite-pi", data_dir=d)
    assert (taosmd_agents._registry(d)
            .effective_fanout("gina", worker_capabilities=None)
            == taosmd_agents.FANOUT_LEVELS["low"])


@pytest.mark.parametrize("strategy", ["thorough", "fast", "minimal", "custom"])
def test_agent_fanout_does_not_override_explicit_result_limit(monkeypatch, strategy):
    class ManyHits(_FakeVector):
        async def search(self, query, **kwargs):
            self.calls.append(kwargs)
            return [
                {"id": f"v{i}", "text": f"hit {i}", "similarity": 1 - i / 20,
                 "metadata": {}}
                for i in range(10)
            ]

    monkeypatch.setattr(taosmd_agents, "effective_fanout", lambda *a, **k: 10)
    vec = ManyHits()
    hits = asyncio.run(retrieval.retrieve(
        "q", sources={"vector": vec}, strategy=strategy, agent="gina", limit=3))
    assert vec.calls[0]["limit"] == 30  # agent K still controls retrieval breadth
    assert len(hits) == 3  # caller's explicit maximum controls returned hits


@pytest.mark.parametrize("strategy", ["thorough", "custom"])
def test_agent_reranker_keeps_caller_limit_when_fanout_is_smaller(monkeypatch, strategy):
    class ManyHits(_FakeVector):
        async def search(self, query, **kwargs):
            self.calls.append(kwargs)
            return [
                {"id": str(i), "text": f"hit {i}", "similarity": 1 - i / 10,
                 "metadata": {}}
                for i in range(8)
            ]

    class TopKReranker:
        available = True

        def rerank(self, query, results, top_k):
            return results[:top_k]

    monkeypatch.setattr(taosmd_agents, "effective_fanout", lambda *a, **k: 3)
    hits = asyncio.run(retrieval.retrieve(
        "q", sources={"vector": ManyHits()}, strategy=strategy,
        agent="gina", limit=5, reranker=TopKReranker()))
    assert len(hits) == 5


@pytest.mark.parametrize("strategy", ["thorough", "fast", "minimal", "custom"])
def test_scope_is_applied_before_capping_returned_hits(strategy):
    class ScopedHits(_FakeVector):
        async def search(self, query, **kwargs):
            self.calls.append(kwargs)
            return [
                {"id": str(i), "text": f"hit {i}", "similarity": 1 - i / 10,
                 "metadata": {"project": "other" if i < 2 else "ours"}}
                for i in range(5)
            ]

    vec = ScopedHits()
    hits = asyncio.run(retrieval.retrieve(
        "q", sources={"vector": vec}, strategy=strategy,
        project="ours", limit=2))
    assert [h["source_id"] for h in hits] == ["2", "3"]
    assert vec.calls[0]["project"] == "ours"


@pytest.mark.parametrize("strategy", ["thorough", "fast", "custom"])
def test_out_of_scope_archive_hit_cannot_deduplicate_scoped_vector_hit(
    monkeypatch, strategy,
):
    from taosmd import intent_classifier

    class VectorHits(_FakeVector):
        async def search(self, query, **kwargs):
            return [
                {"id": "v1", "text": "different first result", "similarity": 1.0,
                 "metadata": {"project": "ours"}},
                {"id": "v2", "text": "shared relevant result", "similarity": 0.9,
                 "metadata": {"project": "ours"}},
            ]

    class ArchiveHits:
        async def search_fts(self, query, limit=5):
            return [{"id": "a1", "summary": "shared relevant result",
                     "project": "elsewhere"}]

    monkeypatch.setattr(intent_classifier, "get_search_strategy", lambda query: {
        "primary": "vector" if strategy == "fast" else "archive",
        "secondary": "archive",
    })
    hits = asyncio.run(retrieval.retrieve(
        "q", sources={"vector": VectorHits(), "archive": ArchiveHits()},
        strategy=strategy, memory_layers=["vector", "archive"],
        project="ours", limit=3,
    ))
    assert [(hit["source"], hit["source_id"]) for hit in hits] == [
        ("vector", "v1"), ("vector", "v2"),
    ]


def test_fast_fills_caller_limit_from_secondary_when_agent_fanout_is_smaller(monkeypatch):
    from taosmd import intent_classifier

    class FourVectorHits(_FakeVector):
        async def search(self, query, **kwargs):
            self.calls.append(kwargs)
            return [
                {"id": str(i), "text": f"vector {i}", "similarity": 1 - i / 10,
                 "metadata": {}}
                for i in range(4)
            ]

    class TwoCatalogHits:
        def __init__(self):
            self.calls = 0

        async def search_topic(self, query, limit=5):
            self.calls += 1
            return [
                {"id": i, "topic": f"catalog {i}", "description": "extra hit"}
                for i in range(2)
            ]

    monkeypatch.setattr(intent_classifier, "get_search_strategy",
                        lambda query: {"primary": "vector", "secondary": "catalog"})
    monkeypatch.setattr(taosmd_agents, "effective_fanout", lambda *a, **k: 3)
    catalog = TwoCatalogHits()
    hits = asyncio.run(retrieval.retrieve(
        "q", sources={"vector": FourVectorHits(), "catalog": catalog},
        strategy="fast", agent="gina", limit=5))
    assert catalog.calls == 1
    assert len(hits) == 5
    assert any(hit["source"] == "catalog" for hit in hits)


@pytest.mark.parametrize("strategy", ["thorough", "custom"])
def test_parallel_source_cancellation_is_not_returned_as_empty_hits(strategy):
    class CancelledVector:
        async def search(self, query, **kwargs):
            raise asyncio.CancelledError("source cancelled")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(retrieval.retrieve(
            "q", sources={"vector": CancelledVector()}, strategy=strategy,
            memory_layers=["vector"], limit=3,
        ))


@pytest.mark.parametrize("strategy", ["thorough", "custom"])
def test_agent_llm_reranker_keeps_caller_limit(monkeypatch, strategy):
    class ManyHits(_FakeVector):
        async def search(self, query, **kwargs):
            return [
                {"id": str(i), "text": f"hit {i}", "similarity": 1 - i / 10,
                 "metadata": {}}
                for i in range(8)
            ]

    async def top_k_only(query, results, reranker, top_k):
        return results[:top_k]

    monkeypatch.setattr(taosmd_agents, "effective_fanout", lambda *a, **k: 3)
    monkeypatch.setattr(retrieval, "_apply_llm_rerank", top_k_only)
    hits = asyncio.run(retrieval.retrieve(
        "q", sources={"vector": ManyHits()}, strategy=strategy,
        agent="gina", limit=5, llm_reranker={"enabled": True}))
    assert len(hits) == 5
