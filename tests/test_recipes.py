from __future__ import annotations
import pytest
from pathlib import Path
import re
from taosmd import recipes


def test_recipe_dataclass_roundtrips_to_dict():
    r = recipes.Recipe(
        id="x", name="X",
        retrieval={"strategy": "thorough", "limit": 5, "candidate_top_k": 20,
                   "fusion": "boost", "reranker": "none",
                   "adjacent_neighbors": 0, "llm_reranker": False},
        ingest={"extraction": True, "extraction_model": "", "embed_verbatim": True},
        generator={"model": ""},
        librarian={"fanout": "low", "worker_aware": True},
        metadata={"tier": "gpu-12gb", "scores": {}, "pros": [], "cons": [],
                  "est_latency": "medium", "est_footprint": "medium", "source": "t"},
    )
    d = r.to_dict()
    assert d["id"] == "x"
    assert d["retrieval"]["fusion"] == "boost"
    assert recipes.Recipe.from_dict(d).to_dict() == d


def test_recipe_schema_is_valid_object_with_four_sections():
    schema = recipes.recipe_schema()
    assert schema["type"] == "object"
    for section in ("retrieval", "ingest", "generator", "librarian", "metadata"):
        assert section in schema["properties"], section
    fusion = schema["properties"]["retrieval"]["properties"]["fusion"]
    assert fusion["enum"] == ["boost", "rrf", "mem0_additive"]
    fanout = schema["properties"]["librarian"]["properties"]["fanout"]
    assert fanout["enum"] == ["off", "low", "med", "high"]


def test_registry_has_unique_ids_and_required_metadata():
    ids = [r.id for r in recipes.list_recipes()]
    assert len(ids) == len(set(ids)), "duplicate recipe ids"
    assert {"maxsim-rerank-9b", "rrf-9b", "fast-8b", "lite-pi"} <= set(ids)
    for r in recipes.list_recipes():
        assert r.metadata.get("tier")
        assert r.metadata.get("source")


def test_leader_scores_match_benchmarks_doc():
    """The leader's scores in the registry must equal docs/benchmarks.md."""
    leader = recipes.get_recipe("maxsim-rerank-9b")
    assert leader is not None
    assert leader.metadata["scores"] == {
        "gemma4:e2b": 0.748,
        "llama3.1:8b": 0.394,
        "qwen3:4b-instruct-2507": 0.659,
    }
    doc = Path("docs/benchmarks.md").read_text()
    # The leader row must contain all three numbers (drift guard).
    for n in ("0.748", "0.394", "0.659"):
        assert n in doc, f"{n} missing from benchmarks.md"


def test_lite_recipe_disables_extraction_and_reranker():
    lite = recipes.get_recipe("lite-pi")
    assert lite.ingest["extraction"] is False
    assert lite.retrieval["reranker"] == "none"


def test_local_probe_shape():
    info = recipes.local_probe()
    assert "host" in info
    host = info["host"]
    for k in ("cpu", "ram_mb", "npu", "gpu"):
        assert k in host, k
    assert "cores" in host["cpu"]
    assert host["gpu"]["type"] in ("nvidia", "amd", "mali", "intel", "metal", "none")


def test_local_probe_tier_classifier():
    # tier_of() maps a probe dict to a coarse tier string.
    assert recipes.tier_of({"host": {"gpu": {"type": "nvidia", "vram_mb": 12000},
                                     "npu": {"type": "none"}, "cpu": {"cores": 8},
                                     "ram_mb": 32000}}) == "gpu-12gb"
    assert recipes.tier_of({"host": {"gpu": {"type": "none"},
                                     "npu": {"type": "rknpu"}, "cpu": {"cores": 8},
                                     "ram_mb": 16000}}) == "pi-npu"
    assert recipes.tier_of({"host": {"gpu": {"type": "none"},
                                     "npu": {"type": "none"}, "cpu": {"cores": 4},
                                     "ram_mb": 8000}}) == "cpu"


def test_recommend_ranks_leader_first_on_gpu12():
    info = {"host": {"gpu": {"type": "nvidia", "vram_mb": 12000},
                     "npu": {"type": "none"}, "cpu": {"cores": 8}, "ram_mb": 32000}}
    ranked = recipes.recommend(info)
    assert ranked[0].id == "maxsim-rerank-9b"
    assert all(hasattr(r, "id") for r in ranked)


def test_recommend_avoids_gpu_recipes_on_pi():
    info = {"host": {"gpu": {"type": "none"}, "npu": {"type": "rknpu"},
                     "cpu": {"cores": 8}, "ram_mb": 16000}}
    ranked = recipes.recommend(info)
    assert ranked[0].id == "lite-pi"
    # A 9B GPU recipe must never outrank the lite recipe on a Pi.
    assert ranked.index(recipes.get_recipe("lite-pi")) < ranked.index(
        recipes.get_recipe("maxsim-rerank-9b"))


def test_recommend_uses_local_probe_when_none(monkeypatch):
    monkeypatch.setattr(recipes, "local_probe", lambda: {
        "host": {"gpu": {"type": "none"}, "npu": {"type": "none"},
                 "cpu": {"cores": 4}, "ram_mb": 8000}})
    ranked = recipes.recommend(None)
    assert ranked[0].metadata["tier"] in ("cpu", "pi-npu")


from taosmd import config as taosmd_config


def test_default_recipe_get_set_clear(tmp_path):
    d = str(tmp_path)
    assert taosmd_config.get_default_recipe(data_dir=d) is None
    taosmd_config.set_default_recipe("rrf-9b", data_dir=d)
    assert taosmd_config.get_default_recipe(data_dir=d) == "rrf-9b"
    taosmd_config.set_default_recipe("", clear=True, data_dir=d)
    assert taosmd_config.get_default_recipe(data_dir=d) is None
