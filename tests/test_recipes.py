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


from taosmd import agents as taosmd_agents


def test_agent_recipe_config_roundtrip(tmp_path):
    # The agents module has no singleton-reset hook or TAOSMD_DATA_DIR env;
    # existing agent tests isolate by rooting an AgentRegistry at tmp_path
    # (see tests/test_agents.py). The recipe helpers take the same data_dir,
    # so we point them at tmp_path rather than poking a global singleton.
    d = str(tmp_path)
    taosmd_agents.ensure_agent("alice", data_dir=d)
    assert taosmd_agents.get_applied_recipe("alice", data_dir=d) is None
    taosmd_agents.set_agent_recipe_config(
        "alice", recipe_id="rrf-9b",
        retrieval_config={"limit": 5, "candidate_top_k": 20, "fusion": "rrf",
                          "reranker": "none", "adjacent_neighbors": 2,
                          "llm_reranker": True, "strategy": "thorough"},
        data_dir=d)
    assert taosmd_agents.get_applied_recipe("alice", data_dir=d) == "rrf-9b"
    cfg = taosmd_agents.get_agent_retrieval_config("alice", data_dir=d)
    assert cfg["fusion"] == "rrf"
    assert cfg["candidate_top_k"] == 20


def test_apply_recipe_writes_through_and_resolves(tmp_path):
    # Isolate by passing data_dir explicitly (no env var / singleton reset;
    # neither exists, see test_agent_recipe_config_roundtrip).
    d = str(tmp_path)
    taosmd_agents.ensure_agent("bob", data_dir=d)

    recipes.apply_recipe("bob", "rrf-9b", data_dir=d)
    assert taosmd_agents.get_applied_recipe("bob", data_dir=d) == "rrf-9b"

    resolved = recipes.resolve_recipe("bob", data_dir=d)
    assert resolved.id == "rrf-9b"
    assert resolved.retrieval["fusion"] == "rrf"
    # lite recipe disables the librarian (write-through to librarian switches)
    recipes.apply_recipe("bob", "lite-pi", data_dir=d)
    rec = taosmd_agents._registry(d).get_agent("bob")
    assert rec["librarian"]["enabled"] is False


def test_apply_recipe_does_not_clobber_existing_memory_model(tmp_path):
    # When the user has already chosen a memory model, applying a recipe that
    # names a generator must NOT overwrite it (fresh-install resolve path would
    # otherwise silently replace the user's choice).
    d = str(tmp_path)
    taosmd_agents.ensure_agent("erin", data_dir=d)
    taosmd_config.set_memory_model("ollama:user-pick", data_dir=d)
    # maxsim-rerank-9b names ollama:qwen3.5:9b as its generator.
    recipes.apply_recipe("erin", "maxsim-rerank-9b", data_dir=d)
    assert taosmd_config.get_memory_model(data_dir=d) == "ollama:user-pick"


def test_apply_recipe_sets_memory_model_when_unset(tmp_path):
    # With no memory model configured, the recipe's generator seeds the global.
    d = str(tmp_path)
    taosmd_agents.ensure_agent("frank", data_dir=d)
    assert taosmd_config.get_memory_model(data_dir=d) is None
    recipes.apply_recipe("frank", "maxsim-rerank-9b", data_dir=d)
    assert taosmd_config.get_memory_model(data_dir=d) == "ollama:qwen3.5:9b"


def test_resolve_falls_back_to_recommend_when_unconfigured(tmp_path, monkeypatch):
    d = str(tmp_path)
    monkeypatch.setattr(recipes, "local_probe", lambda: {
        "host": {"gpu": {"type": "none"}, "npu": {"type": "none"},
                 "cpu": {"cores": 4}, "ram_mb": 8000}})
    taosmd_agents.ensure_agent("carol", data_dir=d)  # no recipe, no global default
    resolved = recipes.resolve_recipe("carol", data_dir=d)
    # fresh install resolves to recommend()[0] for the probed (cpu) tier
    assert resolved.metadata["tier"] in ("cpu", "pi-npu")


def test_ensure_reranker_model_reports_progress_and_is_nonblocking(tmp_path, monkeypatch):
    events = []
    # Fake the actual fetch so the test does no network IO.
    def fake_fetch(dest, on_progress):
        on_progress({"phase": "start", "pct": 0})
        on_progress({"phase": "done", "pct": 100})
        (tmp_path / "model.onnx").write_bytes(b"x")
        return str(tmp_path / "model.onnx")

    monkeypatch.setattr(recipes, "_fetch_reranker_onnx", fake_fetch)
    state = recipes.ensure_reranker_model(
        onnx_path=str(tmp_path), on_progress=events.append, block=True)
    assert state == "ready"
    assert any(e["phase"] == "done" for e in events)


def test_ensure_reranker_model_nonblocking_returns_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(recipes, "_fetch_reranker_onnx",
                        lambda dest, on_progress: None)
    # Missing model, non-blocking: kicks off and reports 'downloading'.
    state = recipes.ensure_reranker_model(
        onnx_path=str(tmp_path / "missing"), on_progress=lambda e: None, block=False)
    assert state in ("downloading", "ready")


def test_public_surface_exported():
    import taosmd
    for name in ("recipe_schema", "list_recipes", "get_recipe",
                 "recommend", "resolve_recipe", "apply_recipe"):
        assert hasattr(taosmd, name), name
