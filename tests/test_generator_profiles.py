from taosmd import generator_profiles as gp


def test_default_is_balanced():
    assert gp.default_profile_id() == "balanced"
    assert gp.get_profile("balanced") is not None


def test_registry_integrity():
    profiles = gp.list_profiles()
    assert {p.id for p in profiles} >= {"balanced", "factual-recall"}
    for p in profiles:
        assert p.models, f"{p.id} has an empty models map"
        for tier in p.models:
            assert tier in gp.TIER_ORDER, f"{p.id} has bad tier {tier!r}"


def test_balanced_mirrors_recipe_generators():
    # backward-compat guard: balanced must equal today's per-tier recipe gens
    bal = gp.get_profile("balanced").models
    assert bal["gpu-12gb"] == "ollama:qwen3.5:9b"
    assert bal["gpu-8gb"] == "ollama:qwen3.5:9b"
    assert bal["gpu-4gb"] == "ollama:llama3.1:8b"
    assert bal["pi-npu"] == ""
    assert bal["cpu"] == ""


def test_factual_recall_tiers():
    fr = gp.get_profile("factual-recall").models
    assert fr["gpu-12gb"] == "ollama:gemma4:12b"
    assert fr["gpu-8gb"] == "ollama:llama3.1:8b"
    assert fr["gpu-4gb"] == "ollama:llama3.1:8b"
    assert "pi-npu" not in fr  # falls through to the recipe (retrieval-only)


def test_unknown_profile_is_none():
    assert gp.get_profile("nope") is None
