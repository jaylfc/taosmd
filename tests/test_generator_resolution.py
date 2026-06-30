import pytest
from taosmd import generator_profiles as gp


@pytest.fixture
def at_12gb(monkeypatch):
    monkeypatch.setattr(gp.recipes, "local_probe", lambda: {"host": {}})
    monkeypatch.setattr(gp.recipes, "tier_of", lambda info: "gpu-12gb")


def test_default_resolves_to_balanced_for_tier(at_12gb, tmp_path):
    # no pin, no profile set -> balanced -> qwen3.5:9b at 12gb
    assert gp.resolve_generator(data_dir=tmp_path) == "ollama:qwen3.5:9b"


def test_global_profile_overrides(at_12gb, tmp_path):
    from taosmd import config
    config.set_generator_profile("factual-recall", data_dir=tmp_path)
    assert gp.resolve_generator(data_dir=tmp_path) == "ollama:gemma4:12b"


def test_pin_beats_profile(at_12gb, tmp_path):
    from taosmd import config
    config.set_generator_profile("factual-recall", data_dir=tmp_path)
    config.set_memory_model("ollama:custom-pin", data_dir=tmp_path)
    assert gp.resolve_generator(data_dir=tmp_path) == "ollama:custom-pin"


def test_absent_tier_falls_through_to_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(gp.recipes, "tier_of", lambda info: "pi-npu")
    monkeypatch.setattr(gp.recipes, "local_probe", lambda: {"host": {}})
    from taosmd import config
    config.set_generator_profile("factual-recall", data_dir=tmp_path)
    # factual-recall has no pi-npu key -> fallback used
    assert gp.resolve_generator(fallback="ollama:recipe-gen", data_dir=tmp_path) == "ollama:recipe-gen"


def test_retrieval_only_empty_string(monkeypatch, tmp_path):
    monkeypatch.setattr(gp.recipes, "tier_of", lambda info: "pi-npu")
    monkeypatch.setattr(gp.recipes, "local_probe", lambda: {"host": {}})
    # balanced pi-npu == "" -> retrieval-only
    assert gp.resolve_generator(data_dir=tmp_path) == ""


def test_per_agent_beats_global(at_12gb, tmp_path):
    from taosmd import config, agents
    config.set_generator_profile("balanced", data_dir=tmp_path)
    agents.AgentRegistry(tmp_path).register_agent("bob")
    agents.set_agent_generator_profile("bob", "factual-recall", data_dir=tmp_path)
    assert gp.resolve_generator("bob", data_dir=tmp_path) == "ollama:gemma4:12b"


def test_resolve_memory_model_uses_per_agent_profile(at_12gb, tmp_path):
    from taosmd import config, agents
    # global profile is balanced (qwen3.5:9b at gpu-12gb)
    config.set_generator_profile("balanced", data_dir=tmp_path)
    agents.AgentRegistry(tmp_path).register_agent("carol")
    # per-agent profile is factual-recall (gemma4:12b at gpu-12gb)
    agents.set_agent_generator_profile("carol", "factual-recall", data_dir=tmp_path)
    result = config.resolve_memory_model(agent="carol", data_dir=tmp_path)
    # must differ from global default (qwen3.5:9b) and match factual-recall
    assert result == "ollama:gemma4:12b"
    global_result = config.resolve_memory_model(data_dir=tmp_path)
    assert global_result == "ollama:qwen3.5:9b"
    assert result != global_result
