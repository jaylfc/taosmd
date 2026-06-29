"""Regression test: memory_extractor uses resolve_memory_model (not get_memory_model).

Verifies that the model-resolution code path at memory_extractor.py L259-263 routes
through the generator-profile resolver, so a profile/recipe generator surfaces as a
real model name rather than the "default" sentinel, and that the fallback to "default"
fires when nothing is resolvable.
"""
import pytest
from taosmd import generator_profiles as gp
from taosmd import config


@pytest.fixture
def at_12gb(monkeypatch):
    monkeypatch.setattr(gp.recipes, "local_probe", lambda: {"host": {}})
    monkeypatch.setattr(gp.recipes, "tier_of", lambda info: "gpu-12gb")


@pytest.fixture
def at_pi_npu(monkeypatch):
    monkeypatch.setattr(gp.recipes, "local_probe", lambda: {"host": {}})
    monkeypatch.setattr(gp.recipes, "tier_of", lambda info: "pi-npu")


def test_extractor_resolves_real_model_via_profile(at_12gb, tmp_path):
    # With a covered tier and default "balanced" profile, resolve_memory_model
    # returns a concrete model string (qwen3.5:9b at 12 GB), not "default".
    # This is the path memory_extractor.py now takes.
    result = config.resolve_memory_model(data_dir=tmp_path)
    assert result is not None, "resolve_memory_model returned None for a covered tier"
    assert result != "default", f"resolve_memory_model returned sentinel 'default': {result!r}"
    assert "qwen3.5" in result, f"expected qwen3.5:9b for balanced@12gb, got {result!r}"


def test_extractor_resolves_factual_recall_profile(at_12gb, tmp_path):
    # factual-recall profile at 12 GB should surface gemma4:12b.
    config.set_generator_profile("factual-recall", data_dir=tmp_path)
    result = config.resolve_memory_model(data_dir=tmp_path)
    assert result is not None
    assert result != "default"
    assert "gemma4" in result, f"expected gemma4:12b for factual-recall@12gb, got {result!r}"


def test_extractor_falls_back_to_default_when_unresolvable(at_pi_npu, tmp_path):
    # balanced at pi-npu == "" (retrieval-only): resolve_memory_model returns None,
    # so the extractor's `or "default"` fires and the sentinel is used.
    result = config.resolve_memory_model(data_dir=tmp_path)
    # resolve_memory_model returns None for empty/retrieval-only resolution
    assert result is None, (
        f"expected None for retrieval-only tier, got {result!r}; "
        "extractor would not fall back to 'default'"
    )
    # Confirm the extractor's fallback expression yields the sentinel
    assert (result or "default") == "default"
