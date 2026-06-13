"""The live serve path must build the vector store in the storage mode the
config specifies (seeded from the recommended recipe at setup), closing the
gap where the lateint recipe was recommended but the store was always dense.
"""

import asyncio
import json

from taosmd import api as taosmd_api
from taosmd import auto_setup


def test_ensure_stores_honors_config_late_interaction(tmp_path, monkeypatch):
    data_dir = tmp_path / "d"
    data_dir.mkdir()
    # qmd embed mode + late_interaction avoids loading a local model in the test
    # while still exercising the config -> store wiring.
    (data_dir / "config.json").write_text(json.dumps({
        "vector_memory": {"embed_mode": "qmd", "late_interaction": True},
    }))
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
    assert stores["vector"].late_interaction is True


def test_ensure_stores_defaults_dense(tmp_path, monkeypatch):
    data_dir = tmp_path / "d"
    data_dir.mkdir()
    (data_dir / "config.json").write_text(json.dumps({
        "vector_memory": {"embed_mode": "qmd"},
    }))
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    stores = asyncio.run(taosmd_api._ensure_stores(str(data_dir)))
    assert stores["vector"].late_interaction is False


def test_recommended_store_mode_shape():
    mode = auto_setup._recommended_store_mode()
    assert set(mode) == {"late_interaction", "colbert_model", "embed_model"}
    assert isinstance(mode["late_interaction"], bool)
    assert isinstance(mode["colbert_model"], str)
    assert isinstance(mode["embed_model"], str)


def test_ensure_stores_honors_config_embed_model(tmp_path, monkeypatch):
    # config.vector_memory.embed_model selects the onnx model dir; with no model
    # present the path resolves None and falls back to qmd, but the resolver was
    # asked for the configured model, not hardcoded minilm.
    import taosmd.api as a
    seen = {}
    real = a._resolve_onnx_path
    def spy(data_dir, model_name="minilm-onnx"):
        seen["model_name"] = model_name
        return real(data_dir, model_name)
    monkeypatch.setattr(a, "_resolve_onnx_path", spy)
    data_dir = tmp_path / "d"; data_dir.mkdir()
    (data_dir / "config.json").write_text(json.dumps({
        "vector_memory": {"embed_mode": "onnx", "embed_model": "arctic-embed-s"},
    }))
    monkeypatch.setattr(a, "_stores_cache", {})
    asyncio.run(a._ensure_stores(str(data_dir)))
    assert seen["model_name"] == "arctic-embed-s"


def test_recommended_store_mode_includes_embed_model():
    mode = auto_setup._recommended_store_mode()
    assert "embed_model" in mode
    assert isinstance(mode["embed_model"], str) and mode["embed_model"]


def test_low_tier_recipes_use_arctic():
    from taosmd import recipes
    for rid in ("lite-pi", "fast-8b"):
        r = recipes.get_recipe(rid)
        assert r.retrieval.get("embed_model") == "arctic-embed-s"
