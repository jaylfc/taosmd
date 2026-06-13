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
    assert set(mode) == {"late_interaction", "colbert_model"}
    assert isinstance(mode["late_interaction"], bool)
    assert isinstance(mode["colbert_model"], str)
