"""Tests for the controls settings store in taosmd.config."""
from __future__ import annotations

import tempfile

import pytest

from taosmd import config as cfg


def test_get_controls_defaults_when_unset():
    with tempfile.TemporaryDirectory() as d:
        c = cfg.get_controls(data_dir=d)
        assert c["prefer_verified"] == "prefer_verified"
        assert c["reranker"] == "off"
        assert c["adjacent_turns"] == 2


def test_set_and_get_runtime_control_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        cfg.set_control("reranker", "bge-v2-m3", data_dir=d)
        cfg.set_control("prefer_verified", "off", data_dir=d)
        c = cfg.get_controls(data_dir=d)
        assert c["reranker"] == "bge-v2-m3"
        assert c["prefer_verified"] == "off"


def test_set_control_rejects_bad_value():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            cfg.set_control("reranker", "bogus", data_dir=d)


def test_set_control_rejects_store_and_consumer_scope():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            cfg.set_control("embedder", "minilm-onnx", data_dir=d)   # store-level
        with pytest.raises(ValueError):
            cfg.set_control("self_verify", True, data_dir=d)         # consumer-side


def test_bad_persisted_value_falls_back_to_default(tmp_path):
    import json
    (tmp_path / "config.json").write_text(json.dumps({"controls": {"adjacent_turns": 999}}))
    c = cfg.get_controls(data_dir=str(tmp_path))
    assert c["adjacent_turns"] == 2  # bad stored value ignored, default kept


def test_runtime_overrides_only_returns_persisted_runtime_controls(tmp_path):
    cfg.set_control("fusion", "rrf", data_dir=str(tmp_path))
    cfg.set_control("adjacent_turns", 3, data_dir=str(tmp_path))
    ov = cfg.get_runtime_overrides(data_dir=str(tmp_path))
    # Only the explicitly-set controls appear; no defaults are filled in.
    assert ov == {"fusion": "rrf", "adjacent_turns": 3}


def test_runtime_overrides_empty_when_nothing_set(tmp_path):
    assert cfg.get_runtime_overrides(data_dir=str(tmp_path)) == {}


def test_runtime_overrides_excludes_prefer_verified(tmp_path):
    # prefer_verified has no recipe baseline; it is resolved via get_controls,
    # not applied as a recipe override, so it must not appear here.
    cfg.set_control("prefer_verified", "off", data_dir=str(tmp_path))
    cfg.set_control("reranker", "bge-v2-m3", data_dir=str(tmp_path))
    ov = cfg.get_runtime_overrides(data_dir=str(tmp_path))
    assert ov == {"reranker": "bge-v2-m3"}


def test_runtime_overrides_skips_bad_persisted_value(tmp_path):
    import json
    (tmp_path / "config.json").write_text(
        json.dumps({"controls": {"adjacent_turns": 999, "fusion": "rrf"}}))
    ov = cfg.get_runtime_overrides(data_dir=str(tmp_path))
    assert ov == {"fusion": "rrf"}  # the out-of-range int is dropped
