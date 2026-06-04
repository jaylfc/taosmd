"""Tests for the system-wide memory model config (taosmd.config).

All tests pin ``TAOSMD_DATA_DIR`` to a temp dir so the real ``~/.taosmd``
is never touched.
"""

from __future__ import annotations

import importlib
import json

import pytest

import taosmd.config as config


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "taosmd-data"
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(d))
    return d


# --- config round-trip ------------------------------------------------------


def test_set_then_get_round_trip(data_dir):
    config.set_memory_model("ollama:qwen3:4b")
    assert config.get_memory_model() == "ollama:qwen3:4b"


def test_clear_returns_none(data_dir):
    config.set_memory_model("ollama:qwen3:4b")
    config.set_memory_model("", clear=True)
    assert config.get_memory_model() is None


def test_unset_is_none(data_dir):
    assert config.get_memory_model() is None


def test_set_empty_string_raises(data_dir):
    with pytest.raises(ValueError):
        config.set_memory_model("")
    with pytest.raises(ValueError):
        config.set_memory_model("   ")


def test_persists_across_fresh_read(data_dir):
    config.set_memory_model("ollama:qwen3:4b")
    # Re-import the module to simulate a fresh process reading the file.
    fresh = importlib.reload(config)
    assert fresh.get_memory_model() == "ollama:qwen3:4b"


def test_corrupt_file_treated_as_unset(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "config.json").write_text("{ this is not valid json")
    assert config.get_memory_model() is None
    # And we can still set on top of it (overwrites the corrupt file).
    config.set_memory_model("ollama:qwen3:4b")
    assert config.get_memory_model() == "ollama:qwen3:4b"
    on_disk = json.loads((data_dir / "config.json").read_text())
    assert on_disk["memory_model"] == "ollama:qwen3:4b"


# --- resolve precedence -----------------------------------------------------


def test_resolve_prefers_global_over_fallback(data_dir):
    config.set_memory_model("ollama:qwen3:4b")
    assert config.resolve_memory_model("fallback-model") == "ollama:qwen3:4b"


def test_resolve_uses_fallback_when_unset(data_dir):
    assert config.resolve_memory_model("fallback-model") == "fallback-model"


def test_resolve_none_fallback_when_unset(data_dir):
    assert config.resolve_memory_model() is None


# --- top-level exports ------------------------------------------------------


def test_top_level_exports(data_dir):
    import taosmd

    taosmd.set_memory_model("ollama:qwen3:4b")
    assert taosmd.get_memory_model() == "ollama:qwen3:4b"


# --- migration: per-agent model is stripped ---------------------------------


def test_legacy_per_agent_model_stripped_on_read(data_dir):
    from taosmd.agents import LIBRARIAN_TASKS, AgentRegistry

    registry = AgentRegistry(data_dir)
    # Hand-write an agents.json carrying a legacy per-agent model key.
    legacy = {
        "agents": [
            {
                "name": "old-agent",
                "display_name": "Old Agent",
                "created_at": 1700000000,
                "last_ingest_at": 0,
                "total_chunks": 0,
                "librarian": {
                    "enabled": True,
                    "model": "ollama:legacy-model",
                    "tasks": {t: True for t in LIBRARIAN_TASKS},
                    "fanout": {"default": "low", "auto_scale": True},
                },
            }
        ]
    }
    (data_dir).mkdir(parents=True, exist_ok=True)
    (data_dir / "agents.json").write_text(json.dumps(legacy))

    lib = registry.get_librarian("old-agent")
    assert "model" not in lib
    assert lib["enabled"] is True


def test_set_librarian_model_redirects_and_warns(data_dir):
    from taosmd.agents import AgentRegistry

    registry = AgentRegistry(data_dir)
    registry.register_agent("alice")

    with pytest.warns(DeprecationWarning):
        registry.set_librarian("alice", model="ollama:qwen3:4b")

    # Nothing stored per-agent; the global config now carries the model.
    assert "model" not in registry.get_librarian("alice")
    assert config.get_memory_model() == "ollama:qwen3:4b"

    with pytest.warns(DeprecationWarning):
        registry.set_librarian("alice", clear_model=True)
    assert config.get_memory_model() is None
