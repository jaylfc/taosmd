"""Tests for taosmd.backend and taosmd.taosmd_backend."""

from __future__ import annotations

import asyncio
import pytest

from taosmd.backend import MemoryBackend
from taosmd.taosmd_backend import TaOSmdBackend


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# MemoryBackend base class
# ---------------------------------------------------------------------------


def test_base_class_raises_not_implemented():
    backend = MemoryBackend()
    with pytest.raises(NotImplementedError):
        _run(backend.get_stats())


def test_base_class_defaults():
    backend = MemoryBackend()

    schema = _run(backend.get_settings_schema())
    assert schema == {"type": "object", "properties": {}}

    settings = _run(backend.get_settings())
    assert settings == {}

    agent_cfg = _run(backend.get_agent_config("some-agent"))
    assert agent_cfg == {"strategy": "thorough", "layers": []}


# ---------------------------------------------------------------------------
# TaOSmdBackend — capabilities
# ---------------------------------------------------------------------------


def test_taosmd_backend_capabilities():
    expected = {
        "kg", "vector", "archive", "catalog", "crystals",
        "pipeline", "retention", "leases", "mesh_sync",
    }
    assert set(TaOSmdBackend.capabilities) == expected
    assert len(TaOSmdBackend.capabilities) == 9


# ---------------------------------------------------------------------------
# TaOSmdBackend — stats with None stores
# ---------------------------------------------------------------------------


def test_taosmd_backend_stats(tmp_path):
    async def _check():
        backend = TaOSmdBackend(settings_db_path=tmp_path / "settings.db")
        stats = await backend.get_stats()
        assert isinstance(stats, dict)
        for section in ("kg", "vector", "archive", "catalog", "crystals"):
            assert section in stats, f"Missing section: {section}"
            assert isinstance(stats[section], dict)
        await backend.close()

    _run(_check())


# ---------------------------------------------------------------------------
# TaOSmdBackend — settings schema
# ---------------------------------------------------------------------------


def test_taosmd_backend_settings_schema(tmp_path):
    async def _check():
        backend = TaOSmdBackend(settings_db_path=tmp_path / "settings.db")
        schema = await backend.get_settings_schema()
        assert schema["type"] == "object"
        props = schema["properties"]
        expected_props = {
            "default_strategy",
            "processing_schedule",
            "enricher_model",
            "crystallize_enabled",
            "secret_filter_mode",
            "retention_hot_threshold",
            "retention_warm_threshold",
            "retention_cold_threshold",
            "archive_compress_days",
        }
        assert expected_props.issubset(set(props.keys()))
        assert props["default_strategy"]["type"] == "string"
        assert set(props["default_strategy"]["enum"]) == {"thorough", "fast", "minimal", "custom"}
        assert props["crystallize_enabled"]["type"] == "boolean"
        assert props["secret_filter_mode"]["type"] == "string"
        assert set(props["secret_filter_mode"]["enum"]) == {"redact", "reject", "warn"}
        assert props["archive_compress_days"]["type"] == "integer"
        await backend.close()

    _run(_check())


# ---------------------------------------------------------------------------
# TaOSmdBackend — settings round-trip
# ---------------------------------------------------------------------------


def test_taosmd_backend_settings_roundtrip(tmp_path):
    async def _check():
        backend = TaOSmdBackend(settings_db_path=tmp_path / "settings.db")

        updates = {
            "default_strategy": "fast",
            "crystallize_enabled": False,
            "archive_compress_days": 14,
            "retention_hot_threshold": 0.9,
        }
        result = await backend.update_settings(updates)
        assert result["default_strategy"] == "fast"
        assert result["crystallize_enabled"] is False
        assert result["archive_compress_days"] == 14
        assert result["retention_hot_threshold"] == 0.9

        # Verify persistence by re-reading
        fetched = await backend.get_settings()
        assert fetched["default_strategy"] == "fast"
        assert fetched["crystallize_enabled"] is False
        assert fetched["archive_compress_days"] == 14

        # Unmodified defaults should still be present
        assert "processing_schedule" in fetched
        assert "enricher_model" in fetched

        await backend.close()

    _run(_check())


# ---------------------------------------------------------------------------
# TaOSmdBackend — agent config round-trip (bonus, not in original 6)
# ---------------------------------------------------------------------------


def test_taosmd_backend_agent_config_roundtrip(tmp_path):
    async def _check():
        backend = TaOSmdBackend(settings_db_path=tmp_path / "settings.db")

        # Default for unknown agent
        cfg = await backend.get_agent_config("alpha")
        assert cfg == {"strategy": "thorough", "layers": []}

        # Update
        new_cfg = {"strategy": "fast", "layers": ["kg", "vector"]}
        result = await backend.update_agent_config("alpha", new_cfg)
        assert result["strategy"] == "fast"
        assert result["layers"] == ["kg", "vector"]

        # Verify persistence
        fetched = await backend.get_agent_config("alpha")
        assert fetched["strategy"] == "fast"

        # Different agent still has default
        other = await backend.get_agent_config("beta")
        assert other == {"strategy": "thorough", "layers": []}

        await backend.close()

    _run(_check())
