"""Tests for the opt-in registry URL config (drives A2A bus auth).

When unset, the bus stays free-handle (standalone). When set (env or config
file), the server builds a registry verifier that authenticates senders.
"""
from __future__ import annotations

import pytest

from taosmd import config


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("TAOSMD_REGISTRY_URL", raising=False)
    return str(tmp_path)


def test_unset_is_none(data_dir):
    assert config.get_registry_url(data_dir) is None


def test_set_then_get_round_trip(data_dir):
    config.set_registry_url("http://reg.local:8000", data_dir=data_dir)
    assert config.get_registry_url(data_dir) == "http://reg.local:8000"


def test_env_overrides_config_file(data_dir, monkeypatch):
    config.set_registry_url("http://from-file", data_dir=data_dir)
    monkeypatch.setenv("TAOSMD_REGISTRY_URL", "http://from-env")
    assert config.get_registry_url(data_dir) == "http://from-env"


def test_clear_returns_none(data_dir):
    config.set_registry_url("http://reg.local", data_dir=data_dir)
    config.set_registry_url("", clear=True, data_dir=data_dir)
    assert config.get_registry_url(data_dir) is None


def test_set_empty_string_raises(data_dir):
    with pytest.raises(ValueError):
        config.set_registry_url("", data_dir=data_dir)
