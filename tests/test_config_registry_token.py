"""Tests for the registry auth token config (the taOS local/admin token).

Per the #710 contract the registry's revoked feed (``/api/agents/registry/revoked``)
moved behind admin auth: the bus must send ``Authorization: Bearer <token>`` to
poll it (the pubkey endpoint stays public). This token is stored here, resolved
exactly like the other server settings (env wins over the config file).
"""
from __future__ import annotations

import pytest

from taosmd import config


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("TAOSMD_REGISTRY_TOKEN", raising=False)
    return str(tmp_path)


def test_unset_is_none(data_dir):
    assert config.get_registry_token(data_dir) is None


def test_set_then_get_round_trip(data_dir):
    config.set_registry_token("local-admin-token", data_dir=data_dir)
    assert config.get_registry_token(data_dir) == "local-admin-token"


def test_env_overrides_config_file(data_dir, monkeypatch):
    config.set_registry_token("from-file", data_dir=data_dir)
    monkeypatch.setenv("TAOSMD_REGISTRY_TOKEN", "from-env")
    assert config.get_registry_token(data_dir) == "from-env"


def test_clear_returns_none(data_dir):
    config.set_registry_token("tok", data_dir=data_dir)
    config.set_registry_token("", clear=True, data_dir=data_dir)
    assert config.get_registry_token(data_dir) is None


def test_set_empty_string_raises(data_dir):
    with pytest.raises(ValueError):
        config.set_registry_token("", data_dir=data_dir)
