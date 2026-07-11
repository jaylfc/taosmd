"""Tests for the admin bearer token config (#154).

The admin token gates the admin surface independently of the data-plane server
token. It resolves like the other server settings: the ``TAOSMD_ADMIN_TOKEN``
env var wins over the ``admin_token`` config-file key, and it is stored
separately from ``server_token``.
"""
from __future__ import annotations

import pytest

from taosmd import config


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("TAOSMD_ADMIN_TOKEN", raising=False)
    return str(tmp_path)


def test_unset_is_none(data_dir):
    assert config.get_admin_token(data_dir) is None


def test_set_then_get_round_trip(data_dir):
    config.set_admin_token("admin-token", data_dir=data_dir)
    assert config.get_admin_token(data_dir) == "admin-token"


def test_env_overrides_config_file(data_dir, monkeypatch):
    config.set_admin_token("from-file", data_dir=data_dir)
    monkeypatch.setenv("TAOSMD_ADMIN_TOKEN", "from-env")
    assert config.get_admin_token(data_dir) == "from-env"


def test_clear_returns_none(data_dir):
    config.set_admin_token("tok", data_dir=data_dir)
    config.set_admin_token("", clear=True, data_dir=data_dir)
    assert config.get_admin_token(data_dir) is None


def test_set_empty_string_raises(data_dir):
    with pytest.raises(ValueError):
        config.set_admin_token("", data_dir=data_dir)


def test_independent_of_server_token(data_dir):
    """Setting the admin token must not touch the data-plane server token."""
    config.set_admin_token("admin-tok", data_dir=data_dir)
    assert config.get_server_token(data_dir) is None
    config.set_server_token("server-tok", data_dir=data_dir)
    assert config.get_admin_token(data_dir) == "admin-tok"
    assert config.get_server_token(data_dir) == "server-tok"
