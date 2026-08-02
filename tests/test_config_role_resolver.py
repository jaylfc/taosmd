"""Tests for the opt-in A2A role-resolver URL config (taOS#2155).

When unset, role recipients (``@taOS-*``) are rejected at send time. When set
(env or config file), the HTTP server builds a :class:`~taosmd.role_resolver
.RoleResolver` and uses it for send-time validation and delivery-time
resolution.
"""
from __future__ import annotations

import pytest

from taosmd import config


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("TAOSMD_A2A_ROLE_RESOLVER_URL", raising=False)
    return str(tmp_path)


def test_unset_is_none(data_dir):
    assert config.get_a2a_role_resolver_url(data_dir) is None


def test_set_then_get_round_trip(data_dir):
    config.set_a2a_role_resolver_url("http://taos.local:8000", data_dir=data_dir)
    assert config.get_a2a_role_resolver_url(data_dir) == "http://taos.local:8000"


def test_env_overrides_config_file(data_dir, monkeypatch):
    config.set_a2a_role_resolver_url("http://from-file", data_dir=data_dir)
    monkeypatch.setenv("TAOSMD_A2A_ROLE_RESOLVER_URL", "http://from-env")
    assert config.get_a2a_role_resolver_url(data_dir) == "http://from-env"


def test_clear_returns_none(data_dir):
    config.set_a2a_role_resolver_url("http://taos.local", data_dir=data_dir)
    config.set_a2a_role_resolver_url("", clear=True, data_dir=data_dir)
    assert config.get_a2a_role_resolver_url(data_dir) is None


def test_set_empty_string_raises(data_dir):
    with pytest.raises(ValueError):
        config.set_a2a_role_resolver_url("", data_dir=data_dir)
