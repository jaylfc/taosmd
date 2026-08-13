"""Tests for the opt-in Files URL config (drives ref-fetch helper).

When unset, the ref-fetch helper falls back to ``registry_url``. When set
(env or config file), the helper resolves ``taos://`` refs against this base.
"""
from __future__ import annotations

import pytest

from taosmd import config


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("TAOSMD_FILES_URL", raising=False)
    return str(tmp_path)


def test_unset_is_none(data_dir):
    assert config.get_files_url(data_dir) is None


def test_set_then_get_round_trip(data_dir):
    config.set_files_url("http://files.local:8000", data_dir=data_dir)
    assert config.get_files_url(data_dir) == "http://files.local:8000"


def test_env_overrides_config_file(data_dir, monkeypatch):
    config.set_files_url("http://from-file:8000", data_dir=data_dir)
    monkeypatch.setenv("TAOSMD_FILES_URL", "http://from-env:8000")
    assert config.get_files_url(data_dir) == "http://from-env:8000"


def test_clear_returns_none(data_dir):
    config.set_files_url("http://files.local:8000", data_dir=data_dir)
    config.set_files_url("", clear=True, data_dir=data_dir)
    assert config.get_files_url(data_dir) is None


def test_set_empty_string_raises(data_dir):
    with pytest.raises(ValueError):
        config.set_files_url("", data_dir=data_dir)
