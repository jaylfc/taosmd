"""Tests for the ``collections.allowed_roots`` config key.

The allowed-roots list is the single most important safety line in the
collections contract: ``source_path`` must resolve inside one of these
directories or collection creation and indexing are refused. The default is
an EMPTY list, which means the collections feature is effectively off until
an operator opts in.

Resolution mirrors the other server settings: the
``TAOSMD_COLLECTIONS_ALLOWED_ROOTS`` env var (``os.pathsep``-separated) wins
over the ``collections.allowed_roots`` config-file key.
"""
from __future__ import annotations

import os

import pytest

from taosmd import config


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("TAOSMD_COLLECTIONS_ALLOWED_ROOTS", raising=False)
    return str(tmp_path)


def test_unset_is_empty_list(data_dir):
    assert config.get_collections_allowed_roots(data_dir) == []


def test_set_then_get_round_trip(data_dir):
    config.set_collections_allowed_roots(["/srv/docs", "/srv/repos"], data_dir=data_dir)
    assert config.get_collections_allowed_roots(data_dir) == ["/srv/docs", "/srv/repos"]


def test_env_overrides_config_file(data_dir, monkeypatch):
    config.set_collections_allowed_roots(["/from-file"], data_dir=data_dir)
    monkeypatch.setenv(
        "TAOSMD_COLLECTIONS_ALLOWED_ROOTS", os.pathsep.join(["/a", "/b"])
    )
    assert config.get_collections_allowed_roots(data_dir) == ["/a", "/b"]


def test_clear_returns_empty(data_dir):
    config.set_collections_allowed_roots(["/srv/docs"], data_dir=data_dir)
    config.set_collections_allowed_roots([], clear=True, data_dir=data_dir)
    assert config.get_collections_allowed_roots(data_dir) == []


def test_set_rejects_non_list(data_dir):
    with pytest.raises(ValueError):
        config.set_collections_allowed_roots("/srv/docs", data_dir=data_dir)


def test_set_rejects_non_string_entries(data_dir):
    with pytest.raises(ValueError):
        config.set_collections_allowed_roots(["/ok", 42], data_dir=data_dir)


def test_blank_entries_are_dropped(data_dir):
    config.set_collections_allowed_roots(["/srv/docs", "", "  "], data_dir=data_dir)
    assert config.get_collections_allowed_roots(data_dir) == ["/srv/docs"]


def test_corrupt_section_is_empty(data_dir, tmp_path):
    (tmp_path / "config.json").write_text('{"collections": "not-a-dict"}')
    assert config.get_collections_allowed_roots(data_dir) == []


def test_independent_of_other_keys(data_dir):
    config.set_collections_allowed_roots(["/srv/docs"], data_dir=data_dir)
    assert config.get_server_token(data_dir) is None
    config.set_server_token("tok", data_dir=data_dir)
    assert config.get_collections_allowed_roots(data_dir) == ["/srv/docs"]
