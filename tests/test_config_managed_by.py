"""Tests for managed_by and serve_dashboard config keys."""
import os
import pytest
from taosmd import config as cfg


def test_managed_by_defaults_to_standalone(tmp_path):
    assert cfg.get_managed_by(tmp_path) == cfg.MANAGED_BY_STANDALONE


def test_managed_by_set_and_get(tmp_path):
    cfg.set_managed_by(cfg.MANAGED_BY_TAOS, tmp_path)
    assert cfg.get_managed_by(tmp_path) == cfg.MANAGED_BY_TAOS


def test_managed_by_roundtrip_standalone(tmp_path):
    cfg.set_managed_by(cfg.MANAGED_BY_TAOS, tmp_path)
    cfg.set_managed_by(cfg.MANAGED_BY_STANDALONE, tmp_path)
    assert cfg.get_managed_by(tmp_path) == cfg.MANAGED_BY_STANDALONE


def test_managed_by_rejects_invalid(tmp_path):
    with pytest.raises(ValueError):
        cfg.set_managed_by("cloud", tmp_path)


def test_managed_by_env_overrides_config(tmp_path, monkeypatch):
    cfg.set_managed_by(cfg.MANAGED_BY_STANDALONE, tmp_path)
    monkeypatch.setenv("TAOSMD_MANAGED_BY", cfg.MANAGED_BY_TAOS)
    assert cfg.get_managed_by(tmp_path) == cfg.MANAGED_BY_TAOS


def test_managed_by_env_invalid_falls_through_to_config(tmp_path, monkeypatch):
    cfg.set_managed_by(cfg.MANAGED_BY_TAOS, tmp_path)
    monkeypatch.setenv("TAOSMD_MANAGED_BY", "notvalid")
    assert cfg.get_managed_by(tmp_path) == cfg.MANAGED_BY_TAOS


# ---------------------------------------------------------------------------
# serve_dashboard
# ---------------------------------------------------------------------------

def test_serve_dashboard_true_when_standalone(tmp_path):
    assert cfg.get_serve_dashboard(tmp_path) is True


def test_serve_dashboard_false_when_taos(tmp_path):
    cfg.set_managed_by(cfg.MANAGED_BY_TAOS, tmp_path)
    assert cfg.get_serve_dashboard(tmp_path) is False


def test_serve_dashboard_explicit_override_true(tmp_path):
    cfg.set_managed_by(cfg.MANAGED_BY_TAOS, tmp_path)
    cfg.set_serve_dashboard(True, tmp_path)
    assert cfg.get_serve_dashboard(tmp_path) is True


def test_serve_dashboard_explicit_override_false(tmp_path):
    cfg.set_managed_by(cfg.MANAGED_BY_STANDALONE, tmp_path)
    cfg.set_serve_dashboard(False, tmp_path)
    assert cfg.get_serve_dashboard(tmp_path) is False


def test_serve_dashboard_env_true(tmp_path, monkeypatch):
    cfg.set_managed_by(cfg.MANAGED_BY_TAOS, tmp_path)
    monkeypatch.setenv("TAOSMD_SERVE_DASHBOARD", "1")
    assert cfg.get_serve_dashboard(tmp_path) is True


def test_serve_dashboard_env_false(tmp_path, monkeypatch):
    cfg.set_managed_by(cfg.MANAGED_BY_STANDALONE, tmp_path)
    monkeypatch.setenv("TAOSMD_SERVE_DASHBOARD", "false")
    assert cfg.get_serve_dashboard(tmp_path) is False
