"""Tests for the ``live_embed_backend`` skip guard and its ONNX/QMD probes.

Regression coverage for tsk-rm57pj: a truncated or empty ``model.onnx`` (the
remnant of an interrupted ``scripts/setup.sh``) must make the guard SKIP
instead of letting an unusable model through to fail at embed time. The QMD
probe already verifies a live response (not mere reachability); the ONNX probe
must do the same in spirit by loading the model, not just checking existence.
"""
from __future__ import annotations

import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests import conftest


def _onnx_dir(tmp_path: Path, name: str = "minilm-onnx", layout: str = "root") -> Path:
    d = tmp_path / name
    if layout == "onnx_subdir":
        (d / "onnx").mkdir(parents=True)
    else:
        d.mkdir(parents=True)
    return d


def _placeholder(d: Path) -> None:
    (d / "model.onnx").write_bytes(b"placeholder")


def _neutralize_default_candidates(tmp_path, monkeypatch):
    """Redirect '~' expansion so the install-location candidates resolve to a
    fake home under tmp_path (where no model exists), making backend probes
    depend solely on ``TAOSMD_ONNX_PATH``."""
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()

    def fake_expanduser(p):
        if p.startswith("~"):
            rel = p.lstrip("~").lstrip("/")
            return str(fake_home / rel)
        return p

    monkeypatch.setattr("os.path.expanduser", fake_expanduser)


# --- resolution ---------------------------------------------------------------


def test_resolve_finds_root_model(tmp_path, monkeypatch):
    d = _onnx_dir(tmp_path, "root-layout")
    _placeholder(d)
    monkeypatch.setenv("TAOSMD_ONNX_PATH", str(d))
    assert conftest._resolve_onnx_model_file() == d / "model.onnx"


def test_resolve_finds_onnx_subdir_model(tmp_path, monkeypatch):
    d = _onnx_dir(tmp_path, "arctic-embed-s", layout="onnx_subdir")
    (d / "onnx" / "model.onnx").write_bytes(b"placeholder")
    monkeypatch.setenv("TAOSMD_ONNX_PATH", str(d))
    assert conftest._resolve_onnx_model_file() == d / "onnx" / "model.onnx"


def test_resolve_returns_none_when_absent(tmp_path, monkeypatch):
    _neutralize_default_candidates(tmp_path, monkeypatch)
    d = tmp_path / "empty"
    d.mkdir()
    monkeypatch.setenv("TAOSMD_ONNX_PATH", str(d))
    assert conftest._resolve_onnx_model_file() is None


# --- the defect: a truncated / empty model.onnx is NOT a working backend ------


def test_empty_model_onnx_is_not_a_backend(tmp_path, monkeypatch):
    d = _onnx_dir(tmp_path)
    (d / "model.onnx").write_bytes(b"")
    monkeypatch.setenv("TAOSMD_ONNX_PATH", str(d))
    assert conftest._has_onnx_model() is False


def test_truncated_model_onnx_is_not_a_backend(tmp_path, monkeypatch):
    d = _onnx_dir(tmp_path)
    (d / "model.onnx").write_bytes(b"\x08\x01\x02 not a real onnx graph")
    monkeypatch.setenv("TAOSMD_ONNX_PATH", str(d))
    assert conftest._has_onnx_model() is False


def test_truncated_model_under_onnx_subdir_is_not_a_backend(tmp_path, monkeypatch):
    d = _onnx_dir(tmp_path, "arctic-embed-s", layout="onnx_subdir")
    (d / "onnx" / "model.onnx").write_bytes(b"")
    monkeypatch.setenv("TAOSMD_ONNX_PATH", str(d))
    assert conftest._has_onnx_model() is False


# --- positive control: a model that loads IS a working backend ----------------
# A real 90MB ONNX model is not available in CI, so only the narrow ONNX
# session constructor is mocked. Without this test the skip-only negative tests
# would be indistinguishable from a guard pinned to always-skip -- the
# "fixed into always skipping" regression the card warns against.


def test_working_onnx_model_is_detected(tmp_path, monkeypatch):
    d = _onnx_dir(tmp_path)
    _placeholder(d)
    monkeypatch.setenv("TAOSMD_ONNX_PATH", str(d))
    with patch("onnxruntime.InferenceSession") as sess:
        sess.return_value = MagicMock()
        assert conftest._has_onnx_model() is True


def test_onnx_load_failure_is_not_a_working_backend(tmp_path, monkeypatch):
    d = _onnx_dir(tmp_path)
    _placeholder(d)
    monkeypatch.setenv("TAOSMD_ONNX_PATH", str(d))
    with patch("onnxruntime.InferenceSession", side_effect=RuntimeError("unloadable")):
        assert conftest._has_onnx_model() is False


# --- combined decision --------------------------------------------------------


def test_live_backend_available_false_when_truncated(tmp_path, monkeypatch):
    d = _onnx_dir(tmp_path)
    (d / "model.onnx").write_bytes(b"")
    monkeypatch.setenv("TAOSMD_ONNX_PATH", str(d))
    monkeypatch.setattr(conftest, "_has_qmd_service", lambda: False)
    assert conftest._live_backend_available() is False


def test_live_backend_available_true_when_onnx_works(tmp_path, monkeypatch):
    d = _onnx_dir(tmp_path)
    _placeholder(d)
    monkeypatch.setenv("TAOSMD_ONNX_PATH", str(d))
    monkeypatch.setattr(conftest, "_has_qmd_service", lambda: False)
    with patch("onnxruntime.InferenceSession") as sess:
        sess.return_value = MagicMock()
        assert conftest._live_backend_available() is True


# --- fixture end-to-end: skip vs run ------------------------------------------


def test_fixture_skips_when_no_backend(monkeypatch):
    monkeypatch.setattr(conftest, "_has_onnx_model", lambda: False)
    monkeypatch.setattr(conftest, "_has_qmd_service", lambda: False)
    with pytest.raises(pytest.skip.Exception, match="No live embed backend"):
        conftest.live_embed_backend.__wrapped__()


def test_fixture_runs_when_backend_works(tmp_path, monkeypatch):
    d = _onnx_dir(tmp_path)
    _placeholder(d)
    monkeypatch.setenv("TAOSMD_ONNX_PATH", str(d))
    monkeypatch.setattr(conftest, "_has_qmd_service", lambda: False)
    with patch("onnxruntime.InferenceSession") as sess:
        sess.return_value = MagicMock()
        assert conftest.live_embed_backend.__wrapped__() is True


# --- QMD probe: caches its result so a filtered port is probed once -----------


def test_qmd_probe_caches_result(monkeypatch):
    conftest._has_qmd_service.cache_clear()
    counter = {"n": 0}
    fake_resp = MagicMock()
    fake_resp.__enter__.return_value.status = 200

    def fake_urlopen(*args, **kwargs):
        counter["n"] += 1
        return fake_resp

    try:
        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        assert conftest._has_qmd_service() is True
        assert conftest._has_qmd_service() is True
        assert counter["n"] == 1
    finally:
        conftest._has_qmd_service.cache_clear()


def test_qmd_probe_refused_is_false(monkeypatch):
    conftest._has_qmd_service.cache_clear()
    try:
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            assert conftest._has_qmd_service() is False
    finally:
        conftest._has_qmd_service.cache_clear()
