"""Shared pytest fixtures for the taosmd test suite.

Network guard: on a GPU/Metal host the fresh-install recipe fallback is
``maxsim-rerank-9b``, whose bge-v2-m3 reranker is absent. An un-mocked
``api.search()`` would therefore spawn a real HuggingFace download thread.

The autouse fixture below neutralises the single network entry point,
``recipes._fetch_reranker_onnx``, for every test, so the suite never performs
network IO. ``ensure_reranker_model`` keeps its real logic, so the Task 9 tests
that exercise it (and patch ``_fetch_reranker_onnx`` themselves) still work; any
un-mocked ``api.search()`` degrade path that spawns a download thread now hits
the no-op fetch instead of HuggingFace. Per-test ``monkeypatch.setattr`` calls
run after this autouse fixture, so any test that patches these symbols
explicitly still wins.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pytest


def _has_onnx_model() -> bool:
    """True if an ONNX embedding model is available on disk."""
    env = os.environ.get("TAOSMD_ONNX_PATH")
    if env:
        p = Path(env)
        if (p / "model.onnx").exists() or (p / "onnx" / "model.onnx").exists():
            return True
    candidates = [
        Path("~/.taosmd/models/minilm-onnx").expanduser(),
        Path(os.environ.get("TAOSMD_DIR", "~/taosmd")).expanduser() / "models" / "minilm-onnx",
    ]
    for candidate in candidates:
        if (candidate / "model.onnx").exists() or (candidate / "onnx" / "model.onnx").exists():
            return True
    return False


def _has_qmd_service() -> bool:
    """True if the QMD embed service is reachable at the default URL."""
    try:
        data = json.dumps({"text": "probe"}).encode()
        req = urllib.request.Request(
            "http://localhost:7832/embed",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _no_reranker_network(monkeypatch):
    """Stop any test from triggering a real reranker model download."""
    from taosmd import recipes

    monkeypatch.setattr(
        recipes, "_fetch_reranker_onnx",
        lambda *a, **k: None, raising=False,
    )
    # Reset the module-level download-status dict so a "downloading" entry from
    # one test cannot leak into the next (it would short-circuit
    # ensure_reranker_model into returning "downloading" without doing work).
    monkeypatch.setattr(recipes, "_RERANKER_DOWNLOADS", {})
    yield


@pytest.fixture
def live_embed_backend():
    """Skip unless a live embed backend (ONNX model or QMD service) is available.

    Configure an ONNX model with ``scripts/setup.sh`` or ``TAOSMD_ONNX_PATH``.
    Start the QMD service with ``qmd serve`` (default http://localhost:7832).
    """
    if _has_onnx_model() or _has_qmd_service():
        return True
    pytest.skip(
        "No live embed backend available. "
        "Install an ONNX model (run scripts/setup.sh or set TAOSMD_ONNX_PATH) "
        "or start the QMD service (qmd serve on http://localhost:7832)."
    )
