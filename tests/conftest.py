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

import pytest


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
