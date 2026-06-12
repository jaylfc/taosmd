"""Tests for the pylate ColBERT loader path in VectorMemory.

All tests are offline: no model downloads, no network calls. pylate and
sentence-transformers are mocked out via monkeypatching so the tests run
in any CI environment that only has numpy + sqlite3.

Coverage:
  (a) pylate path applies a fake projection, stores/reads token blobs at the
      model's dim (128 in the fake), and round-trips through search correctly.
  (b) ST fallback still works at 384 with the "projection head not applied"
      warning logged.
  (c) Mixed-dim guard: adding a token matrix whose dim != self._token_dim raises
      RuntimeError.
  (d) Existing late-interaction tests (ONNX path) still pass unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from taosmd.vector_memory import VectorMemory

# Fake projected dimension for pylate tests.
_FAKE_DIM = 128
# Standard MiniLM backbone dim (ST fallback path).
_ST_DIM = 384

# ---------------------------------------------------------------------------
# Fake pylate model helpers
# ---------------------------------------------------------------------------


def _make_fake_pylate_model(dim: int = _FAKE_DIM) -> MagicMock:
    """Return a mock that behaves like a pylate ColBERT model.

    encode(texts, is_query, ...) returns a list where element[0] is a
    (seq_len, dim) float32 array. seq_len is fixed at 3 tokens.
    """
    model = MagicMock()

    def _encode(texts, *, is_query=False, show_progress_bar=False, convert_to_numpy=True):
        results = []
        for _ in texts:
            # 3 fake tokens, all unit vectors along the first 'dim' axes.
            mat = np.eye(3, dim, dtype=np.float32)
            results.append(mat)
        return results

    model.encode.side_effect = _encode
    return model


def _inject_fake_pylate(monkeypatch, dim: int = _FAKE_DIM):
    """Inject a fake pylate.models module so VectorMemory can import it."""
    fake_pylate = types.ModuleType("pylate")
    fake_models = types.ModuleType("pylate.models")

    def _ColBERT(model_name_or_path: str):
        return _make_fake_pylate_model(dim)

    fake_models.ColBERT = _ColBERT
    fake_pylate.models = fake_models

    monkeypatch.setitem(sys.modules, "pylate", fake_pylate)
    monkeypatch.setitem(sys.modules, "pylate.models", fake_models)


def _make_pylate_store(tmp_path, monkeypatch, dim: int = _FAKE_DIM) -> VectorMemory:
    """Build a VectorMemory with a fake pylate model loaded."""
    _inject_fake_pylate(monkeypatch, dim)
    vmem = VectorMemory(
        db_path=str(tmp_path / "pylate.db"),
        colbert_model="fake-colbert-model",
        # colbert_model forces embed_mode="local" and late_interaction=True
    )
    asyncio.run(vmem.init())
    return vmem


# ---------------------------------------------------------------------------
# (a) pylate path stores/reads at the model's projected dim
# ---------------------------------------------------------------------------


def test_pylate_loader_sets_token_dim(tmp_path, monkeypatch):
    """After init with pylate available, _token_dim equals the model's output dim."""
    vmem = _make_pylate_store(tmp_path, monkeypatch)
    assert vmem._token_dim == _FAKE_DIM
    assert vmem._pylate_model is not None
    assert vmem.late_interaction is True
    asyncio.run(vmem.close())


def test_pylate_path_stores_correct_blob_dim(tmp_path, monkeypatch):
    """Tokens stored through pylate use the projected dim, not 384."""
    vmem = _make_pylate_store(tmp_path, monkeypatch)
    try:
        row_id = asyncio.run(vmem.add("hello world"))
        assert row_id > 0

        import base64
        row = vmem._conn.execute(
            "SELECT embedding FROM vector_memory WHERE id = ?", (row_id,)
        ).fetchone()
        assert row is not None
        blob = base64.b64decode(row[0])
        arr = np.frombuffer(blob, dtype=np.float16)
        # Must be divisible by _FAKE_DIM.
        assert arr.size % _FAKE_DIM == 0, (
            f"blob size {arr.size} not divisible by {_FAKE_DIM}"
        )
        mat = arr.reshape(-1, _FAKE_DIM)
        assert mat.shape[1] == _FAKE_DIM
    finally:
        asyncio.run(vmem.close())


def test_pylate_path_search_returns_hits(tmp_path, monkeypatch):
    """search() works end-to-end with pylate token vectors.

    Regression: in the original pylate loader, search() called embed() first
    and returned [] immediately because _local_model is None in pylate mode
    (only _pylate_model is set). This test must NOT patch embed() so that
    the fix -- bypassing the embed() gate when late_interaction=True -- is
    actually exercised.
    """
    vmem = _make_pylate_store(tmp_path, monkeypatch)
    try:
        asyncio.run(vmem.add("alpha beta gamma"))
        asyncio.run(vmem.add("delta epsilon zeta"))
        # No embed() patch: the gate bypass in search() must handle pylate mode.
        hits = asyncio.run(vmem.search("alpha", limit=2, hybrid=False, fusion="none"))
        assert len(hits) == 2, (
            "search() returned no results in pylate mode — embed() gate bug present"
        )
        for h in hits:
            assert "similarity" in h
            assert 0.0 <= h["similarity"] <= 1.0 + 1e-6
    finally:
        asyncio.run(vmem.close())


def test_pylate_vectors_are_l2_normalised(tmp_path, monkeypatch):
    """Token vectors coming out of _embed_tokens_pylate are L2-normalised."""
    _inject_fake_pylate(monkeypatch)
    vmem = VectorMemory(
        db_path=str(tmp_path / "norm.db"),
        colbert_model="fake-colbert-model",
    )
    asyncio.run(vmem.init())
    try:
        vecs = vmem._embed_tokens_pylate("test text", is_query=False)
        assert vecs, "expected non-empty token list"
        for v in vecs:
            norm = np.linalg.norm(v)
            assert abs(norm - 1.0) < 1e-5, f"token vector not unit-norm: {norm}"
    finally:
        asyncio.run(vmem.close())


# ---------------------------------------------------------------------------
# (b) ST fallback works at 384 with a warning when pylate is not available
# ---------------------------------------------------------------------------


def test_st_fallback_warns_when_pylate_missing(tmp_path, caplog, monkeypatch):
    """When pylate is absent, VectorMemory logs a warning and falls back to ST."""
    # Ensure pylate is NOT importable for this test.
    monkeypatch.setitem(sys.modules, "pylate", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "pylate.models", None)  # type: ignore[arg-type]

    # ST must still work: provide a minimal mock.
    st_model = MagicMock()

    def _st_encode(text, output_value=None, normalize_embeddings=False,
                   convert_to_numpy=True, show_progress_bar=False):
        if output_value == "token_embeddings":
            return np.eye(3, _ST_DIM, dtype=np.float32)
        return np.ones(_ST_DIM, dtype=np.float32)

    st_model.encode.side_effect = _st_encode

    fake_st_module = types.ModuleType("sentence_transformers")

    def _SentenceTransformer(name):
        return st_model

    fake_st_module.SentenceTransformer = _SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_module)

    with caplog.at_level(logging.WARNING, logger="taosmd.vector_memory"):
        vmem = VectorMemory(
            db_path=str(tmp_path / "fallback.db"),
            colbert_model="fake-colbert-model",
        )
        asyncio.run(vmem.init())

    assert vmem._pylate_model is None
    assert vmem._local_model is st_model
    # The warning about the projection head must have been logged.
    assert any("pylate" in r.message.lower() for r in caplog.records), (
        f"Expected a pylate warning in log, got: {[r.message for r in caplog.records]}"
    )
    # _token_dim stays at 384 for the ST fallback path.
    assert vmem._token_dim == _ST_DIM
    asyncio.run(vmem.close())


def test_st_fallback_stores_at_384(tmp_path, monkeypatch):
    """ST fallback path stores token blobs with 384-wide vectors."""
    monkeypatch.setitem(sys.modules, "pylate", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "pylate.models", None)  # type: ignore[arg-type]

    st_model = MagicMock()

    def _st_encode(text, output_value=None, normalize_embeddings=False,
                   convert_to_numpy=True, show_progress_bar=False):
        if output_value == "token_embeddings":
            mat = np.zeros((4, _ST_DIM), dtype=np.float32)
            mat[:, 0] = 1.0  # non-zero so blobs are non-empty
            return mat
        return np.ones(_ST_DIM, dtype=np.float32)

    st_model.encode.side_effect = _st_encode

    fake_st_module = types.ModuleType("sentence_transformers")
    fake_st_module.SentenceTransformer = lambda name: st_model
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_module)

    vmem = VectorMemory(
        db_path=str(tmp_path / "st384.db"),
        colbert_model="fake-colbert-model",
    )
    asyncio.run(vmem.init())
    try:
        row_id = asyncio.run(vmem.add("some text"))
        assert row_id > 0
        import base64
        row = vmem._conn.execute(
            "SELECT embedding FROM vector_memory WHERE id = ?", (row_id,)
        ).fetchone()
        blob = base64.b64decode(row[0])
        arr = np.frombuffer(blob, dtype=np.float16)
        assert arr.size % _ST_DIM == 0
        mat = arr.reshape(-1, _ST_DIM)
        assert mat.shape[1] == _ST_DIM
    finally:
        asyncio.run(vmem.close())


# ---------------------------------------------------------------------------
# (c) Mixed-dim guard
# ---------------------------------------------------------------------------


def test_mixed_dim_raises_runtime_error(tmp_path, monkeypatch):
    """Storing a token matrix with dim != self._token_dim raises RuntimeError."""
    _inject_fake_pylate(monkeypatch, dim=_FAKE_DIM)
    vmem = VectorMemory(
        db_path=str(tmp_path / "mixeddim.db"),
        colbert_model="fake-colbert-model",
    )
    asyncio.run(vmem.init())
    try:
        assert vmem._token_dim == _FAKE_DIM

        # Monkeypatch _embed_tokens_pylate to return wrong-dim vectors.
        def _bad_tokens(text: str, *, is_query: bool = False) -> list[list[float]]:
            wrong_dim = _FAKE_DIM + 64  # intentionally different
            return np.eye(3, wrong_dim, dtype=np.float32).tolist()

        vmem._embed_tokens_pylate = _bad_tokens  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="does not match"):
            asyncio.run(vmem.add("trigger the guard"))
    finally:
        asyncio.run(vmem.close())
