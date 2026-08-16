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

import functools
import json
import os
import urllib.request
from pathlib import Path

import pytest

_NO_BACKEND_SKIP_MSG = (
    "No live embed backend available. "
    "Install an ONNX model (run scripts/setup.sh or set TAOSMD_ONNX_PATH) "
    "or start the QMD service (qmd serve on http://localhost:7832)."
)


def _resolve_onnx_model_file() -> Path | None:
    """Return the ``model.onnx`` path for a discoverable ONNX backend, else None.

    Honours ``$TAOSMD_ONNX_PATH`` and the default install locations
    (``~/.taosmd/models/minilm-onnx`` and ``$TAOSMD_DIR/models/minilm-onnx``),
    accepting either a root ``model.onnx`` or an ``onnx/model.onnx`` layout.
    Existence is necessary but not sufficient: :func:`_has_onnx_model` loads the
    file before a backend is considered available.
    """
    env = os.environ.get("TAOSMD_ONNX_PATH")
    roots = [Path(env)] if env else []
    roots.extend(
        [
            Path("~/.taosmd/models/minilm-onnx").expanduser(),
            Path(os.environ.get("TAOSMD_DIR", "~/taosmd")).expanduser() / "models" / "minilm-onnx",
        ]
    )
    for root in roots:
        for rel in ("model.onnx", "onnx/model.onnx"):
            candidate = root / rel
            if candidate.is_file():
                return candidate
    return None


def _has_onnx_model() -> bool:
    """True if an ONNX embedding backend can actually load and embed.

    Loads the model with ONNX Runtime rather than merely checking that a
    ``model.onnx`` file exists, so a truncated or empty file left by an
    interrupted ``scripts/setup.sh`` is treated as "no backend" and the
    guarded tests SKIP instead of failing at embed time.
    """
    model_file = _resolve_onnx_model_file()
    if model_file is None:
        return False
    try:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 4  # silence the load diagnostics on bad files
        options.intra_op_num_threads = 1
        ort.InferenceSession(
            str(model_file),
            options,
            providers=["CPUExecutionProvider"],
        )
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _has_qmd_service() -> bool:
    """True if the QMD embed service is reachable at the default URL.

    Cached for the process lifetime: the cached value is itself a real probe
    response from this session, and a filtered (rather than refused) port would
    otherwise block for the full timeout on every function-scoped fixture call.
    """
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


def _live_backend_available() -> bool:
    """True if a live ONNX or QMD embed backend is actually usable."""
    return _has_onnx_model() or _has_qmd_service()


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

    The ONNX path is verified by loading the model (not just checking that the
    file exists), so a truncated or empty ``model.onnx`` from an interrupted
    ``scripts/setup.sh`` SKIPs here instead of producing confusing embed
    failures later. Configure with ``scripts/setup.sh`` or
    ``TAOSMD_ONNX_PATH``; start the QMD service with ``qmd serve`` on the
    default http://localhost:7832.
    """
    if _live_backend_available():
        return True
    pytest.skip(_NO_BACKEND_SKIP_MSG)
