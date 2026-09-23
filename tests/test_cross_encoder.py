import sys

import pytest

from taosmd.cross_encoder import CrossEncoderReranker


def test_reranker_fallback_no_model():
    reranker = CrossEncoderReranker(onnx_path="/nonexistent/path")
    results = [
        {"text": "alpha", "score": 0.9},
        {"text": "beta", "score": 0.8},
        {"text": "gamma", "score": 0.7},
    ]
    output = reranker.rerank("test query", results, limit=5)
    assert output == results


def test_reranker_available_property():
    reranker = CrossEncoderReranker(onnx_path="/nonexistent/path")
    assert reranker.available is False


# --- absent-onnxruntime degradation -------------------------------------------
# onnxruntime is an optional extra (pyproject `onnx` group): it is genuinely
# absent on musl hosts (Alpine, postmarketOS), where it has no wheel or sdist.
# CI now always installs it (`uv sync --extra onnx`), so these tests SIMULATE
# absence by putting `None` in sys.modules for "onnxruntime" -- the documented
# way to make `import onnxruntime` raise ImportError regardless of what is
# actually installed -- rather than relying on it being missing in the
# environment. This exercises the guard in
# ``CrossEncoderReranker._load()`` (taosmd/cross_encoder.py), which turns a
# bare ImportError into an actionable one, and the fallback in
# ``CrossEncoderReranker.rerank()``, which must degrade to the unranked
# results rather than raise.


def _model_dir(tmp_path):
    onnx_dir = tmp_path / "cross-encoder-onnx"
    onnx_dir.mkdir()
    (onnx_dir / "model.onnx").write_bytes(b"placeholder")
    return onnx_dir


def test_load_raises_actionable_error_when_onnxruntime_absent(tmp_path, monkeypatch):
    """``_load()`` turns a bare ImportError into one naming the fix, instead of
    letting a raw "No module named 'onnxruntime'" surface from deep in a
    reranker call."""
    reranker = CrossEncoderReranker(onnx_path=str(_model_dir(tmp_path)))
    assert reranker.available is True  # the model file itself is present

    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError, match=r"pip install taosmd\[onnx\]"):
        reranker._load()


def test_rerank_degrades_gracefully_when_onnxruntime_absent(tmp_path, monkeypatch):
    """``rerank()`` must fall back to the unranked results, not raise, when
    onnxruntime is unavailable -- the same graceful-degradation contract
    ``test_reranker_fallback_no_model`` covers for a missing model file."""
    reranker = CrossEncoderReranker(onnx_path=str(_model_dir(tmp_path)))
    assert reranker.available is True

    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    results = [
        {"text": "alpha", "score": 0.9},
        {"text": "beta", "score": 0.8},
    ]
    output = reranker.rerank("test query", results, limit=5)
    assert output == results
