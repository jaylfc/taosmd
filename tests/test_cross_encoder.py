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


def test_case_b_red(tmp_path, monkeypatch):
    """RED test: min_confidence=0.0, scores [-6.9, -4.4, -5.9] -> 
    order A,B,C (input order), all rerank_applied False, 
    each cross_encoder_score equals its own score"""
    
    reranker = CrossEncoderReranker(onnx_path=str(_model_dir(tmp_path)), min_confidence=0.0)
    
    # Patch available property
    monkeypatch.setattr(CrossEncoderReranker, "available", property(lambda self: True))
    # Patch _load to no-op
    monkeypatch.setattr(reranker, "_load", lambda: None)
    # Patch _score to return fixed scores
    monkeypatch.setattr(reranker, "_score", lambda query, results: [-6.9, -4.4, -5.9])
    
    results = [
        {"text": "A", "score": 0.9},
        {"text": "B", "score": 0.8},
        {"text": "C", "score": 0.7},
    ]
    
    output = reranker.rerank("test query", results, limit=5)
    
    # Expected: input order preserved (A, B, C) because max score (-4.4) < min_confidence (0.0)
    # Each should have cross_encoder_score equal to its own score and rerank_applied=False
    assert len(output) == 3
    assert output[0]["text"] == "A"
    assert output[1]["text"] == "B"
    assert output[2]["text"] == "C"
    assert output[0]["cross_encoder_score"] == -6.9
    assert output[1]["cross_encoder_score"] == -4.4
    assert output[2]["cross_encoder_score"] == -5.9
    assert output[0]["rerank_applied"] is False
    assert output[1]["rerank_applied"] is False
    assert output[2]["rerank_applied"] is False


def test_case_a(tmp_path, monkeypatch):
    """New test: min_confidence=None, scores [-6.9, -4.4, -5.9] -> order B,C,A, 
    every hit rerank_applied True"""
    
    reranker = CrossEncoderReranker(onnx_path=str(_model_dir(tmp_path)), min_confidence=None)
    
    # Patch available property
    monkeypatch.setattr(CrossEncoderReranker, "available", property(lambda self: True))
    # Patch _load to no-op
    monkeypatch.setattr(reranker, "_load", lambda: None)
    # Patch _score to return fixed scores
    monkeypatch.setattr(reranker, "_score", lambda query, results: [-6.9, -4.4, -5.9])
    
    results = [
        {"text": "A", "score": 0.9},
        {"text": "B", "score": 0.8},
        {"text": "C", "score": 0.7},
    ]
    
    output = reranker.rerank("test query", results, limit=5)
    
    # Expected: sorted by score descending (higher is better) -> B (-4.4), C (-5.9), A (-6.9)
    # All should have rerank_applied=True
    assert len(output) == 3
    assert output[0]["text"] == "B"
    assert output[1]["text"] == "C"
    assert output[2]["text"] == "A"
    assert output[0]["cross_encoder_score"] == -4.4
    assert output[1]["cross_encoder_score"] == -5.9
    assert output[2]["cross_encoder_score"] == -6.9
    assert output[0]["rerank_applied"] is True
    assert output[1]["rerank_applied"] is True
    assert output[2]["rerank_applied"] is True


def test_case_c(tmp_path, monkeypatch):
    """New test: min_confidence=0.0, scores [-1.0, 2.1, -3.0] -> order B,A,C, 
    rerank_applied True"""
    
    reranker = CrossEncoderReranker(onnx_path=str(_model_dir(tmp_path)), min_confidence=0.0)
    
    # Patch available property
    monkeypatch.setattr(CrossEncoderReranker, "available", property(lambda self: True))
    # Patch _load to no-op
    monkeypatch.setattr(reranker, "_load", lambda: None)
    # Patch _score to return fixed scores
    monkeypatch.setattr(reranker, "_score", lambda query, results: [-1.0, 2.1, -3.0])
    
    results = [
        {"text": "A", "score": 0.9},
        {"text": "B", "score": 0.8},
        {"text": "C", "score": 0.7},
    ]
    
    output = reranker.rerank("test query", results, limit=5)
    
    # Expected: max score (2.1) >= min_confidence (0.0), so sort descending -> B (2.1), A (-1.0), C (-3.0)
    # All should have rerank_applied=True
    assert len(output) == 3
    assert output[0]["text"] == "B"
    assert output[1]["text"] == "A"
    assert output[2]["text"] == "C"
    assert output[0]["cross_encoder_score"] == 2.1
    assert output[1]["cross_encoder_score"] == -1.0
    assert output[2]["cross_encoder_score"] == -3.0
    assert output[0]["rerank_applied"] is True
    assert output[1]["rerank_applied"] is True
    assert output[2]["rerank_applied"] is True


def test_case_d(tmp_path, monkeypatch):
    """New test: limit=2 truncates to 2 hits in BOTH the guardrail path (min_confidence=0.0 with all-negative scores)
    and the sorted path"""
    
    # Test guardrail path (all scores negative)
    guardrail_dir = tmp_path / "guardrail"
    guardrail_dir.mkdir()
    onnx_dir = guardrail_dir / "cross-encoder-onnx"
    onnx_dir.mkdir()
    (onnx_dir / "model.onnx").write_bytes(b"placeholder")
    
    reranker_guardrail = CrossEncoderReranker(onnx_path=str(onnx_dir), min_confidence=0.0)
    
    # Apply patches for guardrail test using monkeypatch pattern
    monkeypatch.setattr(CrossEncoderReranker, "available", property(lambda self: True))
    monkeypatch.setattr(reranker_guardrail, "_load", lambda: None)
    monkeypatch.setattr(reranker_guardrail, "_score", lambda query, results: [-1.0, -2.0, -3.0, -4.0])
    
    guardrail_results = [
        {"text": "A", "score": 0.9},
        {"text": "B", "score": 0.8},
        {"text": "C", "score": 0.7},
        {"text": "D", "score": 0.6},
    ]
    
    guardrail_output = reranker_guardrail.rerank("test query", guardrail_results, limit=2)
    
    # Expected: limit=2, so only A and B in input order with rerank_applied=False
    assert len(guardrail_output) == 2
    assert guardrail_output[0]["text"] == "A"
    assert guardrail_output[1]["text"] == "B"
    assert guardrail_output[0]["rerank_applied"] is False
    assert guardrail_output[1]["rerank_applied"] is False
    
    # Test sorted path (mixed scores, above confidence) - separate test function
    sorted_dir = tmp_path / "sorted"
    sorted_dir.mkdir()
    sorted_onnx_dir = sorted_dir / "cross-encoder-onnx"
    sorted_onnx_dir.mkdir()
    (sorted_onnx_dir / "model.onnx").write_bytes(b"placeholder")
    
    reranker_sorted = CrossEncoderReranker(onnx_path=str(sorted_onnx_dir), min_confidence=0.0)
    
    # Apply patches for sorted test using monkeypatch pattern
    monkeypatch.setattr(CrossEncoderReranker, "available", property(lambda self: True))
    monkeypatch.setattr(reranker_sorted, "_load", lambda: None)
    monkeypatch.setattr(reranker_sorted, "_score", lambda query, results: [1.0, -1.0, 2.0, -2.0])
    
    sorted_results = [
        {"text": "A", "score": 0.9},
        {"text": "B", "score": 0.8},
        {"text": "C", "score": 0.7},
        {"text": "D", "score": 0.6},
    ]
    
    sorted_output = reranker_sorted.rerank("test query", sorted_results, limit=2)
    
    # Expected: sorted by score descending (2.0, 1.0), then C, A
    assert len(sorted_output) == 2
    assert sorted_output[0]["text"] == "C"
    assert sorted_output[1]["text"] == "A"
    assert sorted_output[0]["rerank_applied"] is True
    assert sorted_output[1]["rerank_applied"] is True
