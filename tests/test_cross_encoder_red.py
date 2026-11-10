import sys
import pytest
from unittest.mock import patch, MagicMock

from taosmd.cross_encoder import CrossEncoderReranker


def _model_dir(tmp_path):
    onnx_dir = tmp_path / "cross-encoder-onnx"
    onnx_dir.mkdir()
    (onnx_dir / "model.onnx").write_bytes(b"placeholder")
    return onnx_dir


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
