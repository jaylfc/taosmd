import sys
from pathlib import Path

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


def test_cross_encoder_signature_min_confidence_positional(tmp_path):
    """CrossEncoderReranker('path', 0.5) must set min_confidence=0.5 (2nd positional),
    not provider. Provider must be keyword-only."""
    onnx_dir = tmp_path / "cross-encoder-onnx"
    onnx_dir.mkdir()
    (onnx_dir / "model.onnx").write_bytes(b"placeholder")

    # Positional: onnx_path, min_confidence
    reranker = CrossEncoderReranker(str(onnx_dir), 0.5)
    assert reranker.min_confidence == 0.5, f"min_confidence should be 0.5, got {reranker.min_confidence}"
    assert reranker._provider == "CPUExecutionProvider", f"provider should be default, got {reranker._provider}"

    # Keyword-only provider
    reranker2 = CrossEncoderReranker(str(onnx_dir), min_confidence=0.3, provider="CUDAExecutionProvider")
    assert reranker2.min_confidence == 0.3
    assert reranker2._provider == "CUDAExecutionProvider"


def test_cross_encoder_provider_config_respected(tmp_path, monkeypatch):
    """When cross_encoder_provider='CUDAExecutionProvider' and CUDA is available,
    the recorded providers passed to InferenceSession must start with CUDA."""
    import sys

    onnx_dir = tmp_path / "cross-encoder-onnx"
    onnx_dir.mkdir()
    (onnx_dir / "model.onnx").write_bytes(b"placeholder")

    recorded_providers = {}

    # Create a mock onnxruntime module
    class MockInferenceSession:
        def __init__(self, model_path, providers=None, **kwargs):
            recorded_providers["providers"] = providers
        def get_inputs(self):
            return []
        def run(self, *args, **kwargs):
            import numpy as np
            return [np.array([[0.5]])]

    mock_ort = type(sys)("onnxruntime")
    mock_ort.InferenceSession = MockInferenceSession
    monkeypatch.setitem(sys.modules, "onnxruntime", mock_ort)

    # Patch transformers AutoTokenizer
    import transformers
    def mock_tokenizer(*args, **kwargs):
        class MockTokenizer:
            def __call__(self, *args, **kwargs):
                import numpy as np
                return {"input_ids": np.array([[1, 2, 3]]), "attention_mask": np.array([[1, 1, 1]])}
        return MockTokenizer()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", mock_tokenizer)

    # Patch available and _find_model_file
    monkeypatch.setattr(CrossEncoderReranker, "available", property(lambda self: True))
    monkeypatch.setattr(CrossEncoderReranker, "_find_model_file", lambda self: str(onnx_dir / "model.onnx"))

    # Create reranker with CUDA provider config
    reranker = CrossEncoderReranker(onnx_path=str(onnx_dir), provider="CUDAExecutionProvider")
    reranker._load()

    assert "providers" in recorded_providers, "providers not recorded"
    providers = recorded_providers["providers"]
    assert providers is not None, "providers is None"
    assert providers[0] == "CUDAExecutionProvider", f"expected CUDAExecutionProvider first, got {providers}"


# --- output shape validation (RED tests for the bug) -----------------------------------

def _make_reranker_with_mock_session(tmp_path, output_shape, monkeypatch):
    """Helper to create a reranker with a mocked session returning output_shape."""
    import numpy as np
    from unittest.mock import MagicMock

    onnx_dir = tmp_path / "cross-encoder-onnx"
    onnx_dir.mkdir()
    (onnx_dir / "model.onnx").write_bytes(b"placeholder")

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [
        MagicMock(name="input_ids"),
        MagicMock(name="attention_mask"),
        MagicMock(name="token_type_ids"),
    ]
    mock_session.run.return_value = [np.random.randn(*output_shape).astype(np.float32)]

    def mock_tokenizer(*args, **kwargs):
        class MockTokenizer:
            def __call__(self, *args, **kwargs):
                import numpy as np
                return {
                    "input_ids": np.array([[1, 2, 3, 4, 5]]),
                    "attention_mask": np.array([[1, 1, 1, 1, 1]]),
                    "token_type_ids": np.array([[0, 0, 0, 0, 0]]),
                }
        return MockTokenizer()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", mock_tokenizer)
    monkeypatch.setattr(CrossEncoderReranker, "available", property(lambda self: True))
    monkeypatch.setattr(CrossEncoderReranker, "_find_model_file", lambda self: str(onnx_dir / "model.onnx"))

    reranker = CrossEncoderReranker(onnx_path=str(onnx_dir))
    reranker._session = mock_session
    reranker._tokenizer = mock_tokenizer()
    return reranker


def test_cross_encoder_rejects_3d_output(tmp_path, monkeypatch):
    """3D output (batch, seq, hidden) like ettin last_hidden_state must raise,
    not silently return a hidden-state float as score."""
    reranker = _make_reranker_with_mock_session(tmp_path, (1, 10, 512), monkeypatch)
    results = [{"text": "test document"}]
    with pytest.raises(ValueError, match=r"unexpected output shape|expected logits|3-D|last_hidden_state"):
        reranker._score("test query", results)


def test_cross_encoder_rejects_2d_pooled_output(tmp_path, monkeypatch):
    """2D output (batch, hidden) like pooled hidden states without scoring head
    must raise, not silently return a hidden-state float as score."""
    reranker = _make_reranker_with_mock_session(tmp_path, (1, 512), monkeypatch)
    results = [{"text": "test document"}]
    with pytest.raises(ValueError, match=r"unexpected output shape|expected logits|2-D|pooled"):
        reranker._score("test query", results)


def test_cross_encoder_accepts_logits_1d(tmp_path, monkeypatch):
    """Logits shaped (batch,) must be accepted and scored correctly."""
    reranker = _make_reranker_with_mock_session(tmp_path, (3,), monkeypatch)
    results = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    scores = reranker._score("test query", results)
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)


def test_cross_encoder_accepts_logits_2d_batch_1(tmp_path, monkeypatch):
    """Logits shaped (batch, 1) must be accepted and scored correctly."""
    reranker = _make_reranker_with_mock_session(tmp_path, (3, 1), monkeypatch)
    results = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    scores = reranker._score("test query", results)
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)


def test_cross_encoder_accepts_logits_2d_batch_2_uses_positive_class(tmp_path, monkeypatch):
    """Logits shaped (batch, 2) must use the positive-class logit (index 1) as score."""
    import numpy as np
    from unittest.mock import MagicMock

    onnx_dir = tmp_path / "cross-encoder-onnx"
    onnx_dir.mkdir()
    (onnx_dir / "model.onnx").write_bytes(b"placeholder")

    # Fixed logits per call: each call returns (1, 2) for one sample
    call_logits = [
        np.array([[-2.0, 1.0]], dtype=np.float32),   # sample 0: positive=1.0
        np.array([[-1.0, 3.0]], dtype=np.float32),   # sample 1: positive=3.0
        np.array([[-3.0, 0.5]], dtype=np.float32),   # sample 2: positive=0.5
    ]

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [
        MagicMock(name="input_ids"),
        MagicMock(name="attention_mask"),
        MagicMock(name="token_type_ids"),
    ]
    mock_session.run.side_effect = [[logits] for logits in call_logits]

    def mock_tokenizer(*args, **kwargs):
        class MockTokenizer:
            def __call__(self, *args, **kwargs):
                import numpy as np
                return {
                    "input_ids": np.array([[1, 2, 3, 4, 5]]),
                    "attention_mask": np.array([[1, 1, 1, 1, 1]]),
                    "token_type_ids": np.array([[0, 0, 0, 0, 0]]),
                }
        return MockTokenizer()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", mock_tokenizer)
    monkeypatch.setattr(CrossEncoderReranker, "available", property(lambda self: True))
    monkeypatch.setattr(CrossEncoderReranker, "_find_model_file", lambda self: str(onnx_dir / "model.onnx"))

    reranker = CrossEncoderReranker(onnx_path=str(onnx_dir))
    reranker._session = mock_session
    reranker._tokenizer = mock_tokenizer()

    results = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    scores = reranker._score("test query", results)
    # Should use positive class (index 1): [1.0, 3.0, 0.5]
    assert scores == [1.0, 3.0, 0.5]


# --- optional integration test for ettin-reranker-68m-v1 ---------------------------
# This test requires the ettin-reranker-68m-v1 model to be downloaded locally.
# It is skipped by default and runs only when TAOSMD_TEST_ETTIN=1 is set and
# the model files exist. It validates that taOSmd's ettin scores match
# sentence-transformers CrossEncoder scores within 1e-3 and ranking order matches.

@pytest.mark.skipif(
    not __import__("os").environ.get("TAOSMD_TEST_ETTIN"),
    reason="Set TAOSMD_TEST_ETTIN=1 to run ettin integration test (requires model download)"
)
def test_ettin_reranker_matches_sentence_transformers(tmp_path, monkeypatch):
    """ettin-reranker-68m-v1 scores through taOSmd must match sentence-transformers
    CrossEncoder scores for the same model on 5 fixture pairs within 1e-3,
    and ranking order must match.
    """
    import os
    import numpy as np

    # Check if model exists in the default location or HF cache
    model_path = os.environ.get("TAOSMD_ETTIN_PATH")
    if not model_path:
        from huggingface_hub import try_to_load_from_cache
        # Try to find in HF cache
        model_path = try_to_load_from_cache(
            repo_id="cross-encoder/ettin-reranker-68m-v1",
            filename="config.json",
        )
        if model_path:
            model_path = str(Path(model_path).parent)
    
    if not model_path or not Path(model_path).exists():
        pytest.skip("ettin-reranker-68m-v1 model not found locally")

    # Check for required files
    if not (Path(model_path) / "config.json").exists():
        pytest.skip("ettin-reranker-68m-v1 config.json not found")
    if not list(Path(model_path).glob("*.safetensors")):
        pytest.skip("ettin-reranker-68m-v1 safetensors not found")

    # Fixture pairs for testing
    fixture_pairs = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is the capital of France?", "Berlin is the capital of Germany."),
        ("How does photosynthesis work?", "Plants convert sunlight into energy through photosynthesis."),
        ("How does photosynthesis work?", "Animals eat plants for energy."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    ]

    # Get scores from taOSmd
    reranker = CrossEncoderReranker(onnx_path=model_path)
    if not reranker.available:
        pytest.skip("taOSmd reranker not available for this model path")
    
    taosmd_scores = []
    for query, text in fixture_pairs:
        scores = reranker._score(query, [{"text": text}])
        taosmd_scores.append(scores[0] if scores else 0.0)

    # Get scores from sentence-transformers CrossEncoder
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        pytest.skip("sentence-transformers not installed")

    st_model = CrossEncoder(model_path)
    st_scores = st_model.predict(fixture_pairs).tolist()

    # Compare scores side by side
    print("\nScore comparison (taOSmd vs sentence-transformers):")
    for i, ((query, text), ts, st) in enumerate(zip(fixture_pairs, taosmd_scores, st_scores)):
        print(f"  Pair {i}: taOSmd={ts:.6f}, ST={st:.6f}, diff={abs(ts-st):.6f}")

    # Assert scores match within 1e-3
    for ts, st in zip(taosmd_scores, st_scores):
        assert abs(ts - st) < 1e-3, f"Score mismatch: taOSmd={ts:.6f}, ST={st:.6f}"

    # Assert ranking order matches
    taosmd_ranking = np.argsort(taosmd_scores)[::-1]  # descending
    st_ranking = np.argsort(st_scores)[::-1]
    assert np.array_equal(taosmd_ranking, st_ranking), (
        f"Ranking mismatch: taOSmd={taosmd_ranking.tolist()}, ST={st_ranking.tolist()}"
    )
