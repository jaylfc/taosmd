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
