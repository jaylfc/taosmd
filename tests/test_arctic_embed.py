"""ONNX embed prefix and pooling selection for asymmetric models (arctic-embed).

The actual embedding needs the model and runs on the bench host; these
tests pin the prefix and pooling-mode logic that, if wrong, silently
degrades a model and invalidates a benchmark comparison.
"""

from taosmd.vector_memory import (
    _ARCTIC_QUERY_PREFIX,
    _onnx_apply_prefix,
    _onnx_pooling_mode,
)


def test_arctic_query_gets_prefix():
    out = _onnx_apply_prefix("models/snowflake-arctic-embed-s", "who is alice", "search_query")
    assert out == f"{_ARCTIC_QUERY_PREFIX}who is alice"


def test_arctic_document_gets_no_prefix():
    out = _onnx_apply_prefix("models/snowflake-arctic-embed-s", "alice adopted a dog", "search_document")
    assert out == "alice adopted a dog"


def test_nomic_prefixes_both_query_and_document():
    q = _onnx_apply_prefix("models/nomic-embed-text-v1.5", "x", "search_query")
    d = _onnx_apply_prefix("models/nomic-embed-text-v1.5", "x", "search_document")
    assert q == "search_query: x"
    assert d == "search_document: x"


def test_minilm_gets_no_prefix():
    for task in ("search_query", "search_document"):
        assert _onnx_apply_prefix("models/cross-encoder-onnx-minilm", "x", task) == "x"


def test_prefix_handles_none_path():
    assert _onnx_apply_prefix(None, "x", "search_query") == "x"


def test_arctic_pools_cls():
    assert _onnx_pooling_mode("models/snowflake-arctic-embed-xs") == "cls"


def test_default_pools_mean():
    assert _onnx_pooling_mode("models/all-MiniLM-L6-v2-onnx") == "mean"
    assert _onnx_pooling_mode(None) == "mean"
