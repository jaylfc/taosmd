import importlib.util
import os
import sys

# Import the benchmark script from the benchmarks directory
spec = importlib.util.spec_from_file_location(
    "longmemeval_enhanced",
    os.path.join(os.path.dirname(__file__), "..", "benchmarks", "longmemeval_enhanced.py")
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

from longmemeval_enhanced import (
    arm_config_from_env,
    mrl_truncate,
    rescore,
    make_embed_wrapper,
)

def test_arm_config_from_env_defaults():
    """(a) arm_config_from_env({}) equals the default dict with onnx_path == the module's ONNX_PATH and everything else off."""
    # The module's ONNX_PATH is defined in the script
    expected_onnx_path = module.ONNX_PATH
    result = arm_config_from_env({})
    assert result["onnx_path"] == expected_onnx_path
    assert result["binary_quant"] is False
    
    assert result["mrl_dim"] is None
    assert result["rescore_oversample"] is None
    assert result["fusion"] == "boost"
    assert result["late_interaction"] is False
def test_arm_config_from_env_parsing():
    """(b) env parsing: dim '512' -> 512; dim '0', '-1', 'x' -> ValueError; oversample without binary -> ValueError."""
    # Test dim parsing
    env = {"TAOSMD_BENCH_MRL_DIM": "512"}
    result = arm_config_from_env(env)
    assert result["mrl_dim"] == 512

    # Test invalid dim values
    for val in ["0", "-1", "x"]:
        env = {"TAOSMD_BENCH_MRL_DIM": val}
        try:
            arm_config_from_env(env)
            assert False, f"Expected ValueError for MRL_DIM={val}"
        except ValueError:
            pass

    # Test oversample without binary quant
    env = {"TAOSMD_BENCH_RESCORE_OVERSAMPLE": "2"}
    try:
        arm_config_from_env(env)
        assert False, "Expected ValueError for rescore oversample without binary quant"
    except ValueError:
        pass

    # Test valid oversample with binary quant
    env = {
        "TAOSMD_BENCH_BINARY_QUANT": "1",
        "TAOSMD_BENCH_RESCORE_OVERSAMPLE": "2"
    }
    result = arm_config_from_env(env)
    assert result["binary_quant"] is True
    assert result["rescore_oversample"] == 2

def test_mrl_truncate():
    """(c) mrl_truncate([3,4,12], 2) == [0.6, 0.8]; zero vector stays zero."""
    # Test normal vector
    vec = [3.0, 4.0, 12.0]
    result = mrl_truncate(vec, 2)
    assert len(result) == 2
    assert abs(result[0] - 0.6) < 1e-6
    assert abs(result[1] - 0.8) < 1e-6

    # Test zero vector
    zero_vec = [0.0, 0.0, 0.0]
    result = mrl_truncate(zero_vec, 2)
    assert result == [0.0, 0.0]

def test_rescore():
    """(d) rescore: three candidates whose binary order is A,B,C but whose float cosines to the query are C > A > B, k=2 -> [C, A]."""
    # We'll create mock candidates with text keys and pre-recorded floats in floats_by_text
    query_vec = [1.0, 0.0, 0.0]  # unit vector along x-axis
    # We want cosines: C > A > B
    # Let A have cosine 0.6, B 0.0, C 0.8
    # A: [0.6, 0.8, 0.0] -> dot with [1,0,0] = 0.6, norm = sqrt(0.6^2+0.8^2)=1.0 -> cosine=0.6
    # B: [0.0, 1.0, 0.0] -> cosine = 0.0
    # C: [0.8, 0.0, 0.6] -> dot with [1,0,0] = 0.8, norm = sqrt(0.8^2+0.6^2)=1.0 -> cosine=0.8
    candidates = [
        {"text": "A", "id": 1},
        {"text": "B", "id": 2},
        {"text": "C", "id": 3},
    ]
    floats_by_text = {
        "A": [0.6, 0.8, 0.0],   # cosine 0.6
        "B": [0.0, 1.0, 0.0],   # cosine 0.0
        "C": [0.8, 0.0, 0.6],   # cosine 0.8
    }

    result = rescore(query_vec, candidates, floats_by_text, 2)
    # Expect [C, A]
    assert len(result) == 2
    assert result[0]["text"] == "C"
    assert result[1]["text"] == "A"

def test_embed_wrapper():
    """(e) the embed wrapper: with a fake async embed returning [3,4,12,0], mrl_dim=2 makes the wrapped embed return [0.6, 0.8] and records it in the floats dict."""
    # We'll create a mock original embed function that returns [3,4,12,0]
    async def mock_original_embed(text, task):
        return [3.0, 4.0, 12.0, 0.0]

    floats_by_text = {}
    wrapped = make_embed_wrapper(mock_original_embed, mrl_dim=2, floats_by_text=floats_by_text)

    # Call the wrapped embed
    import asyncio
    vec = asyncio.run(wrapped("hello", "search_document"))
    # Should be [0.6, 0.8] (normalized [3,4])
    assert len(vec) == 2
    assert abs(vec[0] - 0.6) < 1e-6
    assert abs(vec[1] - 0.8) < 1e-6
    # Should have recorded the vector in floats_by_text
    assert "hello" in floats_by_text
    recorded = floats_by_text["hello"]
    assert len(recorded) == 2
    assert abs(recorded[0] - 0.6) < 1e-6
    assert abs(recorded[1] - 0.8) < 1e-6


class _RecordingVectorMemory:
    """Stands in for VectorMemory: records the search limit and query."""

    calls: list[dict] = []
    init_kwargs: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _RecordingVectorMemory.init_kwargs.append(kwargs)

    async def init(self):
        return None

    async def add(self, text, metadata=None):
        return 1

    async def embed(self, text, task="search_document"):
        return [1.0, 0.0, 0.0]

    async def search(self, query, limit=5, hybrid=True, fusion="boost"):
        _RecordingVectorMemory.calls.append({"query": query, "limit": limit, "fusion": fusion})
        return []

    async def close(self):
        return None


def _item():
    return {
        "question": "When did Alice move to Paris?",
        "answer_session_ids": ["s0"],
        "haystack_sessions": [[{"role": "user", "content": "Alice moved to Paris in 2020"}]],
        "haystack_session_ids": ["s0"],
    }


def _run_named_arm(monkeypatch, config_name, top_k=5):
    import asyncio
    monkeypatch.setattr(module, "VectorMemory", _RecordingVectorMemory)
    _RecordingVectorMemory.calls = []
    _RecordingVectorMemory.init_kwargs = []
    cfg = arm_config_from_env({})
    asyncio.run(module.run_question(_item(), top_k, config_name, cfg))
    assert len(_RecordingVectorMemory.calls) == 1
    return _RecordingVectorMemory.calls[0]


def test_named_arm_hybrid_baseline_is_untouched(monkeypatch):
    """Config A: the search uses the raw question at limit top_k."""
    call = _run_named_arm(monkeypatch, "hybrid_baseline")
    assert call["limit"] == 5
    assert call["query"] == _item()["question"]


def test_named_arm_wider_retrieval_doubles_limit(monkeypatch):
    """Config E: retrieve_k is top_k * 2, independent of the env arm dict."""
    call = _run_named_arm(monkeypatch, "wider_retrieval")
    assert call["limit"] == 10


def test_named_arm_query_expand_changes_query(monkeypatch):
    """Config B: the search query carries expansion terms, not the bare question."""
    call = _run_named_arm(monkeypatch, "query_expand")
    assert call["query"] != _item()["question"]
    assert call["query"].startswith(_item()["question"])


def test_named_arm_temporal_boost_reranks(monkeypatch):
    """Config C: temporal_rerank runs for a temporal question."""
    import asyncio
    monkeypatch.setattr(module, "VectorMemory", _RecordingVectorMemory)
    seen = []

    def fake_rerank(results, question, boost_factor=0.25):
        seen.append(question)
        return results

    monkeypatch.setattr(module, "temporal_rerank", fake_rerank)
    cfg = arm_config_from_env({})
    asyncio.run(module.run_question(_item(), 5, "temporal_boost", cfg))
    assert seen == [_item()["question"]]


def test_rescore_oversample_multiplies_limit(monkeypatch):
    """The env arm multiplies retrieve_k on top of the named arm's value."""
    import asyncio
    monkeypatch.setattr(module, "VectorMemory", _RecordingVectorMemory)
    _RecordingVectorMemory.calls = []
    cfg = arm_config_from_env({"TAOSMD_BENCH_BINARY_QUANT": "1", "TAOSMD_BENCH_RESCORE_OVERSAMPLE": "3"})
    asyncio.run(module.run_question(_item(), 5, "wider_retrieval", cfg))
    assert _RecordingVectorMemory.calls[0]["limit"] == 30


def test_arm_config_from_env_fusion():
    """fusion: unset -> 'boost'; TAOSMD_BENCH_FUSION=bm25_rrf -> 'bm25_rrf'; bogus -> ValueError."""
    result = arm_config_from_env({})
    assert result["fusion"] == "boost"

    result = arm_config_from_env({"TAOSMD_BENCH_FUSION": "bm25_rrf"})
    assert result["fusion"] == "bm25_rrf"

    for val in ["bogus", "unknown", "cosine"]:
        try:
            arm_config_from_env({"TAOSMD_BENCH_FUSION": val})
            assert False, f"Expected ValueError for FUSION={val}"
        except ValueError:
            pass


def test_arm_config_from_env_late_interaction():
    """late_interaction: unset -> False; '1' -> True; '0' -> False."""
    result = arm_config_from_env({})
    assert result["late_interaction"] is False

    result = arm_config_from_env({"TAOSMD_BENCH_LATE_INTERACTION": "1"})
    assert result["late_interaction"] is True

    result = arm_config_from_env({"TAOSMD_BENCH_LATE_INTERACTION": "0"})
    assert result["late_interaction"] is False


def test_named_arm_passes_fusion_to_search(monkeypatch):
    """The fusion arm value is forwarded to vmem.search via the env arm dict."""
    import asyncio
    monkeypatch.setattr(module, "VectorMemory", _RecordingVectorMemory)
    _RecordingVectorMemory.calls = []
    cfg = arm_config_from_env({"TAOSMD_BENCH_FUSION": "rrf"})
    asyncio.run(module.run_question(_item(), 5, "hybrid_baseline", cfg))
    assert _RecordingVectorMemory.calls[0]["fusion"] == "rrf"


def test_mrl_dim_larger_than_embedding_is_rejected():
    """An MRL dim above the model dimension is an error, not a silent no-op."""
    import asyncio
    import pytest

    async def mock_original_embed(text, task):
        return [3.0, 4.0, 12.0, 0.0]

    wrapped = make_embed_wrapper(mock_original_embed, mrl_dim=8, floats_by_text={})
    with pytest.raises(ValueError):
        asyncio.run(wrapped("hello", "search_document"))


def test_arm_config_from_env_reranker_default():
    """Default env -> reranker off."""
    result = arm_config_from_env({})
    assert result["reranker"] == "off"


def test_arm_config_from_env_reranker_bge():
    """TAOSMD_BENCH_RERANKER=bge-v2-m3 -> bge-v2-m3."""
    result = arm_config_from_env({"TAOSMD_BENCH_RERANKER": "bge-v2-m3"})
    assert result["reranker"] == "bge-v2-m3"


def test_arm_config_from_env_reranker_bogus():
    """TAOSMD_BENCH_RERANKER=bogus raises ValueError."""
    try:
        arm_config_from_env({"TAOSMD_BENCH_RERANKER": "bogus"})
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_late_interaction_reaches_vector_memory_ctor(monkeypatch):
    """late_interaction from config is passed to VectorMemory constructor."""
    import asyncio
    monkeypatch.setattr(module, "VectorMemory", _RecordingVectorMemory)
    _RecordingVectorMemory.calls = []
    _RecordingVectorMemory.init_kwargs = []

    # Config with late_interaction=True
    cfg_true = arm_config_from_env({"TAOSMD_BENCH_LATE_INTERACTION": "1"})
    assert cfg_true["late_interaction"] is True
    asyncio.run(module.run_question(_item(), 5, "hybrid_baseline", cfg_true))
    assert _RecordingVectorMemory.init_kwargs[-1]["late_interaction"] is True

    # Config with late_interaction=False
    _RecordingVectorMemory.calls = []
    _RecordingVectorMemory.init_kwargs = []
    cfg_false = arm_config_from_env({})
    assert cfg_false["late_interaction"] is False
    asyncio.run(module.run_question(_item(), 5, "hybrid_baseline", cfg_false))
    assert _RecordingVectorMemory.init_kwargs[-1]["late_interaction"] is False


def test_late_interaction_only_exact_one():
    """Only the string '1' yields late_interaction=True; all other values yield False."""
    for v in ["true", "yes", "2", "0", ""]:
        result = arm_config_from_env({"TAOSMD_BENCH_LATE_INTERACTION": v})
        assert result["late_interaction"] is False, f"Expected False for value={v!r}"

    result = arm_config_from_env({"TAOSMD_BENCH_LATE_INTERACTION": "1"})
    assert result["late_interaction"] is True


def test_reranker_unavailable_refuses(monkeypatch):
    """Fake reranker with available False makes run_benchmark refuse."""
    import asyncio

    class Dummy:
        available = False

    # Monkeypatch _load_reranker
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
    import locomo_runner

    monkeypatch.setattr(locomo_runner, "_load_reranker", lambda choice: Dummy())
    monkeypatch.setenv("TAOSMD_BENCH_RERANKER", "ms-marco")

    try:
        asyncio.run(module.run_benchmark(limit=1, top_k=5))
        assert False, "Expected SystemExit"
    except SystemExit as e:
        assert e.code != 0
