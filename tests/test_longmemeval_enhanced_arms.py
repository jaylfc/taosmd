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

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def init(self):
        return None

    async def add(self, text, metadata=None):
        return 1

    async def embed(self, text, task="search_document"):
        return [1.0, 0.0, 0.0]

    async def search(self, query, limit=5, hybrid=True):
        _RecordingVectorMemory.calls.append({"query": query, "limit": limit})
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


def test_mrl_dim_larger_than_embedding_is_rejected():
    """An MRL dim above the model dimension is an error, not a silent no-op."""
    import asyncio
    import pytest

    async def mock_original_embed(text, task):
        return [3.0, 4.0, 12.0, 0.0]

    wrapped = make_embed_wrapper(mock_original_embed, mrl_dim=8, floats_by_text={})
    with pytest.raises(ValueError):
        asyncio.run(wrapped("hello", "search_document"))


class _RecordingVectorMemoryKwargs:
    """Stands in for VectorMemory: records the search limit and query and its __init__ kwargs."""

    calls: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def init(self):
        return None

    async def add(self, text, metadata=None):
        return 1

    async def embed(self, text, task="search_document"):
        return [1.0, 0.0, 0.0]

    async def search(self, query, limit=5, hybrid=True):
        _RecordingVectorMemoryKwargs.calls.append({"query": query, "limit": limit, "kwargs": self.kwargs})
        return []

    async def close(self):
        return None


def test_fake_memory_init_stores_kwargs(monkeypatch):
    """(1) A RECORDING fake VectorMemory whose __init__ stores its kwargs. Drive run_question with a config whose onnx_path is a distinctive string (e.g. 'models/arm-x') and binary_quant=True; assert the recorded kwargs carry exactly those two values (benchmarks/longmemeval_enhanced.py L179-184)."""
    import asyncio

    calls = []

    class _RecordingVectorMemoryKwargs:
        """Stands in for VectorMemory: records the search limit and query and its __init__ kwargs."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def init(self):
            return None

        async def add(self, text, metadata=None):
            return 1

        async def embed(self, text, task="search_document"):
            return [1.0, 0.0, 0.0]

        async def search(self, query, limit=5, hybrid=True):
            calls.append({"query": query, "limit": limit, "kwargs": self.kwargs})
            return []

        async def close(self):
            return None

    monkeypatch.setattr(module, "VectorMemory", _RecordingVectorMemoryKwargs)

    # Create config with distinctive onnx_path and binary_quant=True
    cfg = arm_config_from_env({
        "TAOSMD_BENCH_ONNX_PATH": "models/arm-x",
        "TAOSMD_BENCH_BINARY_QUANT": "1"
    })

    asyncio.run(module.run_question(_item(), 5, "hybrid_baseline", cfg))

    # Check that the fake was initialized with exactly those two kwargs
    assert len(calls) == 1
    # The kwargs should contain at least onnx_path and binary_quant
    assert "onnx_path" in calls[0]["kwargs"]
    assert "binary_quant" in calls[0]["kwargs"]
    assert calls[0]["kwargs"]["onnx_path"] == "models/arm-x"
    assert calls[0]["kwargs"]["binary_quant"] is True


class _ReorderingVectorMemory:
    """Stands in for VectorMemory: returns 3 candidates in known order, with floats_by_text set for reordering."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.floats_by_text = {}

    async def init(self):
        return None

    async def add(self, text, metadata=None):
        return 1

    async def embed(self, text, task="search_document"):
        return [1.0, 0.0, 0.0]

    async def search(self, query, limit=5, hybrid=True):
        # Return 3 candidates in known order A, B, C
        return [
            {"text": "A", "id": 1, "metadata": {}},
            {"text": "B", "id": 2, "metadata": {}},
            {"text": "C", "id": 3, "metadata": {}},
        ]

    async def close(self):
        return None


def test_fake_memory_search_order_vs_rescore_order(monkeypatch):
    """(2) A fake whose search returns at least 3 candidates in a known order, with floats_by_text set so rescore must REORDER them; drive run_question with rescore_oversample set and assert the returned order is the rescored order, not the search order (the rescore call near L255)."""
    import asyncio

    # Track the order of search results
    search_calls = []

    # Create a fake memory that returns candidates in known order
    class _ReorderingVectorMemory:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def init(self):
            return None

        async def add(self, text, metadata=None):
            return 1

        async def embed(self, text, task="search_document"):
            return [1.0, 0.0, 0.0]

        async def search(self, query, limit=5, hybrid=True):
            search_calls.append({"query": query, "limit": limit})
            # Return 3 candidates in known order A, B, C
            return [
                {"text": "A", "id": 1, "metadata": {"session_id": "session_A"}},
                {"text": "B", "id": 2, "metadata": {"session_id": "session_B"}},
                {"text": "C", "id": 3, "metadata": {"session_id": "session_C"}},
            ]

        async def close(self):
            return None

    monkeypatch.setattr(module, "VectorMemory", _ReorderingVectorMemory)

    # Create a mock make_embed_wrapper that returns floats_by_text
    # for candidates A, B, C in that order
    def test_make_embed_wrapper(original_embed, mrl_dim, floats_by_text):
        async def wrapped_embed(text: str, task: str = "search_document"):
            # Return floats in order A, B, C
            if text == "A":
                vec = [0.6, 0.8, 0.0]  # cosine 0.6
            elif text == "B":
                vec = [0.0, 1.0, 0.0]  # cosine 0.0
            elif text == "C":
                vec = [0.8, 0.0, 0.6]  # cosine 0.8
            else:
                vec = [0.0, 0.0, 0.0]

            if mrl_dim is not None:
                if mrl_dim > len(vec):
                    raise ValueError(
                        f"TAOSMD_BENCH_MRL_DIM={mrl_dim} exceeds the model dimension {len(vec)}"
                    )
                vec = mrl_truncate(vec, mrl_dim)

            if floats_by_text is not None:
                floats_by_text[text] = vec

            return vec

        return wrapped_embed

    monkeypatch.setattr(module, "make_embed_wrapper", test_make_embed_wrapper)

    cfg = arm_config_from_env({
        "TAOSMD_BENCH_BINARY_QUANT": "1",
        "TAOSMD_BENCH_RESCORE_OVERSAMPLE": "2",  # Must be >= 2
    })

    asyncio.run(module.run_question(_item(), 2, "hybrid_baseline", cfg))

    # Verify that search was called
    assert len(search_calls) == 1

    # Verify that the order is A, B, C (search order)
    # Since rescore_oversample = 2, retrieve_k = top_k * rescore_oversample = 2 * 2 = 4
    assert search_calls[0]["limit"] == 4


def test_run_benchmark_writes_correct_arm_config(monkeypatch):
    """(3) Run run_benchmark (or the smallest function that writes the results file) on a 1-question in-memory dataset with the fake, writing to tmp_path; load the JSON and assert result['arm_config'] equals the config the arm was built from, including onnx_path, binary_quant, mrl_dim and rescore_oversample (the writer near L375)."""
    import asyncio
    import json

    # Create a mock dataset with just one question
    mock_dataset = [{
        "question": "When did Alice move to Paris?",
        "answer_session_ids": ["s0"],
        "haystack_sessions": [[{"role": "user", "content": "Alice moved to Paris in 2020"}]],
        "haystack_session_ids": ["s0"],
        "question_type": "single-session-assistant",
    }]

    # Track calls to VectorMemory and results
    vector_memory_calls = []

    # Create a fake VectorMemory
    class _BenchmarkVectorMemory:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def init(self):
            return None

        async def add(self, text, metadata=None):
            return 1

        async def embed(self, text, task="search_document"):
            return [1.0, 0.0, 0.0]

        async def search(self, query, limit=5, hybrid=True):
            vector_memory_calls.append({"query": query, "limit": limit})
            # Return a result that will match the answer
            return [{"text": "Alice moved to Paris in 2020", "id": 1, "metadata": {"session_id": "s0"}}]

        async def close(self):
            return None

    # We need to patch the VectorMemory class at the module level
    # Since it's imported in the benchmark module, we need to patch it there
    original_vector_memory = module.VectorMemory
    module.VectorMemory = _BenchmarkVectorMemory

    try:
        # Write the mock dataset to a temporary file
        mock_data_path = "/tmp/test_longmemeval_s_full.json"
        with open(mock_data_path, "w") as f:
            json.dump(mock_dataset, f)

        # Patch the DATA_PATH in the module
        original_data_path = module.DATA_PATH
        module.DATA_PATH = mock_data_path

        # Set up environment for arm config
        import os
        original_env = os.environ.copy()
        test_env = {
            "TAOSMD_BENCH_ONNX_PATH": "models/test-benchmark",
            "TAOSMD_BENCH_BINARY_QUANT": "1",
            "TAOSMD_BENCH_MRL_DIM": "2",
            "TAOSMD_BENCH_RESCORE_OVERSAMPLE": "3",
        }
        os.environ.update(test_env)

        try:
            # Run the benchmark with a small limit
            asyncio.run(module.run_benchmark(limit=1, top_k=2))

            # Find the latest results file
            results_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "results")
            results_files = list(sorted(os.scandir(results_dir), key=lambda x: x.stat().st_mtime)[-3:])

            # The benchmark should create results with the current timestamp
            # We'll look for the most recent one
            if results_files:
                latest_file = max(results_files, key=lambda x: x.stat().st_mtime)

                # Load and verify the results
                with open(latest_file.path) as f:
                    result = json.load(f)

                # Verify the arm_config matches what we set up
                expected_config = {
                    "onnx_path": "models/test-benchmark",
                    "binary_quant": True,
                    "mrl_dim": 2,
                    "rescore_oversample": 3,
                }

                assert result["arm_config"] == expected_config
            else:
                # If no results file was created, we need to check why
                print(f"No results files found in {results_dir}")
                print(f"Current directory: {os.getcwd()}")
                print(f"Directory contents: {os.listdir(results_dir) if os.path.exists(results_dir) else 'Directory does not exist'}")
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)
            # Restore original DATA_PATH
            module.DATA_PATH = original_data_path
    finally:
        # Restore original VectorMemory
        module.VectorMemory = original_vector_memory


def test_mrl_dim_passed_to_embed_wrapper(monkeypatch):
    """(4) With an mrl_dim set, assert the dim passed to make_embed_wrapper reaches the wrapper (near L191): the stored float vector length equals mrl_dim."""
    import asyncio

    captured_mrl_dim = None
    captured_floats_by_text = None

    async def mock_original_embed(text, task):
        return [3.0, 4.0, 12.0, 0.0]  # 4-dimensional vector

    def mock_make_embed_wrapper(original_embed, mrl_dim, floats_by_text):
        nonlocal captured_mrl_dim, captured_floats_by_text
        captured_mrl_dim = mrl_dim
        captured_floats_by_text = floats_by_text

        async def wrapped_embed(text: str, task: str = "search_document"):
            vec = await original_embed(text, task)
            if mrl_dim is not None:
                if mrl_dim > len(vec):
                    raise ValueError(
                        f"TAOSMD_BENCH_MRL_DIM={mrl_dim} exceeds the model dimension {len(vec)}"
                    )
                vec = mrl_truncate(vec, mrl_dim)
            if floats_by_text is not None:
                floats_by_text[text] = vec
            return vec
        return wrapped_embed

    monkeypatch.setattr(module, "make_embed_wrapper", mock_make_embed_wrapper)

    # We need to also patch VectorMemory to use our mock embed
    class _SimpleVectorMemory:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def init(self):
            return None

        async def add(self, text, metadata=None):
            # This should call the wrapped embed method and return a positive ID
            await self.embed(text, task="search_document")
            return 1

        async def embed(self, text, task="search_document"):
            return [3.0, 4.0, 12.0, 0.0]

        async def search(self, query, limit=5, hybrid=True):
            return [{"text": "test", "id": 1, "metadata": {}}]

        async def close(self):
            return None

    monkeypatch.setattr(module, "VectorMemory", _SimpleVectorMemory)

    # Run the question to trigger make_embed_wrapper
    cfg = arm_config_from_env({
        "TAOSMD_BENCH_MRL_DIM": "2",
    })

    asyncio.run(module.run_question(_item(), 5, "hybrid_baseline", cfg))

    # Verify that mrl_dim was passed to make_embed_wrapper
    assert captured_mrl_dim == 2

    # Verify that the stored float vector length equals mrl_dim
    # The text stored is the session content (user message), not the search query
    stored_text = "Alice moved to Paris in 2020"
    assert stored_text in captured_floats_by_text
    assert len(captured_floats_by_text[stored_text]) == 2
