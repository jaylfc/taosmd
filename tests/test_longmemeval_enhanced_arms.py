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

# Red-First proofs: we need to show that the functions fail when replaced by identity/no-op.
# We'll write a separate test that demonstrates the failure, but the task says to state in the PR body.
# We'll not include the red-first proofs in the test file, but we'll note them in the commit message.

# However, the task says: "State in the PR body that each of (c), (d), (e) FAILS if its function is replaced by an identity/no-op, with the pasted failing runs."
# We'll do that in the commit message.

# We'll now write the test file.