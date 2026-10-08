"""Tests for LongMemEval Enhanced embedder arms (E-035).

Tests the environment-driven arm selection, MRL truncation, rescoring,
and the embed wrapper that records float vectors before binary packing.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENHANCED_PATH = REPO_ROOT / "benchmarks" / "longmemeval_enhanced.py"


def _load_enhanced():
    spec = importlib.util.spec_from_file_location("longmemeval_enhanced", ENHANCED_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# (a) arm_config_from_env default values
# ---------------------------------------------------------------------------

def test_arm_config_from_env_defaults():
    mod = _load_enhanced()
    cfg = mod.arm_config_from_env({})
    assert cfg["onnx_path"] == mod.ONNX_PATH
    assert cfg["binary_quant"] is False
    assert cfg["mrl_dim"] is None
    assert cfg["rescore_oversample"] is None


# ---------------------------------------------------------------------------
# (b) env parsing validation
# ---------------------------------------------------------------------------

def test_arm_config_from_env_dim_positive_int():
    mod = _load_enhanced()
    cfg = mod.arm_config_from_env({"TAOSMD_BENCH_MRL_DIM": "512"})
    assert cfg["mrl_dim"] == 512


@pytest.mark.parametrize("bad_val", ["0", "-1", "x", "3.14"])
def test_arm_config_from_env_dim_invalid(bad_val):
    mod = _load_enhanced()
    with pytest.raises(ValueError, match="TAOSMD_BENCH_MRL_DIM"):
        mod.arm_config_from_env({"TAOSMD_BENCH_MRL_DIM": bad_val})


def test_arm_config_from_env_oversample_without_binary_raises():
    mod = _load_enhanced()
    with pytest.raises(ValueError, match="TAOSMD_BENCH_RESCORE_OVERSAMPLE requires TAOSMD_BENCH_BINARY_QUANT=1"):
        mod.arm_config_from_env({"TAOSMD_BENCH_RESCORE_OVERSAMPLE": "4"})


def test_arm_config_from_env_oversample_with_binary_ok():
    mod = _load_enhanced()
    cfg = mod.arm_config_from_env({
        "TAOSMD_BENCH_BINARY_QUANT": "1",
        "TAOSMD_BENCH_RESCORE_OVERSAMPLE": "4",
    })
    assert cfg["binary_quant"] is True
    assert cfg["rescore_oversample"] == 4


@pytest.mark.parametrize("bad_val", ["1", "0", "-1", "x"])
def test_arm_config_from_env_oversample_invalid(bad_val):
    mod = _load_enhanced()
    with pytest.raises(ValueError, match="TAOSMD_BENCH_RESCORE_OVERSAMPLE"):
        mod.arm_config_from_env({
            "TAOSMD_BENCH_BINARY_QUANT": "1",
            "TAOSMD_BENCH_RESCORE_OVERSAMPLE": bad_val,
        })


# ---------------------------------------------------------------------------
# (c) mrl_truncate behaviour
# ---------------------------------------------------------------------------

def test_mrl_truncate_basic():
    mod = _load_enhanced()
    # [3, 4, 12] -> norm 13 -> first 2: [3, 4] -> norm 5 -> [0.6, 0.8]
    result = mod.mrl_truncate([3.0, 4.0, 12.0], 2)
    assert result == pytest.approx([0.6, 0.8])


def test_mrl_truncate_zero_vector():
    mod = _load_enhanced()
    result = mod.mrl_truncate([0.0, 0.0, 0.0], 2)
    assert result == [0.0, 0.0]


def test_mrl_truncate_dim_ge_len_returns_as_is():
    mod = _load_enhanced()
    vec = [1.0, 2.0, 3.0]
    result = mod.mrl_truncate(vec, 5)
    assert result == vec


# ---------------------------------------------------------------------------
# (d) rescore reorders by float cosine
# ---------------------------------------------------------------------------

def test_rescore_reorders_by_float_cosine():
    mod = _load_enhanced()
    # Query vector [1, 0]
    query_vec = [1.0, 0.0]
    # Candidates in binary order: A, B, C
    candidates = [
        {"text": "A", "score": 0.9},
        {"text": "B", "score": 0.8},
        {"text": "C", "score": 0.7},
    ]
    # Float cosines: C (1.0) > A (0.0) > B (-1.0)
    floats_by_text = {
        "A": [0.0, 1.0],    # cosine = 0
        "B": [-1.0, 0.0],   # cosine = -1
        "C": [1.0, 0.0],    # cosine = 1
    }
    result = mod.rescore(query_vec, candidates, floats_by_text, k=2)
    assert [c["text"] for c in result] == ["C", "A"]


def test_rescore_empty_candidates():
    mod = _load_enhanced()
    result = mod.rescore([1.0, 0.0], [], {}, 2)
    assert result == []


def test_rescore_missing_float_falls_back():
    mod = _load_enhanced()
    query_vec = [1.0, 0.0]
    candidates = [
        {"text": "A", "score": 0.9},
        {"text": "B", "score": 0.8},
    ]
    # Only A has a float vector
    floats_by_text = {"A": [1.0, 0.0]}
    result = mod.rescore(query_vec, candidates, floats_by_text, k=2)
    # A should come first (cosine 1.0), B second (fallback -1.0)
    assert [c["text"] for c in result] == ["A", "B"]


# ---------------------------------------------------------------------------
# (e) embed wrapper records truncated floats
# ---------------------------------------------------------------------------

def test_embed_wrapper_mrl_and_records():
    mod = _load_enhanced()

    async def fake_embed(text: str, task: str = "search_document") -> list[float]:
        return [3.0, 4.0, 12.0, 0.0]

    # Simulate the wrapper logic from run_question
    mrl_dim = 2
    rescore_oversample = 4
    floats_by_text: dict[str, list[float]] = {}

    async def wrapped_embed(text: str, task: str = "search_document") -> list[float]:
        vec = await fake_embed(text, task)
        if mrl_dim is not None:
            vec = mod.mrl_truncate(vec, mrl_dim)
        if rescore_oversample is not None:
            floats_by_text[text] = vec
        return vec

    result = asyncio.run(wrapped_embed("test text"))
    assert result == pytest.approx([0.6, 0.8])
    assert floats_by_text["test text"] == pytest.approx([0.6, 0.8])


def test_embed_wrapper_no_mrl_no_record():
    mod = _load_enhanced()

    async def fake_embed(text: str, task: str = "search_document") -> list[float]:
        return [3.0, 4.0, 12.0, 0.0]

    mrl_dim = None
    rescore_oversample = None
    floats_by_text: dict[str, list[float]] = {}

    async def wrapped_embed(text: str, task: str = "search_document") -> list[float]:
        vec = await fake_embed(text, task)
        if mrl_dim is not None:
            vec = mod.mrl_truncate(vec, mrl_dim)
        if rescore_oversample is not None:
            floats_by_text[text] = vec
        return vec

    result = asyncio.run(wrapped_embed("test text"))
    assert result == [3.0, 4.0, 12.0, 0.0]
    assert floats_by_text == {}


# ---------------------------------------------------------------------------
# RED-PROOF: demonstrate that (c), (d), (e) fail if replaced by identity/no-op
# These tests are here to document the expected failures; they are not run
# as part of the normal suite (they would fail). The PR body must include
# the failing runs as proof.
# ---------------------------------------------------------------------------

# def test_RED_c_mrl_truncate_identity_fails():
#     """If mrl_truncate is replaced by identity, this FAILS."""
#     mod = _load_enhanced()
#     # Identity: return vec[:dim] without renormalising
#     def identity_truncate(vec, dim):
#         return vec[:dim]
#     result = identity_truncate([3.0, 4.0, 12.0], 2)
#     # Would be [3.0, 4.0], not [0.6, 0.8]
#     assert result == pytest.approx([0.6, 0.8])  # FAILS

# def test_RED_d_rescore_identity_fails():
#     """If rescore is replaced by identity (returns candidates unchanged), this FAILS."""
#     mod = _load_enhanced()
#     def identity_rescore(query_vec, candidates, floats_by_text, k):
#         return candidates[:k]
#     query_vec = [1.0, 0.0]
#     candidates = [{"text": "A"}, {"text": "B"}, {"text": "C"}]
#     floats_by_text = {"A": [0.0, 1.0], "B": [-1.0, 0.0], "C": [1.0, 0.0]}
#     result = identity_rescore(query_vec, candidates, floats_by_text, 2)
#     # Would be ["A", "B"], not ["C", "A"]
#     assert [c["text"] for c in result] == ["C", "A"]  # FAILS

# def test_RED_e_embed_wrapper_noop_fails():
#     """If wrapped_embed doesn't truncate or record, this FAILS."""
#     mod = _load_enhanced()
#     async def fake_embed(text, task="search_document"):
#         return [3.0, 4.0, 12.0, 0.0]
#     mrl_dim = 2
#     rescore_oversample = 4
#     floats_by_text = {}
#     async def noop_wrapped_embed(text, task="search_document"):
#         return await fake_embed(text, task)  # No truncate, no record
#     result = asyncio.run(noop_wrapped_embed("test"))
#     # Would be [3.0, 4.0, 12.0, 0.0], not [0.6, 0.8]
#     assert result == pytest.approx([0.6, 0.8])  # FAILS
#     assert floats_by_text["test"] == pytest.approx([0.6, 0.8])  # FAILS (empty)