"""Tests for benchmarks/bench_checkpoint.py and the runner's pause/resume flags.

Covers:
  1. config_hash: stable across key order, insensitive to excluded keys,
     sensitive to a model change.
  2. write_header + append_conversation + load_checkpoint round-trip.
  3. load_checkpoint raises ValueError on hash mismatch (names both hashes).
  4. Truncated final line is skipped with a warning; earlier content loads fine.
  5. Malformed middle line raises ValueError.
  6. pause_requested returns True/False based on flag file presence.
  7. locomo_runner exposes --ckpt, --resume, and --pause-flag via _build_parser.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = REPO_ROOT / "benchmarks" / "bench_checkpoint.py"
RUNNER_PATH = REPO_ROOT / "benchmarks" / "locomo_runner.py"


def _load_ckpt():
    spec = importlib.util.spec_from_file_location("bench_checkpoint", CKPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner():
    spec = importlib.util.spec_from_file_location("locomo_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. config_hash
# ---------------------------------------------------------------------------

def test_config_hash_stable_across_key_order():
    ckpt = _load_ckpt()
    a = {"model": "qwen3:4b", "top_k": 10, "strategy": "vector-only"}
    b = {"strategy": "vector-only", "top_k": 10, "model": "qwen3:4b"}
    assert ckpt.config_hash(a) == ckpt.config_hash(b)


def test_config_hash_insensitive_to_excluded_keys():
    ckpt = _load_ckpt()
    base = {"model": "qwen3:4b", "top_k": 10}
    with_extras = {
        "model": "qwen3:4b",
        "top_k": 10,
        "timestamp": "20260611_120000",
        "git_sha": "abc1234",
        "run_id": "my-run",
        "failed_qa": 3,
    }
    assert ckpt.config_hash(base) == ckpt.config_hash(with_extras)


def test_config_hash_sensitive_to_model_change():
    ckpt = _load_ckpt()
    a = {"model": "qwen3:4b", "top_k": 10}
    b = {"model": "llama3.1:8b", "top_k": 10}
    assert ckpt.config_hash(a) != ckpt.config_hash(b)


# ---------------------------------------------------------------------------
# 2. Round-trip: write_header + append_conversation + load_checkpoint
# ---------------------------------------------------------------------------

def test_round_trip_two_conversations():
    ckpt = _load_ckpt()
    cfg = {"model": "qwen3:4b", "top_k": 10}
    h = ckpt.config_hash(cfg)

    rows_a = [{"conv_id": "c1", "f1": 0.5, "judge": 1.0}]
    rows_b = [{"conv_id": "c2", "f1": 0.3, "judge": 0.0},
              {"conv_id": "c2", "f1": 0.7, "judge": 1.0}]

    with tempfile.NamedTemporaryFile(suffix=".ckpt.jsonl", delete=False) as tf:
        path = tf.name
    try:
        ckpt.write_header(path, h, "/data/locomo10.json")
        ckpt.append_conversation(path, "c1", rows_a)
        ckpt.append_conversation(path, "c2", rows_b)

        done_ids, all_rows = ckpt.load_checkpoint(path, h)

        assert done_ids == {"c1", "c2"}
        assert all_rows == rows_a + rows_b
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 3. load_checkpoint raises ValueError on hash mismatch
# ---------------------------------------------------------------------------

def test_load_raises_on_hash_mismatch():
    ckpt = _load_ckpt()
    cfg = {"model": "qwen3:4b"}
    h = ckpt.config_hash(cfg)

    with tempfile.NamedTemporaryFile(suffix=".ckpt.jsonl", delete=False) as tf:
        path = tf.name
    try:
        ckpt.write_header(path, h, "/data/locomo10.json")
        wrong_hash = "deadbeef00000000"
        with pytest.raises(ValueError) as exc_info:
            ckpt.load_checkpoint(path, wrong_hash)
        msg = str(exc_info.value)
        assert h in msg
        assert wrong_hash in msg
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 4. Truncated final line is skipped with a warning
# ---------------------------------------------------------------------------

def test_truncated_final_line_skipped(capsys):
    ckpt = _load_ckpt()
    cfg = {"model": "qwen3:4b"}
    h = ckpt.config_hash(cfg)

    rows_a = [{"f1": 0.5}]

    with tempfile.NamedTemporaryFile(suffix=".ckpt.jsonl", delete=False,
                                     mode="w") as tf:
        path = tf.name
        tf.write(json.dumps({"kind": "header", "config_hash": h, "dataset": "d"}) + "\n")
        tf.write(json.dumps({"kind": "conv", "conv_id": "c1", "rows": rows_a}) + "\n")
        # Partial line, no closing brace or newline: simulate crash mid-write.
        tf.write('{"kind": "conv", "conv_id": "c2", "rows": [{"f1"')
    try:
        done_ids, all_rows = ckpt.load_checkpoint(path, h)
        assert done_ids == {"c1"}
        assert all_rows == rows_a
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "truncated" in captured.out.lower()
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 5. Malformed middle line raises ValueError
# ---------------------------------------------------------------------------

def test_malformed_middle_line_raises():
    ckpt = _load_ckpt()
    cfg = {"model": "qwen3:4b"}
    h = ckpt.config_hash(cfg)

    with tempfile.NamedTemporaryFile(suffix=".ckpt.jsonl", delete=False,
                                     mode="w") as tf:
        path = tf.name
        tf.write(json.dumps({"kind": "header", "config_hash": h, "dataset": "d"}) + "\n")
        tf.write("{not valid json\n")
        tf.write(json.dumps({"kind": "conv", "conv_id": "c2", "rows": []}) + "\n")
    try:
        with pytest.raises(ValueError, match="malformed checkpoint line"):
            ckpt.load_checkpoint(path, h)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 6. pause_requested
# ---------------------------------------------------------------------------

def test_pause_requested_false_when_no_file():
    ckpt = _load_ckpt()
    with tempfile.TemporaryDirectory() as d:
        flag = os.path.join(d, "pause-flag")
        assert ckpt.pause_requested(flag) is False


def test_pause_requested_true_when_file_exists():
    ckpt = _load_ckpt()
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        flag = tf.name
    try:
        assert ckpt.pause_requested(flag) is True
    finally:
        os.unlink(flag)


# ---------------------------------------------------------------------------
# 7. locomo_runner exposes --ckpt, --resume, --pause-flag via _build_parser
# ---------------------------------------------------------------------------

def test_runner_parser_has_checkpoint_flags():
    runner = _load_runner()
    assert hasattr(runner, "_build_parser"), (
        "locomo_runner must expose _build_parser() for parser introspection"
    )
    parser = runner._build_parser()
    args = parser.parse_args([])
    # Verify the three new flags exist with their defaults.
    assert hasattr(args, "ckpt"), "--ckpt flag missing from parser"
    assert args.ckpt is False
    assert hasattr(args, "resume"), "--resume flag missing from parser"
    assert args.resume is False
    assert hasattr(args, "pause_flag"), "--pause-flag flag missing from parser"
    assert args.pause_flag == "/tmp/taosmd-bench-pause"
