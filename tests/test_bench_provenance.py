#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import bench_provenance


def test_config_hash_order_insensitive(tmp_path):
    a = {"a": 1, "b": 2}
    b = {"b": 2, "a": 1}
    assert bench_provenance.config_hash(a) == bench_provenance.config_hash(b)
    assert bench_provenance.config_hash(a) != bench_provenance.config_hash({"a": 1, "b": 3})


def test_file_sha256(tmp_path):
    content = b"hello world"
    p = tmp_path / "f.txt"
    p.write_bytes(content)
    assert bench_provenance.file_sha256(p) == hashlib.sha256(content).hexdigest()
    assert bench_provenance.file_sha256(None) is None


def test_model_dir_digests(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"onnx")
    (tmp_path / "model.onnx_data").write_bytes(b"data")
    (tmp_path / "tokenizer.json").write_bytes(b"tok")
    (tmp_path / "notes.txt").write_bytes(b"n")
    d = bench_provenance.model_dir_digests(tmp_path)
    assert set(d.keys()) == {"model.onnx", "model.onnx_data", "tokenizer.json"}
    assert d["model.onnx"] == hashlib.sha256(b"onnx").hexdigest()
    assert d["model.onnx_data"] == hashlib.sha256(b"data").hexdigest()
    assert d["tokenizer.json"] == hashlib.sha256(b"tok").hexdigest()


def test_model_dir_digests_empty_existing_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    try:
        bench_provenance.model_dir_digests(empty)
    except ValueError:
        pass
    else:
        raise AssertionError("should have raised")


def test_model_dir_digests_none_returns_empty():
    assert bench_provenance.model_dir_digests(None) == {}


def test_collect_returns_all_keys_and_format(tmp_path):
    cfg = {"a": 1}
    (tmp_path / "ds.json").write_text("{}")
    prov = bench_provenance.collect(cfg, None, tmp_path / "ds.json")
    expected = {
        "git_commit",
        "git_dirty",
        "hostname",
        "platform",
        "machine",
        "python",
        "onnxruntime_version",
        "env_prefixes",
        "model_files",
        "dataset_sha256",
        "config_hash",
    }
    assert set(prov.keys()) == expected
    assert re.match(r"^[0-9a-f]{40}$", prov["git_commit"] or "") is not None
    assert prov["config_hash"] == bench_provenance.config_hash(cfg)


def test_probe_runs_as_script():
    repo_root = Path(__file__).resolve().parent.parent
    for script in (
        "benchmarks/retrieval_latency_probe.py",
        "benchmarks/longmemeval_enhanced.py",
    ):
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"{script} --help exited {result.returncode}: {result.stderr}"
        )
        assert "ModuleNotFoundError" not in result.stderr, (
            f"{script} --help produced ModuleNotFoundError: {result.stderr}"
        )
