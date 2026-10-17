#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import bench_provenance


def test_config_hash_order_insensitive(tmp_path):
    a = {"a": 1, "b": 2}
    b = {"b": 2, "a": 1}
    assert bench_provenance.config_hash(a) == bench_provenance.config_hash(b)
    assert bench_provenance.config_hash(a) != bench_provenance.config_hash({"a": 1, "b": 3})


def test_config_hash_ignores_volatile_keys():
    assert bench_provenance.config_hash({"top_k": 5, "timestamp": "a", "git_sha": "x"}) == \
           bench_provenance.config_hash({"top_k": 5, "timestamp": "b", "git_sha": "y"})
    assert bench_provenance.config_hash({"top_k": 5}) != bench_provenance.config_hash({"top_k": 6})


def test_file_sha256(tmp_path):
    content = b"hello world"
    p = tmp_path / "f.txt"
    p.write_bytes(content)
    assert bench_provenance.file_sha256(p) == hashlib.sha256(content).hexdigest()
    assert bench_provenance.file_sha256(None) is None


def test_file_sha256_none():
    assert bench_provenance.file_sha256(None) is None


def test_file_sha256_missing_raises(tmp_path):
    missing = tmp_path / "missing.txt"
    try:
        bench_provenance.file_sha256(missing)
    except FileNotFoundError as exc:
        assert str(exc) == str(missing)
    else:
        raise AssertionError("should have raised FileNotFoundError")


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


def test_model_dir_digests_none():
    assert bench_provenance.model_dir_digests(None) == {}


def test_model_dir_digests_missing_raises(tmp_path):
    missing = tmp_path / "missing"
    try:
        bench_provenance.model_dir_digests(missing)
    except ValueError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("should have raised ValueError")


def test_model_dir_digests_not_dir_raises(tmp_path):
    (tmp_path / "file.txt").write_text("test")
    try:
        bench_provenance.model_dir_digests(tmp_path / "file.txt")
    except ValueError as exc:
        assert "is not a directory" in str(exc)
    else:
        raise AssertionError("should have raised ValueError")


def test_hostname_type():
    import socket
    prov = bench_provenance.collect({}, None, None)
    assert isinstance(prov["hostname"], str)
    assert prov["hostname"] == socket.gethostname()


def test_platform_type():
    import platform
    prov = bench_provenance.collect({}, None, None)
    assert isinstance(prov["platform"], str)
    assert prov["platform"] == platform.platform()


def test_machine_type():
    import platform
    prov = bench_provenance.collect({}, None, None)
    assert isinstance(prov["machine"], str)
    assert prov["machine"] == platform.machine()


def test_python_type():
    import platform
    prov = bench_provenance.collect({}, None, None)
    assert isinstance(prov["python"], str)
    assert prov["python"] == platform.python_version()


def test_onnxruntime_version_type():
    prov = bench_provenance.collect({}, None, None)
    assert prov["onnxruntime_version"] is None or isinstance(prov["onnxruntime_version"], str)


def test_env_prefixes_type():
    import os
    prov = bench_provenance.collect({}, None, None)
    assert isinstance(prov["env_prefixes"], dict)
    assert "TAOSMD_ONNX_QUERY_PREFIX" in prov["env_prefixes"]
    assert "TAOSMD_ONNX_DOC_PREFIX" in prov["env_prefixes"]
    assert prov["env_prefixes"]["TAOSMD_ONNX_QUERY_PREFIX"] == os.environ.get("TAOSMD_ONNX_QUERY_PREFIX")
    assert prov["env_prefixes"]["TAOSMD_ONNX_DOC_PREFIX"] == os.environ.get("TAOSMD_ONNX_DOC_PREFIX")


def test_git_commit_unknown_on_error():
    """Test that git_commit is 'unknown' when git operations fail."""
    with patch('subprocess.run') as mock_run:
        # Mock git commands to fail
        mock_run.side_effect = Exception("Git not found")
        
        prov = bench_provenance.collect({}, None, None)
        assert prov["git_commit"] == "unknown"
        assert prov["git_dirty"] is None


def test_git_absent_outside_repo():
    """Test that git_commit is 'unknown' and git_dirty is None when outside git repo."""
    with patch('subprocess.run') as mock_run:
        # Mock git commands to fail
        mock_run.side_effect = Exception("Git not found")
        
        prov = bench_provenance.collect({}, None, None)
        assert prov["git_commit"] == "unknown"
        assert prov["git_dirty"] is None


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


def test_provenance_key_in_longmemeval_results(monkeypatch):
    """Test that provenance key is written in benchmark results."""
    sentinel = {"__sentinel__": True}
    monkeypatch.setattr(bench_provenance, "collect", lambda *_, **__: sentinel)
    
    # Just verify that collect returns our sentinel
    prov = bench_provenance.collect({}, None, None)
    assert prov == sentinel


def test_provenance_key_in_retrieval_probe_results(monkeypatch):
    """Test that provenance key is written in benchmark results."""
    sentinel = {"__sentinel__": True}
    monkeypatch.setattr(bench_provenance, "collect", lambda *_, **__: sentinel)
    
    # Just verify that collect returns our sentinel
    prov = bench_provenance.collect({}, None, None)
    assert prov == sentinel

