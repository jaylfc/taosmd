#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import bench_provenance


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=repo, check=True, capture_output=True,
    )


@pytest.fixture
def clean_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def test_config_hash_order_insensitive():
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


def test_file_sha256_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        bench_provenance.file_sha256(tmp_path / "missing.json")


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
    with pytest.raises(ValueError):
        bench_provenance.model_dir_digests(empty)


def test_model_dir_digests_missing_raises(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        bench_provenance.model_dir_digests(tmp_path / "missing")


def test_model_dir_digests_file_raises(tmp_path):
    f = tmp_path / "model.onnx"
    f.write_bytes(b"onnx")
    with pytest.raises(ValueError, match="not a directory"):
        bench_provenance.model_dir_digests(f)


def test_model_dir_digests_none_returns_empty():
    assert bench_provenance.model_dir_digests(None) == {}


def test_collect_field_values(tmp_path, monkeypatch, clean_repo):
    monkeypatch.setenv("TAOSMD_ONNX_QUERY_PREFIX", "query: ")
    monkeypatch.setenv("TAOSMD_ONNX_DOC_PREFIX", "passage: ")
    models = tmp_path / "models"
    models.mkdir()
    (models / "a.onnx").write_bytes(b"weights")
    (models / "tokenizer.json").write_bytes(b"vocab")
    ds = tmp_path / "ds.json"
    ds.write_bytes(b"[1, 2, 3]")
    cfg = {"top_k": 5, "fusion": "rrf", "timestamp": "volatile"}

    prov = bench_provenance.collect(cfg, models, ds, repo_root=clean_repo)

    assert prov["hostname"] == socket.gethostname()
    assert prov["platform"] == platform.platform()
    assert prov["machine"] == platform.machine()
    assert prov["python"] == platform.python_version()
    assert prov["env_prefixes"] == {
        "TAOSMD_ONNX_QUERY_PREFIX": "query: ",
        "TAOSMD_ONNX_DOC_PREFIX": "passage: ",
    }
    assert prov["model_files"] == {
        "a.onnx": hashlib.sha256(b"weights").hexdigest(),
        "tokenizer.json": hashlib.sha256(b"vocab").hexdigest(),
    }
    assert prov["dataset_sha256"] == hashlib.sha256(b"[1, 2, 3]").hexdigest()
    # Independent of config_hash(): the volatile key is dropped, the rest is canonical JSON.
    expected_cfg = json.dumps({"fusion": "rrf", "top_k": 5}, sort_keys=True, separators=(",", ":"))
    assert prov["config_hash"] == hashlib.sha256(expected_cfg.encode("utf-8")).hexdigest()
    try:
        import onnxruntime
        expected_ort = onnxruntime.__version__
    except Exception:
        expected_ort = None
    assert prov["onnxruntime_version"] == expected_ort
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clean_repo,
                          capture_output=True, text=True, check=True).stdout.strip()
    assert prov["git_commit"] == head


def test_collect_git_dirty_false_in_clean_repo(clean_repo):
    prov = bench_provenance.collect({}, None, None, repo_root=clean_repo)
    assert prov["git_dirty"] is False


def test_collect_git_dirty_true_with_untracked_file(clean_repo):
    (clean_repo / "untracked.txt").write_text("x\n")
    prov = bench_provenance.collect({}, None, None, repo_root=clean_repo)
    assert prov["git_dirty"] is True


def test_collect_outside_repo_is_unknown(tmp_path):
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    prov = bench_provenance.collect({}, None, None, repo_root=not_a_repo)
    assert prov["git_commit"] == "unknown"
    assert prov["git_dirty"] is None


def _fake_git(monkeypatch, returncode: int, stdout: str) -> None:
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")
    monkeypatch.setattr(bench_provenance.subprocess, "run", fake_run)


def test_collect_rev_parse_nonzero_rc_with_stdout_is_unknown(monkeypatch, tmp_path):
    _fake_git(monkeypatch, 1, "deadbeef\n")
    prov = bench_provenance.collect({}, None, None, repo_root=tmp_path)
    assert prov["git_commit"] == "unknown"
    assert prov["git_dirty"] is None


def test_collect_rev_parse_zero_rc_empty_stdout_is_unknown(monkeypatch, tmp_path):
    _fake_git(monkeypatch, 0, "")
    prov = bench_provenance.collect({}, None, None, repo_root=tmp_path)
    assert prov["git_commit"] == "unknown"
    assert prov["git_dirty"] is None


def test_collect_git_binary_missing_is_unknown(monkeypatch, tmp_path):
    def raise_missing(cmd, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(bench_provenance.subprocess, "run", raise_missing)
    prov = bench_provenance.collect({}, None, None, repo_root=tmp_path)
    assert prov["git_commit"] == "unknown"
    assert prov["git_dirty"] is None


def test_collect_status_failure_after_good_rev_parse_is_none(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        if cmd[1:3] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n", stderr="")
        return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="fatal")
    monkeypatch.setattr(bench_provenance.subprocess, "run", fake_run)
    prov = bench_provenance.collect({}, None, None, repo_root=tmp_path)
    assert prov["git_commit"] == "abc123"
    assert prov["git_dirty"] is None


def test_collect_onnxruntime_version_from_module(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(__version__="9.9.9-test"))
    prov = bench_provenance.collect({}, None, None, repo_root=tmp_path)
    assert prov["onnxruntime_version"] == "9.9.9-test"


def test_collect_onnxruntime_absent_is_none(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    prov = bench_provenance.collect({}, None, None, repo_root=tmp_path)
    assert prov["onnxruntime_version"] is None


SENTINEL = {"sentinel": "provenance-reached-the-file"}


def test_longmemeval_results_carry_provenance(tmp_path):
    from benchmarks import longmemeval_enhanced
    out = tmp_path / "results" / "enhanced.json"
    longmemeval_enhanced._write_results(str(out), 5, {"hybrid_baseline": {}}, {"fusion": "rrf"}, SENTINEL)
    written = json.loads(out.read_text())
    assert written["provenance"] == SENTINEL


def test_longmemeval_run_benchmark_writes_provenance(tmp_path, monkeypatch):
    """Provenance wiring in run_benchmark: a successful run writes the provenance block into results."""
    from benchmarks import longmemeval_enhanced
    import json
    import asyncio

    data_file = tmp_path / "test_data.json"
    test_question = {
        "question": "When did Alice move to Paris?",
        "question_type": "single-session-user",
        "answer_session_ids": ["s0"],
        "haystack_sessions": [[{"role": "user", "content": "Alice moved to Paris in 2020"}]],
        "haystack_session_ids": ["s0"],
    }
    data_file.write_text(json.dumps([test_question]))

    # Monkey-patch DATA_PATH so run_benchmark uses our test data
    monkeypatch.setattr(longmemeval_enhanced, "DATA_PATH", str(data_file))

    # Create a model directory so bench_provenance.collect finds model files
    arm_dir = tmp_path / "models" / "custom-arm"
    arm_dir.mkdir(parents=True)
    (arm_dir / "model.onnx").write_text("")
    (arm_dir / "tokenizer.json").write_text("{}")

    # Monkey-patch VectorMemory to avoid ONNX dependency
    captured_kwargs = {}

    def make_fake_vmem(**kwargs):
        captured_kwargs.update(kwargs)
        class FakeVMem:
            async def init(self): pass
            async def add(self, text, metadata=None): return 1
            async def embed(self, text, task="search_document"): return [1.0]*384
            async def search(self, query, limit=5, hybrid=True, fusion="boost"): return []
            async def close(self): pass
        return FakeVMem()

    monkeypatch.setattr(longmemeval_enhanced, "VectorMemory", make_fake_vmem)
    monkeypatch.setattr(longmemeval_enhanced, "collect", lambda *a, **kw: SENTINEL)
    # Monkeypatch __file__ so _write_results writes to tmp_path instead of benchmarks/
    monkeypatch.setattr(longmemeval_enhanced, "__file__", str(tmp_path / "longmemeval_enhanced.py"))

    # Set required env vars
    import os
    old_onnx_path = os.environ.get("TAOSMD_BENCH_ONNX_PATH")
    os.environ["TAOSMD_BENCH_ONNX_PATH"] = str(arm_dir)
    os.environ["TAOSMD_BENCH_BINARY_QUANT"] = "1"
    old_cwd = os.getcwd()
    os.chdir(str(tmp_path))

    try:
        asyncio.run(longmemeval_enhanced.run_benchmark(limit=1, top_k=5))
    finally:
        if old_onnx_path is not None:
            os.environ["TAOSMD_BENCH_ONNX_PATH"] = old_onnx_path
        elif "TAOSMD_BENCH_ONNX_PATH" in os.environ:
            del os.environ["TAOSMD_BENCH_ONNX_PATH"]
        os.chdir(old_cwd)

    # Check that results were written with provenance
    output_files = list((tmp_path / "results").glob("enhanced_*.json"))
    assert len(output_files) == 1, f"Expected one output file, found {output_files}"

    with open(output_files[0]) as f:
        output = json.load(f)

    assert "provenance" in output, "provenance block missing from run_benchmark output"
    assert output["arm_config"]["onnx_path"] == str(arm_dir)
    assert output["arm_config"]["binary_quant"] is True


def test_retrieval_probe_results_carry_provenance(tmp_path, monkeypatch):
    from benchmarks import retrieval_latency_probe
    seen = {}

    def fake_collect(config, onnx_dir, dataset_path):
        seen["args"] = (onnx_dir, dataset_path)
        return SENTINEL

    monkeypatch.setattr(retrieval_latency_probe, "collect", fake_collect)
    out = tmp_path / "probe.json"
    args = argparse.Namespace(out=str(out), embed_mode="onnx", onnx_path="/m", dataset="/d.json")
    config = {"timestamp": "20261019_000000"}
    path = retrieval_latency_probe._write_results(args, config, [], [], {})
    assert path == out
    written = json.loads(out.read_text())
    assert written["config"]["provenance"] == SENTINEL
    assert seen["args"] == ("/m", "/d.json")


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
