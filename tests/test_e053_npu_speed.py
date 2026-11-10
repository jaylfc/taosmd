"""Tests for benchmarks/e053_npu_speed.py -- E-053 NPU speed harness.

All tests are offline; no network, no real tokenizer download.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Stub HTTP server for /api/generate
# ---------------------------------------------------------------------------

class StubGenerateHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, recorded_requests=None, **kwargs):
        self.recorded_requests = recorded_requests
        super().__init__(*args, **kwargs)

    def do_POST(self):
        if self.path != "/api/generate":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            req_json = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return

        if self.recorded_requests is not None:
            self.recorded_requests.append(req_json)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()

        # Chunk 1: first text
        chunk1 = {"message": {"content": "Hello"}, "done": False}
        self.wfile.write((json.dumps(chunk1) + "\n").encode())

        # Chunk 2: more text
        chunk2 = {"message": {"content": " world"}, "done": False}
        self.wfile.write((json.dumps(chunk2) + "\n").encode())

        # Chunk 3: done with all fields
        chunk3 = {
            "done": True,
            "total_duration": 1000000000,
            "load_duration": 100000000,
            "prompt_eval_count": 128,
            "prompt_eval_duration": 200000000,
            "eval_count": 10,
            "eval_duration": 500000000,
            "done_reason": "stop",
        }
        self.wfile.write((json.dumps(chunk3) + "\n").encode())

    def log_message(self, format, *args):
        pass


def _make_stub_server(recorded_requests):
    def handler(*args, **kwargs):
        return StubGenerateHandler(*args, recorded_requests=recorded_requests, **kwargs)
    server = HTTPServer(("127.0.0.1", 0), handler)
    return server


@pytest.fixture
def stub_server():
    recorded = []
    server = _make_stub_server(recorded)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", recorded
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Fixtures: tiny dataset and monkeypatched tokenizer
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_dataset(tmp_path):
    """Minimal LongMemEval-S shaped dataset with one question."""
    data = [{
        "haystack_sessions": [
            [
                {"role": "user", "content": "This is the first user message with enough words to pass the twenty word minimum threshold for loop prompt selection."},
                {"role": "assistant", "content": "First assistant response."},
                {"role": "user", "content": "Second user message also long enough to qualify for the loop extraction prompt generation."},
                {"role": "assistant", "content": "Second assistant response."},
            ],
            [
                {"role": "user", "content": "Third user message in a new session with sufficient length."},
                {"role": "assistant", "content": "Third assistant response."},
            ],
        ]
    }]
    path = tmp_path / "longmemeval_s_full.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def mock_tokenizers(monkeypatch):
    """Monkeypatch transformers.AutoTokenizer to return fake tokenizers."""

    class FakeTokenizer:
        def __init__(self, name):
            self.name = name

        def encode(self, text, add_special_tokens=False):
            # Simple whitespace split, each word = 1 token
            return list(range(len(text.split())))

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(["token"] * len(ids))

    qwen_tok = FakeTokenizer("qwen")
    granite_tok = FakeTokenizer("granite")

    def mock_from_pretrained(repo, revision=None):
        if "Qwen" in repo:
            return qwen_tok
        if "granite" in repo.lower() or "ibm" in repo.lower():
            return granite_tok
        return FakeTokenizer("unknown")

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", mock_from_pretrained)


# ---------------------------------------------------------------------------
# Test: prepare subcommand
# ---------------------------------------------------------------------------

def test_prepare_creates_prompts(tiny_dataset, tmp_path, mock_tokenizers):
    from benchmarks.e053_npu_speed import cmd_prepare
    import argparse

    out_path = tmp_path / "prompts.json"
    args = argparse.Namespace(dataset=str(tiny_dataset), out=str(out_path))
    rc = cmd_prepare(args)
    assert rc == 0
    assert out_path.exists()

    with out_path.open() as f:
        data = json.load(f)

    assert "header" in data
    assert data["header"]["dataset_sha256"]
    assert data["header"]["qwen_tokenizer_repo"] == "Qwen/Qwen3.5-2B"
    assert data["header"]["qwen_tokenizer_revision"] == "15852e8c16360a2fea060d615a32b45270f8a8fc"
    assert data["header"]["granite_tokenizer_repo"] == "ibm-granite/granite-4.2-3b"
    assert data["header"]["granite_tokenizer_revision"] == "4e9c275e232f10211ec88644d106a1f6a1b60e06"

    prompts = data["prompts"]
    assert len(prompts) > 0

    length_prompts = [p for p in prompts if p["kind"] == "length"]
    assert len(length_prompts) == 3
    for p in length_prompts:
        assert p["length"] in (128, 512, 2048)
        assert "tokens_qwen" in p
        assert "tokens_granite" in p

    loop_prompts = [p for p in prompts if p["kind"] == "loop"]
    assert len(loop_prompts) <= 20
    for p in loop_prompts:
        assert "index" in p
        assert "tokens_qwen" in p
        assert "tokens_granite" in p
        assert p["text"].startswith("You are the librarian")


# ---------------------------------------------------------------------------
# Test: run subcommand -- request body validation
# ---------------------------------------------------------------------------

def test_run_request_body_has_correct_fields(stub_server, tiny_dataset, tmp_path, mock_tokenizers):
    from benchmarks.e053_npu_speed import cmd_prepare, cmd_run
    import argparse

    url, recorded = stub_server

    # Prepare prompts first
    prompts_path = tmp_path / "prompts.json"
    prep_args = argparse.Namespace(dataset=str(tiny_dataset), out=str(prompts_path))
    cmd_prepare(prep_args)

    out_dir = tmp_path / "run_out"
    run_args = argparse.Namespace(
        prompts=str(prompts_path),
        model="test-model",
        arm="Q",
        worker_pid=os.getpid(),
        url=url,
        repeats=1,
        rest_seconds=0,
        out_dir=str(out_dir),
        ctx_limit=None,
    )
    rc = cmd_run(run_args)
    assert rc == 0

    assert len(recorded) > 0
    for req in recorded:
        assert req["model"] == "test-model"
        assert req["stream"] is True
        assert req["think"] is False
        assert req["options"]["temperature"] == 0
        # length prompts use 128, loop prompts use 512
        assert req["options"]["max_new_tokens"] in (128, 512)


# ---------------------------------------------------------------------------
# Test: run subcommand -- context guard skips over-limit prompts
# ---------------------------------------------------------------------------

def test_run_context_guard_skips(stub_server, tiny_dataset, tmp_path, mock_tokenizers):
    from benchmarks.e053_npu_speed import cmd_prepare, cmd_run
    import argparse

    url, recorded = stub_server

    prompts_path = tmp_path / "prompts.json"
    prep_args = argparse.Namespace(dataset=str(tiny_dataset), out=str(prompts_path))
    cmd_prepare(prep_args)

    out_dir = tmp_path / "run_out"
    # Set ctx_limit very low so all prompts are skipped
    run_args = argparse.Namespace(
        prompts=str(prompts_path),
        model="test-model",
        arm="Q",
        worker_pid=os.getpid(),
        url=url,
        repeats=1,
        rest_seconds=0,
        out_dir=str(out_dir),
        ctx_limit=10,  # Very small limit
    )
    rc = cmd_run(run_args)
    assert rc == 0

    # No requests should have been sent
    assert len(recorded) == 0

    # Check raw.jsonl has skipped rows
    raw_path = out_dir / "e053-Q-raw.jsonl"
    assert raw_path.exists()
    with raw_path.open() as f:
        lines = f.readlines()
    assert len(lines) > 0
    for line in lines:
        row = json.loads(line)
        assert row.get("skipped") == "context"


# ---------------------------------------------------------------------------
# Test: run subcommand -- server_prefill_fabricated flag
# ---------------------------------------------------------------------------

class FabricatedPrefillHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, fabricated=True, recorded_requests=None, **kwargs):
        self.fabricated = fabricated
        self.recorded_requests = recorded_requests
        super().__init__(*args, **kwargs)

    def do_POST(self):
        if self.path != "/api/generate":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            req_json = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return

        if self.recorded_requests is not None:
            self.recorded_requests.append(req_json)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()

        chunk1 = {"message": {"content": "Hello"}, "done": False}
        self.wfile.write((json.dumps(chunk1) + "\n").encode())

        chunk2 = {"message": {"content": " world"}, "done": False}
        self.wfile.write((json.dumps(chunk2) + "\n").encode())

        total_dur = 1000000000
        if self.fabricated:
            prompt_eval_dur = int(0.1 * total_dur)
        else:
            prompt_eval_dur = 200000000

        chunk3 = {
            "done": True,
            "total_duration": total_dur,
            "load_duration": 100000000,
            "prompt_eval_count": 128,
            "prompt_eval_duration": prompt_eval_dur,
            "eval_count": 10,
            "eval_duration": 500000000,
            "done_reason": "stop",
        }
        self.wfile.write((json.dumps(chunk3) + "\n").encode())

    def log_message(self, format, *args):
        pass


def _make_fabricated_server(fabricated, recorded):
    def handler(*args, **kwargs):
        return FabricatedPrefillHandler(*args, fabricated=fabricated, recorded_requests=recorded, **kwargs)
    return HTTPServer(("127.0.0.1", 0), handler)


def test_run_server_prefill_fabricated_flag_true(tmp_path, tiny_dataset, mock_tokenizers):
    from benchmarks.e053_npu_speed import cmd_prepare, cmd_run
    import argparse

    recorded = []
    server = _make_fabricated_server(fabricated=True, recorded=recorded)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        prompts_path = tmp_path / "prompts.json"
        prep_args = argparse.Namespace(dataset=str(tiny_dataset), out=str(prompts_path))
        cmd_prepare(prep_args)

        out_dir = tmp_path / "run_out"
        run_args = argparse.Namespace(
            prompts=str(prompts_path),
            model="test-model",
            arm="Q",
            worker_pid=os.getpid(),
            url=url,
            repeats=1,
            rest_seconds=0,
            out_dir=str(out_dir),
            ctx_limit=None,
        )
        cmd_run(run_args)

        raw_path = out_dir / "e053-Q-raw.jsonl"
        with raw_path.open() as f:
            for line in f:
                row = json.loads(line)
                if "error" not in row and "skipped" not in row:
                    assert row.get("server_prefill_fabricated") is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_run_server_prefill_fabricated_flag_false(tmp_path, tiny_dataset, mock_tokenizers):
    from benchmarks.e053_npu_speed import cmd_prepare, cmd_run
    import argparse

    recorded = []
    server = _make_fabricated_server(fabricated=False, recorded=recorded)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        prompts_path = tmp_path / "prompts.json"
        prep_args = argparse.Namespace(dataset=str(tiny_dataset), out=str(prompts_path))
        cmd_prepare(prep_args)

        out_dir = tmp_path / "run_out"
        run_args = argparse.Namespace(
            prompts=str(prompts_path),
            model="test-model",
            arm="Q",
            worker_pid=os.getpid(),
            url=url,
            repeats=1,
            rest_seconds=0,
            out_dir=str(out_dir),
            ctx_limit=None,
        )
        cmd_run(run_args)

        raw_path = out_dir / "e053-Q-raw.jsonl"
        with raw_path.open() as f:
            for line in f:
                row = json.loads(line)
                if "error" not in row and "skipped" not in row:
                    assert row.get("server_prefill_fabricated") is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Test: run subcommand -- survives unreadable worker pid
# ---------------------------------------------------------------------------

def test_run_survives_unreadable_pid(stub_server, tiny_dataset, tmp_path, mock_tokenizers):
    from benchmarks.e053_npu_speed import cmd_prepare, cmd_run
    import argparse

    url, _ = stub_server

    prompts_path = tmp_path / "prompts.json"
    prep_args = argparse.Namespace(dataset=str(tiny_dataset), out=str(prompts_path))
    cmd_prepare(prep_args)

    out_dir = tmp_path / "run_out"
    # Use a non-existent PID
    run_args = argparse.Namespace(
        prompts=str(prompts_path),
        model="test-model",
        arm="Q",
        worker_pid=999999999,
        url=url,
        repeats=1,
        rest_seconds=0,
        out_dir=str(out_dir),
        ctx_limit=None,
    )
    rc = cmd_run(run_args)
    assert rc == 0

    host_path = out_dir / "e053-Q-host.json"
    assert host_path.exists()
    with host_path.open() as f:
        host = json.load(f)
    # Should record error string, not crash
    assert "error" in host["mem_before"]
    assert "error" in host["mem_after"]


# ---------------------------------------------------------------------------
# Test: score subcommand -- kill checks
# ---------------------------------------------------------------------------

def _write_synthetic_raw(out_dir, arm, prefill_ratio, cap_hits):
    """Write a synthetic raw.jsonl for scoring tests."""
    raw_path = out_dir / f"e053-{arm}-raw.jsonl"
    host_path = out_dir / f"e053-{arm}-host.json"

    # Host with VmHWM
    host = {
        "arm": arm,
        "mem_before": {"VmRSS_kB": 100000, "VmHWM_kB": 200000},
        "mem_after": {"VmRSS_kB": 150000, "VmHWM_kB": 200000},
        "host": {"cpu_governors": ["performance"], "npu_freqs_hz": [1000000000], "uname": {}},
    }
    with host_path.open("w") as f:
        json.dump(host, f)

    with raw_path.open("w") as f:
        # Length prompts: 3 lengths x 5 repeats = 15 rows
        for L in (128, 512, 2048):
            for r in range(1, 6):
                if arm == "G":
                    prefill_tps = 100.0 * prefill_ratio
                elif arm == "QH":
                    prefill_tps = 100.0
                else:
                    prefill_tps = 100.0
                row = {
                    "arm": arm,
                    "kind": "length",
                    "length": L,
                    "repeat": r,
                    "ttft_s": 0.1,
                    "prefill_tps": prefill_tps,
                    "decode_tps": 50.0,
                    "output_text": "x" * 100,
                    "eval_count": 100,
                    "total_duration_ns": 1000000000,
                    "load_duration_ns": 100000000,
                    "prompt_eval_count": 128,
                    "prompt_eval_duration_ns": 200000000,
                    "eval_duration_ns": 500000000,
                    "done_reason": "stop",
                    "server_prefill_fabricated": False,
                    "utc_timestamp": time.time(),
                }
                f.write(json.dumps(row) + "\n")

        # Loop prompts: 20 prompts x 1 repeat = 20 rows
        for i in range(20):
            output_text = "token " * (512 if i < cap_hits else 100)
            row = {
                "arm": arm,
                "kind": "loop",
                "index": i,
                "repeat": 1,
                "ttft_s": 0.1,
                "prefill_tps": 100.0,
                "decode_tps": 50.0,
                "output_text": output_text,
                "eval_count": len(output_text.split()),
                "total_duration_ns": 1000000000,
                "load_duration_ns": 100000000,
                "prompt_eval_count": 300,
                "prompt_eval_duration_ns": 200000000,
                "eval_duration_ns": 500000000,
                "done_reason": "stop",
                "server_prefill_fabricated": False,
                "utc_timestamp": time.time(),
            }
            f.write(json.dumps(row) + "\n")


def test_score_kill_check_prefill_fail(tmp_path):
    from benchmarks.e053_npu_speed import cmd_score
    import argparse

    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(json.dumps({"header": {}, "prompts": []}))

    out_dir = tmp_path / "score_in"
    out_dir.mkdir()

    # G prefill = 0.79x QH -> FAIL
    _write_synthetic_raw(out_dir, "G", prefill_ratio=0.79, cap_hits=0)
    _write_synthetic_raw(out_dir, "QH", prefill_ratio=1.0, cap_hits=0)
    _write_synthetic_raw(out_dir, "Q", prefill_ratio=1.0, cap_hits=0)

    out_path = tmp_path / "summary.json"
    args = argparse.Namespace(
        prompts=str(prompts_path),
        dir=str(out_dir),
        out=str(out_path),
    )
    rc = cmd_score(args)
    assert rc == 0

    with out_path.open() as f:
        summary = json.load(f)

    check1 = summary["kill_checks"]["prefill_512"]
    assert check1["status"] == "FAIL"
    assert check1["ratio"] == pytest.approx(0.79, rel=0.01)


def test_score_kill_check_prefill_pass(tmp_path):
    from benchmarks.e053_npu_speed import cmd_score
    import argparse

    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(json.dumps({"header": {}, "prompts": []}))

    out_dir = tmp_path / "score_in"
    out_dir.mkdir()

    # G prefill = 0.81x QH -> PASS
    _write_synthetic_raw(out_dir, "G", prefill_ratio=0.81, cap_hits=0)
    _write_synthetic_raw(out_dir, "QH", prefill_ratio=1.0, cap_hits=0)
    _write_synthetic_raw(out_dir, "Q", prefill_ratio=1.0, cap_hits=0)

    out_path = tmp_path / "summary.json"
    args = argparse.Namespace(
        prompts=str(prompts_path),
        dir=str(out_dir),
        out=str(out_path),
    )
    rc = cmd_score(args)
    assert rc == 0

    with out_path.open() as f:
        summary = json.load(f)

    check1 = summary["kill_checks"]["prefill_512"]
    assert check1["status"] == "PASS"
    assert check1["ratio"] == pytest.approx(0.81, rel=0.01)


def test_score_kill_check_cap_hits_fail(tmp_path):
    from benchmarks.e053_npu_speed import cmd_score
    import argparse

    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(json.dumps({"header": {}, "prompts": []}))

    out_dir = tmp_path / "score_in"
    out_dir.mkdir()

    # G cap hits = 3 -> FAIL (threshold is 2)
    _write_synthetic_raw(out_dir, "G", prefill_ratio=1.0, cap_hits=3)
    _write_synthetic_raw(out_dir, "QH", prefill_ratio=1.0, cap_hits=0)
    _write_synthetic_raw(out_dir, "Q", prefill_ratio=1.0, cap_hits=0)

    out_path = tmp_path / "summary.json"
    args = argparse.Namespace(
        prompts=str(prompts_path),
        dir=str(out_dir),
        out=str(out_path),
    )
    rc = cmd_score(args)
    assert rc == 0

    with out_path.open() as f:
        summary = json.load(f)

    check4 = summary["kill_checks"]["cap_hits"]
    assert check4["status"] == "FAIL"
    assert check4["g_value"] == 3


# ---------------------------------------------------------------------------
# Test: module import does not import transformers
# ---------------------------------------------------------------------------

def test_import_does_not_load_transformers():
    """Importing the module must not import transformers at top level."""
    # Run in subprocess to get a clean sys.modules
    code = """
import sys
import benchmarks.e053_npu_speed as m
# Check transformers is not in sys.modules
transformers_loaded = 'transformers' in sys.modules
print(f'transformers_loaded={transformers_loaded}')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd="/tmp/exec-tsk-k2z4mn",
    )
    assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
    assert "transformers_loaded=False" in result.stdout


# ---------------------------------------------------------------------------
# Test: score uses monkeypatched tokenizer (whitespace splitter)
# ---------------------------------------------------------------------------

def test_score_uses_tokenizer_for_cap_hits(tmp_path):
    """Score should use the arm's tokenizer to count generated tokens."""
    from benchmarks.e053_npu_speed import cmd_score
    import argparse

    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(json.dumps({"header": {}, "prompts": []}))

    out_dir = tmp_path / "score_in"
    out_dir.mkdir()

    # Write raw with known output text lengths
    raw_path = out_dir / "e053-G-raw.jsonl"
    host_path = out_dir / "e053-G-host.json"
    host = {"arm": "G", "mem_before": {"VmHWM_kB": 200000}, "mem_after": {"VmHWM_kB": 200000}, "host": {}}
    host_path.write_text(json.dumps(host))

    with raw_path.open("w") as f:
        # 20 loop prompts, each with output_text that splits to 600 words (cap hit)
        for i in range(20):
            row = {
                "arm": "G",
                "kind": "loop",
                "index": i,
                "repeat": 1,
                "ttft_s": 0.1,
                "prefill_tps": 100.0,
                "decode_tps": 50.0,
                "output_text": "word " * 600,  # 600 tokens by whitespace split
                "eval_count": 600,
                "total_duration_ns": 1000000000,
                "load_duration_ns": 100000000,
                "prompt_eval_count": 300,
                "prompt_eval_duration_ns": 200000000,
                "eval_duration_ns": 500000000,
                "done_reason": "stop",
                "server_prefill_fabricated": False,
                "utc_timestamp": time.time(),
            }
            f.write(json.dumps(row) + "\n")
        # Also need QH and Q for kill checks
        for arm in ("QH", "Q"):
            rp = out_dir / f"e053-{arm}-raw.jsonl"
            hp = out_dir / f"e053-{arm}-host.json"
            hp.write_text(json.dumps({"arm": arm, "mem_before": {"VmHWM_kB": 200000}, "mem_after": {"VmHWM_kB": 200000}, "host": {}}))
            with rp.open("w") as f2:
                for L in (128, 512, 2048):
                    for r in range(5):
                        f2.write(json.dumps({"arm": arm, "kind": "length", "length": L, "repeat": r, "ttft_s": 0.1, "prefill_tps": 100.0, "decode_tps": 50.0, "output_text": "x", "eval_count": 1, "utc_timestamp": time.time()}) + "\n")
                for i in range(20):
                    f2.write(json.dumps({"arm": arm, "kind": "loop", "index": i, "repeat": 1, "ttft_s": 0.1, "prefill_tps": 100.0, "decode_tps": 50.0, "output_text": "word " * 100, "eval_count": 100, "utc_timestamp": time.time()}) + "\n")

    out_path = tmp_path / "summary.json"
    args = argparse.Namespace(prompts=str(prompts_path), dir=str(out_dir), out=str(out_path))
    rc = cmd_score(args)
    assert rc == 0

    with out_path.open() as f:
        summary = json.load(f)

    # With whitespace tokenizer, 600 words >= 512 -> cap hit
    assert summary["arms"]["G"]["loop"]["cap_hits"] == 20


import subprocess  # noqa: E402 (needed for test_import_does_not_load_transformers)