"""Tests for the new LongMemEval runner features (tsk-sfs2s7).

Each MUST from the card is covered by at least one test that FAILS when the
fix is reverted (RED-FIRST).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "benchmarks" / "longmemeval_runner.py"

_SYNTHETIC_DATASET = [
    {
        "question_id": "q-1",
        "question_type": "temporal",
        "question": "Who is the CEO?",
        "answer": "Bob",
        "haystack_sessions": [],
    }
]

_load_counter = 0


def _load_runner():
    """Load the runner module fresh (module-level constants re-evaluated)."""
    global _load_counter
    _load_counter += 1
    name = f"longmemeval_runner_tsk_sfs2s7_{_load_counter}"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeResp:
    def __init__(self, content, status=200):
        self.status_code = status
        self._content = content

    def json(self):
        return {"message": {"content": self._content}}


class _RecordingClient:
    """Fake HTTP client: returns fixed content, records every .post() call."""

    def __init__(self, content, status=200):
        self._content = content
        self._status = status
        self.calls = []

    async def post(self, *args, **kwargs):
        self.calls.append(kwargs)
        return _FakeResp(self._content, self._status)

    async def aclose(self):
        pass


class _FakeVectorMemory:
    def __init__(self, texts):
        self.texts = texts

    async def init(self, http_client=None):
        pass

    async def search(self, query, limit=5, **kw):
        return [
            {"id": i, "text": t, "similarity": 1.0 - i * 0.01, "metadata": {}}
            for i, t in enumerate(self.texts[:limit])
        ]

    async def close(self):
        pass

    async def aclose(self):
        pass


class _FakeArchive:
    async def init(self):
        pass

    async def search_fts(self, term, limit=5):
        return []

    async def close(self):
        pass

    async def aclose(self):
        pass


class _FakeKG:
    async def init(self):
        pass

    async def close(self):
        pass

    async def aclose(self):
        pass


class _RecordingAssembler:
    def __init__(self, kg=None, archive=None):
        pass

    async def assemble(self, query, depth="auto", max_total_tokens=4000):
        return {"context": ""}


async def _fake_llm_answer(client, ctx, q):
    return "Alice"


async def _fake_self_verify(client, ctx, q, a):
    return a


# ---------------------------------------------------------------------------
# MUST 1: Default substring mode (use_llm=False) must still complete.
# `answer` must be initialised on every path.
# ---------------------------------------------------------------------------


def test_use_llm_false_returns_and_writes_rows(tmp_path, monkeypatch, capsys):
    """run_benchmark with use_llm=False must return a float and write a JSON file."""
    mod = _load_runner()
    monkeypatch.setattr(mod, "ContextAssembler", _RecordingAssembler)
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda db_path: _FakeKG())
    monkeypatch.setattr(mod, "ArchiveStore", lambda archive_dir, index_path: _FakeArchive())
    monkeypatch.setattr(mod, "VectorMemory", lambda db_path, embed_mode, onnx_path: _FakeVectorMemory(["chunk"]))
    monkeypatch.setattr(mod, "process_conversation_turn", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "load_dataset", lambda: list(_SYNTHETIC_DATASET))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_file = str(out_dir / "results.json")

    import argparse as _ap
    args = _ap.Namespace(
        limit=1, type=None, llm=False,
        graph_expansion=0, retrieval_path="retrieve",
        report_retrieval_delta=False, out=out_file,
    )

    import asyncio
    result = asyncio.run(mod.run_benchmark(limit=1, use_llm=False, args=args))

    assert isinstance(result, float), f"expected float return, got {type(result)}: {result}"
    with open(out_file) as fh:
        doc = json.load(fh)
    assert doc["metrics"]["n"] == 1


# ---------------------------------------------------------------------------
# MUST 2: TAOSMD_LME_NO_INLINE_JUDGE=1
# ---------------------------------------------------------------------------


def test_no_inline_judge_no_correct_or_accuracy_in_metrics(tmp_path, monkeypatch, capsys):
    """With NO_INLINE_JUDGE=1 the metrics must not contain correct/accuracy."""
    monkeypatch.setenv("TAOSMD_LME_NO_INLINE_JUDGE", "1")
    mod = _load_runner()
    monkeypatch.setattr(mod, "ContextAssembler", _RecordingAssembler)
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda db_path: _FakeKG())
    monkeypatch.setattr(mod, "ArchiveStore", lambda archive_dir, index_path: _FakeArchive())
    monkeypatch.setattr(mod, "VectorMemory", lambda db_path, embed_mode, onnx_path: _FakeVectorMemory(["chunk"]))
    monkeypatch.setattr(mod, "process_conversation_turn", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "load_dataset", lambda: list(_SYNTHETIC_DATASET))
    recording_client = _RecordingClient("Alice")

    def fake_async_client(*args, **kwargs):
        return recording_client

    monkeypatch.setattr("httpx.AsyncClient", fake_async_client)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_file = str(out_dir / "results.json")

    import argparse as _ap
    args = _ap.Namespace(
        limit=1, type=None, llm=True,
        graph_expansion=0, retrieval_path="retrieve",
        report_retrieval_delta=False, out=out_file,
    )

    import asyncio
    result = asyncio.run(mod.run_benchmark(limit=1, use_llm=True, args=args))

    assert result is None, f"expected None return with NO_INLINE_JUDGE, got {result}"

    with open(out_file) as fh:
        doc = json.load(fh)

    assert "correct" not in doc["metrics"], f"'correct' leaked into metrics: {doc['metrics']}"
    assert "accuracy" not in doc["metrics"], f"'accuracy' leaked into metrics: {doc['metrics']}"
    assert doc["metrics"]["n"] == 1

    captured = capsys.readouterr().out
    assert "Overall:" not in captured, f"'Overall:' line found when NO_INLINE_JUDGE=1:\n{captured}"


# ---------------------------------------------------------------------------
# MUST 3: TAOSMD_LME_GEN_TEMP validation
# ---------------------------------------------------------------------------


def test_gen_temp_empty_defaults_to_zero():
    assert _load_runner()._parse_gen_temp("") == 0.0


def test_gen_temp_non_numeric_defaults_to_zero():
    assert _load_runner()._parse_gen_temp("abc") == 0.0


def test_gen_temp_negative_defaults_to_zero():
    assert _load_runner()._parse_gen_temp("-1") == 0.0


def test_gen_temp_nan_defaults_to_zero():
    assert _load_runner()._parse_gen_temp("nan") == 0.0


def test_gen_temp_inf_defaults_to_zero():
    assert _load_runner()._parse_gen_temp("inf") == 0.0


def test_gen_temp_valid_returned():
    mod = _load_runner()
    assert mod._parse_gen_temp("0.7") == 0.7


# ---------------------------------------------------------------------------
# MUST 4: Byte-identical payloads with defaults
# ---------------------------------------------------------------------------


def test_generation_payload_temperature_is_int_zero(tmp_path, monkeypatch):
    mod = _load_runner()
    monkeypatch.setattr(mod, "ContextAssembler", _RecordingAssembler)
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda db_path: _FakeKG())
    monkeypatch.setattr(mod, "ArchiveStore", lambda archive_dir, index_path: _FakeArchive())
    monkeypatch.setattr(mod, "VectorMemory", lambda db_path, embed_mode, onnx_path: _FakeVectorMemory(["chunk"]))
    monkeypatch.setattr(mod, "process_conversation_turn", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "load_dataset", lambda: list(_SYNTHETIC_DATASET))
    recording_client = _RecordingClient("Alice")

    def fake_async_client(*args, **kwargs):
        return recording_client

    monkeypatch.setattr("httpx.AsyncClient", fake_async_client)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_file = str(out_dir / "results.json")

    import argparse as _ap
    args = _ap.Namespace(
        limit=1, type=None, llm=True,
        graph_expansion=0, retrieval_path="retrieve",
        report_retrieval_delta=False, out=out_file,
    )

    import asyncio
    asyncio.run(mod.run_benchmark(limit=1, use_llm=True, args=args))

    gen_calls = [c for c in recording_client.calls if c.get("json", {}).get("model") == mod.REMOTE_LLM_MODEL]
    assert gen_calls, "expected at least one generation call"
    gen_opts = gen_calls[0]["json"]["options"]
    assert gen_opts["temperature"] == 0, f"generation temperature must be int 0, got {gen_opts['temperature']!r}"
    assert isinstance(gen_opts["temperature"], int), f"generation temperature must be int, got {type(gen_opts['temperature'])}"


def test_judge_payload_unchanged(tmp_path, monkeypatch):
    mod = _load_runner()
    monkeypatch.setattr(mod, "ContextAssembler", _RecordingAssembler)
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda db_path: _FakeKG())
    monkeypatch.setattr(mod, "ArchiveStore", lambda archive_dir, index_path: _FakeArchive())
    monkeypatch.setattr(mod, "VectorMemory", lambda db_path, embed_mode, onnx_path: _FakeVectorMemory(["chunk"]))
    monkeypatch.setattr(mod, "process_conversation_turn", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "load_dataset", lambda: list(_SYNTHETIC_DATASET))
    recording_client = _RecordingClient("CORRECT")

    def fake_async_client(*args, **kwargs):
        return recording_client

    monkeypatch.setattr("httpx.AsyncClient", fake_async_client)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_file = str(out_dir / "results.json")

    import argparse as _ap
    args = _ap.Namespace(
        limit=1, type=None, llm=True,
        graph_expansion=0, retrieval_path="retrieve",
        report_retrieval_delta=False, out=out_file,
    )

    import asyncio
    asyncio.run(mod.run_benchmark(limit=1, use_llm=True, args=args))

    judge_calls = [c for c in recording_client.calls if c.get("json", {}).get("model") == mod.JUDGE_MODEL]
    assert judge_calls, "expected at least one judge call"
    judge_payload = judge_calls[-1]["json"]
    assert judge_payload["options"] == {"temperature": 0, "num_predict": 16}, (
        f"judge payload drifted: {judge_payload['options']}"
    )


# ---------------------------------------------------------------------------
# MUST 5: Tests exercise shipped run_benchmark, not a reimplementation.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MUST 6: Mutation table (included in PR body; this file provides the tests).
# ---------------------------------------------------------------------------


def test_result_doc_has_gen_temp_and_inline_judge(tmp_path, monkeypatch):
    mod = _load_runner()
    monkeypatch.setattr(mod, "ContextAssembler", _RecordingAssembler)
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda db_path: _FakeKG())
    monkeypatch.setattr(mod, "ArchiveStore", lambda archive_dir, index_path: _FakeArchive())
    monkeypatch.setattr(mod, "VectorMemory", lambda db_path, embed_mode, onnx_path: _FakeVectorMemory(["chunk"]))
    monkeypatch.setattr(mod, "process_conversation_turn", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "load_dataset", lambda: list(_SYNTHETIC_DATASET))
    monkeypatch.setattr(mod, "llm_answer", _fake_llm_answer)
    monkeypatch.setattr(mod, "self_verify_answer", _fake_self_verify)
    recording_client = _RecordingClient("Alice")

    def fake_async_client(*args, **kwargs):
        return recording_client

    monkeypatch.setattr("httpx.AsyncClient", fake_async_client)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_file = str(out_dir / "results.json")

    import argparse as _ap
    args = _ap.Namespace(
        limit=1, type=None, llm=True,
        graph_expansion=0, retrieval_path="retrieve",
        report_retrieval_delta=False, out=out_file,
    )

    import asyncio
    asyncio.run(mod.run_benchmark(limit=1, use_llm=True, args=args))

    with open(out_file) as fh:
        doc = json.load(fh)

    assert "gen_temp" in doc, f"'gen_temp' missing from result doc: {list(doc.keys())}"
    assert "inline_judge" in doc, f"'inline_judge' missing from result doc: {list(doc.keys())}"
    assert doc["gen_temp"] == 0.0
    assert doc["inline_judge"] is True


def test_answer_and_gold_persisted(tmp_path, monkeypatch):
    mod = _load_runner()
    monkeypatch.setattr(mod, "ContextAssembler", _RecordingAssembler)
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda db_path: _FakeKG())
    monkeypatch.setattr(mod, "ArchiveStore", lambda archive_dir, index_path: _FakeArchive())
    monkeypatch.setattr(mod, "VectorMemory", lambda db_path, embed_mode, onnx_path: _FakeVectorMemory(["chunk"]))
    monkeypatch.setattr(mod, "process_conversation_turn", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "load_dataset", lambda: list(_SYNTHETIC_DATASET))
    monkeypatch.setattr(mod, "llm_answer", _fake_llm_answer)
    monkeypatch.setattr(mod, "self_verify_answer", _fake_self_verify)
    recording_client = _RecordingClient("Alice")

    def fake_async_client(*args, **kwargs):
        return recording_client

    monkeypatch.setattr("httpx.AsyncClient", fake_async_client)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_file = str(out_dir / "results.json")

    import argparse as _ap
    args = _ap.Namespace(
        limit=1, type=None, llm=True,
        graph_expansion=0, retrieval_path="retrieve",
        report_retrieval_delta=False, out=out_file,
    )

    import asyncio
    asyncio.run(mod.run_benchmark(limit=1, use_llm=True, args=args))

    with open(out_file) as fh:
        doc = json.load(fh)

    first_result = doc["results"][0]
    assert "answer" in first_result, f"'answer' missing from result row: {first_result.keys()}"
    assert "gold_answer" in first_result, f"'gold_answer' missing from result row: {first_result.keys()}"
    assert "question_id" in first_result, f"'question_id' missing from result row: {first_result.keys()}"
    assert first_result["answer"] == "Alice"
    assert first_result["gold_answer"] == "Bob"
    assert first_result["question_id"] == "q-1"


def test_no_inline_judge_correct_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("TAOSMD_LME_NO_INLINE_JUDGE", "1")
    mod = _load_runner()
    monkeypatch.setattr(mod, "ContextAssembler", _RecordingAssembler)
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda db_path: _FakeKG())
    monkeypatch.setattr(mod, "ArchiveStore", lambda archive_dir, index_path: _FakeArchive())
    monkeypatch.setattr(mod, "VectorMemory", lambda db_path, embed_mode, onnx_path: _FakeVectorMemory(["chunk"]))
    monkeypatch.setattr(mod, "process_conversation_turn", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "load_dataset", lambda: list(_SYNTHETIC_DATASET))
    recording_client = _RecordingClient("Alice")

    def fake_async_client(*args, **kwargs):
        return recording_client

    monkeypatch.setattr("httpx.AsyncClient", fake_async_client)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_file = str(out_dir / "results.json")

    import argparse as _ap
    args = _ap.Namespace(
        limit=1, type=None, llm=True,
        graph_expansion=0, retrieval_path="retrieve",
        report_retrieval_delta=False, out=out_file,
    )

    import asyncio
    asyncio.run(mod.run_benchmark(limit=1, use_llm=True, args=args))

    with open(out_file) as fh:
        doc = json.load(fh)

    first_result = doc["results"][0]
    assert first_result.get("correct") is None, (
        f"correct must be None with NO_INLINE_JUDGE, got {first_result.get('correct')!r}"
    )
