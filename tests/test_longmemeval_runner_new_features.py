"""Tests for the LongMemEval runner's new features (GEN_TEMP, NO_INLINE_JUDGE, result fields).

This file pins the behaviour added to unblock the judge-runner work:

  1. Default substring mode (use_llm=False) still completes and writes rows.
  2. TAOSMD_LME_NO_INLINE_JUDGE=1 suppresses inline scoring: no correct/accuracy
     in metrics, no Overall line, return is None, metrics count n only.
  3. TAOSMD_LME_GEN_TEMP warns on empty, non-numeric, negative, nan and inf and
     falls back to 0; stays silent on unset or a valid value like "0.7".
  4. With TAOSMD_LME_GEN_TEMP=0.7 the real run_benchmark routes 0.7 to the
     generation call and 0 to the judge call.
  5. With TAOSMD_LME_NO_INLINE_JUDGE=1 no judge-prompt call is posted.
  6. The runner loader is hermetic: tests control env vars explicitly before
     loading so the module-level defaults are deterministic.
  7. all_results rows carry question_type, question_id, answer and gold_answer.

The LongMemEval-S / oracle dataset is gitignored and generally absent, so
every test here uses synthetic fixtures and never touches benchmarks/data.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "benchmarks" / "longmemeval_runner.py"


def _load_runner():
    """Load the runner module fresh.

    Tests are responsible for setting or deleting the two new env vars
    before calling this so the module-level defaults are deterministic.
    """
    spec = importlib.util.spec_from_file_location("longmemeval_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _RecordingClient:
    """Fake HTTP client that records every POST call."""

    def __init__(self):
        self.calls = []

    async def post(self, url, json=None, **kwargs):
        self.calls.append({"url": url, "json": json, "kwargs": kwargs})
        return _FakeResponse()

    async def aclose(self):
        pass


class _FakeResponse:
    status_code = 200

    def json(self):
        return {"message": {"content": "fake answer"}}


class _FakeVectorMemory:
    """Minimal vector source with the close() the runner expects."""

    def __init__(self, texts):
        self.texts = texts
        self.calls = []

    async def search(self, query, limit=5, hybrid=True, fusion="boost",
                     project=None, search_agents=None):
        self.calls.append({"query": query, "limit": limit, "fusion": fusion})
        return [
            {"id": i, "text": t, "similarity": 1.0 - i * 0.01, "metadata": {}}
            for i, t in enumerate(self.texts[:limit])
        ]

    async def close(self):
        pass


class _FakeArchive:
    async def search_fts(self, term, limit=5):
        return []

    async def close(self):
        pass


class _FakeKG:
    async def close(self):
        pass


class _RecordingAssembler:
    """Stands in for ContextAssembler; contributes nothing to the context."""

    def __init__(self, kg=None, archive=None):
        pass

    async def assemble(self, query, depth="auto", max_total_tokens=4000):
        return {"context": ""}


@pytest.fixture()
def runner(monkeypatch):
    monkeypatch.delenv("TAOSMD_LME_GEN_TEMP", raising=False)
    monkeypatch.delenv("TAOSMD_LME_NO_INLINE_JUDGE", raising=False)
    mod = _load_runner()
    monkeypatch.setattr(mod, "ContextAssembler", _RecordingAssembler)
    return mod


def _args(**overrides):
    base = dict(
        limit=1,
        type=None,
        llm=False,
        graph_expansion=0,
        retrieval_path="retrieve",
        report_retrieval_delta=False,
        out="",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _install_fake_httpx(monkeypatch, mod, client):
    """Replace httpx in sys.modules so run_benchmark's local import gets our fake."""
    fake_httpx = types.SimpleNamespace(AsyncClient=lambda **kw: client)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)


# ---------------------------------------------------------------------------
# MUST 1: default substring mode still completes and writes rows
# ---------------------------------------------------------------------------

def test_substring_mode_completes_and_writes_rows(runner, monkeypatch, tmp_path):
    """use_llm=False must return, write a result file, and populate all_results."""
    out_path = str(tmp_path / "results.json")
    dataset = [
        {
            "question_type": "temporal",
            "question": "When did X happen?",
            "answer": "2024-01-01",
            "haystack_sessions": [],
        }
    ]
    monkeypatch.setattr(runner, "load_dataset", lambda: dataset)
    monkeypatch.setattr(runner, "_prepare_out_dir", lambda path: os.path.dirname(path))

    result = asyncio.run(runner.run_benchmark(args=_args(limit=1, llm=False, out=out_path)))

    assert isinstance(result, float)
    assert os.path.exists(out_path)
    doc = json.loads(Path(out_path).read_text())
    assert doc["metrics"]["n"] == 1
    assert len(doc["results"]) == 1
    assert doc["results"][0]["question_type"] == "temporal"


# ---------------------------------------------------------------------------
# MUST 2: NO_INLINE_JUDGE=1 suppresses fake scoring
# ---------------------------------------------------------------------------

def test_no_inline_judge_no_fake_score(monkeypatch, tmp_path, capsys):
    """With NO_INLINE_JUDGE=1 there is no correct/accuracy in metrics, no
    Overall line, the return is None, and metrics count n only."""
    monkeypatch.setenv("TAOSMD_LME_NO_INLINE_JUDGE", "1")
    mod = _load_runner()
    monkeypatch.setattr(mod, "ContextAssembler", _RecordingAssembler)

    out_path = str(tmp_path / "results.json")
    dataset = [
        {
            "question_type": "single",
            "question": "Q?",
            "answer": "gold",
            "haystack_sessions": [],
        }
    ]
    monkeypatch.setattr(mod, "load_dataset", lambda: dataset)
    monkeypatch.setattr(mod, "_prepare_out_dir", lambda path: os.path.dirname(path))

    result = asyncio.run(mod.run_benchmark(args=_args(limit=1, llm=True, out=out_path)))

    assert result is None
    doc = json.loads(Path(out_path).read_text())
    assert "correct" not in doc["metrics"], "no correct count when inline judge is off"
    assert "accuracy" not in doc["metrics"], "no accuracy when inline judge is off"
    assert doc["metrics"]["n"] == 1
    captured = capsys.readouterr()
    assert "Overall:" not in captured.out


# ---------------------------------------------------------------------------
# MUST 3: _parse_gen_temp warns on bad values, silent on valid/unset
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MUST 3: _parse_gen_temp warns on bad values, silent on valid/unset
# ---------------------------------------------------------------------------

def test_parse_gen_temp_empty_warns(monkeypatch, capsys):
    monkeypatch.delenv("TAOSMD_LME_GEN_TEMP", raising=False)
    mod = _load_runner()
    result = mod._parse_gen_temp("")
    assert result == 0.0
    assert "TAOSMD_LME_GEN_TEMP" in capsys.readouterr().err


def test_parse_gen_temp_non_numeric_warns(monkeypatch, capsys):
    monkeypatch.delenv("TAOSMD_LME_GEN_TEMP", raising=False)
    mod = _load_runner()
    result = mod._parse_gen_temp("abc")
    assert result == 0.0
    assert "TAOSMD_LME_GEN_TEMP" in capsys.readouterr().err


def test_parse_gen_temp_negative_warns(monkeypatch, capsys):
    monkeypatch.delenv("TAOSMD_LME_GEN_TEMP", raising=False)
    mod = _load_runner()
    result = mod._parse_gen_temp("-1")
    assert result == 0.0
    assert "TAOSMD_LME_GEN_TEMP" in capsys.readouterr().err


def test_parse_gen_temp_nan_warns(monkeypatch, capsys):
    monkeypatch.delenv("TAOSMD_LME_GEN_TEMP", raising=False)
    mod = _load_runner()
    result = mod._parse_gen_temp("nan")
    assert result == 0.0
    assert "TAOSMD_LME_GEN_TEMP" in capsys.readouterr().err


def test_parse_gen_temp_inf_warns(monkeypatch, capsys):
    monkeypatch.delenv("TAOSMD_LME_GEN_TEMP", raising=False)
    mod = _load_runner()
    result = mod._parse_gen_temp("inf")
    assert result == 0.0
    assert "TAOSMD_LME_GEN_TEMP" in capsys.readouterr().err


def test_parse_gen_temp_valid_no_warn(monkeypatch, capsys):
    monkeypatch.delenv("TAOSMD_LME_GEN_TEMP", raising=False)
    mod = _load_runner()
    result = mod._parse_gen_temp("0.7")
    assert result == 0.7
    assert "TAOSMD_LME_GEN_TEMP" not in capsys.readouterr().err


def test_parse_gen_temp_unset_no_warn(monkeypatch, capsys):
    monkeypatch.delenv("TAOSMD_LME_GEN_TEMP", raising=False)
    mod = _load_runner()
    assert mod.GEN_TEMP == 0.0
    assert "TAOSMD_LME_GEN_TEMP" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# MUST 4: GEN_TEMP=0.7 plumbed to generation, judge stays at 0
# ---------------------------------------------------------------------------

def test_gen_temp_0_7_plumbed_to_generation(monkeypatch, tmp_path):
    """With TAOSMD_LME_GEN_TEMP=0.7 the generation call gets temperature 0.7
    and the judge call keeps temperature 0."""
    monkeypatch.setenv("TAOSMD_LME_GEN_TEMP", "0.7")
    monkeypatch.setenv("TAOSMD_JUDGE_MODEL", "judge-model:7b")
    mod = _load_runner()
    monkeypatch.setattr(mod, "ContextAssembler", _RecordingAssembler)

    client = _RecordingClient()
    _install_fake_httpx(monkeypatch, mod, client)

    dataset = [
        {
            "question_type": "single",
            "question": "Q?",
            "answer": "gold",
            "haystack_sessions": [],
        }
    ]
    monkeypatch.setattr(mod, "load_dataset", lambda: dataset)
    monkeypatch.setattr(mod, "_prepare_out_dir", lambda path: os.path.dirname(path))

    asyncio.run(mod.run_benchmark(args=_args(limit=1, llm=True, out=str(tmp_path / "r.json"))))

    gen_calls = [c for c in client.calls if c.get("json", {}).get("model") == mod.REMOTE_LLM_MODEL]
    assert len(gen_calls) >= 1
    assert gen_calls[0]["json"]["options"]["temperature"] == 0.7
    # Judge call uses JUDGE_MODEL and must keep temperature 0
    judge_calls = [c for c in client.calls if c.get("json", {}).get("model") == mod.JUDGE_MODEL]
    if judge_calls:
        assert judge_calls[0]["json"]["options"]["temperature"] == 0


# ---------------------------------------------------------------------------
# MUST 5: NO_INLINE_JUDGE=1 produces no judge calls
# ---------------------------------------------------------------------------

def test_no_inline_judge_no_judge_calls(monkeypatch, tmp_path):
    """With TAOSMD_LME_NO_INLINE_JUDGE=1 only generation calls are posted."""
    monkeypatch.setenv("TAOSMD_LME_NO_INLINE_JUDGE", "1")
    mod = _load_runner()
    monkeypatch.setattr(mod, "ContextAssembler", _RecordingAssembler)

    client = _RecordingClient()
    _install_fake_httpx(monkeypatch, mod, client)

    dataset = [
        {
            "question_type": "single",
            "question": "Q?",
            "answer": "gold",
            "haystack_sessions": [],
        }
    ]
    monkeypatch.setattr(mod, "load_dataset", lambda: dataset)
    monkeypatch.setattr(mod, "_prepare_out_dir", lambda path: os.path.dirname(path))

    asyncio.run(mod.run_benchmark(args=_args(limit=1, llm=True, out=str(tmp_path / "r.json"))))

    for call in client.calls:
        if "model" not in call.get("json", {}):
            continue
        assert call["json"]["model"] == mod.REMOTE_LLM_MODEL, (
            f"unexpected judge call with model {call['json']['model']}"
        )


# ---------------------------------------------------------------------------
# R1: question_type, question_id, answer, gold_answer persisted
# ---------------------------------------------------------------------------

def test_answer_and_gold_persisted(runner, monkeypatch, tmp_path):
    """all_results rows carry question_type, question_id, answer and gold_answer."""
    out_path = str(tmp_path / "results.json")
    dataset = [
        {
            "question_type": "temporal",
            "question": "Q?",
            "answer": "gold1",
            "haystack_sessions": [],
        },
        {
            "question_id": "q-42",
            "question_type": "single",
            "question": "Q2?",
            "answer": "gold2",
            "haystack_sessions": [],
        },
    ]
    monkeypatch.setattr(runner, "load_dataset", lambda: dataset)
    monkeypatch.setattr(runner, "_prepare_out_dir", lambda path: os.path.dirname(path))

    asyncio.run(runner.run_benchmark(args=_args(limit=2, llm=False, out=out_path)))

    doc = json.loads(Path(out_path).read_text())
    rows = doc["results"]
    assert rows[0]["question_type"] == "temporal"
    assert rows[0]["question_id"] == 0
    assert rows[0]["answer"] == ""
    assert rows[0]["gold_answer"] == "gold1"
    assert rows[1]["question_type"] == "single"
    assert rows[1]["question_id"] == "q-42"
    assert rows[1]["answer"] == ""
    assert rows[1]["gold_answer"] == "gold2"


# ---------------------------------------------------------------------------
# R5: hermetic loader defaults
# ---------------------------------------------------------------------------

def test_runner_loader_hermetic_defaults(monkeypatch):
    """Both env vars must default to off after a clean load."""
    monkeypatch.delenv("TAOSMD_LME_GEN_TEMP", raising=False)
    monkeypatch.delenv("TAOSMD_LME_NO_INLINE_JUDGE", raising=False)
    mod = _load_runner()
    assert mod.GEN_TEMP == 0.0
    assert mod.NO_INLINE_JUDGE is False
