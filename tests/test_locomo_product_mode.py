"""``--context product`` in the LoCoMo runner."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from taosmd import api as taosmd_api

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "benchmarks" / "locomo_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("locomo_runner_product", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _words(si, ti, n=240):
    return " ".join(f"w{si}x{ti}x{k}" for k in range(n))


def _conv():
    conv = {"sample_id": "conv-1", "conversation": {"speaker_a": "Ann", "speaker_b": "Bo"}, "qa": []}
    dates = ["1:56 pm on 8 May, 2023", "9:00 am on 20 May, 2023", "6:30 pm on 1 June, 2023"]
    for si, dt in enumerate(dates, start=1):
        conv["conversation"][f"session_{si}_date_time"] = dt
        conv["conversation"][f"session_{si}"] = [
            {"speaker": "Ann" if t % 2 == 0 else "Bo", "dia_id": f"D{si}:{t + 1}",
             "text": _words(si, t)}
            for t in range(8)
        ]
    conv["qa"] = [{"question": "What breed is Rex?", "answer": "beagle", "category": 1,
                   "evidence": ["D1:3", "D3:2"]}]
    return conv


ENV = {"recipe": "r1", "embedder": "fake-embed", "reranker": None, "reranker_available": True,
       "fusion": "boost", "strategy": "thorough", "candidate_top_k": None}


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload

    status_code = 200


class _Client:
    posts: list = []

    def __init__(self, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def aclose(self):
        pass

    async def post(self, url, json=None, **kw):
        _Client.posts.append(json)
        return _Resp({"response": "beagle"})


@pytest.fixture
def mod(monkeypatch):
    for var in ("TAOSMD_CONTEXT_MODE", "TAOSMD_PRODUCT_CONFIG"):
        monkeypatch.delenv(var, raising=False)
    m = _load_runner()
    orig = taosmd_api._ensure_stores

    async def ensure(data_dir=None):
        stores = await orig(data_dir)

        async def fake_embed(text, task="search_document"):
            h = hash(text) & 0xFFFFFFFF
            return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]

        stores["vector"].embed = fake_embed
        return stores

    async def env(data_dir, agent="bench_eval"):
        return dict(ENV)

    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    monkeypatch.setattr(taosmd_api, "_ensure_stores", ensure)
    monkeypatch.setattr(m, "product_environment", env)
    _Client.posts = []
    return m


def _run(mod, monkeypatch, tmp_path, extra=()):
    ds = tmp_path / "ds.json"
    ds.write_text(json.dumps([_conv()]))
    out = tmp_path / "out.json"
    monkeypatch.setattr(mod.httpx, "AsyncClient", _Client)
    args = mod._build_parser().parse_args([
        "--dataset", str(ds), "--conversations", "1", "--no-inline-judge", "--out", str(out),
        "--context", "product", "--reader-ctx-tokens", "100000", "--answer-limit", "24",
        "--concurrency", "1", *extra])
    rc = asyncio.run(mod.run(args))
    return rc, json.loads(out.read_text())


def test_product_run_no_char_cut_guidance_and_records(mod, monkeypatch, tmp_path):
    rc, doc = _run(mod, monkeypatch, tmp_path)
    assert rc == 0
    prompt = _Client.posts[0]["prompt"]
    from taosmd.answer_context import GUIDANCE
    assert prompt.startswith(GUIDANCE)
    assert len(prompt) > 16_000
    assert _Client.posts[0]["options"]["num_ctx"] == 100_000 + 1536
    row = doc["results"][0]
    ctx = prompt[len(GUIDANCE) + 2: prompt.index("\n\nQuestion:")]
    assert row["product"]["context_sha256"] == hashlib.sha256(ctx.encode()).hexdigest()
    assert row["product"]["sessions_dropped"] == 0
    assert row["product"]["tokens_used"] > 0
    # sessions are dated from the LoCoMo timestamps, oldest first, speaker as role
    assert "Session 2023-05-08 (Mon)" in ctx
    assert ctx.index("Session 2023-05-08") < ctx.index("Session 2023-05-20") < ctx.index("Session 2023-06-01")
    assert "Ann: w1x0x0" in ctx
    assert "Current date: 2023-06-01 (Thu)" in ctx
    # both evidence turns reached the reader, and the report says so
    assert row["evidence_hits"] == 2 and row["evidence_dropped"] == 0
    assert [b["session_id"] for b in row["product"]["included"]] == ["session_1", "session_2", "session_3"]
    assert row["product"]["included"][0]["span"] == "whole"
    assert row["product"]["dropped"] == []
    assert doc["meta"]["context_mode"] == "product"
    assert doc["meta"]["reader_ctx_tokens"] == 100_000


def test_product_evidence_dropped_is_reported_under_a_small_budget(mod, monkeypatch, tmp_path):
    ds = tmp_path / "ds.json"
    ds.write_text(json.dumps([_conv()]))
    monkeypatch.setattr(mod.httpx, "AsyncClient", _Client)
    args = mod._build_parser().parse_args([
        "--dataset", str(ds), "--conversations", "1", "--no-inline-judge",
        "--out", str(tmp_path / "o.json"), "--context", "product",
        "--reader-ctx-tokens", "6000", "--answer-limit", "24", "--concurrency", "1"])
    asyncio.run(mod.run(args))
    row = json.loads((tmp_path / "o.json").read_text())["results"][0]
    assert (row["product"]["sessions_included"], row["product"]["sessions_dropped"]) == (1, 2)
    assert row["product"]["tokens_used"] > 0
    assert row["evidence_hits"] + row["evidence_dropped"] == 2
    assert row["evidence_dropped"] >= 1


def test_product_mode_requires_reader_ctx_tokens(mod, tmp_path, capsys):
    ds = tmp_path / "ds.json"
    ds.write_text(json.dumps([_conv()]))
    args = mod._build_parser().parse_args(["--dataset", str(ds), "--context", "product"])
    with pytest.raises(SystemExit) as exc:
        asyncio.run(mod.run(args))
    assert exc.value.code == 2
    assert "--reader-ctx-tokens" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--cove", "--full-context", "--hyde", "--few-shot", "--llm-rerank",
                                  "--emem-edu", "--timeline-format", "--empty-retry"])
def test_product_mode_refuses_levers_it_would_ignore(mod, tmp_path, flag):
    ds = tmp_path / "ds.json"
    ds.write_text(json.dumps([_conv()]))
    args = mod._build_parser().parse_args([
        "--dataset", str(ds), "--context", "product", "--reader-ctx-tokens", "1000", flag])
    with pytest.raises(SystemExit) as exc:
        asyncio.run(mod.run(args))
    assert exc.value.code == 2


def test_unreadable_session_date_is_refused(mod):
    with pytest.raises(ValueError):
        mod._locomo_session_date("session_1", "sometime in spring")


def test_legacy_default_is_unchanged(mod, monkeypatch):
    args = mod._build_parser().parse_args([])
    assert args.context is None and mod.resolve_context_mode(args) == "legacy"


def test_environment_is_recorded(mod, monkeypatch, tmp_path):
    rc, doc = _run(mod, monkeypatch, tmp_path)
    assert doc["meta"]["product_environment"]["embedder"] == "fake-embed"


def test_degraded_ingest_stops_the_run(mod, monkeypatch, tmp_path):
    orig = taosmd_api.ingest

    async def degraded(*a, **k):
        return {**(await orig(*a, **k)), "degraded": True, "vector_failures": 2}

    monkeypatch.setattr(taosmd_api, "ingest", degraded)
    with pytest.raises(RuntimeError, match="not embedded"):
        _run(mod, monkeypatch, tmp_path)


def test_missing_reranker_stops_the_run(mod, monkeypatch, tmp_path):
    async def env(data_dir, agent="bench_eval"):
        return {**ENV, "reranker": "bge-v2-m3", "reranker_available": False}

    monkeypatch.setattr(mod, "product_environment", env)
    with pytest.raises(SystemExit) as exc:
        _run(mod, monkeypatch, tmp_path)
    assert exc.value.code == 1
