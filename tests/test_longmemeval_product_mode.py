"""``--context product`` in the LongMemEval runner: the context comes only from
taosmd.answer_context, with no character cut, and self-verify is off."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

from taosmd import api as taosmd_api

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "benchmarks" / "longmemeval_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("longmemeval_runner_product", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ENV = {"recipe": "r1", "embedder": "fake-embed", "reranker": None, "reranker_available": True,
       "fusion": "boost", "strategy": "thorough", "candidate_top_k": None}


class _Resp:
    status_code = 200

    def __init__(self, content):
        self._content = content

    def json(self):
        return {"message": {"content": self._content}, "prompt_eval_count": 4321}


class _Client:
    def __init__(self):
        self.posts = []

    async def post(self, url, json=None, timeout=30):
        self.posts.append(json)
        return _Resp("Rex is a beagle")

    async def aclose(self):
        pass


def _words(si, ti, n=240):
    return " ".join(f"w{si}x{ti}x{k}" for k in range(n))


def _item(sessions=3, turns=8):
    return {
        "question_id": "q1",
        "question": "What breed is Rex?",
        "question_type": "temporal",
        "answer": "beagle",
        "question_date": "2023/06/10 (Sat) 10:00",
        "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/22 (Mon) 09:00", "2023/06/01 (Thu) 18:30"][:sessions],
        "haystack_sessions": [
            [{"role": "user" if t % 2 == 0 else "assistant", "content": _words(si, t)}
             for t in range(turns)]
            for si in range(sessions)
        ],
    }


@pytest.fixture
def mod(monkeypatch):
    for var in ("TAOSMD_CONTEXT_MODE", "TAOSMD_LME_DATE_FRAME", "TAOSMD_LME_STREAMS",
                "TAOSMD_RERANK", "TAOSMD_DECOMPOSE", "TAOSMD_SELF_VERIFY", "TAOSMD_PRODUCT_CONFIG"):
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
    return m


def _args(tmp_path, **kw):
    base = dict(limit=1, type=None, llm=True, graph_expansion=0, retrieval_path="retrieve",
                report_retrieval_delta=False, out=str(tmp_path / "out.json"), writer_url=None,
                writer_model="default", context="product", reader_ctx_tokens=100_000,
                answer_limit=24, answer_neighbors=None)
    base.update(kw)
    return type("Args", (), base)


async def _run(mod, monkeypatch, tmp_path, client, item=None, **kw):
    monkeypatch.setattr(mod, "load_dataset", lambda: [item or _item()])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **k: client)
    await mod.run_benchmark(args=_args(tmp_path, **kw))
    return json.loads((tmp_path / "out.json").read_text())


@pytest.mark.asyncio
async def test_product_mode_has_no_character_cut_and_uses_guidance(mod, monkeypatch, tmp_path):
    client = _Client()
    doc = await _run(mod, monkeypatch, tmp_path, client)
    prompt = client.posts[0]["messages"][0]["content"]
    assert len(prompt) > mod.CONTEXT_CHARS, "the old 16,000 character cut must not apply"
    from taosmd.answer_context import GUIDANCE
    assert prompt.startswith(GUIDANCE)
    assert "Current date: 2023-06-10 (Sat)" in prompt
    assert "Session 2023-05-20 (Sat)" in prompt
    assert prompt.rstrip().endswith("Answer:")
    # The reader window is the budget the context was built for.
    assert client.posts[0]["options"]["num_ctx"] == 100_000
    # per-question record
    row = doc["results"][0]
    ctx = prompt[len(GUIDANCE) + 2: prompt.index("\n\nQuestion:")]
    assert row["product"]["context_sha256"] == hashlib.sha256(ctx.encode()).hexdigest()
    assert row["product"]["tokens_used"] > 0
    assert row["product"]["sessions_dropped"] == 0
    assert row["context_chars"] == len(ctx) > mod.CONTEXT_CHARS
    assert doc["context_mode"] == "product" and doc["reader_ctx_tokens"] == 100_000


@pytest.mark.asyncio
async def test_small_budget_drops_whole_sessions_and_says_so(mod, monkeypatch, tmp_path):
    client = _Client()
    doc = await _run(mod, monkeypatch, tmp_path, client, reader_ctx_tokens=3_000)
    row = doc["results"][0]["product"]
    assert row["sessions_dropped"] >= 1
    assert row["tokens_used"] <= row["budget_available"] <= 3_000


@pytest.mark.asyncio
async def test_self_verify_is_off_in_product_mode(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "SELF_VERIFY", True)
    client = _Client()
    doc = await _run(mod, monkeypatch, tmp_path, client)
    assert not any("Draft answer:" in p["messages"][0]["content"] for p in client.posts)
    assert doc["self_verify"] is False


def test_product_mode_requires_reader_ctx_tokens(mod, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mod, "load_dataset", lambda: [_item()])
    with pytest.raises(SystemExit) as exc:
        import asyncio
        asyncio.run(mod.run_benchmark(args=_args(tmp_path, reader_ctx_tokens=None)))
    assert exc.value.code == 2
    assert "--reader-ctx-tokens" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [
    {"graph_expansion": 3}, {"retrieval_path": "legacy"}, {"report_retrieval_delta": True},
])
def test_product_mode_refuses_levers_it_would_ignore(mod, monkeypatch, tmp_path, bad):
    import asyncio
    with pytest.raises(SystemExit) as exc:
        asyncio.run(mod.run_benchmark(args=_args(tmp_path, **bad)))
    assert exc.value.code == 2


def test_context_mode_from_env_and_invalid(mod, monkeypatch):
    monkeypatch.setenv("TAOSMD_CONTEXT_MODE", "product")
    assert mod._resolve_context_mode(type("A", (), {})) == "product"
    monkeypatch.setenv("TAOSMD_CONTEXT_MODE", "bogus")
    with pytest.raises(SystemExit):
        mod._resolve_context_mode(type("A", (), {}))
    monkeypatch.delenv("TAOSMD_CONTEXT_MODE")
    assert mod._resolve_context_mode(type("A", (), {"context": None})) == "legacy"


def test_haystack_turns_carry_session_order_and_dates(mod):
    turns = mod.haystack_turns(_item(sessions=2, turns=3))
    assert [(t["session_id"], t["turn_index"]) for t in turns] == [
        ("s0", 0), ("s0", 1), ("s0", 2), ("s1", 0), ("s1", 1), ("s1", 2)]
    assert turns[0]["session_date"] == "2023/05/20 (Sat) 02:21"
    assert turns[3]["session_date"] == "2023/05/22 (Mon) 09:00"
    assert turns[1]["role"] == "assistant"


def test_cli_exposes_the_flags(mod):
    src = RUNNER_PATH.read_text()
    for flag in ("--context", "--reader-ctx-tokens", "--answer-limit", "--answer-neighbors"):
        assert f'"{flag}"' in src


@pytest.mark.asyncio
async def test_environment_and_server_prompt_count_are_recorded(mod, monkeypatch, tmp_path):
    doc = await _run(mod, monkeypatch, tmp_path, _Client())
    assert doc["product_environment"]["embedder"] == "fake-embed"
    assert doc["results"][0]["product"]["prompt_eval_count"] == 4321


@pytest.mark.asyncio
async def test_reader_failure_is_not_scored_as_a_wrong_answer(mod, monkeypatch, tmp_path):
    class Boom(_Client):
        async def post(self, url, json=None, timeout=30):
            raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(mod, "load_dataset", lambda: [_item()])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **k: Boom())
    with pytest.raises(SystemExit) as exc:
        await mod.run_benchmark(args=_args(tmp_path))
    assert exc.value.code == 1
    doc = json.loads((tmp_path / "out.json").read_text())
    assert doc["answer_failures"] == 1
    assert "ReadTimeout" in doc["results"][0]["product"]["answer_error"]


@pytest.mark.asyncio
async def test_reader_timeout_is_configurable(mod, monkeypatch, tmp_path):
    seen = []

    class Spy(_Client):
        async def post(self, url, json=None, timeout=30):
            seen.append(timeout)
            return await super().post(url, json=json, timeout=timeout)

    await _run(mod, monkeypatch, tmp_path, Spy(), reader_timeout=77)
    assert 77.0 in seen


@pytest.mark.asyncio
async def test_degraded_ingest_stops_the_run(mod, monkeypatch, tmp_path):
    orig_ingest = taosmd_api.ingest

    async def degraded(*a, **k):
        r = await orig_ingest(*a, **k)
        return {**r, "degraded": True, "vector_failures": 3}

    monkeypatch.setattr(taosmd_api, "ingest", degraded)
    monkeypatch.setattr(mod, "load_dataset", lambda: [_item()])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **k: _Client())
    with pytest.raises(RuntimeError, match="not embedded"):
        await mod.run_benchmark(args=_args(tmp_path))


@pytest.mark.asyncio
async def test_missing_reranker_stops_the_run(mod, monkeypatch, tmp_path):
    async def env(data_dir, agent="bench_eval"):
        return {**ENV, "reranker": "bge-v2-m3", "reranker_available": False}

    monkeypatch.setattr(mod, "product_environment", env)
    monkeypatch.setattr(mod, "load_dataset", lambda: [_item()])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **k: _Client())
    with pytest.raises(SystemExit) as exc:
        await mod.run_benchmark(args=_args(tmp_path))
    assert exc.value.code == 1


@pytest.mark.asyncio
async def test_included_dropped_and_gold_evidence_are_written_per_question(mod, monkeypatch, tmp_path):
    item = _item()
    item["haystack_session_ids"] = ["sess_a", "sess_b", "sess_c"]
    item["answer_session_ids"] = ["sess_b"]
    item["haystack_sessions"][1][3]["has_answer"] = True
    doc = await _run(mod, monkeypatch, tmp_path, _Client(), item=item)
    p = doc["results"][0]["product"]
    assert {b["session_id"] for b in p["included"]} == {"sess_a", "sess_b", "sess_c"}
    assert p["included"][0]["span"] == "whole" and p["included"][0]["tokens"] > 0
    assert p["included"][1]["turns"] == list(range(8))
    assert p["dropped"] == []
    assert p["evidence"] == {"turns_included": 1, "turns_dropped": 0, "turns_total": 1,
                             "sessions_included": 1, "sessions_dropped": 0, "sessions_total": 1}


@pytest.mark.asyncio
async def test_evidence_in_a_dropped_session_is_reported_as_dropped(mod, monkeypatch, tmp_path):
    item = _item()
    item["haystack_session_ids"] = ["sess_a", "sess_b", "sess_c"]
    item["answer_session_ids"] = ["sess_a", "sess_b", "sess_c"]
    for si in range(3):
        item["haystack_sessions"][si][3]["has_answer"] = True
    doc = await _run(mod, monkeypatch, tmp_path, _Client(), item=item, reader_ctx_tokens=3_000)
    ev = doc["results"][0]["product"]["evidence"]
    assert ev["sessions_dropped"] >= 1
    assert ev["sessions_included"] + ev["sessions_dropped"] == 3
    assert ev["turns_included"] + ev["turns_dropped"] == 3


@pytest.mark.asyncio
async def test_judge_failure_is_counted_not_scored(mod, monkeypatch, tmp_path):
    class JudgeDown(_Client):
        async def post(self, url, json=None, timeout=30):
            prompt = json["messages"][0]["content"]
            if "strict answer evaluator" in prompt:
                raise httpx.ReadTimeout("judge swapping models")
            return await super().post(url, json=json, timeout=timeout)

    monkeypatch.setattr(mod, "load_dataset", lambda: [_item()])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **k: JudgeDown())
    with pytest.raises(SystemExit) as exc:
        await mod.run_benchmark(args=_args(tmp_path))
    assert exc.value.code == 1
    doc = json.loads((tmp_path / "out.json").read_text())
    assert doc["answer_failures"] == 1
    assert "judge_error" in doc["results"][0]["product"]
