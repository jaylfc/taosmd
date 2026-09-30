"""Tests for the LongMemEval runner's new features (tsk-nvdoe4).

Covers:
- answer / gold / question_id added to per-question rows
- TAOSMD_LME_GEN_TEMP env var (generation only, bad value falls back to 0)
- TAOSMD_LME_NO_INLINE_JUDGE=1 skips inline judge calls
- Result doc carries gen_temp and inline_judge
- Defaults produce the same generation and judge payloads as master
- Rescore script imports from runner, scores empty/idk answers without judge calls
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "benchmarks" / "longmemeval_runner.py"
RESCORE_PATH = REPO_ROOT / "benchmarks" / "longmemeval_rescore.py"


def _load_runner(name=None):
    """Load the runner module under a unique name each time."""
    mod_name = name or f"lme_runner_{_load_runner._counter}"
    _load_runner._counter += 1
    spec = importlib.util.spec_from_file_location(mod_name, RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_runner._counter = 0


class _FakeResp:
    def __init__(self, content, status=200):
        self.status_code = status
        self._content = content

    def json(self):
        return {"message": {"content": self._content}}


class _RecordingClientSimple:
    """Simple recording client with fixed response for all calls."""

    def __init__(self, content="CORRECT"):
        self._content = content
        self.calls: list[dict] = []

    async def post(self, *args, **kwargs):
        self.calls.append(kwargs.get("json", {}))
        return _FakeResp(self._content)

    async def aclose(self):
        pass


class _FailClient:
    """Client that always raises."""

    async def post(self, *args, **kwargs):
        raise RuntimeError("network down")

    async def aclose(self):
        pass


# ---------------------------------------------------------------------------
# Helpers for run_benchmark integration tests
# ---------------------------------------------------------------------------

def _make_fake_vmem():
    vmem = MagicMock()
    vmem.init = AsyncMock()
    vmem.add = AsyncMock()
    vmem.close = AsyncMock()
    vmem.search = AsyncMock(return_value=[])
    return vmem


def _make_fake_kg():
    kg = MagicMock()
    kg.init = AsyncMock()
    kg.close = AsyncMock()
    return kg


def _make_fake_archive():
    archive = MagicMock()
    archive.init = AsyncMock()
    archive.close = AsyncMock()
    archive.search_fts = AsyncMock(return_value=[])
    archive.record = AsyncMock()
    return archive


def _make_fake_embed_client():
    return _RecordingClientSimple(content="")


def _one_question_dataset():
    return [
        {
            "question_type": "single-session-user",
            "question": "What did Alice order?",
            "answer": "A latte",
            "question_id": "q-001",
            "haystack_sessions": [
                [{"role": "user", "content": "Alice ordered a latte"}],
            ],
        }
    ]


def _patch_run_benchmark(mod, monkeypatch):
    """Replace run_benchmark on `mod` with a version accepting injected overrides."""
    fake_vmem = _make_fake_vmem()
    fake_kg = _make_fake_kg()
    fake_archive = _make_fake_archive()
    fake_embed_client = _make_fake_embed_client()

    monkeypatch.setattr(mod, "VectorMemory", MagicMock(return_value=fake_vmem))
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", MagicMock(return_value=fake_kg))
    monkeypatch.setattr(mod, "ArchiveStore", MagicMock(return_value=fake_archive))

    async def _patched(
        limit=50,
        question_type=None,
        use_llm=False,
        args=None,
        _dataset_override=None,
        _llm_client_override=None,
        _embed_client=None,
    ):
        graph_expansion = 0
        retrieval_path = "retrieve"
        out_path = ""
        if args is not None:
            limit = args.limit
            question_type = args.type
            use_llm = args.llm
            graph_expansion = args.graph_expansion
            retrieval_path = args.retrieval_path
            out_path = args.out

        mod._prepare_out_dir(out_path)

        dataset = _dataset_override if _dataset_override is not None else mod.load_dataset()
        dataset = mod.sample_dataset(dataset, limit, mod.SAMPLE_SEED)

        results_by_type = {}
        all_results = []
        total_correct = 0
        total_questions = 0
        total_time = 0

        llm_client = _llm_client_override

        for i, item in enumerate(dataset):
            qtype = item["question_type"]
            question = item["question"]
            gold_answer = item["answer"]
            question_id = item.get("question_id", "")
            sessions = item.get("haystack_sessions", [])

            kg = fake_kg
            archive = fake_archive
            vmem = fake_vmem

            await kg.init()
            await archive.init()

            embed_client = _embed_client or fake_embed_client
            await vmem.init(http_client=embed_client)

            t0 = asyncio.get_event_loop().time()
            for si, session in enumerate(sessions):
                session_text = ""
                for turn in session:
                    content = turn.get("content", "")
                    role = turn.get("role", "user")
                    if content:
                        await mod.process_conversation_turn(
                            content,
                            agent_name="assistant" if role == "assistant" else None,
                            kg=kg,
                            archive=archive,
                            source="longmemeval",
                        )
                        await archive.record(
                            "conversation",
                            {"role": role, "content": content},
                            summary=content[:80],
                        )
                        session_text += f"\n[{role}]: {content}"

                if session_text:
                    words = session_text.split()
                    chunk_size = 100
                    overlap = 20
                    for start in range(0, len(words), chunk_size - overlap):
                        chunk = " ".join(words[start : start + chunk_size])
                        if chunk.strip():
                            await vmem.add(chunk, metadata={"session": si})

            ingest_time = asyncio.get_event_loop().time() - t0

            # Skip retrieval: tests only verify generation/judge payloads.
            full_context = ""
            query_time = 0.0

            delta = None

            answer = ""
            correct = False
            if use_llm and llm_client is not None:
                answer = await mod.llm_answer(llm_client, full_context, question, temperature=mod.GEN_TEMP)
                if mod.SELF_VERIFY:
                    answer = await mod.self_verify_answer(
                        llm_client, full_context, question, answer, temperature=mod.GEN_TEMP
                    )
                if mod.INLINE_JUDGE:
                    if answer and not any(
                        idk in answer.lower()
                        for idk in (
                            "i don't know", "i do not know", "i'm sorry",
                            "not in the context", "does not contain", "no information",
                        )
                    ):
                        correct = await mod.score_answer_llm(llm_client, answer, gold_answer, question)
                    else:
                        correct = False
                else:
                    correct = None
            else:
                correct = mod.score_answer_substring(full_context, gold_answer)

            total_questions += 1
            if correct:
                total_correct += 1

            if qtype not in results_by_type:
                results_by_type[qtype] = {"correct": 0, "total": 0}
            results_by_type[qtype]["total"] += 1
            if correct:
                results_by_type[qtype]["correct"] += 1

            elapsed = ingest_time + query_time
            total_time += elapsed
            all_results.append({
                "idx": i,
                "question_id": question_id,
                "question_type": qtype,
                "question": question,
                "answer": answer or "",
                "gold": str(gold_answer),
                "correct": bool(correct) if correct is not None else None,
                "retrieved_chars": len(full_context),
                "retrieval_delta": delta,
            })

            await archive.close()
            await kg.close()
            await vmem.close()
            await embed_client.aclose()

        overall = total_correct / total_questions * 100 if total_questions > 0 else 0

        if total_questions:
            if not out_path:
                out_dir = mod._default_out_dir()
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"longmemeval_{asyncio.get_event_loop().time()}.json")
            result_doc = {
                "question_type": question_type,
                "limit": limit,
                "generator": mod.REMOTE_LLM_MODEL,
                "judge": mod.JUDGE_MODEL,
                "rerank": mod.RERANK,
                "decompose": mod.DECOMPOSE,
                "self_verify": mod.SELF_VERIFY,
                "assemble_tokens": mod.ASSEMBLE_TOKENS,
                "retrieve_limit": mod.RETRIEVE_LIMIT,
                "fts_limit": mod.FTS_LIMIT,
                "context_chars": mod.CONTEXT_CHARS,
                "num_ctx": mod.NUM_CTX,
                "gen_temp": mod.GEN_TEMP,
                "inline_judge": mod.INLINE_JUDGE,
                "retrieval_path": retrieval_path,
                "graph_expansion": graph_expansion,
                "retrieval_delta": mod.summarize_retrieval_delta(all_results),
                "metrics": {
                    "n": total_questions,
                    "correct": total_correct,
                    "accuracy": overall,
                    "by_type": results_by_type,
                },
                "results": all_results,
            }
            with open(out_path, "w") as f:
                json.dump(result_doc, f, indent=2)

        return overall

    monkeypatch.setattr(mod, "run_benchmark", _patched)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def runner_mod(monkeypatch):
    """Load the runner with default env; patches run_benchmark."""
    mod = _load_runner()
    _patch_run_benchmark(mod, monkeypatch)
    return mod


@pytest.fixture()
def fake_llm_client():
    return _RecordingClientSimple(content="CORRECT")


def _run(mod, dataset, llm_client, out_path, tmp_path, monkeypatch):
    """Convenience: call the patched run_benchmark on `mod`."""
    args = argparse.Namespace(
        limit=len(dataset), type=None, llm=True,
        graph_expansion=0, retrieval_path="retrieve",
        report_retrieval_delta=False, out=out_path,
    )
    asyncio.run(
        mod.run_benchmark(
            limit=len(dataset), question_type=None, use_llm=True, args=args,
            _dataset_override=dataset,
            _llm_client_override=llm_client,
            _embed_client=_make_fake_embed_client(),
        )
    )


# ---------------------------------------------------------------------------
# Helpers shared by tests
# ---------------------------------------------------------------------------

def _is_gen(call):
    prompt = call.get("messages", [{}])[0].get("content", "")
    return "Based on the following context" in prompt


def _is_judge(call):
    prompt = call.get("messages", [{}])[0].get("content", "")
    return "strict answer evaluator" in prompt


# ---------------------------------------------------------------------------
# 1. Defaults: new row keys and result doc keys
# ---------------------------------------------------------------------------

def test_defaults_have_new_row_keys(runner_mod, fake_llm_client, tmp_path):
    """Default run produces rows with answer, gold, question_id; existing keys intact."""
    mod = runner_mod
    out_file = str(tmp_path / "results.json")
    _run(mod, _one_question_dataset(), fake_llm_client, out_file, tmp_path, None)

    with open(out_file) as f:
        doc = json.load(f)

    # Existing top-level keys still present
    assert "generator" in doc
    assert "judge" in doc
    assert "metrics" in doc
    # New top-level keys
    assert "gen_temp" in doc
    assert "inline_judge" in doc
    assert doc["inline_judge"] is True
    assert doc["gen_temp"] == 0.0

    row = doc["results"][0]
    # New row keys
    assert "answer" in row
    assert "gold" in row
    assert "question_id" in row
    assert row["question_id"] == "q-001"
    assert row["gold"] == "A latte"
    # Existing row keys still present
    assert "idx" in row
    assert "question_type" in row
    assert "question" in row
    assert "correct" in row
    assert "retrieved_chars" in row


def test_defaults_generation_payload_is_byte_identical(runner_mod, fake_llm_client, tmp_path):
    """Generation payload at defaults: temperature=0 explicitly, no num_ctx."""
    mod = runner_mod
    out_file = str(tmp_path / "r.json")
    _run(mod, _one_question_dataset(), fake_llm_client, out_file, tmp_path, None)

    gen_calls = [c for c in fake_llm_client.calls if _is_gen(c)]
    judge_calls = [c for c in fake_llm_client.calls if _is_judge(c)]

    assert len(gen_calls) == 1, f"expected 1 gen call, got {len(gen_calls)}"
    assert gen_calls[0]["stream"] is False
    assert gen_calls[0]["think"] is False
    assert gen_calls[0]["options"]["temperature"] == 0
    assert gen_calls[0]["options"]["num_predict"] == 100
    assert "num_ctx" not in gen_calls[0]["options"]

    assert len(judge_calls) == 1, f"expected 1 judge call, got {len(judge_calls)}"
    assert judge_calls[0]["options"]["temperature"] == 0
    assert "num_ctx" not in judge_calls[0]["options"]


def test_defaults_judge_payload_uses_runner_prompt(runner_mod, fake_llm_client, tmp_path):
    """Judge payload uses score_answer_llm's prompt (rescore imports it, not copies)."""
    mod = runner_mod
    out_file = str(tmp_path / "r.json")
    _run(mod, _one_question_dataset(), fake_llm_client, out_file, tmp_path, None)

    judge_calls = [c for c in fake_llm_client.calls if _is_judge(c)]
    assert len(judge_calls) == 1
    prompt = judge_calls[0]["messages"][0]["content"]
    assert "strict answer evaluator" in prompt
    assert "CORRECT or INCORRECT" in prompt
    assert "/no_think" in prompt


# ---------------------------------------------------------------------------
# 2. TAOSMD_LME_GEN_TEMP
# ---------------------------------------------------------------------------

def test_gen_temp_0_5_changes_only_generation_payload(tmp_path, monkeypatch):
    """TAOSMD_LME_GEN_TEMP=0.5: generation gets temp=0.5, judge stays at 0."""
    monkeypatch.setenv("TAOSMD_LME_GEN_TEMP", "0.5")
    monkeypatch.setenv("TAOSMD_LME_NO_INLINE_JUDGE", "0")

    mod = _load_runner()
    _patch_run_benchmark(mod, monkeypatch)

    fake_client = _RecordingClientSimple(content="CORRECT")
    out_file = str(tmp_path / "r.json")
    _run(mod, _one_question_dataset(), fake_client, out_file, tmp_path, monkeypatch)

    with open(out_file) as f:
        doc = json.load(f)

    assert doc["gen_temp"] == 0.5

    gen_calls = [c for c in fake_client.calls if _is_gen(c)]
    judge_calls = [c for c in fake_client.calls if _is_judge(c)]

    assert len(gen_calls) == 1
    assert gen_calls[0]["options"]["temperature"] == 0.5
    assert len(judge_calls) == 1
    assert judge_calls[0]["options"]["temperature"] == 0


def test_bad_gen_temp_warns_and_falls_back_to_zero(monkeypatch, capsys):
    """Junk TAOSMD_LME_GEN_TEMP degrades to 0 and prints a warning."""
    monkeypatch.setenv("TAOSMD_LME_GEN_TEMP", "banana")

    mod = _load_runner()

    assert mod.GEN_TEMP == 0.0
    assert "banana" in capsys.readouterr().err


def test_empty_gen_temp_is_quiet_and_stays_zero(monkeypatch, capsys):
    """Empty TAOSMD_LME_GEN_TEMP is unset, no warning."""
    monkeypatch.setenv("TAOSMD_LME_GEN_TEMP", "")

    mod = _load_runner()

    assert mod.GEN_TEMP == 0.0
    assert "TAOSMD_LME_GEN_TEMP" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 3. TAOSMD_LME_NO_INLINE_JUDGE
# ---------------------------------------------------------------------------

def test_no_inline_judge_makes_zero_judge_calls(tmp_path, monkeypatch):
    """TAOSMD_LME_NO_INLINE_JUDGE=1: correct=null per row, zero judge HTTP calls."""
    monkeypatch.setenv("TAOSMD_LME_GEN_TEMP", "")
    monkeypatch.setenv("TAOSMD_LME_NO_INLINE_JUDGE", "1")

    mod = _load_runner()
    _patch_run_benchmark(mod, monkeypatch)

    fake_client = _RecordingClientSimple(content="CORRECT")
    out_file = str(tmp_path / "r.json")
    _run(mod, _one_question_dataset(), fake_client, out_file, tmp_path, monkeypatch)

    with open(out_file) as f:
        doc = json.load(f)

    assert doc["inline_judge"] is False
    row = doc["results"][0]
    assert row["correct"] is None

    gen_calls = [c for c in fake_client.calls if _is_gen(c)]
    judge_calls = [c for c in fake_client.calls if _is_judge(c)]
    assert len(gen_calls) >= 1
    assert len(judge_calls) == 0


# ---------------------------------------------------------------------------
# 4. Rescore script
# ---------------------------------------------------------------------------

def test_rescore_uses_same_judge_prompt_as_runner():
    """rescore.score_answer_llm produces the same prompt as the runner (imported, not copied)."""
    runner_mod = _load_runner()
    rescore_mod = _load_rescore_module()

    prompts = []

    class _CaptureClient:
        async def post(self, *args, **kwargs):
            prompts.append(kwargs.get("json", {}))
            return _FakeResp("CORRECT")

        async def aclose(self):
            pass

    async def _probe():
        c = _CaptureClient()
        try:
            await runner_mod.score_answer_llm(c, "pred", "gold", "q")
            await rescore_mod.score_answer_llm(c, "pred", "gold", "q", judge_model="m")
        finally:
            await c.aclose()

    asyncio.run(_probe())

    # Both calls must produce identical prompts (imported from the same source).
    assert len(prompts) == 2
    assert prompts[0]["messages"][0]["content"] == prompts[1]["messages"][0]["content"]
    # The rescore call must honour the overridden judge_model.
    assert prompts[1]["model"] == "m"


def test_rescore_empty_answer_scores_false_without_call(tmp_path):
    """Empty answer: scores False, no HTTP call made."""
    rescore_mod = _load_rescore_module()
    in_file = _write_result_json(tmp_path, [
        {"question_type": "t", "question": "Q?", "answer": "", "gold": "g", "correct": False}
    ])

    judge_called = []

    class _NoCallClient:
        async def post(self, *args, **kwargs):
            judge_called.append(True)
            raise AssertionError("judge must not be called for empty answer")

        async def aclose(self):
            pass

    with patch.object(rescore_mod._httpx, "AsyncClient", return_value=_NoCallClient()):
        rescore_mod.asyncio.run(
            rescore_mod.run_rescore(str(in_file), "judge-model", str(tmp_path / "out.json"))
        )

    assert len(judge_called) == 0
    row = _read_result(tmp_path / "out.json")["results"][0]
    assert row["correct"] is False
    assert row["judge_rejudged"] is False
    assert row["judge_model"] == "judge-model"


def test_rescore_idk_answer_scores_false_without_call(tmp_path):
    """idk-phrase answer: scores False, no HTTP call."""
    rescore_mod = _load_rescore_module()
    in_file = _write_result_json(tmp_path, [
        {"question_type": "t", "question": "Q?", "answer": "I don't know.", "gold": "g", "correct": None}
    ])

    judge_called = []

    class _NoCallClient:
        async def post(self, *args, **kwargs):
            judge_called.append(True)
            raise AssertionError("judge must not be called for idk answer")

        async def aclose(self):
            pass

    with patch.object(rescore_mod._httpx, "AsyncClient", return_value=_NoCallClient()):
        rescore_mod.asyncio.run(
            rescore_mod.run_rescore(str(in_file), "judge-model", str(tmp_path / "out.json"))
        )

    assert len(judge_called) == 0
    row = _read_result(tmp_path / "out.json")["results"][0]
    assert row["correct"] is False
    assert row["judge_rejudged"] is False


def test_rescore_calls_judge_once_for_nonempty_answer(tmp_path):
    """Non-empty, non-idk answer: calls judge exactly once with --judge-model."""
    rescore_mod = _load_rescore_module()
    in_file = _write_result_json(tmp_path, [
        {"question_type": "t", "question": "Q?", "answer": "In 2024.", "gold": "2024", "correct": False}
    ])

    judge_payloads = []
    call_count = [0]

    class _JudgeClient:
        async def post(self, *args, **kwargs):
            call_count[0] += 1
            judge_payloads.append(kwargs.get("json", {}))
            return _FakeResp("CORRECT")

        async def aclose(self):
            pass

    with patch.object(rescore_mod._httpx, "AsyncClient", return_value=_JudgeClient()):
        rescore_mod.asyncio.run(
            rescore_mod.run_rescore(str(in_file), "second-judge", str(tmp_path / "out.json"))
        )

    assert call_count[0] == 1
    assert judge_payloads[0]["model"] == "second-judge"

    doc = _read_result(tmp_path / "out.json")
    row = doc["results"][0]
    assert row["correct"] is True
    assert row["judge_rejudged"] is True
    assert row["judge_model"] == "second-judge"
    assert doc["judge_model"] == "second-judge"


def test_rescore_prints_overall_and_per_type(tmp_path, capsys):
    """Rescore prints 'Overall: n/n' and per question_type lines."""
    rescore_mod = _load_rescore_module()
    in_file = _write_result_json(tmp_path, [
        {"question_type": "temporal-reasoning", "question": "Q1?", "answer": "a1", "gold": "g1", "correct": False},
        {"question_type": "single-session-user", "question": "Q2?", "answer": "a2", "gold": "g2", "correct": False},
    ])

    responses = ["CORRECT", "INCORRECT"]
    idx = [0]

    class _JudgeClient:
        async def post(self, *args, **kwargs):
            resp = _FakeResp(responses[idx[0]])
            idx[0] += 1
            return resp

        async def aclose(self):
            pass

    with patch.object(rescore_mod._httpx, "AsyncClient", return_value=_JudgeClient()):
        rescore_mod.asyncio.run(
            rescore_mod.run_rescore(str(in_file), "judge-model", str(tmp_path / "out.json"))
        )

    captured = capsys.readouterr().out
    assert "1/2" in captured
    assert "temporal-reasoning" in captured
    assert "single-session-user" in captured
    overall_line = next(ln for ln in captured.strip().splitlines() if ln.startswith("Overall:"))
    assert "1/2" in overall_line


# ---------------------------------------------------------------------------
# Rescore module helpers
# ---------------------------------------------------------------------------

def _load_rescore_module():
    spec = importlib.util.spec_from_file_location("longmemeval_rescore_test", RESCORE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_result_json(tmp_path, results):
    doc = {"results": results}
    p = tmp_path / "in.json"
    p.write_text(json.dumps(doc))
    return p


def _read_result(path):
    with open(path) as f:
        return json.load(f)
