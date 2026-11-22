"""Tests for LongMemEval abstention-question scoring.

30 of the 500 LongMemEval questions are abstention questions (question_id
ending "_abs"): the correct behaviour is to say the information is not
available. The runner must not apply the "I don't know" string filter to them
and must always send them to the judge with the abstention prompt, while
non-abstention behaviour stays byte-identical to master.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "benchmarks" / "longmemeval_runner.py"

# Master's (pre-fix) non-abstention judge prompt, byte for byte, used by
# test_non_abstention_prompt_unchanged to pin unchanged behaviour.
MASTER_JUDGE_PROMPT = """You are a strict answer evaluator. Determine if the predicted answer contains the same factual information as the reference answer.

Rules:
- "I don't know" or similar non-answers are ALWAYS incorrect
- The predicted answer must contain the key facts from the reference answer
- Paraphrasing is fine, but the core information must match
- If the predicted answer is vague or generic while the reference is specific, that is INCORRECT

Reply with exactly one word: CORRECT or INCORRECT

Question: {question}
Reference answer: {gold}
Predicted answer: {predicted}

Verdict: /no_think"""


def _load_runner():
    spec = importlib.util.spec_from_file_location("longmemeval_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeVectorMemory:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.adds = []

    async def init(self, http_client=None):
        return None

    async def add(self, text, metadata=None):
        self.adds.append((text, dict(metadata or {})))
        return 1

    async def embed(self, text, task="search_document"):
        return [1.0, 0.0, 0.0]

    async def search(self, query, limit=5, hybrid=True, fusion="boost"):
        return []

    async def close(self):
        return None


class _FakeArchive:
    async def init(self):
        return None

    async def record(self, event_type, data, agent_name=None, app_id=None, summary="", project=None):
        return 1

    async def search_fts(self, term, limit=5):
        return []

    async def close(self):
        return None


class _FakeKG:
    async def init(self):
        return None

    async def close(self):
        return None


class _FakeAssembler:
    def __init__(self, kg=None, archive=None):
        pass

    async def assemble(self, query, depth="auto", max_total_tokens=4000):
        return {"context": ""}


class _CapturingClient:
    """Fake httpx.AsyncClient: records every POST payload, serves canned replies in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.payloads = []

    async def post(self, url, json=None, timeout=30):
        self.payloads.append(json)
        content = self.replies.pop(0) if self.replies else "INCORRECT"
        resp = MagicMock()
        resp.status_code = 200
        resp.json = MagicMock(return_value={"message": {"content": content}})
        return resp

    async def aclose(self):
        return None


def _item(question_id, question, answer, question_type="temporal"):
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": question,
        "answer": answer,
        "haystack_sessions": [[{"role": "user", "content": "Session content."}]],
        "haystack_dates": ["2023/05/20 (Sat) 02:21"],
    }


def _make_args(tmp_path, llm=True, limit=1, out_name="results.json"):
    return type(
        "Args",
        (),
        dict(
            limit=limit,
            type=None,
            llm=llm,
            graph_expansion=0,
            retrieval_path="retrieve",
            report_retrieval_delta=False,
            out=str(tmp_path / out_name),
            writer_url=None,
            writer_model="default",
        ),
    )


def _install_fakes(mod, monkeypatch, client):
    monkeypatch.setattr(mod, "VectorMemory", lambda **kw: _FakeVectorMemory())
    monkeypatch.setattr(mod, "ArchiveStore", lambda **kw: _FakeArchive())
    monkeypatch.setattr(mod, "TemporalKnowledgeGraph", lambda **kw: _FakeKG())
    monkeypatch.setattr(mod, "ContextAssembler", _FakeAssembler)
    monkeypatch.setattr(mod, "process_conversation_turn", AsyncMock(return_value={"method": "regex"}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client)


async def _run_items(mod, monkeypatch, tmp_path, items, replies, limit=None, out_name="results.json"):
    if limit is None:
        limit = len(items)
    monkeypatch.setattr(mod, "load_dataset", lambda: items)
    client = _CapturingClient(replies)
    _install_fakes(mod, monkeypatch, client)
    await mod.run_benchmark(args=_make_args(tmp_path, llm=True, limit=limit, out_name=out_name))
    with open(tmp_path / out_name) as f:
        result = json.load(f)
    return result, client


# ---------------------------------------------------------------------------
# (a) abstention idk answer goes to the abstention judge and is scored correct
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_abstention_idk_goes_to_abstention_judge(monkeypatch, tmp_path):
    mod = _load_runner()
    question = ("Which task did I complete first, fixing the fence or purchasing "
                "three cows from Peter?")
    gold = ("The information provided is not enough. You mentioned fixing the fence "
            "but did not mention purchasing cows from Peter.")
    item = _item("x_abs", question, gold)
    # First reply: reader answer. Second reply: judge verdict.
    replies = ["I don't know, you never mentioned buying cows.", "CORRECT"]
    result, client = await _run_items(mod, monkeypatch, tmp_path, [item], replies)

    # The judge MUST be called: one post for the answer, one for the verdict.
    # On master the string filter skips the judge, so only one post happens.
    assert len(client.payloads) == 2, (
        f"expected 2 posts (answer + judge), got {len(client.payloads)}: judge not called"
    )
    judge_prompt = client.payloads[1]["messages"][0]["content"]
    expected = mod.ABSTENTION_JUDGE_PROMPT.format(
        question=question, gold=gold, predicted="I don't know, you never mentioned buying cows."
    )
    assert judge_prompt == expected
    assert result["results"][0]["correct"] is True


# ---------------------------------------------------------------------------
# (b) abstention specific answer judged INCORRECT
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_abstention_specific_answer_judged(monkeypatch, tmp_path):
    mod = _load_runner()
    question = ("Which task did I complete first, fixing the fence or purchasing "
                "three cows from Peter?")
    gold = ("The information provided is not enough. You mentioned fixing the fence "
            "but did not mention purchasing cows from Peter.")
    item = _item("x_abs", question, gold)
    replies = ["You fixed the fence first.", "INCORRECT"]
    result, client = await _run_items(mod, monkeypatch, tmp_path, [item], replies)

    assert len(client.payloads) == 2, "judge must be called for abstention questions"
    judge_prompt = client.payloads[1]["messages"][0]["content"]
    expected = mod.ABSTENTION_JUDGE_PROMPT.format(
        question=question, gold=gold, predicted="You fixed the fence first."
    )
    assert judge_prompt == expected
    assert result["results"][0]["correct"] is False


# ---------------------------------------------------------------------------
# (c) non-abstention idk answer still filtered, judge not called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_abstention_idk_still_filtered(monkeypatch, tmp_path):
    mod = _load_runner()
    item = _item("y", "When did Alice move to Paris?", "Alice moved to Paris in 2020")
    replies = ["I don't know"]  # judge must NOT be called
    result, client = await _run_items(mod, monkeypatch, tmp_path, [item], replies)

    assert len(client.payloads) == 1, "judge must not be called for a filtered idk answer"
    assert result["results"][0]["correct"] is False


# ---------------------------------------------------------------------------
# (d) non-abstention judge prompt byte-identical to master
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_abstention_prompt_unchanged(monkeypatch, tmp_path):
    mod = _load_runner()
    question = "When did Alice move to Paris?"
    gold = "Alice moved to Paris in 2020"
    item = _item("y", question, gold)
    replies = ["Alice moved to Paris in 2020", "CORRECT"]
    result, client = await _run_items(mod, monkeypatch, tmp_path, [item], replies)

    assert len(client.payloads) == 2, "judge must be called for a substantive answer"
    judge_prompt = client.payloads[1]["messages"][0]["content"]
    expected = MASTER_JUDGE_PROMPT.format(
        question=question, gold=gold, predicted="Alice moved to Paris in 2020"
    )
    assert judge_prompt == expected
    assert result["results"][0]["correct"] is True


# ---------------------------------------------------------------------------
# (e) summary counts abstention questions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summary_counts_abstention(monkeypatch, tmp_path):
    mod = _load_runner()
    items = [
        _item("a1_abs", "q1", "gold1"),
        _item("a2_abs", "q2", "gold2"),
        _item("n1", "q3", "gold3"),
    ]
    replies = [
        "not mentioned", "CORRECT",   # item 1: abstention, correct
        "not available", "CORRECT",   # item 2: abstention, correct
        "Paris", "CORRECT",           # item 3: non-abstention, correct
    ]
    result, _client = await _run_items(mod, monkeypatch, tmp_path, items, replies, limit=3)

    assert result["metrics"]["abstention_total"] == 2
    assert result["metrics"]["abstention_correct"] == 2
