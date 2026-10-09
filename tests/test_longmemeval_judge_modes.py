"""Tests for LongMemEval judge modes and context saving (A5 harness, E-036).

The runner's LLM judge prompt (score_answer_llm, line 165 on master) sees only
question, gold, and prediction. MemStrata (arXiv 2610.05343) reports that a judge
which also sees the retrieved evidence swings scores by up to 12.7 points on
LoCoMo and 2.4 on LongMemEval. These tests pin the two judge modes and the
context-saving flag so re-judging past runs becomes possible.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "benchmarks" / "longmemeval_runner.py"
REJUDGE_PATH = REPO_ROOT / "benchmarks" / "longmemeval_rejudge.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("longmemeval_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_rejudge():
    spec = importlib.util.spec_from_file_location("longmemeval_rejudge", REJUDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _StubHTTPXClient:
    """Records the prompt sent to the judge and returns a canned verdict."""

    def __init__(self, verdict: str = "CORRECT"):
        self.verdict = verdict
        self.last_prompt = ""
        self.call_count = 0

    async def post(self, url, json=None, timeout=None):
        self.call_count += 1
        self.last_prompt = json["messages"][0]["content"] if json and "messages" in json else ""

        class _Resp:
            status_code = 200

            def json(self):
                return {"message": {"content": self.verdict}}

        return _Resp()

    async def aclose(self):
        pass


# The pinned reference prompt for a fixed (question, gold, predicted) triple.
# This is the EXACT string produced by the current score_answer_llm (reference mode).
FIXED_QUESTION = "When did Alice meet Bob?"
FIXED_GOLD = "Alice met Bob on Tuesday."
FIXED_PREDICTED = "Alice met Bob on Tuesday."
REFERENCE_PROMPT = f"""You are a strict answer evaluator. Determine if the predicted answer contains the same factual information as the reference answer.

Rules:
- "I don't know" or similar non-answers are ALWAYS incorrect
- The predicted answer must contain the key facts from the reference answer
- Paraphrasing is fine, but the core information must match
- If the predicted answer is vague or generic while the reference is specific, that is INCORRECT

Reply with exactly one word: CORRECT or INCORRECT

Question: {FIXED_QUESTION}
Reference answer: {FIXED_GOLD}
Predicted answer: {FIXED_PREDICTED}

Verdict: /no_think"""


class _FakeVectorMemory:
    def __init__(self, texts):
        self.texts = texts

    async def search(self, query, limit=5, hybrid=True, fusion="boost", project=None, search_agents=None):
        return [
            {"id": i, "text": t, "similarity": 1.0 - i * 0.01, "metadata": {}}
            for i, t in enumerate(self.texts[:limit])
        ]


class _FakeArchive:
    async def search_fts(self, term, limit=5):
        return []


class _FakeKG:
    pass


class _RecordingAssembler:
    def __init__(self, kg=None, archive=None):
        pass

    async def assemble(self, query, depth="auto", max_total_tokens=4000):
        return {"context": "assembled context from ContextAssembler"}


@pytest.fixture()
def runner(monkeypatch):
    mod = _load_runner()
    monkeypatch.setattr(mod, "ContextAssembler", _RecordingAssembler)
    return mod


def _context(mod, **kwargs):
    vmem = kwargs.pop("vmem", None) or _FakeVectorMemory(["alpha chunk", "beta chunk"])
    return asyncio.run(
        mod.retrieve_context(
            "who met the captain", _FakeKG(), _FakeArchive(), vmem, **kwargs
        )
    ), vmem


# ---------------------------------------------------------------------------
# 1. Reference mode prompt is byte-identical to today's prompt
# ---------------------------------------------------------------------------

def test_reference_mode_prompt_is_pinned(runner, monkeypatch):
    """The reference judge prompt must be byte-identical to the current one."""
    mod = runner

    async def fake_retrieve(query, **kwargs):
        return [{"text": "alpha chunk"}]

    monkeypatch.setattr(mod, "_retrieve", fake_retrieve)

    client = _StubHTTPXClient(verdict="CORRECT")

    asyncio.run(mod.score_answer_llm(client, FIXED_PREDICTED, FIXED_GOLD, FIXED_QUESTION))

    assert client.last_prompt == REFERENCE_PROMPT, (
        "Reference mode prompt must be byte-identical to the pinned string. "
        f"Got:\n{client.last_prompt}\n\nExpected:\n{REFERENCE_PROMPT}"
    )


def test_evidence_mode_prompt_contains_context_and_extra_rule(runner, monkeypatch):
    """Evidence mode adds the context section and the extra rule."""
    mod = runner

    async def fake_retrieve(query, **kwargs):
        return [{"text": "alpha chunk"}]

    monkeypatch.setattr(mod, "_retrieve", fake_retrieve)

    client = _StubHTTPXClient(verdict="CORRECT")
    context = "This is the retrieved context passed to the generator."

    # Call with evidence mode by setting the env var
    monkeypatch.setenv("TAOSMD_JUDGE_MODE", "evidence")
    try:
        asyncio.run(mod.score_answer_llm(client, FIXED_PREDICTED, FIXED_GOLD, FIXED_QUESTION, context=context))
    finally:
        monkeypatch.delenv("TAOSMD_JUDGE_MODE", raising=False)

    prompt = client.last_prompt
    # Must contain the context section
    assert "Retrieved context the answer was generated from:" in prompt
    assert context in prompt
    # Must contain the extra rule
    assert "Judge against the reference answer; the context is there so you can tell a grounded paraphrase from an unsupported guess, it never overrides the reference" in prompt
    # Must still have the original rules
    assert '"I don\'t know" or similar non-answers are ALWAYS incorrect' in prompt
    # Must still require one-word verdict
    assert "Reply with exactly one word: CORRECT or INCORRECT" in prompt


def test_evidence_mode_defaults_to_reference_when_env_unset(runner, monkeypatch):
    """Without TAOSMD_JUDGE_MODE=evidence, the prompt is reference (no context)."""
    mod = runner

    async def fake_retrieve(query, **kwargs):
        return [{"text": "alpha chunk"}]

    monkeypatch.setattr(mod, "_retrieve", fake_retrieve)

    client = _StubHTTPXClient(verdict="CORRECT")
    context = "This context should NOT appear in reference mode."

    # Do NOT set TAOSMD_JUDGE_MODE
    asyncio.run(mod.score_answer_llm(client, FIXED_PREDICTED, FIXED_GOLD, FIXED_QUESTION, context=context))

    prompt = client.last_prompt
    assert "Retrieved context the answer was generated from:" not in prompt
    assert context not in prompt
    assert prompt == REFERENCE_PROMPT


def test_parse_verdict_unchanged(runner):
    """_parse_verdict contract unchanged: INCORRECT checked first, empty=incorrect."""
    mod = runner
    assert mod._parse_verdict("CORRECT") is True
    assert mod._parse_verdict("INCORRECT") is False
    assert mod._parse_verdict("correct") is True
    assert mod._parse_verdict("incorrect") is False
    assert mod._parse_verdict("CORRECT!") is True
    assert mod._parse_verdict("INCORRECT because") is False
    assert mod._parse_verdict("") is False
    assert mod._parse_verdict("   ") is False
    assert mod._parse_verdict(None) is False


# ---------------------------------------------------------------------------
# 2. SAVE_CONTEXT flag controls presence of "context" key in result rows
# ---------------------------------------------------------------------------

def test_save_context_env_adds_context_to_result_row(runner, monkeypatch, tmp_path):
    """With TAOSMD_LME_SAVE_CONTEXT=1, each result row has a 'context' key."""
    mod = runner
    monkeypatch.setenv("TAOSMD_LME_SAVE_CONTEXT", "1")

    async def fake_retrieve(query, **kwargs):
        return [{"text": "retrieved context chunk"}]

    monkeypatch.setattr(mod, "_retrieve", fake_retrieve)
    monkeypatch.setattr(mod, "load_dataset", lambda: [{
        "question_type": "temporal",
        "question": "test question",
        "answer": "test answer",
        "haystack_sessions": [],
    }])

    monkeypatch.setattr(mod, "REMOTE_LLM_MODEL", "test-model")
    monkeypatch.setattr(mod, "JUDGE_MODEL", "test-model")

    async def run():
        await mod.run_benchmark(limit=1, use_llm=True, args=argparse.Namespace(
            limit=1, type=None, llm=True, graph_expansion=0,
            retrieval_path="retrieve", report_retrieval_delta=False, out=str(tmp_path / "out.json")
        ))

    asyncio.run(run())

    # Check the written results file
    out_files = list(tmp_path.glob("out.json"))
    assert out_files, "results file should be written"
    with open(out_files[0]) as f:
        doc = json.load(f)

    results = doc["results"]
    assert len(results) == 1
    row = results[0]
    assert "context" in row, "context key must be present when SAVE_CONTEXT=1"
    assert row["context"] == "assembled context from ContextAssembler  retrieved context chunk"


def test_save_context_unset_omits_context_key(runner, monkeypatch, tmp_path):
    """Without TAOSMD_LME_SAVE_CONTEXT=1, the 'context' key is absent."""
    mod = runner
    monkeypatch.delenv("TAOSMD_LME_SAVE_CONTEXT", raising=False)

    async def fake_retrieve(query, **kwargs):
        return [{"text": "retrieved context chunk"}]

    monkeypatch.setattr(mod, "_retrieve", fake_retrieve)
    monkeypatch.setattr(mod, "load_dataset", lambda: [{
        "question_type": "temporal",
        "question": "test question",
        "answer": "test answer",
        "haystack_sessions": [],
    }])

    monkeypatch.setattr(mod, "REMOTE_LLM_MODEL", "test-model")
    monkeypatch.setattr(mod, "JUDGE_MODEL", "test-model")

    async def run():
        await mod.run_benchmark(limit=1, use_llm=True, args=argparse.Namespace(
            limit=1, type=None, llm=True, graph_expansion=0,
            retrieval_path="retrieve", report_retrieval_delta=False, out=str(tmp_path / "out.json")
        ))

    asyncio.run(run())

    out_files = list(tmp_path.glob("out.json"))
    assert out_files, "results file should be written"
    with open(out_files[0]) as f:
        doc = json.load(f)

    results = doc["results"]
    assert len(results) == 1
    row = results[0]
    assert "context" not in row, "context key must be absent when SAVE_CONTEXT is not set"


def test_judge_mode_recorded_in_result_doc(runner, monkeypatch, tmp_path):
    """The result_doc must include the judge_mode used."""
    mod = runner

    for mode in ("reference", "evidence"):
        monkeypatch.setenv("TAOSMD_JUDGE_MODE", mode)
        monkeypatch.setenv("TAOSMD_LME_SAVE_CONTEXT", "1")

        async def fake_retrieve(query, **kwargs):
            return [{"text": "retrieved context chunk"}]

        monkeypatch.setattr(mod, "_retrieve", fake_retrieve)
        monkeypatch.setattr(mod, "load_dataset", lambda: [{
            "question_type": "temporal",
            "question": "test question",
            "answer": "test answer",
            "haystack_sessions": [],
        }])

        monkeypatch.setattr(mod, "REMOTE_LLM_MODEL", "test-model")
        monkeypatch.setattr(mod, "JUDGE_MODEL", "test-model")

        async def run():
            await mod.run_benchmark(limit=1, use_llm=True, args=argparse.Namespace(
                limit=1, type=None, llm=True, graph_expansion=0,
                retrieval_path="retrieve", report_retrieval_delta=False, out=str(tmp_path / f"out_{mode}.json")
            ))

        asyncio.run(run())

        out_files = list(tmp_path.glob(f"out_{mode}.json"))
        assert out_files
        with open(out_files[0]) as f:
            doc = json.load(f)

        assert doc.get("judge_mode") == mode, f"judge_mode must be recorded as {mode}"


# ---------------------------------------------------------------------------
# 3. Re-judge script tests
# ---------------------------------------------------------------------------

def test_rejudge_refuses_file_missing_context(monkeypatch, tmp_path):
    """Re-judge exits with code 2 if any row lacks 'context'."""
    # Create a results file WITHOUT context
    results_file = tmp_path / "results_no_context.json"
    doc = {
        "judge_mode": "reference",
        "results": [
            {"idx": 0, "question_type": "temporal", "question": "q1", "correct": True, "retrieved_chars": 100},
            {"idx": 1, "question_type": "temporal", "question": "q2", "correct": False, "retrieved_chars": 100},
        ],
    }
    with open(results_file, "w") as f:
        json.dump(doc, f)

    mod = _load_rejudge()

    class _StubClient:
        async def post(self, *a, **kw):
            class R:
                status_code = 200
                def json(self): return {"message": {"content": "CORRECT"}}
            return R()
        async def aclose(self): pass

    monkeypatch.setattr(mod, "httpx", type("mod", (), {"AsyncClient": lambda *a, **kw: _StubClient()}))
    runner_mod = _load_runner()
    mod.set_runner_mod_for_testing(runner_mod)
    try:
        monkeypatch.setattr(sys, "argv", [
            "longmemeval_rejudge.py",
            "--results", str(results_file),
            "--judge-model", "test-model",
        ])

        with pytest.raises(SystemExit) as exc:
            mod.main()

        assert exc.value.code == 2
    finally:
        mod.set_runner_mod_for_testing(None)


def test_rejudge_runs_both_modes_and_computes_agreement(monkeypatch, tmp_path):
    """Re-judge runs both modes, computes accuracy and agreement."""
    # Create a results file WITH context (3 rows)
    results_file = tmp_path / "results_with_context.json"
    doc = {
        "judge_mode": "reference",
        "results": [
            {"idx": 0, "question_type": "temporal", "question": "q1", "answer": "a1", "correct": True, "retrieved_chars": 100, "context": "ctx1"},
            {"idx": 1, "question_type": "semantic", "question": "q2", "answer": "a2", "correct": False, "retrieved_chars": 100, "context": "ctx2"},
            {"idx": 2, "question_type": "temporal", "question": "q3", "answer": "a3", "correct": True, "retrieved_chars": 100, "context": "ctx3"},
        ],
    }
    with open(results_file, "w") as f:
        json.dump(doc, f)

    mod = _load_rejudge()

    # Stub judge that returns CORRECT, INCORRECT, CORRECT in sequence for EACH mode (6 total)
    verdicts = ["CORRECT", "INCORRECT", "CORRECT", "CORRECT", "INCORRECT", "CORRECT"]

    class _StubClient:
        def __init__(self):
            self.call_idx = 0
        async def post(self, *a, **kw):
            v = verdicts[self.call_idx]
            self.call_idx += 1
            class R:
                status_code = 200
                def json(self_inner): return {"message": {"content": v}}
            return R()
        async def aclose(self): pass

    stub_client = _StubClient()
    monkeypatch.setattr(mod, "httpx", type("mod", (), {"AsyncClient": lambda *a, **kw: stub_client}))

    # Mock importlib loading of runner to use our stub
    async def mock_score_answer_llm(client, predicted, gold, question, context=None):
        # This will be called by the rejudge script
        return await mod.score_answer_llm(client, predicted, gold, question, context)

    # The rejudge script imports the runner module and calls its score_answer_llm
    # We need to monkeypatch the runner's score_answer_llm
    runner_mod = _load_runner()
    original_score_llm = runner_mod.score_answer_llm
    call_log = []

    async def tracked_score_llm(client, predicted, gold, question, context=None):
        call_log.append({"context": context is not None})
        return await original_score_llm(client, predicted, gold, question, context)

    monkeypatch.setattr(runner_mod, "score_answer_llm", tracked_score_llm)
    mod.set_runner_mod_for_testing(runner_mod)

    try:
        monkeypatch.setattr(sys, "argv", [
            "longmemeval_rejudge.py",
            "--results", str(results_file),
            "--judge-model", "test-model",
            "--modes", "reference,evidence",
        ])

        mod.main()

        out_file = tmp_path / "results_with_context.rejudge.json"
        assert out_file.exists()

        with open(out_file) as f:
            rejudge_doc = json.load(f)

        # Check structure
        assert "modes" in rejudge_doc
        assert "reference" in rejudge_doc["modes"]
        assert "evidence" in rejudge_doc["modes"]
        assert "agreement" in rejudge_doc

        # With our stub returning CORRECT, INCORRECT, CORRECT:
        # reference mode (no context): 3 calls -> CORRECT, INCORRECT, CORRECT = 2/3 = 66.7%
        # evidence mode (with context): 3 calls -> CORRECT, INCORRECT, CORRECT = 2/3 = 66.7%
        ref_acc = rejudge_doc["modes"]["reference"]["accuracy"]
        ev_acc = rejudge_doc["modes"]["evidence"]["accuracy"]
        assert abs(ref_acc - 66.666) < 0.1 or abs(ref_acc - 66.667) < 0.1
        assert abs(ev_acc - 66.666) < 0.1 or abs(ev_acc - 66.667) < 0.1

        # Agreement: all 3 rows agree (both modes give same verdict per row)
        agree = rejudge_doc["agreement"]
        assert agree["total"] == 3
        assert agree["agree"] == 3
        assert agree["disagree"] == 0
        assert agree["reference_correct_evidence_incorrect"] == 0
        assert agree["reference_incorrect_evidence_correct"] == 0
    finally:
        mod.set_runner_mod_for_testing(None)


def test_rejudge_prints_summary_table(monkeypatch, tmp_path, capsys):
    """Re-judge prints a 3-column table: mode, accuracy, n."""
    results_file = tmp_path / "results_with_context.json"
    doc = {
        "judge_mode": "reference",
        "results": [
            {"idx": 0, "question_type": "temporal", "question": "q1", "answer": "a1", "correct": True, "retrieved_chars": 100, "context": "ctx1"},
            {"idx": 1, "question_type": "temporal", "question": "q2", "answer": "a2", "correct": True, "retrieved_chars": 100, "context": "ctx2"},
        ],
    }
    with open(results_file, "w") as f:
        json.dump(doc, f)

    mod = _load_rejudge()

    class _StubClient:
        async def post(self, *a, **kw):
            class R:
                status_code = 200
                def json(self): return {"message": {"content": "CORRECT"}}
            return R()
        async def aclose(self): pass

    monkeypatch.setattr(mod, "httpx", type("mod", (), {"AsyncClient": lambda *a, **kw: _StubClient()}))
    runner_mod = _load_runner()
    monkeypatch.setattr(mod, "runner_mod", runner_mod)

    monkeypatch.setattr(sys, "argv", [
        "longmemeval_rejudge.py",
        "--results", str(results_file),
        "--judge-model", "test-model",
        "--modes", "reference,evidence",
    ])

    mod.main()

    out = capsys.readouterr().out
    # Should print a table with mode, accuracy, n
    lines = out.strip().split("\n")
    table_lines = [l for l in lines if "reference" in l or "evidence" in l]
    assert len(table_lines) >= 2
