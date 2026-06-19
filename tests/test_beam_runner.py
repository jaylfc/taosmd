"""Unit tests for benchmarks/beam_runner.py — GPU-free wiring verification.

Covers the load-bearing pure helpers (dataset-row parsing, the ported mem0 judge
prompt + scoring, chat normalization) and the LLM-calling helpers with a mocked
Ollama client (mirrors tests/test_e012_levers.py). No live LLM / GPU needed.
"""

import asyncio
import importlib.util
import json
import pathlib

_RUNNER = pathlib.Path(__file__).resolve().parent.parent / "benchmarks" / "beam_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("beam_runner", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeResp:
    def __init__(self, content, status=200):
        self.status_code = status
        self._content = content

    def json(self):
        return {"message": {"content": self._content}}


class _FakeClient:
    """Returns a fixed Ollama chat response for any .post()."""

    def __init__(self, content, status=200):
        self._content = content
        self._status = status

    async def post(self, *args, **kwargs):
        return _FakeResp(self._content, self._status)


class _BoomClient:
    async def post(self, *args, **kwargs):
        raise RuntimeError("boom")


# A trimmed BEAM row in the real HF shape: probing_questions is a Python-repr
# string keyed by ability type; each question carries a list[str] rubric and a
# category-specific answer key.
_SAMPLE_ROW = {
    "conversation_id": "1",
    "chat": [[{"role": "user", "content": "My sprint ends March 29."},
              {"role": "assistant", "content": "Noted, sprint ends March 29."}]],
    "probing_questions": (
        "{'information_extraction': ["
        "{'question': 'When does my first sprint end?', "
        "'answer': 'March 29', 'difficulty': 'easy', "
        "'rubric': ['LLM response should state: March 29']}], "
        "'abstention': ["
        "{'question': 'What is my favourite colour?', "
        "'ideal_response': 'No information is provided.', 'difficulty': 'medium', "
        "'rubric': ['Based on the provided chat, there is no information about a favourite colour']}]}"
    ),
}


# ── dataset-row parsing ──────────────────────────────────────────────


def test_parse_probing_questions_flattens_with_type_and_rubric():
    mod = _load_runner()
    qs = mod.parse_probing_questions(_SAMPLE_ROW)
    assert len(qs) == 2
    by_type = {q["question_type"]: q for q in qs}
    ie = by_type["information_extraction"]
    assert ie["question"] == "When does my first sprint end?"
    assert ie["rubric"] == ["LLM response should state: March 29"]
    assert ie["gold"] == "March 29"
    # abstention uses the ideal_response key for its gold answer
    assert by_type["abstention"]["gold"] == "No information is provided."


def test_parse_probing_questions_handles_dict_input():
    mod = _load_runner()
    row = {"probing_questions": {"summarization": [
        {"question": "Summarize.", "ideal_summary": "X happened.", "rubric": ["mentions X"]}]}}
    qs = mod.parse_probing_questions(row)
    assert len(qs) == 1
    assert qs[0]["question_type"] == "summarization"
    assert qs[0]["gold"] == "X happened."
    assert qs[0]["rubric"] == ["mentions X"]


def test_extract_rubric_nuggets_list_dict_and_string():
    mod = _load_runner()
    assert mod.extract_rubric_nuggets({"rubric": ["a", "b"]}) == ["a", "b"]
    assert mod.extract_rubric_nuggets(
        {"rubric": {"nuggets": [{"description": "x"}, "y"]}}) == ["x", "y"]
    assert mod.extract_rubric_nuggets({"rubric": "single"}) == ["single"]
    assert mod.extract_rubric_nuggets({}) == []


def test_parse_beam_chat_flattens_nested_and_dict_turns():
    mod = _load_runner()
    turns = mod.parse_beam_chat(_SAMPLE_ROW["chat"])
    assert turns == [("user", "My sprint ends March 29."),
                     ("assistant", "Noted, sprint ends March 29.")]
    # batch-dict (10M) shape
    assert mod.parse_beam_chat([{"turns": [{"role": "user", "content": "hi"}]}]) == [("user", "hi")]
    # bare-string turns default to user
    assert mod.parse_beam_chat(["just text"]) == [("user", "just text")]


# ── mem0 judge prompt + scoring (the head-to-head surface) ───────────


def test_judge_prompt_contains_mem0_anchors():
    mod = _load_runner()
    p = mod.get_beam_nugget_judge_prompt("Q?", "should state March 29", "It ends March 29.")
    # the load-bearing structure that makes the score comparable to mem0
    assert "RUBRIC CRITERION:" in p
    assert "POSITIVE requirement" in p and "NEGATIVE constraint" in p
    assert "1.0 (Complete Compliance)" in p
    assert '{"score":' in p  # JSON output contract
    assert "should state March 29" in p and "It ends March 29." in p


def test_clamp_nugget_score_snaps_to_three_buckets():
    mod = _load_runner()
    assert mod.clamp_nugget_score(0.9) == 1.0
    assert mod.clamp_nugget_score(0.75) == 1.0
    assert mod.clamp_nugget_score(0.5) == 0.5
    assert mod.clamp_nugget_score(0.25) == 0.5
    assert mod.clamp_nugget_score(0.1) == 0.0


def test_parse_judge_json_plain_wrapped_and_fallback():
    mod = _load_runner()
    assert mod.parse_judge_json('{"score": 1.0, "reason": "ok"}')["score"] == 1.0
    # wrapped in prose / code fence
    assert mod.parse_judge_json('Sure!\n```json\n{"score": 0.5, "reason": "partial"}\n```')["score"] == 0.5
    # non-JSON fallback scan
    assert mod.parse_judge_json("I would say 1.0 here")["score"] == 1.0
    assert mod.parse_judge_json("garbage")["score"] == 0.0


def test_score_question_mean_and_pass_threshold():
    mod = _load_runner()
    out = mod.score_question([{"score": 1.0}, {"score": 0.0}])
    assert out["score"] == 0.5 and out["judgment"] == "PASS"
    out2 = mod.score_question([{"score": 0.0}, {"score": 0.5}])
    assert out2["score"] == 0.25 and out2["judgment"] == "FAIL"


# ── LLM-calling helpers with a mocked client ─────────────────────────


def test_judge_nugget_parses_score_from_mocked_client():
    mod = _load_runner()
    out = asyncio.run(mod.judge_nugget(
        _FakeClient('{"score": 1.0, "reason": "states March 29"}'),
        "When does my sprint end?", "should state March 29", "It ends March 29."))
    assert out["score"] == 1.0 and out["nugget"] == "should state March 29"


def test_judge_nugget_zero_on_client_error():
    mod = _load_runner()
    out = asyncio.run(mod.judge_nugget(_BoomClient(), "Q", "nugget", "ans"))
    assert out["score"] == 0.0


def test_llm_answer_strips_answer_prefix():
    mod = _load_runner()
    out = asyncio.run(mod.llm_answer(_FakeClient("ANSWER: It ends March 29."),
                                     ["mem1"], "When does it end?"))
    assert out == "It ends March 29."


def test_self_verify_keeps_draft_on_error_and_empty():
    mod = _load_runner()
    assert asyncio.run(mod.self_verify_answer(_BoomClient(), ["m"], "Q", "draft")) == "draft"
    assert asyncio.run(mod.self_verify_answer(_FakeClient("   "), ["m"], "Q", "draft")) == "draft"
    assert asyncio.run(mod.self_verify_answer(_FakeClient("anything"), ["m"], "Q", "")) == ""


# ── ingest chunking (the sweepable score lever) ──────────────────────


def test_chunk_text_defaults_match_old_hardcoded_behaviour():
    mod = _load_runner()
    # The old code: chunk_size=100, overlap=20 -> step 80, windows of <=100 words.
    words = [f"w{i}" for i in range(250)]
    text = " ".join(words)
    chunks = mod.chunk_text(text)  # defaults = 100 words / 20 overlap
    # Reproduce the exact original loop and compare verbatim.
    expected = []
    for start in range(0, len(words), 100 - 20):
        c = " ".join(words[start:start + 100])
        if c.strip():
            expected.append(c)
    assert chunks == expected
    # 250 words, step 80 -> starts at 0,80,160,240 -> 4 chunks.
    assert len(chunks) == 4


def test_chunk_text_count_and_size_bound():
    mod = _load_runner()
    text = " ".join(f"w{i}" for i in range(20))
    chunks = mod.chunk_text(text, words=5, overlap=1)  # step 4
    # starts: 0,4,8,12,16 -> 5 chunks
    assert len(chunks) == 5
    # no chunk exceeds the configured word size
    assert all(len(c.split()) <= 5 for c in chunks)
    # overlap is real: chunk[1] starts at word index 4, chunk[0] ends at 4
    assert chunks[0].split()[-1] == "w4"
    assert chunks[1].split()[0] == "w4"


def test_chunk_text_empty_and_overlap_floor():
    mod = _load_runner()
    assert mod.chunk_text("") == []
    assert mod.chunk_text("   ") == []
    # overlap >= words must not deadlock (step floored at 1); still bounded.
    chunks = mod.chunk_text(" ".join(f"w{i}" for i in range(5)), words=3, overlap=5)
    assert chunks and all(len(c.split()) <= 3 for c in chunks)


def test_chunk_text_sentences_never_splits_mid_sentence():
    mod = _load_runner()
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota kappa lambda mu nu."
    chunks = mod.chunk_text_sentences(text, words=6)
    # every chunk is composed of whole sentences (ends with terminal punctuation)
    assert all(c.rstrip().endswith((".", "!", "?")) for c in chunks)
    # the two 3-word sentences pack together (<=6); the 7-word one is its own chunk
    assert chunks[0] == "Alpha beta gamma. Delta epsilon zeta."
    assert chunks[1] == "Eta theta iota kappa lambda mu nu."


# ── metrics aggregation ──────────────────────────────────────────────


def test_compute_metrics_overall_and_per_type():
    mod = _load_runner()
    evals = [
        {"question_type": "information_extraction", "score": 1.0},
        {"question_type": "information_extraction", "score": 0.0},
        {"question_type": "abstention", "score": 0.5},
        {"question_type": "abstention", "rubric": [], "retrieved_count": 0},  # dry, no score -> skipped
    ]
    m = mod.compute_metrics(evals)
    assert m["overall"]["total"] == 3  # the no-score dry row is excluded
    assert m["overall"]["correct"] == 2  # 1.0 and 0.5 pass >= 0.5
    assert m["by_question_type"]["information_extraction"]["accuracy"] == 50.0
    assert m["by_question_type"]["abstention"]["correct"] == 1


# ── two-phase (generate -> judge) wiring ─────────────────────────────

# Two fake generate-phase predictions in the exact shape the generate phase
# writes (no judge fields yet). These exercise the predictions round-trip and the
# judge-from-prediction path with a mocked Ollama client (no live LLM).
_FAKE_PREDICTIONS = [
    {
        "conversation_id": "1",
        "question_type": "information_extraction",
        "question": "When does my first sprint end?",
        "rubric": ["LLM response should state: March 29"],
        "gold": "March 29",
        "generated_answer": "Your first sprint ends March 29.",
        "retrieved_count": 7,
    },
    {
        "conversation_id": "1",
        "question_type": "abstention",
        "question": "What is my favourite colour?",
        "rubric": [],  # no rubric -> judge marks ERROR, score 0.0
        "gold": "No information is provided.",
        "generated_answer": "I don't have enough information to answer this question.",
        "retrieved_count": 4,
    },
]


def test_predictions_round_trip_through_writer_and_reader(tmp_path):
    mod = _load_runner()
    path = str(tmp_path / "preds.json")
    mod.write_predictions(_FAKE_PREDICTIONS, path, split="100K", limit=1)

    # File is valid JSON with the wrapped shape + the predictions payload.
    on_disk = json.loads(pathlib.Path(path).read_text())
    assert on_disk["split"] == "100K" and on_disk["limit"] == 1
    assert len(on_disk["predictions"]) == 2

    # read_predictions returns the list back, field-for-field.
    back = mod.read_predictions(path)
    assert back == _FAKE_PREDICTIONS


def test_read_predictions_tolerates_bare_list(tmp_path):
    mod = _load_runner()
    path = tmp_path / "bare.json"
    path.write_text(json.dumps(_FAKE_PREDICTIONS))
    assert mod.read_predictions(str(path)) == _FAKE_PREDICTIONS


def test_judge_prediction_scores_a_prediction_dict():
    mod = _load_runner()
    # Mocked judge always returns score 1.0; the rubric'd prediction should PASS
    # and carry the scored shape the original interleaved judge path produced.
    client = _FakeClient('{"score": 1.0, "reason": "states March 29"}')
    rec = asyncio.run(mod.judge_prediction(client, dict(_FAKE_PREDICTIONS[0])))
    assert rec["score"] == 1.0 and rec["judgment"] == "PASS"
    assert rec["generated_answer"] == "Your first sprint ends March 29."
    assert rec["question_type"] == "information_extraction"
    assert len(rec["nugget_scores"]) == 1


def test_judge_prediction_marks_error_when_no_rubric():
    mod = _load_runner()
    client = _FakeClient('{"score": 1.0, "reason": "ignored"}')
    rec = asyncio.run(mod.judge_prediction(client, dict(_FAKE_PREDICTIONS[1])))
    assert rec["judgment"] == "ERROR" and rec["score"] == 0.0
    assert rec["error"] == "No rubric nuggets"
    # the generated answer is preserved through the judge step
    assert rec["generated_answer"].startswith("I don't have enough")


def test_judge_prediction_empty_answer_auto_fails_n014():
    mod = _load_runner()
    # A judge that would PASS everything if consulted (score 1.0). The N-014
    # guard must FAIL an empty answer WITHOUT calling the judge, because mem0's
    # nugget judge scores a non-responsive answer as full compliance on a
    # negative/abstention rubric, so a blank (e.g. a generation timeout) would
    # otherwise PASS and inflate the score.
    client = _FakeClient('{"score": 1.0, "reason": "no colour mentioned"}')
    rec = {
        "question": "What is my favourite colour?",
        "question_type": "abstention",
        "rubric": ["Based on the provided chat, there is no information about a favourite colour"],
        "generated_answer": "",
    }
    out = asyncio.run(mod.judge_prediction(client, rec))
    assert out["judgment"] == "FAIL"
    assert out["score"] == 0.0
    assert out["empty_answer"] is True
    assert out["nugget_scores"][0]["score"] == 0.0

    # whitespace-only is equally non-responsive and must also auto-fail
    rec2 = dict(rec, generated_answer="   \n  ")
    out2 = asyncio.run(mod.judge_prediction(client, rec2))
    assert out2["judgment"] == "FAIL" and out2["score"] == 0.0 and out2["empty_answer"] is True


def test_judge_phase_iterates_predictions_and_computes_metrics(tmp_path):
    """The judge phase reads predictions, judges each, and feeds compute_metrics.

    Done without a live LLM by judging each prediction with the mocked client and
    asserting compute_metrics aggregates exactly the rows the judge phase would.
    """
    mod = _load_runner()
    path = str(tmp_path / "preds.json")
    mod.write_predictions(_FAKE_PREDICTIONS, path, split="100K", limit=1)

    client = _FakeClient('{"score": 1.0, "reason": "ok"}')
    scored = [asyncio.run(mod.judge_prediction(client, dict(p)))
              for p in mod.read_predictions(path)]

    m = mod.compute_metrics(scored)
    # both rows carry a score (the no-rubric one scores 0.0, not skipped)
    assert m["overall"]["total"] == 2
    assert m["overall"]["correct"] == 1  # only the rubric'd row passes
    assert m["overall"]["errors"] == 1   # the no-rubric row is flagged
    assert m["by_question_type"]["information_extraction"]["correct"] == 1
