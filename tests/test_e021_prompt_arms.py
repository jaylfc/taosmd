"""E-021 generator-prompt ablation arms.

Asserts each default-off flag changes the prompt or flow as designed, and that
with all E-021 flags off the generation path is byte-identical to the E-012
baseline. CPU-only: no LLM, no GPU. The Ollama client is faked, so these run
without a model.
"""
import importlib.util
import pathlib
import asyncio

_RUNNER = pathlib.Path(__file__).resolve().parent.parent / "benchmarks" / "longmemeval_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("lme_runner", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeResp:
    def __init__(self, content, status=200):
        self.status_code = status
        self._content = content

    def json(self):
        return {"message": {"content": self._content}}


class _SeqClient:
    """Returns a queued sequence of contents; records every request body."""
    def __init__(self, contents):
        self._contents = list(contents)
        self.requests = []

    async def post(self, *args, **kwargs):
        self.requests.append(kwargs.get("json"))
        content = self._contents.pop(0) if self._contents else ""
        return _FakeResp(content)


class _FixedClient:
    def __init__(self, content):
        self._content = content
        self.requests = []

    async def post(self, *args, **kwargs):
        self.requests.append(kwargs.get("json"))
        return _FakeResp(self._content)


def _reset_arms(mod):
    mod.EVIDENCE_GROUNDING = False
    mod.PERSONA = False
    mod.SELF_VERIFY = False
    mod.SELF_CONSISTENCY = 0


# --- baseline byte-identity -------------------------------------------------

def test_all_arms_off_reproduces_baseline_prompt():
    mod = _load_runner()
    _reset_arms(mod)
    built = mod.build_answer_prompt("CTX", "Q")
    baseline = mod.ANSWER_PROMPT.format(context="CTX"[: mod.CONTEXT_CHARS], question="Q")
    assert built == baseline


def test_all_arms_off_request_body_is_baseline():
    mod = _load_runner()
    _reset_arms(mod)
    client = _FixedClient("an answer")
    asyncio.run(mod.llm_answer(client, "CTX", "Q"))
    opts = client.requests[0]["options"]
    # Byte-identical to the E-012 path: temperature 0, no seed key.
    assert opts == {"temperature": 0, "num_predict": 100}
    assert "seed" not in opts


# --- arm 2: evidence grounding ----------------------------------------------

def test_evidence_grounding_changes_prompt():
    mod = _load_runner()
    _reset_arms(mod)
    mod.EVIDENCE_GROUNDING = True
    built = mod.build_answer_prompt("CTX", "Q")
    assert built == mod.EVIDENCE_GROUNDING_PROMPT.format(context="CTX", question="Q")
    assert built != mod.ANSWER_PROMPT.format(context="CTX", question="Q")
    # The stricter rubric demands span quoting and grounding language.
    assert "ONLY" in built and "quot" in built.lower()


# --- arm 3: persona (control) -----------------------------------------------

def test_persona_only_prepends_prefix_no_other_change():
    mod = _load_runner()
    _reset_arms(mod)
    mod.PERSONA = True
    built = mod.build_answer_prompt("CTX", "Q")
    baseline = mod.ANSWER_PROMPT.format(context="CTX", question="Q")
    # Persona is a pure prefix: stripping it returns the byte-identical baseline.
    assert built.startswith(mod.PERSONA_PREFIX)
    assert built[len(mod.PERSONA_PREFIX):] == baseline


def test_persona_composes_with_evidence_grounding():
    mod = _load_runner()
    _reset_arms(mod)
    mod.PERSONA = True
    mod.EVIDENCE_GROUNDING = True
    built = mod.build_answer_prompt("CTX", "Q")
    expected = mod.PERSONA_PREFIX + mod.EVIDENCE_GROUNDING_PROMPT.format(context="CTX", question="Q")
    assert built == expected


# --- arm 1: self-consistency aggregation (pure) -----------------------------

def test_aggregate_majority_wins():
    mod = _load_runner()
    out = mod.aggregate_answers(["Paris", "paris.", "London"])
    assert out == "Paris"  # two votes for paris vs one for london


def test_aggregate_ignores_blank_candidates():
    mod = _load_runner()
    out = mod.aggregate_answers(["", "   ", "Berlin"])
    assert out == "Berlin"


def test_aggregate_empty_when_nothing_usable():
    mod = _load_runner()
    assert mod.aggregate_answers(["", "   "]) == ""


def test_aggregate_tie_break_prefers_most_entailed():
    mod = _load_runner()
    # One vote each: scorer prefers "Rome".
    scorer = lambda ans: 1.0 if ans == "Rome" else 0.0
    out = mod.aggregate_answers(["Rome", "Madrid"], entailment_scorer=scorer)
    assert out == "Rome"


def test_aggregate_tie_break_deterministic_without_scorer():
    mod = _load_runner()
    a = mod.aggregate_answers(["Madrid", "Rome"])
    b = mod.aggregate_answers(["Rome", "Madrid"])
    assert a == b  # sorted-key tie-break is order-independent and stable


def test_self_consistency_samples_n_and_uses_distinct_seeds():
    mod = _load_runner()
    _reset_arms(mod)
    mod.SELF_CONSISTENCY = 3
    mod.SELF_CONSISTENCY_TEMP = 0.7
    mod.SELF_CONSISTENCY_SEED = 1000
    # 3 generation drafts (self-verify off) + entailment scoring for unique keys.
    client = _SeqClient(["Tokyo", "tokyo", "Osaka", "yes", "no"])
    out = asyncio.run(mod.self_consistency_answer(client, "CTX", "Q", 3))
    assert out == "Tokyo"  # majority of the 3 samples
    # First three requests are the generation samples at non-zero temp + seeds.
    gen_reqs = [r for r in client.requests if r["options"].get("temperature") == 0.7]
    assert len(gen_reqs) == 3
    seeds = [r["options"]["seed"] for r in gen_reqs]
    assert seeds == [1000, 1001, 1002]


def test_self_consistency_runner_uses_aggregation_path_when_n_gt_1():
    mod = _load_runner()
    _reset_arms(mod)
    mod.SELF_CONSISTENCY = 3
    # All samples agree -> aggregation returns that answer.
    client = _SeqClient(["Cairo", "Cairo", "Cairo"])
    out = asyncio.run(mod.self_consistency_answer(client, "CTX", "Q", 3))
    assert out == "Cairo"


def test_self_consistency_off_does_not_change_seed_in_baseline_call():
    # N<=1 means the runner takes the plain llm_answer path (no seed, temp 0).
    mod = _load_runner()
    _reset_arms(mod)
    mod.SELF_CONSISTENCY = 1
    client = _FixedClient("baseline answer")
    asyncio.run(mod.llm_answer(client, "CTX", "Q"))
    assert "seed" not in client.requests[0]["options"]
