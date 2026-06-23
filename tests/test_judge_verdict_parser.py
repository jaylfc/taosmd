"""Regression test for the LongMemEval judge verdict parser.

The runner scores an LLM judge's reply (prompted to be exactly one word,
CORRECT or INCORRECT). A naive ``"CORRECT" in judgment`` mis-scores every
INCORRECT reply as a pass, because "INCORRECT" contains the substring "CORRECT".
That bug silently inflated every --llm "Judge" number from this runner. These
tests pin the corrected behavior: INCORRECT is checked first, unrecognised
replies fail closed.
"""

import importlib.util
import pathlib

_RUNNER = pathlib.Path(__file__).resolve().parent.parent / "benchmarks" / "longmemeval_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("lme_runner", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_incorrect_is_not_scored_correct():
    # the core bug: "CORRECT" in "INCORRECT" is True, but the verdict is a FAIL
    parse = _load_runner()._parse_verdict
    assert parse("INCORRECT") is False


def test_correct_is_scored_correct():
    parse = _load_runner()._parse_verdict
    assert parse("CORRECT") is True


def test_case_insensitive():
    parse = _load_runner()._parse_verdict
    assert parse("incorrect") is False
    assert parse("correct") is True


def test_verdict_embedded_in_a_sentence():
    parse = _load_runner()._parse_verdict
    assert parse("The predicted answer is INCORRECT because the dates differ.") is False
    assert parse("Verdict: CORRECT") is True


def test_empty_or_unrecognised_fails_closed():
    parse = _load_runner()._parse_verdict
    assert parse("") is False
    assert parse("   ") is False
    assert parse("WRONG") is False
    assert parse("NO") is False
