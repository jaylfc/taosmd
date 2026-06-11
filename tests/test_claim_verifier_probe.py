"""Unit tests for benchmarks/claim_verifier_probe.py.

Tests are fully offline (no Ollama required). Covers:
  - Verdict parsing including noisy responses
  - Same-family assertion
  - Checkpoint resume skips already-done work
  - Extraction + verification integration with mocked HTTP
"""

from __future__ import annotations

import json
import sys
import os
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest

# Make sure benchmarks/ is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Import the module under test
import benchmarks.claim_verifier_probe as cvp


# ---------------------------------------------------------------------------
# Verdict parsing tests
# ---------------------------------------------------------------------------

class TestParseVerdict:
    def test_exact_supported(self):
        assert cvp._parse_verdict("SUPPORTED") == "SUPPORTED"

    def test_exact_partial(self):
        assert cvp._parse_verdict("PARTIAL") == "PARTIAL"

    def test_exact_unsupported(self):
        assert cvp._parse_verdict("UNSUPPORTED") == "UNSUPPORTED"

    def test_lowercase(self):
        assert cvp._parse_verdict("supported") == "SUPPORTED"

    def test_mixed_case(self):
        assert cvp._parse_verdict("Unsupported") == "UNSUPPORTED"

    def test_leading_whitespace(self):
        assert cvp._parse_verdict("  SUPPORTED  ") == "SUPPORTED"

    def test_first_token_extraction(self):
        # Model often puts the token first then explains
        assert cvp._parse_verdict("SUPPORTED. The claim is grounded.") == "SUPPORTED"

    def test_answer_prefix(self):
        # Some models say "Answer: UNSUPPORTED"
        assert cvp._parse_verdict("Answer: UNSUPPORTED") == "UNSUPPORTED"

    def test_verdict_label(self):
        assert cvp._parse_verdict("Verdict: PARTIAL") == "PARTIAL"

    def test_think_tag_stripped(self):
        # Qwen-style <think>...</think> block before the actual verdict
        noisy = "<think>Let me analyze this carefully.</think>\nSUPPORTED"
        assert cvp._parse_verdict(noisy) == "SUPPORTED"

    def test_markdown_code_block_stripped(self):
        noisy = "```\nUNSUPPORTED\n```"
        assert cvp._parse_verdict(noisy) == "UNSUPPORTED"

    def test_unparseable_returns_none(self):
        assert cvp._parse_verdict("I am not sure what to say here.") is None

    def test_empty_string_returns_none(self):
        assert cvp._parse_verdict("") is None

    def test_only_whitespace_returns_none(self):
        assert cvp._parse_verdict("   \n  ") is None

    def test_bullet_prefix(self):
        # Some models add "- SUPPORTED" or "* PARTIAL"
        assert cvp._parse_verdict("- SUPPORTED") == "SUPPORTED"
        assert cvp._parse_verdict("* PARTIAL") == "PARTIAL"

    def test_sentence_with_verdict_in_head(self):
        # Within first 80 chars
        assert (
            cvp._parse_verdict(
                "UNSUPPORTED because the source does not mention Alice's role."
            )
            == "UNSUPPORTED"
        )


# ---------------------------------------------------------------------------
# Same-family assertion tests
# ---------------------------------------------------------------------------

class TestModelFamilyCheck:
    def test_gemma_vs_qwen_ok(self, capsys):
        # Should not exit
        cvp.assert_different_families("gemma4:e2b", "qwen3:4b")
        captured = capsys.readouterr()
        assert "FATAL" not in captured.err

    def test_qwen_vs_qwen_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            cvp.assert_different_families("qwen3:4b", "qwen3:8b")
        assert exc_info.value.code == 1

    def test_gemma_vs_gemma_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            cvp.assert_different_families("gemma2:2b", "gemma4:e2b")
        assert exc_info.value.code == 1

    def test_llama_vs_llama_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            cvp.assert_different_families("llama3.1:8b", "llama3.2:3b")
        assert exc_info.value.code == 1

    def test_unknown_vs_known_warns(self, capsys):
        # Unknown family should not exit but warn
        cvp.assert_different_families("my-custom-model:v1", "qwen3:4b")
        captured = capsys.readouterr()
        assert "WARN" in captured.err or "warn" in captured.err.lower()

    def test_gemma_vs_llama_ok(self, capsys):
        cvp.assert_different_families("gemma4:e2b", "llama3.1:8b")
        captured = capsys.readouterr()
        assert "FATAL" not in captured.err


# ---------------------------------------------------------------------------
# Checkpoint resume tests
# ---------------------------------------------------------------------------

class TestCheckpointResume:
    def test_load_pairs_empty_when_file_missing(self, tmp_path):
        result = cvp._load_pairs(tmp_path / "nonexistent.jsonl")
        assert result == {}

    def test_append_and_reload_pairs(self, tmp_path):
        pairs_path = tmp_path / "pairs.jsonl"
        pairs = [
            {
                "pair_id": "conv-1:session_1:12345",
                "conversation_id": "conv-1",
                "session_id": "session_1",
                "claim": "Alice uses Python",
                "source": "Alice uses Python for her work.",
                "fact_raw": {"subject": "Alice", "predicate": "uses", "object": "Python"},
            }
        ]
        cvp._append_pairs(pairs_path, pairs)
        loaded = cvp._load_pairs(pairs_path)
        assert "conv-1:session_1" in loaded
        assert len(loaded["conv-1:session_1"]["pairs"]) == 1
        assert loaded["conv-1:session_1"]["pairs"][0]["claim"] == "Alice uses Python"

    def test_resume_skips_done_session(self, tmp_path):
        """Pairs from a done session are loaded from checkpoint, not re-extracted."""
        pairs_path = tmp_path / "pairs.jsonl"
        pairs = [
            {
                "pair_id": "conv-26:session_1:99",
                "conversation_id": "conv-26",
                "session_id": "session_1",
                "claim": "Bob works on AI",
                "source": "Bob works on AI projects.",
                "fact_raw": {},
            }
        ]
        cvp._append_pairs(pairs_path, pairs)
        done = cvp._load_pairs(pairs_path)
        # The session should be present and would be skipped in the extraction loop
        assert "conv-26:session_1" in done

    def test_load_verdicts_empty_when_missing(self, tmp_path):
        result = cvp._load_verdicts(tmp_path / "nope.json")
        assert result == {}

    def test_save_and_reload_verdicts(self, tmp_path):
        out_path = tmp_path / "results.json"
        pairs = [
            {
                "pair_id": "p1",
                "conversation_id": "c1",
                "session_id": "s1",
                "claim": "X",
                "source": "X is here.",
            }
        ]
        verdicts = {"p1": "SUPPORTED"}
        new_verdicts = [
            {"pair_id": "p1", "conversation_id": "c1", "session_id": "s1",
             "claim": "X", "verdict": "SUPPORTED"}
        ]
        cvp._save_results(out_path, pairs, verdicts, new_verdicts)

        loaded = cvp._load_verdicts(out_path)
        assert loaded["p1"] == "SUPPORTED"

    def test_save_incremental_merge(self, tmp_path):
        """Second _save_results call merges new verdicts without duplicating old ones."""
        out_path = tmp_path / "results.json"
        pairs1 = [{"pair_id": "p1", "conversation_id": "c1", "session_id": "s1",
                   "claim": "A", "source": "A is here."}]
        cvp._save_results(out_path, pairs1, {"p1": "SUPPORTED"},
                          [{"pair_id": "p1", "conversation_id": "c1",
                            "session_id": "s1", "claim": "A", "verdict": "SUPPORTED"}])

        # Second save with a new pair
        pairs2 = pairs1 + [
            {"pair_id": "p2", "conversation_id": "c1", "session_id": "s1",
             "claim": "B", "source": "B is here."}
        ]
        cvp._save_results(out_path, pairs2,
                          {"p1": "SUPPORTED", "p2": "UNSUPPORTED"},
                          [{"pair_id": "p2", "conversation_id": "c1",
                            "session_id": "s1", "claim": "B",
                            "verdict": "UNSUPPORTED"}])

        loaded = cvp._load_verdicts(out_path)
        assert loaded["p1"] == "SUPPORTED"
        assert loaded["p2"] == "UNSUPPORTED"

        data = json.loads(out_path.read_text())
        pair_ids = [v["pair_id"] for v in data["verdicts"]]
        # No duplicates
        assert len(pair_ids) == len(set(pair_ids))


# ---------------------------------------------------------------------------
# Mocked integration: extraction + verification
# ---------------------------------------------------------------------------

class TestMockedIntegration:
    """Drive _extract_session_pairs and _verify_pair with mocked HTTP responses."""

    def _make_extraction_response(self, facts: list[dict]) -> MagicMock:
        """Build a mock httpx response that returns a JSON array of facts."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(facts)
                    }
                }
            ]
        }
        return mock_resp

    def _make_verdict_response(self, verdict_text: str) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": verdict_text
                    }
                }
            ]
        }
        return mock_resp

    def test_extract_session_pairs_basic(self):
        """Extraction builds correct claim text and preserves source."""
        facts = [
            {"subject": "Alice", "predicate": "uses", "object": "Python",
             "source_line": "Alice uses Python."}
        ]
        mock_client = MagicMock()
        mock_client.post = AsyncMock(
            return_value=self._make_extraction_response(facts)
        )

        conversation = {
            "session_1": [
                {"speaker": "Bob", "text": "Alice uses Python."},
            ],
            "session_1_date_time": "2024-01-01",
        }

        result = asyncio.run(
            cvp._extract_session_pairs(
                mock_client,
                "http://localhost:11434",
                "gemma4:e2b",
                "conv-test",
                conversation,
                "session_1",
                "2024-01-01",
            )
        )

        assert len(result) == 1
        pair = result[0]
        assert pair["claim"] == "Alice uses Python"
        assert "Alice uses Python." in pair["source"]
        assert pair["conversation_id"] == "conv-test"
        assert pair["session_id"] == "session_1"
        # pair_id is deterministic for the same claim
        assert pair["pair_id"].startswith("conv-test:session_1:")

    def test_extract_session_pairs_empty_session(self):
        """Empty session produces no pairs without calling the LLM."""
        mock_client = MagicMock()
        mock_client.post = AsyncMock()

        conversation = {"session_1": [], "session_1_date_time": "2024-01-01"}

        result = asyncio.run(
            cvp._extract_session_pairs(
                mock_client,
                "http://localhost:11434",
                "gemma4:e2b",
                "conv-empty",
                conversation,
                "session_1",
                "2024-01-01",
            )
        )

        assert result == []
        mock_client.post.assert_not_called()

    def test_verify_pair_supported(self):
        mock_client = MagicMock()
        mock_client.post = AsyncMock(
            return_value=self._make_verdict_response("SUPPORTED")
        )
        pair = {
            "pair_id": "p1",
            "source": "Alice uses Python for her work.",
            "claim": "Alice uses Python",
        }
        verdict = asyncio.run(
            cvp._verify_pair(
                mock_client,
                "http://localhost:11434",
                "qwen3:4b",
                pair,
            )
        )
        assert verdict == "SUPPORTED"

    def test_verify_pair_unsupported(self):
        mock_client = MagicMock()
        mock_client.post = AsyncMock(
            return_value=self._make_verdict_response("UNSUPPORTED")
        )
        pair = {
            "pair_id": "p2",
            "source": "Bob likes hiking.",
            "claim": "Bob uses Python",
        }
        verdict = asyncio.run(
            cvp._verify_pair(
                mock_client,
                "http://localhost:11434",
                "qwen3:4b",
                pair,
            )
        )
        assert verdict == "UNSUPPORTED"

    def test_verify_pair_retry_on_unparseable(self):
        """First call returns unparseable text; second call returns PARTIAL."""
        responses = [
            self._make_verdict_response("I cannot determine this."),
            self._make_verdict_response("PARTIAL"),
        ]
        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=responses)
        pair = {
            "pair_id": "p3",
            "source": "Carol sometimes uses Go.",
            "claim": "Carol uses Go",
        }
        verdict = asyncio.run(
            cvp._verify_pair(
                mock_client,
                "http://localhost:11434",
                "qwen3:4b",
                pair,
            )
        )
        assert verdict == "PARTIAL"
        assert mock_client.post.call_count == 2

    def test_verify_pair_verifier_error_after_two_failures(self):
        """Both retries fail to parse -> VERIFIER_ERROR."""
        mock_client = MagicMock()
        mock_client.post = AsyncMock(
            return_value=self._make_verdict_response("I have no idea about this claim.")
        )
        pair = {
            "pair_id": "p4",
            "source": "Some text.",
            "claim": "Some claim",
        }
        verdict = asyncio.run(
            cvp._verify_pair(
                mock_client,
                "http://localhost:11434",
                "qwen3:4b",
                pair,
            )
        )
        assert verdict == "VERIFIER_ERROR"
        assert mock_client.post.call_count == 2

    def test_verify_pair_network_error_becomes_verifier_error(self):
        """Network exception on both attempts -> VERIFIER_ERROR."""
        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
        pair = {
            "pair_id": "p5",
            "source": "Some text.",
            "claim": "Some claim",
        }
        verdict = asyncio.run(
            cvp._verify_pair(
                mock_client,
                "http://localhost:11434",
                "qwen3:4b",
                pair,
            )
        )
        assert verdict == "VERIFIER_ERROR"


# ---------------------------------------------------------------------------
# Session text builder
# ---------------------------------------------------------------------------

class TestSessionText:
    def test_includes_datetime(self):
        conversation = {
            "session_1": [
                {"speaker": "Alice", "text": "Hello."},
            ],
            "session_1_date_time": "2024-03-15",
        }
        text = cvp._session_text(conversation, "session_1", "2024-03-15")
        assert "[2024-03-15]" in text
        assert "[Alice] Hello." in text

    def test_skips_empty_turns(self):
        conversation = {
            "session_1": [
                {"speaker": "Alice", "text": ""},
                {"speaker": "Bob", "text": "Hi there."},
            ],
        }
        text = cvp._session_text(conversation, "session_1", "")
        assert "[Alice]" not in text
        assert "[Bob] Hi there." in text

    def test_no_datetime_header_when_empty(self):
        conversation = {
            "session_1": [{"speaker": "X", "text": "Test."}],
        }
        text = cvp._session_text(conversation, "session_1", "")
        assert text.startswith("[X]")


# ---------------------------------------------------------------------------
# Session keys helper
# ---------------------------------------------------------------------------

class TestSessionKeys:
    def test_returns_sorted_numeric_keys(self):
        conversation = {
            "session_3": [{"speaker": "A", "text": "x"}],
            "session_1": [{"speaker": "B", "text": "y"}],
            "session_2": [{"speaker": "C", "text": "z"}],
            "session_3_date_time": "dt3",
            "session_1_date_time": "dt1",
            "session_2_date_time": "dt2",
        }
        keys = cvp._session_keys(conversation)
        assert [k for k, _ in keys] == ["session_1", "session_2", "session_3"]

    def test_ignores_non_session_keys(self):
        conversation = {
            "qa": [],
            "session_1": [{"speaker": "A", "text": "x"}],
            "event_summary": {},
        }
        keys = cvp._session_keys(conversation)
        assert len(keys) == 1
        assert keys[0][0] == "session_1"

    def test_skips_non_list_sessions(self):
        conversation = {
            "session_1": {"not": "a list"},
            "session_2": [{"speaker": "A", "text": "real"}],
        }
        keys = cvp._session_keys(conversation)
        assert len(keys) == 1
        assert keys[0][0] == "session_2"
