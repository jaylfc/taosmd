"""Tests for temporal anchoring: prompt placement and anchor_relative_dates."""

from __future__ import annotations

import hashlib
from datetime import datetime

from taosmd.prompts import (
    PRESERVE_DATES_INSTRUCTION,
    crystallization_prompt,
    extraction_prompt,
    session_enrichment_prompt,
)
from taosmd.temporal import anchor_relative_dates

# ---------------------------------------------------------------------------
# Master (f6f8074a) prompt sha256s for the default (preserve_dates=False) case.
# Computed from the checked-out master tree so any drift is caught.
# ---------------------------------------------------------------------------
_MASTER_EXTRACTION_SHA = "38cca36343515c2fe875810c3e66179e87ee70d0a274bbea219f3dbbe08ef3fa"
_MASTER_SESSION_SHA = "37ee3d2bba496df3f340cc50b0f32274e664a6eba9bd03d581b8ef05ddec3b5f"
_MASTER_CRYSTALLIZATION_SHA = "97e92d70957a268fb9af6393919e936e8206cf04c5e6141ef1ff82f9030ca9b3"

_FIXED_TEXT = "hello world"
_FIXED_SESSION_LOG = "session content"
_FIXED_SESSION_TEXT = "session text"


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Prompt placement tests (D1, D2)
# ---------------------------------------------------------------------------


def test_extraction_prompt_default_is_master_byte_identical():
    assert _sha256(extraction_prompt(_FIXED_TEXT)) == _MASTER_EXTRACTION_SHA


def test_session_enrichment_prompt_default_is_master_byte_identical():
    assert _sha256(session_enrichment_prompt(_FIXED_SESSION_LOG)) == _MASTER_SESSION_SHA


def test_crystallization_prompt_default_is_master_byte_identical():
    assert _sha256(crystallization_prompt(_FIXED_SESSION_TEXT)) == _MASTER_CRYSTALLIZATION_SHA


def test_extraction_prompt_preserve_dates_sentence_appears_once():
    p = extraction_prompt(_FIXED_TEXT, preserve_dates=True)
    assert p.count(PRESERVE_DATES_INSTRUCTION) == 1


def test_session_enrichment_prompt_preserve_dates_sentence_appears_once():
    p = session_enrichment_prompt(_FIXED_SESSION_LOG, preserve_dates=True)
    assert p.count(PRESERVE_DATES_INSTRUCTION) == 1


def test_crystallization_prompt_preserve_dates_sentence_appears_once():
    p = crystallization_prompt(_FIXED_SESSION_TEXT, preserve_dates=True)
    assert p.count(PRESERVE_DATES_INSTRUCTION) == 1


def test_extraction_prompt_preserve_dates_ends_with_json():
    assert extraction_prompt(_FIXED_TEXT, preserve_dates=True).endswith("JSON:")


def test_session_enrichment_prompt_preserve_dates_ends_with_json():
    assert session_enrichment_prompt(_FIXED_SESSION_LOG, preserve_dates=True).endswith("JSON:")


def test_crystallization_prompt_preserve_dates_ends_with_json():
    assert crystallization_prompt(_FIXED_SESSION_TEXT, preserve_dates=True).endswith("JSON:")


def test_extraction_prompt_preserve_dates_instruction_before_json():
    p = extraction_prompt(_FIXED_TEXT, preserve_dates=True)
    instr_pos = p.find(PRESERVE_DATES_INSTRUCTION)
    json_pos = p.rfind("JSON:")
    assert instr_pos != -1
    assert json_pos != -1
    assert instr_pos < json_pos


# ---------------------------------------------------------------------------
# anchor_relative_dates tests
# ---------------------------------------------------------------------------

_REF = datetime(2023, 5, 10, 12, 0, 0)


def test_anchors_yesterday():
    result = anchor_relative_dates("we met yesterday", _REF)
    assert result == "we met yesterday [2023-05-09]"


def test_anchors_n_days_ago():
    result = anchor_relative_dates("happened 3 days ago", _REF)
    # _ago_range returns (from_dt, ref); from_dt is 2023-05-07 00:00, ref is 2023-05-10 12:00
    # Different dates -> range format
    assert result == "happened 3 days ago [2023-05-07..2023-05-10]"


def test_anchors_last_week():
    result = anchor_relative_dates("last week was busy", _REF)
    # Anchor is inserted directly after the original words
    assert result.startswith("last week [2023-05-01..2023-05-07] was busy")


def test_anchors_last_month():
    result = anchor_relative_dates("last month was rainy", _REF)
    # Anchor is inserted directly after the original words
    assert result.startswith("last month [2023-04-01..2023-04-30] was rainy")


def test_absolute_date_unchanged():
    # Explicit day dates are absolute and must be left untouched.
    result = anchor_relative_dates("on 8 May 2023 we met", _REF)
    assert result == "on 8 May 2023 we met"


def test_reference_none_returns_unchanged():
    text = "we met yesterday"
    assert anchor_relative_dates(text, None) is text


def test_two_relative_expressions_both_anchored():
    text = "yesterday and last week were busy"
    result = anchor_relative_dates(text, _REF)
    assert "[2023-05-09]" in result
    assert "[2023-05-01..2023-05-07]" in result


def test_garbage_input_does_not_raise():
    # No parseable temporal expressions: must not raise, text returned unchanged.
    assert anchor_relative_dates("hello world", _REF) == "hello world"
    assert anchor_relative_dates("", _REF) == ""
    assert anchor_relative_dates("sometime next month maybe", _REF) == "sometime next month maybe"
