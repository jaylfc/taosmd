"""Tests for taosmd.temporal: parser, extractor, hit-datetime, and stage."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from taosmd.temporal import (
    apply_temporal_stage,
    extract_temporal_expression,
    parse_hit_datetime,
    parse_temporal_expression,
)

# ---------------------------------------------------------------------------
# Fixed reference date for all parser tests: Wednesday, March 18, 2026 noon
# ---------------------------------------------------------------------------
REF = datetime(2026, 3, 18, 12, 0, 0)


# ---------------------------------------------------------------------------
# Ported from engram-review temporal.test.ts
# ---------------------------------------------------------------------------


def test_parses_today():
    r = parse_temporal_expression("today", REF)
    assert r is not None
    from_dt, to_dt = r
    assert from_dt.day == 18 and from_dt.month == 3 and from_dt.year == 2026
    assert to_dt.day == 18
    assert from_dt.hour == 0 and from_dt.minute == 0 and from_dt.second == 0
    assert to_dt.hour == 23 and to_dt.minute == 59 and to_dt.second == 59


def test_parses_yesterday():
    r = parse_temporal_expression("yesterday", REF)
    assert r is not None
    from_dt, to_dt = r
    assert from_dt.day == 17
    assert to_dt.day == 17


def test_parses_last_week_correct_range():
    # REF is Wednesday Mar 18.  7 days back = Mar 11 (Wednesday).
    # Monday of that week = Mar 9.
    r = parse_temporal_expression("last week", REF)
    assert r is not None
    from_dt, to_dt = r
    assert from_dt.day == 9
    assert from_dt.month == 3


def test_parses_this_month():
    r = parse_temporal_expression("this month", REF)
    assert r is not None
    from_dt, to_dt = r
    assert from_dt.day == 1
    assert from_dt.month == 3
    assert to_dt.day == 31


def test_parses_3_days_ago():
    r = parse_temporal_expression("3 days ago", REF)
    assert r is not None
    from_dt, _ = r
    assert from_dt.day == 15  # Mar 18 - 3 = Mar 15


def test_parses_in_march():
    r = parse_temporal_expression("in March", REF)
    assert r is not None
    from_dt, to_dt = r
    assert from_dt.month == 3
    assert from_dt.day == 1
    assert to_dt.day == 31


def test_parses_q1_2026():
    r = parse_temporal_expression("Q1 2026", REF)
    assert r is not None
    from_dt, to_dt = r
    assert from_dt.month == 1  # January
    assert to_dt.month == 3    # March
    assert to_dt.day == 31


def test_parses_last_month():
    r = parse_temporal_expression("last month", REF)
    assert r is not None
    from_dt, _ = r
    assert from_dt.month == 2  # February


def test_returns_none_for_unparseable():
    # "before the migration": "before" requires a parseable inner expression
    assert parse_temporal_expression("before the migration", REF) is None
    assert parse_temporal_expression("sometime", REF) is None


def test_parses_bare_month_name():
    r = parse_temporal_expression("january", REF)
    assert r is not None
    from_dt, _ = r
    assert from_dt.month == 1


def test_parses_this_week():
    # REF is Wednesday Mar 18; Monday of that week = Mar 16.
    r = parse_temporal_expression("this week", REF)
    assert r is not None
    from_dt, to_dt = r
    assert from_dt.day == 16
    assert from_dt.month == 3
    assert to_dt.day == 22  # Sunday


def test_parses_this_year():
    r = parse_temporal_expression("this year", REF)
    assert r is not None
    from_dt, to_dt = r
    assert from_dt.year == 2026
    assert from_dt.month == 1
    assert to_dt.year == 2026
    assert to_dt.month == 12
    assert to_dt.day == 31


def test_parses_last_year():
    r = parse_temporal_expression("last year", REF)
    assert r is not None
    from_dt, _ = r
    assert from_dt.year == 2025


# ---------------------------------------------------------------------------
# Additional cases: before/after, last N units, in <month> <year>
# ---------------------------------------------------------------------------


def test_parses_before_last_month():
    # "before last month" = before Feb 2026 start
    r = parse_temporal_expression("before last month", REF)
    assert r is not None
    from_dt, to_dt = r
    assert from_dt == datetime.min
    assert to_dt.month == 2 and to_dt.year == 2026


def test_parses_after_yesterday():
    r = parse_temporal_expression("after yesterday", REF)
    assert r is not None
    from_dt, to_dt = r
    # from_dt = end of yesterday = 23:59:59
    assert from_dt.day == 17
    assert to_dt == REF


def test_parses_last_2_weeks():
    r = parse_temporal_expression("last 2 weeks", REF)
    assert r is not None
    from_dt, _ = r
    # 2 weeks before Mar 18 = Mar 4, floored to day start
    assert from_dt.day == 4
    assert from_dt.month == 3


def test_parses_in_may_2023():
    r = parse_temporal_expression("in May 2023", REF)
    assert r is not None
    from_dt, to_dt = r
    assert from_dt.year == 2023
    assert from_dt.month == 5
    assert from_dt.day == 1
    assert to_dt.day == 31


def test_parses_in_month_with_comma():
    # "in May, 2023" with comma before year
    r = parse_temporal_expression("in May, 2023", REF)
    assert r is not None
    from_dt, _ = r
    assert from_dt.year == 2023
    assert from_dt.month == 5


# ---------------------------------------------------------------------------
# Explicit day date additions
# ---------------------------------------------------------------------------


def test_parses_day_month_year():
    # "8 May 2023"
    r = parse_temporal_expression("8 May 2023", REF)
    assert r is not None
    from_dt, to_dt = r
    assert from_dt.year == 2023 and from_dt.month == 5 and from_dt.day == 8
    assert to_dt.day == 8 and to_dt.hour == 23


def test_parses_month_day_comma_year():
    # "May 8, 2023"
    r = parse_temporal_expression("May 8, 2023", REF)
    assert r is not None
    from_dt, to_dt = r
    assert from_dt.year == 2023 and from_dt.month == 5 and from_dt.day == 8


def test_parses_on_day_month_comma_year():
    # "on 20 July, 2023"
    r = parse_temporal_expression("on 20 July, 2023", REF)
    assert r is not None
    from_dt, to_dt = r
    assert from_dt.year == 2023 and from_dt.month == 7 and from_dt.day == 20


# ---------------------------------------------------------------------------
# parse_hit_datetime
# ---------------------------------------------------------------------------


def test_hit_dt_epoch_float():
    # 2023-05-08 00:00:00 UTC - just check we get a datetime
    dt = parse_hit_datetime(1683504000.0)
    assert isinstance(dt, datetime)


def test_hit_dt_epoch_int():
    dt = parse_hit_datetime(1683504000)
    assert isinstance(dt, datetime)


def test_hit_dt_iso_string():
    dt = parse_hit_datetime("2023-05-08T10:30:00")
    assert dt is not None
    assert dt.year == 2023 and dt.month == 5 and dt.day == 8
    assert dt.hour == 10 and dt.minute == 30


def test_hit_dt_iso_with_z():
    dt = parse_hit_datetime("2023-07-20T20:56:00Z")
    assert dt is not None
    assert dt.tzinfo is None  # must be naive
    assert dt.year == 2023 and dt.month == 7 and dt.day == 20


def test_hit_dt_locomo_format():
    # "8:56 pm on 20 July, 2023"
    dt = parse_hit_datetime("8:56 pm on 20 July, 2023")
    assert dt is not None
    assert dt.year == 2023 and dt.month == 7 and dt.day == 20
    assert dt.hour == 20 and dt.minute == 56


def test_hit_dt_date_only_locomo():
    # "20 July, 2023"
    dt = parse_hit_datetime("20 July, 2023")
    assert dt is not None
    assert dt.year == 2023 and dt.month == 7 and dt.day == 20


def test_hit_dt_garbage_returns_none():
    assert parse_hit_datetime("not a date at all") is None


def test_hit_dt_none_input():
    assert parse_hit_datetime(None) is None


def test_hit_dt_empty_string():
    assert parse_hit_datetime("") is None


# ---------------------------------------------------------------------------
# extract_temporal_expression
# ---------------------------------------------------------------------------


def test_extract_last_week_in_sentence():
    result = extract_temporal_expression(
        "what did we decide last week about the database"
    )
    assert result is not None
    assert "last week" in result.lower()


def test_extract_in_may_2023():
    result = extract_temporal_expression(
        "what meetings did we have in May 2023 about the project?"
    )
    assert result is not None
    assert "may" in result.lower()
    assert "2023" in result


def test_extract_bare_month_false_positive_guard():
    # "did May approve the plan": bare month name should NOT match
    result = extract_temporal_expression("did May approve the plan")
    assert result is None


def test_extract_no_temporal_content():
    result = extract_temporal_expression("what is the meaning of life")
    assert result is None


def test_extract_explicit_date_in_sentence():
    result = extract_temporal_expression(
        "what happened on 8 May, 2023 in the park"
    )
    assert result is not None
    assert "8" in result or "may" in result.lower()


def test_extract_returns_none_for_modal_may():
    # "may" as a modal verb should not match
    result = extract_temporal_expression("this may cause issues later")
    assert result is None


# ---------------------------------------------------------------------------
# apply_temporal_stage: boost mode
# ---------------------------------------------------------------------------


def _make_hit(hit_id: int, score: float, dt_str: str | None) -> dict:
    meta = {}
    if dt_str is not None:
        meta["datetime"] = dt_str
    return {"id": hit_id, "score": score, "text": f"hit {hit_id}", "metadata": meta}


def test_boost_mode_in_range_reordered():
    # May 2023 window; boost=1.0 so in-range hits double their score.
    # Hit 1: score 0.5, datetime in May 2023 (in range)
    # Hit 2: score 0.8, datetime in July 2023 (out of range)
    # After boost: hit 1 = 1.0, hit 2 = 0.8 -> hit 1 should rank first.
    hits = [
        _make_hit(2, 0.8, "2023-07-15T00:00:00"),
        _make_hit(1, 0.5, "2023-05-10T00:00:00"),
    ]
    result = apply_temporal_stage(
        hits,
        {"window": "in May 2023", "mode": "boost", "boost": 1.0},
        "query",
    )
    assert result[0]["id"] == 1
    assert result[0]["score"] == pytest.approx(1.0)
    assert result[1]["id"] == 2
    assert result[1]["score"] == pytest.approx(0.8)


def test_boost_mode_no_timestamp_untouched():
    hits = [
        _make_hit(1, 0.9, None),  # no timestamp
        _make_hit(2, 0.5, "2023-05-10T00:00:00"),  # in range
    ]
    result = apply_temporal_stage(
        hits,
        {"window": "in May 2023", "mode": "boost", "boost": 0.25},
        "query",
    )
    # Hit 1 (no ts) stays at 0.9; hit 2 boosted to 0.625 -> still sorted 1 first
    ids = [h["id"] for h in result]
    assert ids[0] == 1  # 0.9 > 0.625
    # Score of hit 1 should be unchanged (no "score" key written back when hit had it)
    h1 = next(h for h in result if h["id"] == 1)
    assert h1["score"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# apply_temporal_stage: filter mode
# ---------------------------------------------------------------------------


def test_filter_mode_drops_out_of_range():
    hits = [
        _make_hit(1, 0.9, "2023-05-10T00:00:00"),  # in range
        _make_hit(2, 0.8, "2023-07-15T00:00:00"),  # out of range
        _make_hit(3, 0.7, None),                    # no timestamp, kept
    ]
    result = apply_temporal_stage(
        hits,
        {"window": "in May 2023", "mode": "filter"},
        "query",
    )
    ids = {h["id"] for h in result}
    assert 1 in ids  # in range
    assert 2 not in ids  # out of range
    assert 3 in ids  # no timestamp: fail-open


def test_filter_mode_zero_survivors_returns_original():
    hits = [
        _make_hit(1, 0.9, "2023-07-15T00:00:00"),  # out of range
        _make_hit(2, 0.8, "2023-08-01T00:00:00"),  # out of range
    ]
    result = apply_temporal_stage(
        hits,
        {"window": "in May 2023", "mode": "filter"},
        "query",
    )
    # All timestamped hits are out of range; fail-open globally
    assert len(result) == 2
    assert {h["id"] for h in result} == {1, 2}


# ---------------------------------------------------------------------------
# apply_temporal_stage: no-expression and unparseable-window cases
# ---------------------------------------------------------------------------


def test_no_expression_returns_unchanged():
    hits = [_make_hit(1, 0.9, "2023-05-10T00:00:00")]
    result = apply_temporal_stage(hits, {}, "query with no temporal")
    assert result == hits


def test_unparseable_window_returns_unchanged():
    hits = [_make_hit(1, 0.9, "2023-05-10T00:00:00")]
    result = apply_temporal_stage(
        hits, {"window": "not a real expression !@#"}, "query"
    )
    assert result == hits


# ---------------------------------------------------------------------------
# apply_temporal_stage: reference handling
# ---------------------------------------------------------------------------


def test_reference_locomo_format_last_month():
    # Reference = "8:56 pm on 20 July, 2023" -> last month = June 2023.
    hits = [
        _make_hit(1, 0.9, "2023-06-15T00:00:00"),  # June 2023: in range
        _make_hit(2, 0.8, "2023-05-01T00:00:00"),  # May 2023: out of range
    ]
    result = apply_temporal_stage(
        hits,
        {
            "window": "last month",
            "mode": "filter",
            "reference": "8:56 pm on 20 July, 2023",
        },
        "query",
    )
    ids = {h["id"] for h in result}
    assert 1 in ids   # June 2023 hit in range
    assert 2 not in ids  # May hit out of range


# ---------------------------------------------------------------------------
# Integration: retrieve() with temporal kwarg
# ---------------------------------------------------------------------------


from taosmd.retrieval import retrieve  # noqa: E402


class _DateVectorMemory:
    """Minimal fake vector source with three hits carrying LoCoMo datetimes.

    Returns raw vector-memory dicts; _adapt_vector normalises them into hits
    with source_id (str) and metadata pointing at the original dict.
    """

    async def search(self, query, limit=5, hybrid=True, fusion="rrf", project=None, search_agents=None):
        return [
            {
                "id": 10,
                "text": "May event",
                "similarity": 0.9,
                "metadata": {"datetime": "10:00 am on 5 May, 2023"},
                "created_at": 0.0,
            },
            {
                "id": 11,
                "text": "May event 2",
                "similarity": 0.85,
                "metadata": {"datetime": "3:30 pm on 20 May, 2023"},
                "created_at": 0.0,
            },
            {
                "id": 12,
                "text": "July event",
                "similarity": 0.8,
                "metadata": {"datetime": "8:56 pm on 20 July, 2023"},
                "created_at": 0.0,
            },
        ]


def test_retrieve_temporal_filter_integration():
    """retrieve() with temporal=filter drops out-of-window hits.

    After _adapt_vector normalisation, original id becomes source_id (str) and
    metadata is the original dict (which carries the "datetime" key).
    """
    results = asyncio.run(
        retrieve(
            query="what happened in May 2023",
            strategy="custom",
            memory_layers=["vector"],
            sources={"vector": _DateVectorMemory()},
            limit=10,
            temporal={"window": "in May 2023", "mode": "filter"},
        )
    )

    assert isinstance(results, list)
    # source_id is a string after _adapt_vector normalisation
    source_ids = {r["source_id"] for r in results}
    assert "10" in source_ids   # May hit
    assert "11" in source_ids   # May hit
    assert "12" not in source_ids  # July hit filtered out


def test_boost_mode_rrf_score_hits():
    # Hits from an RRF merge carry "rrf_score" instead of "score"; boost
    # must reorder those too.
    hits = [
        {"id": 2, "rrf_score": 0.8, "metadata": {"datetime": "2023-07-15T00:00:00"}},
        {"id": 1, "rrf_score": 0.5, "metadata": {"datetime": "2023-05-10T00:00:00"}},
    ]
    result = apply_temporal_stage(
        hits,
        {"window": "in May 2023", "mode": "boost", "boost": 1.0},
        "query",
    )
    assert result[0]["id"] == 1
    assert result[0]["rrf_score"] == pytest.approx(1.0)
    assert "score" not in result[0]
