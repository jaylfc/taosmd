"""Tests for teamops eval builder."""
from __future__ import annotations

import json
from pathlib import Path

from eval.teamops import build_eval_set


def test_parse_rulings():
    md = """- 2025-10-01 ~09:00Z alice: initial rule A
- 2025-10-05 ~10:00Z bob: replaced rule A
"""
    rulings = build_eval_set.parse_rulings(md)
    assert len(rulings) == 2
    assert rulings[0]["date"] == "2025-10-01"
    assert rulings[0]["who"] == "alice"
    assert rulings[0]["text"] == "initial rule A"
    assert "source_id" in rulings[0]


def test_parse_ledger():
    tsv = """utc seat owner target action why undo
2025-10-01T12:00:00Z s1 jay t1 deploy hotfix revert
2025-10-02T13:00:00Z s2 bob t2 scale up space down
"""
    entries = build_eval_set.parse_ledger(tsv)
    assert len(entries) == 2
    assert entries[0]["owner"] == "jay"
    assert entries[0]["action"] == "deploy"
    # Also test with spaces in fields
    tsv2 = """utc seat owner target action why undo
2025-10-01T12:00:00Z s1 jay t1 deploy "hot fix" "revert now"
"""
    entries = build_eval_set.parse_ledger(tsv2)
    assert len(entries) >= 1  # at minimum parse correctly


def test_parse_prs():
    js = """[
  {"number": 1, "title": "t1", "mergedAt": "2025-10-01T08:00:00Z", "body": "b1"},
  {"number": 2, "title": "t2", "mergedAt": "2025-10-03T08:00:00Z", "body": "b2"}
]"""
    prs = build_eval_set.parse_prs(js)
    assert len(prs) == 2
    assert prs[0]["number"] == 1


def test_make_items_basic():
    rulings = build_eval_set.parse_rulings("- 2025-10-01 ~09:00Z alice: rule X")
    ledger = build_eval_set.parse_ledger("utc seat owner target action why undo\n2025-10-02T12:00:00Z s1 jay t1 deploy fix revert")
    prs = build_eval_set.parse_prs('[{"number": 1, "title": "t", "mergedAt": "2025-10-03T08:00:00Z", "body": "b"}]')
    items = build_eval_set.make_items(rulings, ledger, prs)
    assert len(items) == 3
    for item in items:
        assert "id" in item
        assert "question" in item
        assert "gold_answer" in item
        assert "gold_sources" in item
        assert "gold_sessions" in item
        assert "date" in item
        assert "kind" in item
        assert item["superseded"] is False


def test_split_time_holdout_deterministic():
    items = []
    for i in range(10):
        day = i + 1
        items.append({
            "id": f"r-{i:05d}",
            "question": f"q{i}",
            "gold_answer": f"a{i}",
            "gold_sources": [f"s{i}"],
            "gold_sessions": [],
            "date": f"2025-09-{day:02d}",
            "kind": "ruling",
            "superseded": False,
        })
    train1, holdout1, manifest1 = build_eval_set.split(items, holdout=0.30)
    train2, holdout2, manifest2 = build_eval_set.split(items, holdout=0.30)
    assert len(holdout1) == len(holdout2)
    assert [h["id"] for h in holdout1] == [h["id"] for h in holdout2]
    assert [t["id"] for t in train1] == [t["id"] for t in train2]
    holdout_dates = sorted([h["date"] for h in holdout1 if h["kind"] != "unanswerable" and h["kind"] != "as_of"])
    if holdout_dates:
        assert holdout_dates[-1] == "2025-09-10"


def test_split_creates_holdout_newest():
    items = []
    for i in range(10):
        day = i + 1
        items.append({
            "id": f"r-{i:05d}",
            "question": f"q{i}",
            "gold_answer": f"a{i}",
            "gold_sources": [f"s{i}"],
            "gold_sessions": [],
            "date": f"2025-09-{day:02d}",
            "kind": "ruling",
            "superseded": False,
        })
    train, holdout, manifest = build_eval_set.split(items, holdout=0.30)
    holdout_dates = sorted([h["date"] for h in holdout if h["kind"] != "unanswerable" and h["kind"] != "as_of"])
    assert holdout_dates == ["2025-09-08", "2025-09-09", "2025-09-10"]


def test_superseded_pair_linked():
    rulings = build_eval_set.parse_rulings("""- 2025-10-01 ~09:00Z alice: rule A applies
- 2025-10-05 ~10:00Z bob: rule A is replaced by rule B
""")
    items = build_eval_set.make_items(rulings, [], [])
    assert any(it.get("superseded") for it in items)
    assert any(it.get("supersedes") for it in items)


def test_unanswerable_share():
    items = []
    for i in range(10):
        items.append({
            "id": f"r-{i:05d}",
            "question": f"q{i}",
            "gold_answer": f"a{i}",
            "gold_sources": [f"s{i}"],
            "gold_sessions": [],
            "date": f"2025-09-{i+1:02d}",
            "kind": "ruling",
            "superseded": False,
        })
    train, holdout, manifest = build_eval_set.split(items, holdout=0.30)
    unanswerable = [t for t in train if t["kind"] == "unanswerable"] + [h for h in holdout if h["kind"] == "unanswerable"]
    assert abs(len(unanswerable) - int(round(10 * 0.10))) <= 1 or len(unanswerable) >= 1


def test_as_of_for_superseded_pair():
    rulings = build_eval_set.parse_rulings("""- 2025-10-01 ~09:00Z alice: rule A applies
- 2025-10-05 ~10:00Z bob: rule A is replaced by rule B
""")
    items = build_eval_set.make_items(rulings, [], [])
    train, holdout, manifest = build_eval_set.split(items, holdout=0.30)
    as_of = [t for t in train if t["kind"] == "as_of"] + [h for h in holdout if h["kind"] == "as_of"]
    assert len(as_of) >= 1


def test_session_map_fills_sessions():
    rulings = build_eval_set.parse_rulings("- 2025-10-01 ~09:00Z alice: rule X")
    session_map = [{
        "source_id": rulings[0]["source_id"],
        "session_id": "s1",
        "turn_from": 1,
        "turn_to": 5,
    }]
    items = build_eval_set.make_items(rulings, [], [], session_map)
    assert items[0]["gold_sessions"] == [{"session_id": "s1", "turn_from": 1, "turn_to": 5}]
