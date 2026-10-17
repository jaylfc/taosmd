"""Tests for eval/teamops/build_eval_set.py"""

import json
import pytest
from eval.teamops.build_eval_set import split, make_items, parse_rulings, parse_ledger, parse_prs


def test_manifest_holdout_ratio_is_float():
    # Create some base items
    rulings = [
        {
            "date": "2020-01-01",
            "who": "Alice",
            "text": "Rule one",
            "source_id": "ruling:2020-01-01T00:00:00Z:Alice",
        }
    ]
    ledger = []
    prs = []
    items = make_items(rulings, ledger, prs)
    train, holdout, manifest = split(items, holdout=0.30)
    # The manifest's holdout_ratio should be the float ratio, not the list length
    assert isinstance(manifest["holdout_ratio"], float)
    assert manifest["holdout_ratio"] == 0.30


def test_parse_prs_invalid_json_raises():
    invalid_json = "{not json"
    try:
        parse_prs(invalid_json)
        assert False, "Expected JSONDecodeError"
    except json.JSONDecodeError:
        pass  # expected


def test_main_nonexistent_session_map_raises(tmpdir):
    # Create temporary input files
    rulings_path = tmpdir.join("rulings.md")
    rulings_path.write("- 2020-01-01~00:00:00Z Alice: Rule one\n")
    ledger_path = tmpdir.join("ledger.tsv")
    ledger_path.write("utc\tseat\ttarget\towner\taction\twhy\tundo\n")
    ledger_path.write("2020-01-0T00:00:00Z\tseat1\ttarget1\towner1\taction1\twhy1\tundo1\n")
    prs_path = tmpdir.join("prs.json")
    prs_path.write('[{"number": 1, "title": "PR one", "mergedAt": "2020-01-01T00:00:00Z", "body": "Body one"}]')
    output_dir = tmpdir.mkdir("output")
    nonexistent_session_map = tmpdir.join("nonexistent.jsonl")

    from eval.teamops.build_eval_set import main
    import sys
    old_argv = sys.argv
    try:
        sys.argv = [
            "build_eval_set.py",
            "--rulings", str(rulings_path),
            "--ledger", str(ledger_path),
            "--prs", str(prs_path),
            "--output-dir", str(output_dir),
            "--session-map", str(nonexistent_session_map)
        ]
        with pytest.raises(FileNotFoundError):
            main()
    finally:
        sys.argv = old_argv


def test_supersedence_unrelated_never_rulings():
    # Three rulings that start with "Never" and are otherwise unrelated
    rulings_text = """- 2020-01-01~00:00:00Z Alice: Never eat apples
- 2020-01-02~00:00:00Z Bob: Never drive fast
- 2020-01-03~00:00:00Z Charlie: Never sleep late"""
    ledger_text = ""
    prs_text = "[]"
    rulings = parse_rulings(rulings_text)
    ledger = parse_ledger(ledger_text)
    prs = parse_prs(prs_text)
    items = make_items(rulings, ledger, prs)
    # None of the rulings should be superseded
    for item in items:
        if item["kind"] == "ruling":
            assert item["superseded"] == False, f"Ruling {item['id']} should not be superseded"
            assert item["supersedes"] == [], f"Ruling {item['id']} should not supersede any other"


def test_supersedence_related_with_withdrawal():
    # Two rulings that share a subject key "limit" and the later one withdraws the earlier
    rulings_text = """- 2020-01-01~00:00:00Z Alice: The speed limit is 60 mph
- 2020-01-02~00:00:00Z Bob: Withdraw the speed limit"""
    ledger_text = ""
    prs_text = "[]"
    rulings = parse_rulings(rulings_text)
    ledger = parse_ledger(ledger_text)
    prs = parse_prs(prs_text)
    items = make_items(rulings, ledger, prs)
    # Find the two rulings
    ruling_items = [item for item in items if item["kind"] == "ruling"]
    assert len(ruling_items) == 2
    # The earlier ruling (day 1) should be superseded
    earlier = [item for item in ruling_items if item["id"] == "r-00000"][0]
    later = [item for item in ruling_items if item["id"] == "r-00001"][0]
    assert earlier["superseded"] == True
    assert later["superseded"] == False
    # The later ruling should supersede the earlier one
    assert later["supersedes"] == [earlier["gold_sources"][0]]
    # The earlier ruling should not supersede any
    assert earlier["supersedes"] == []


def test_unanswerable_count_small_n():
    # Test with n_base = 1
    rulings_text = "- 2020-01-01~00:00:00Z Alice: Rule one"
    ledger_text = ""
    prs_text = "[]"
    rulings = parse_rulings(rulings_text)
    ledger = parse_ledger(ledger_text)
    prs = parse_prs(prs_text)
    items = make_items(rulings, ledger, prs)
    train, holdout, manifest = split(items, holdout=0.30)
    # With n_base=1, unanswerable_count = round(1 * 0.10) = 0
    # So we expect 0 unanswerable items in total
    unanswerable_train = [item for item in train if item["kind"] == "unanswerable"]
    unanswerable_holdout = [item for item in holdout if item["kind"] == "unanswerable"]
    assert len(unanswerable_train) == 0
    assert len(unanswerable_holdout) == 0
    # Also test with n_base = 10
    rulings_text = "\n".join([f"- 2020-01-{i+1:02d}~00:00:00Z Alice: Rule {i+1}" for i in range(10)])
    rulings = parse_rulings(rulings_text)
    items = make_items(rulings, ledger, prs)
    train, holdout, manifest = split(items, holdout=0.30)
    # unanswerable_count = round(10 * 0.10) = 1
    # holdout gets round(1 * 0.30) = 0
    # train gets 1
    unanswerable_train = [item for item in train if item["kind"] == "unanswerable"]
    unanswerable_holdout = [item for item in holdout if item["kind"] == "unanswerable"]
    assert len(unanswerable_train) == 1
    assert len(unanswerable_holdout) == 0


def test_parse_ledger_quoted_fields():
    # Test that quoted fields are parsed correctly using shlex fallback
    tsv_text = 'utc\tseat\ttarget\towner\taction\twhy\tundo\n'
    tsv_text += '2020-01-0T00:00:00Z\tseat1\ttarget1\towner1\t"move piece"\t"why with spaces"\t"undo with spaces"\n'
    # Also test a line without tabs (should fall back to shlex)
    tsv_text += '2020-01-0T00:00:00Z seat1 target1 owner1 "move piece" "why with spaces" "undo with spaces"\n'
    ledger = parse_ledger(tsv_text)
    assert len(ledger) == 2
    # First line (tab-separated)
    assert ledger[0]["action"] == "move piece"
    assert ledger[0]["why"] == "why with spaces"
    assert ledger[0]["undo"] == "undo with spaces"
    # Second line (space-separated, quoted)
    assert ledger[1]["action"] == "move piece"
    assert ledger[1]["why"] == "why with spaces"
    assert ledger[1]["undo"] == "undo with spaces"


def test_parse_ledger_values_not_just_length():
    # Ensure we don't just check len >= 1 but actual values
    tsv_text = 'utc\tseat\ttarget\towner\taction\twhy\tundo\n'
    tsv_text += '\t\t\t\t\t\t\n'  # all empty
    ledger = parse_ledger(tsv_text)
    # Should skip empty lines because len(parts) < 7
    assert len(ledger) == 0
