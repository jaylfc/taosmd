"""Tests for benchmarks/e048/owndata.py (red-first).

Uses pytest fixtures and temporary directories to test the own-data loader.
All network, model downloads, and GPU operations are mocked/injected.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

# Make the benchmark module importable from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.e048.owndata import assert_outside_repo, load_owndata, recall_at_k_qrels

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_owndata_dir(tmp_path):
    """Create a valid own-data directory structure for testing."""
    dir_path = tmp_path / "owndata"
    dir_path.mkdir()

    # Create dummy files for items
    item1 = dir_path / "item1.png"
    item1.write_bytes(b"fake image 1")
    item2 = dir_path / "item2.png"
    item2.write_bytes(b"fake image 2")
    item3 = dir_path / "subdir" / "item3.png"
    item3.parent.mkdir()
    item3.write_bytes(b"fake image 3")

    # Write items.jsonl
    items = [
        {"id": "img1", "path": "item1.png", "mime": "image/png"},
        {"id": "img2", "path": "item2.png", "mime": "image/png"},
        {"id": "img3", "path": "subdir/item3.png", "mime": "image/png"},
    ]
    (dir_path / "items.jsonl").write_text("\n".join(json.dumps(it) for it in items))

    # Write texts.jsonl (optional)
    texts = [
        {"id": "txt1", "text": "Some text document"},
        {"id": "txt2", "text": "Another text document"},
    ]
    (dir_path / "texts.jsonl").write_text("\n".join(json.dumps(it) for it in texts))

    # Write queries.jsonl
    queries = [
        {"qid": "q1", "text": "First query", "author": "alice", "written_at": "2024-01-01T00:00:00Z"},
        {"qid": "q2", "text": "Second query", "author": "bob", "written_at": "2024-01-02T00:00:00Z"},
        {"qid": "q3", "text": "Third query", "author": "charlie", "written_at": "2024-01-03T00:00:00Z"},
    ]
    (dir_path / "queries.jsonl").write_text("\n".join(json.dumps(it) for it in queries))

    # Write qrels.tsv
    qrels_lines = [
        "q1\timg1\t1",
        "q1\timg2\t1",
        "q2\timg2\t1",
        "q2\timg3\t1",
        "q3\timg1\t1",
        "q3\ttxt1\t1",
    ]
    (dir_path / "qrels.tsv").write_text("\n".join(qrels_lines))

    return dir_path, {
        "items": items,
        "texts": texts,
        "queries": queries,
        "qrels": {"q1": {"img1", "img2"}, "q2": {"img2", "img3"}, "q3": {"img1", "txt1"}},
        "qrels_sha256": hashlib.sha256("\n".join(qrels_lines).encode()).hexdigest(),
    }


@pytest.fixture
def minimal_owndata_dir(tmp_path):
    """Create a minimal own-data directory with just one item."""
    dir_path = tmp_path / "owndata"
    dir_path.mkdir()

    # Create dummy file
    item1 = dir_path / "single.png"
    item1.write_bytes(b"single image")

    # Write items.jsonl
    (dir_path / "items.jsonl").write_text(json.dumps({"id": "single", "path": "single.png", "mime": "image/png"}))

    # Write queries.jsonl
    (dir_path / "queries.jsonl").write_text(json.dumps({"qid": "q1", "text": "One query", "author": "test", "written_at": "2024-01-01T00:00:00Z"}))

    # Write qrels.tsv
    (dir_path / "qrels.tsv").write_text("q1\tsingle\t1")

    return dir_path

# ---------------------------------------------------------------------------
# Tests for assert_outside_repo
# ---------------------------------------------------------------------------


def test_assert_outside_repo_valid_path(tmp_path):
    """Test that assert_outside_repo accepts paths outside repo."""
    dummy = tmp_path / "dummy.txt"
    dummy.write_text("test")
    assert_outside_repo(dummy)


def test_assert_outside_repo_git_repo_path(tmp_path):
    """Test that assert_outside_repo refuses paths inside a git repo."""
    git_dir = tmp_path / "git_repo"
    git_dir.mkdir()
    (git_dir / ".git").mkdir()  # Git directory

    path_inside = git_dir / "file.txt"
    path_inside.write_text("test")

    with pytest.raises(ValueError, match="own-data path is inside a git tree"):
        assert_outside_repo(path_inside)

    # Also test with .git file (worktree style)
    git_file = tmp_path / "git_worktree"
    git_file.mkdir()
    (git_file / ".git").write_text("gitdir: /some/other/path")

    path_with_gitfile = git_file / "file.txt"
    path_with_gitfile.write_text("test")

    with pytest.raises(ValueError, match="own-data path is inside a git tree"):
        assert_outside_repo(path_with_gitfile)


def test_assert_outside_repo_symlink_to_git_tree(tmp_path):
    """Test that assert_outside_repo refuses symlinks pointing into git tree."""
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / ".git").mkdir()

    link_path = tmp_path / "link_to_target"
    link_path.symlink_to(target_dir / "file.txt")

    with pytest.raises(ValueError, match="own-data path is inside a git tree"):
        assert_outside_repo(link_path)


# ---------------------------------------------------------------------------
# Tests for load_owndata
# ---------------------------------------------------------------------------


def test_load_owndata_valid(valid_owndata_dir):
    """Test that valid own-data loads correctly."""
    dir_path, expected = valid_owndata_dir

    result = load_owndata(dir_path)

    assert len(result["items"]) == 3
    assert len(result["texts"]) == 2
    assert len(result["queries"]) == 3
    assert len(result["qrels"]) == 3

    # Check items
    item_ids = {item["id"] for item in result["items"]}
    assert item_ids == {"img1", "img2", "img3"}

    # Check queries
    qids = {query["qid"] for query in result["queries"]}
    assert qids == {"q1", "q2", "q3"}

    # Check qrels
    assert result["qrels"]["q1"] == {"img1", "img2"}
    assert result["qrels"]["q2"] == {"img2", "img3"}
    assert result["qrels"]["q3"] == {"img1", "txt1"}

    # Check sha256
    assert result["qrels_sha256"] == expected["qrels_sha256"]


def test_load_owndata_git_repo_path(tmp_path):
    """Test that load_owndata refuses paths inside a git repo."""
    git_dir = tmp_path / "git_repo"
    git_dir.mkdir()
    (git_dir / ".git").mkdir()

    with pytest.raises(ValueError, match="own-data path is inside a git tree"):
        load_owndata(git_dir)


def test_load_owndata_symlink_to_git_tree(tmp_path):
    """Test that load_owndata refuses symlinks pointing into git tree."""
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / ".git").mkdir()

    link_path = tmp_path / "link_to_target"
    link_path.symlink_to(target_dir)

    with pytest.raises(ValueError, match="own-data path is inside a git tree"):
        load_owndata(link_path)


def test_load_owndata_item_path_outside_dir(tmp_path):
    """Test that load_owndata refuses item paths that resolve outside directory."""
    dir_path = tmp_path / "owndata"
    dir_path.mkdir()

    # Create items.jsonl with path pointing outside directory
    items = [{"id": "outside", "path": "../outside.png", "mime": "image/png"}]
    (dir_path / "items.jsonl").write_text(json.dumps(items[0]))

    # Create the outside file
    outside_file = tmp_path / "outside.png"
    outside_file.write_bytes(b"outside image")

    with pytest.raises(ValueError, match=re.escape("Path ../outside.png resolves outside directory")):
        load_owndata(dir_path)


def test_load_owndata_duplicate_ids_across_items_text(tmp_path):
    """Test that load_owndata refuses duplicate IDs across items and texts."""
    dir_path = tmp_path / "owndata"
    dir_path.mkdir()

    # Create dummy files
    (dir_path / "img.png").write_bytes(b"image")
    (dir_path / "txt.txt").write_bytes(b"text")

    # Write items.jsonl and texts.jsonl with same ID
    items = [{"id": "same", "path": "img.png", "mime": "image/png"}]
    texts = [{"id": "same", "text": "duplicate id"}]

    (dir_path / "items.jsonl").write_text(json.dumps(items[0]))
    (dir_path / "texts.jsonl").write_text(json.dumps(texts[0]))

    (dir_path / "queries.jsonl").write_text(json.dumps({"qid": "q1", "text": "query", "author": "a", "written_at": "2024-01-01T00:00:00Z"}))
    (dir_path / "qrels.tsv").write_text("q1\tsame\t1")

    with pytest.raises(ValueError, match=re.escape("duplicate id across items+texts")):
        load_owndata(dir_path)


def test_load_owndata_duplicate_qids(tmp_path):
    """Test that load_owndata refuses duplicate qids."""
    dir_path = tmp_path / "owndata"
    dir_path.mkdir()

    # Create dummy files
    (dir_path / "img1.png").write_bytes(b"image 1")
    (dir_path / "img2.png").write_bytes(b"image 2")

    # Write items.jsonl
    items = [
        {"id": "img1", "path": "img1.png", "mime": "image/png"},
        {"id": "img2", "path": "img2.png", "mime": "image/png"},
    ]
    (dir_path / "items.jsonl").write_text("\n".join(json.dumps(it) for it in items))

    # Write queries.jsonl with duplicate qid
    queries = [
        {"qid": "q1", "text": "First query", "author": "a", "written_at": "2024-01-01T00:00:00Z"},
        {"qid": "q1", "text": "Second query", "author": "b", "written_at": "2024-01-02T00:00:00Z"},
    ]
    (dir_path / "queries.jsonl").write_text("\n".join(json.dumps(it) for it in queries))

    (dir_path / "qrels.tsv").write_text("q1\timg1\t1\nq1\timg2\t1")

    with pytest.raises(ValueError, match="duplicate qid"):
        load_owndata(dir_path)


def test_load_owndata_missing_keys(tmp_path):
    """Test that load_owndata refuses missing required keys."""
    dir_path = tmp_path / "owndata"
    dir_path.mkdir()

    # Create dummy file
    (dir_path / "img.png").write_bytes(b"image")

    # Write items.jsonl with missing 'mime' key
    items = [{"id": "img1", "path": "img.png"}]  # Missing 'mime'

    (dir_path / "items.jsonl").write_text(json.dumps(items[0]))

    (dir_path / "queries.jsonl").write_text(json.dumps({"qid": "q1", "text": "query", "author": "a", "written_at": "2024-01-01T00:00:00Z"}))
    (dir_path / "qrels.tsv").write_text("q1\timg1\t1")

    with pytest.raises(ValueError, match="missing key"):
        load_owndata(dir_path)


def test_load_owndata_qrels_line_format(tmp_path):
    """Test that load_owndata refuses invalid qrels line format."""
    dir_path = tmp_path / "owndata"
    dir_path.mkdir()

    # Create dummy files
    (dir_path / "img1.png").write_bytes(b"image 1")
    (dir_path / "img2.png").write_bytes(b"image 2")

    # Write items.jsonl
    items = [
        {"id": "img1", "path": "img1.png", "mime": "image/png"},
        {"id": "img2", "path": "img2.png", "mime": "image/png"},
    ]
    (dir_path / "items.jsonl").write_text("\n".join(json.dumps(it) for it in items))

    # Write queries.jsonl
    (dir_path / "queries.jsonl").write_text(json.dumps({"qid": "q1", "text": "query", "author": "a", "written_at": "2024-01-01T00:00:00Z"}))

    # Write qrels.tsv with invalid line format (only 2 fields)
    (dir_path / "qrels.tsv").write_text("q1\timg1")  # Missing rel field

    with pytest.raises(ValueError, match="invalid qrels line format"):
        load_owndata(dir_path)


def test_load_owndata_qrels_rel_not_one(tmp_path):
    """Test that load_owndata refuses qrels rel not equal to 1."""
    dir_path = tmp_path / "owndata"
    dir_path.mkdir()

    # Create dummy files
    (dir_path / "img1.png").write_bytes(b"image")

    # Write items.jsonl
    (dir_path / "items.jsonl").write_text(json.dumps({"id": "img1", "path": "img1.png", "mime": "image/png"}))

    # Write queries.jsonl
    (dir_path / "queries.jsonl").write_text(json.dumps({"qid": "q1", "text": "query", "author": "a", "written_at": "2024-01-01T00:00:00Z"}))

    # Write qrels.tsv with rel not equal to 1
    (dir_path / "qrels.tsv").write_text("q1\timg1\t0")  # rel=0, not 1

    with pytest.raises(ValueError, match="qrels rel not equal to 1"):
        load_owndata(dir_path)


def test_load_owndata_qrels_qid_not_in_queries(tmp_path):
    """Test that load_owndata refuses qrels qid not in queries."""
    dir_path = tmp_path / "owndata"
    dir_path.mkdir()

    # Create dummy files
    (dir_path / "img1.png").write_bytes(b"image")

    # Write items.jsonl
    (dir_path / "items.jsonl").write_text(json.dumps({"id": "img1", "path": "img1.png", "mime": "image/png"}))

    # Write queries.jsonl
    (dir_path / "queries.jsonl").write_text(json.dumps({"qid": "q1", "text": "query", "author": "a", "written_at": "2024-01-01T00:00:00Z"}))

    # Write qrels.tsv with qid not in queries
    (dir_path / "qrels.tsv").write_text("q2\timg1\t1")  # q2 not in queries

    with pytest.raises(ValueError, match="qrels qid not in queries"):
        load_owndata(dir_path)


def test_load_owndata_qrels_doc_id_not_in_items_text(tmp_path):
    """Test that load_owndata refuses qrels doc_id not in items or texts."""
    dir_path = tmp_path / "owndata"
    dir_path.mkdir()

    # Create dummy files
    (dir_path / "img1.png").write_bytes(b"image")

    # Write items.jsonl
    (dir_path / "items.jsonl").write_text(json.dumps({"id": "img1", "path": "img1.png", "mime": "image/png"}))

    # Write queries.jsonl
    (dir_path / "queries.jsonl").write_text(json.dumps({"qid": "q1", "text": "query", "author": "a", "written_at": "2024-01-01T00:00:00Z"}))

    # Write qrels.tsv with doc_id not in items
    (dir_path / "qrels.tsv").write_text("q1\tmissing_id\t1")  # missing_id not in items

    with pytest.raises(ValueError, match="qrels doc_id not in items or texts"):
        load_owndata(dir_path)


def test_load_owndata_query_no_qrels(tmp_path):
    """Test that load_owndata refuses queries with no qrels line."""
    dir_path = tmp_path / "owndata"
    dir_path.mkdir()

    # Create dummy files
    (dir_path / "img1.png").write_bytes(b"image")

    # Write items.jsonl
    (dir_path / "items.jsonl").write_text(json.dumps({"id": "img1", "path": "img1.png", "mime": "image/png"}))

    # Write queries.jsonl
    queries = [
        {"qid": "q1", "text": "First query", "author": "a", "written_at": "2024-01-01T00:00:00Z"},
        {"qid": "q2", "text": "Second query", "author": "b", "written_at": "2024-01-02T00:00:00Z"},
    ]
    (dir_path / "queries.jsonl").write_text("\n".join(json.dumps(it) for it in queries))

    # Write qrels.tsv with only one qrel (q1)
    (dir_path / "qrels.tsv").write_text("q1\timg1\t1")  # q2 has no qrels

    with pytest.raises(ValueError, match="query with no qrels line"):
        load_owndata(dir_path)


# ---------------------------------------------------------------------------
# Tests for recall_at_k_qrels
# ---------------------------------------------------------------------------


def test_recall_at_k_qrels_basic():
    """Test basic recall calculation."""
    ranked_doc_ids = {
        "q1": ["img1", "img2", "img3"],
        "q2": ["img2", "img3", "img1"],
        "q3": ["img1", "img3", "img2"],
    }
    qrels = {
        "q1": {"img1", "img2"},
        "q2": {"img1", "img3"},  # q2's relevant docs are img1 and img3
        "q3": {"img1", "img3"},
    }

    metrics = recall_at_k_qrels(ranked_doc_ids, qrels, ks=(1, 5, 10))

    # Use pytest.approx with reasonable tolerance
    assert metrics["R@1"] == pytest.approx(2/3)
    assert metrics["R@5"] == pytest.approx(1.0)
    assert metrics["R@10"] == pytest.approx(1.0)


def test_recall_at_k_qrels_hit_at_k():
    """Test that recall counts correctly when relevant doc is in top-K but not top-1."""
    ranked_doc_ids = {
        "q1": ["img2", "img3", "img1"],  # img2 at position 0, img3 at position 1, img1 at position 2
        "q2": ["img1", "img2", "img3"],  # img1 at position 0, img2 at position 1, img3 at position 2
    }
    qrels = {
        "q1": {"img3"},  # img3 is at position 1 (K=1 miss, K=5 hit)
        "q2": {"img1"},  # img1 is at position 0 (K=1 hit)
    }

    metrics = recall_at_k_qrels(ranked_doc_ids, qrels, ks=(1, 5, 10))

    assert metrics["R@1"] == 0.5  # Only q2 hits at K=1
    assert metrics["R@5"] == 1.0  # Both queries hit at K=5
    assert metrics["R@10"] == 1.0  # Both queries hit at K=10


def test_recall_at_k_qrels_missing_qid_counts_miss():
    """Test that qids missing from ranked_doc_ids count as misses."""
    ranked_doc_ids = {
        "q1": ["img1", "img2"],  # Only q1 present
    }
    qrels = {
        "q1": {"img1"},  # q1 has relevant doc
        "q2": {"img3"},  # q2 not in ranked_doc_ids
    }

    metrics = recall_at_k_qrels(ranked_doc_ids, qrels, ks=(1, 5, 10))

    assert metrics["R@1"] == 0.5  # Only q1 hits, q2 is missing -> miss
    assert metrics["R@5"] == 0.5  # Only q1 hits, q2 is missing -> miss
    assert metrics["R@10"] == 0.5  # Only q1 hits, q2 is missing -> miss


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


def test_load_owndata_empty_files(tmp_path):
    """Test that load_owndata handles empty files correctly."""
    dir_path = tmp_path / "owndata"
    dir_path.mkdir()

    # Write empty files
    (dir_path / "items.jsonl").write_text("")
    (dir_path / "queries.jsonl").write_text("")
    (dir_path / "qrels.tsv").write_text("")

    result = load_owndata(dir_path)

    assert result["items"] == []
    assert result["queries"] == []
    assert result["qrels"] == {}
    assert result["qrels_sha256"] == hashlib.sha256(b"").hexdigest()


def test_load_owndata_no_texts_file(tmp_path):
    """Test that load_owndata works without texts.jsonl file."""
    dir_path = tmp_path / "owndata"
    dir_path.mkdir()

    # Create dummy file
    (dir_path / "img1.png").write_bytes(b"image")
    (dir_path / "img2.png").write_bytes(b"image")

    # Write items.jsonl
    items = [
        {"id": "img1", "path": "img1.png", "mime": "image/png"},
        {"id": "img2", "path": "img2.png", "mime": "image/png"},
    ]
    (dir_path / "items.jsonl").write_text("\n".join(json.dumps(it) for it in items))

    # Write queries.jsonl
    (dir_path / "queries.jsonl").write_text(json.dumps({"qid": "q1", "text": "query", "author": "a", "written_at": "2024-01-01T00:00:00Z"}))

    # Write qrels.tsv
    (dir_path / "qrels.tsv").write_text("q1\timg1\t1")

    result = load_owndata(dir_path)

    assert len(result["items"]) == 2
    assert result["texts"] == []  # No texts file
    assert len(result["queries"]) == 1
    assert result["qrels"] == {"q1": {"img1"}}

    # Should have sha256 of the actual file content
    expected_sha = hashlib.sha256("q1\timg1\t1".encode()).hexdigest()
    assert result["qrels_sha256"] == expected_sha


def test_load_owndata_blank_lines(tmp_path):
    """Test that load_owndata skips blank lines in JSONL files."""
    dir_path = tmp_path / "owndata"
    dir_path.mkdir()

    # Create actual files that the items reference
    (dir_path / "img1.png").write_bytes(b"image")
    (dir_path / "img2.png").write_bytes(b"image")

    # Write items.jsonl with blank lines
    items = [
        {"id": "img1", "path": "img1.png", "mime": "image/png"},
        {"id": "img2", "path": "img2.png", "mime": "image/png"},
    ]
    (dir_path / "items.jsonl").write_text(f"{json.dumps(items[0])}\n\n{json.dumps(items[1])}\n\n")

    # Write queries.jsonl with blank lines
    queries = [
        {"qid": "q1", "text": "Query 1", "author": "a", "written_at": "2024-01-01T00:00:00Z"},
        {"qid": "q2", "text": "Query 2", "author": "b", "written_at": "2024-01-02T00:00:00Z"},
    ]
    (dir_path / "queries.jsonl").write_text(f"{json.dumps(queries[0])}\n\n{json.dumps(queries[1])}\n\n")

    # Write qrels.tsv with blank lines
    qrels_lines = ["q1\timg1\t1", "", "q2\timg2\t1"]
    (dir_path / "qrels.tsv").write_text("\n".join(qrels_lines))

    result = load_owndata(dir_path)

    assert len(result["items"]) == 2
    assert len(result["queries"]) == 2
    assert len(result["qrels"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
