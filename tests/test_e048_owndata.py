"""Tests for E-048 own-data loader and privacy guard."""

from __future__ import annotations

import hashlib
from pathlib import Path

from benchmarks.e048.owndata import assert_outside_repo, load_owndata


def _create_fixture(tmp_path: Path) -> Path:
    """Create a valid own-data fixture directory with minimal data.
    
    Structure:
    - items.jsonl: 2 items with tiny files
    - texts.jsonl: 1 text entry  
    - queries.jsonl: 2 queries
    - qrels.tsv: 3 qrels lines
    
    Returns the path to the fixture directory.
    """
    fixture_dir = tmp_path / "owndata_fixture"
    fixture_dir.mkdir()
    
    # Create tiny image files
    (fixture_dir / "img1.png").write_bytes(b"PNG\x89\x00\x0D\x0A\x1A\x0A")
    (fixture_dir / "img2.jpg").write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01")
    
    # items.jsonl
    items_lines = [
        '{"id": "img1", "path": "img1.png", "mime": "image/png"}',
        '{"id": "img2", "path": "img2.jpg", "mime": "image/jpeg"}',
    ]
    (fixture_dir / "items.jsonl").write_text("\n".join(items_lines) + "\n")
    
    # texts.jsonl (optional)
    (fixture_dir / "texts.jsonl").write_text('{"id": "text1", "text": "This is a sample text entry."}\n')
    
    # queries.jsonl
    queries_lines = [
        '{"qid": "q1", "text": "What is this image?", "author": "lab-user1", "written_at": "2024-01-15T10:30:00Z"}',
        '{"qid": "q2", "text": "Describe the scene", "author": "lab-user2", "written_at": "2024-01-16T14:45:00Z"}',
    ]
    (fixture_dir / "queries.jsonl").write_text("\n".join(queries_lines) + "\n")
    
    # qrels.tsv
    qrels_lines = [
        "q1\timg1\t1",
        "q1\timg2\t1", 
        "q2\ttext1\t1",
    ]
    (fixture_dir / "qrels.tsv").write_text("\n".join(qrels_lines) + "\n")
    
    return fixture_dir


def test_load_valid_fixture(tmp_path):
    """Test that a valid fixture loads with correct counts and qrels_sha256."""
    fixture_dir = _create_fixture(tmp_path)
    
    result = load_owndata(fixture_dir)
    
    # Check directory path
    assert "dir" in result
    assert result["dir"] == str(fixture_dir.resolve())
    
    # Check items count
    assert "items" in result
    assert len(result["items"]) == 2
    assert result["items"][0]["id"] == "img1"
    assert result["items"][0]["path"] == "img1.png"
    assert result["items"][0]["mime"] == "image/png"
    assert "abs_path" in result["items"][0]
    assert Path(result["items"][0]["abs_path"]).exists()
    
    assert result["items"][1]["id"] == "img2"
    
    # Check texts count (optional field)
    assert "texts" in result
    assert len(result["texts"]) == 1
    assert result["texts"][0]["id"] == "text1"
    
    # Check queries count
    assert "queries" in result
    assert len(result["queries"]) == 2
    assert result["queries"][0]["qid"] == "q1"
    assert result["queries"][0]["text"] == "What is this image?"
    
    # Check qrels structure
    assert "qrels" in result
    assert result["qrels"]["q1"] == {"img1", "img2"}
    assert result["qrels"]["q2"] == {"text1"}
    
    # Check qrels_sha256 (compute expected hash)
    qrels_content = "q1\timg1\t1\nq1\timg2\t1\nq2\ttext1\t1\n"
    expected_hash = hashlib.sha256(qrels_content.encode()).hexdigest()
    assert result["qrels_sha256"] == expected_hash


def test_fixture_with_git_directory_raises(tmp_path):
    """Test that a fixture containing a .git directory raises ValueError."""
    fixture_dir = tmp_path / "owndata_fixture_git"
    fixture_dir.mkdir()
    
    # Create .git directory (not file)
    (fixture_dir / ".git").mkdir()
    
    # Create minimal valid files
    (fixture_dir / "items.jsonl").write_text('{"id": "img1", "path": "img1.png", "mime": "image/png"}\n')
    (fixture_dir / "queries.jsonl").write_text('{"qid": "q1", "text": "test", "author": "user", "written_at": "2024-01-01T00:00:00Z"}\n')
    (fixture_dir / "qrels.tsv").write_text("q1\timg1\t1\n")
    
    # Try to load - should raise ValueError
    try:
        load_owndata(fixture_dir)
        assert False, "Expected ValueError when fixture contains .git directory"
    except ValueError as e:
        assert "own-data path is inside a git tree" in str(e)
        assert str(fixture_dir) in str(e)


def test_fixture_with_git_file_raises(tmp_path):
    """Test that a fixture containing a .git file raises ValueError."""
    fixture_dir = tmp_path / "owndata_fixture_git_file"
    fixture_dir.mkdir()
    
    # Create .git file (not directory)
    (fixture_dir / ".git").write_text("git repo file")
    
    # Create minimal valid files
    (fixture_dir / "items.jsonl").write_text('{"id": "img1", "path": "img1.png", "mime": "image/png"}\n')
    (fixture_dir / "queries.jsonl").write_text('{"qid": "q1", "text": "test", "author": "user", "written_at": "2024-01-01T00:00:00Z"}\n')
    (fixture_dir / "qrels.tsv").write_text("q1\timg1\t1\n")
    
    # Try to load - should raise ValueError
    try:
        load_owndata(fixture_dir)
        assert False, "Expected ValueError when fixture contains .git file"
    except ValueError as e:
        assert "own-data path is inside a git tree" in str(e)
        assert str(fixture_dir) in str(e)


def test_symlink_from_outside_into_git_tree_raises(tmp_path):
    """Test that a symlink pointing into a .git tree raises ValueError."""
    fixture_dir = tmp_path / "owndata_fixture_normal"
    fixture_dir.mkdir()
    
    # Create .git directory inside fixture
    (fixture_dir / ".git").mkdir()
    
    # Create minimal valid files
    (fixture_dir / "items.jsonl").write_text('{"id": "img1", "path": "img1.png", "mime": "image/png"}\n')
    (fixture_dir / "queries.jsonl").write_text('{"qid": "q1", "text": "test", "author": "user", "written_at": "2024-01-01T00:00:00Z"}\n')
    (fixture_dir / "qrels.tsv").write_text("q1\timg1\t1\n")
    
    # Create a symlink from outside the fixture pointing inside
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    symlink_path = outside_dir / "link_to_fixture"
    symlink_path.symlink_to(fixture_dir)
    
    # Try to load via symlink - should raise ValueError
    try:
        load_owndata(symlink_path)
        assert False, "Expected ValueError when loading via symlink to .git tree"
    except ValueError as e:
        assert "own-data path is inside a git tree" in str(e)
        # Path might be resolved to fixture_dir


def test_item_path_escapes_directory_raises(tmp_path):
    """Test that an item with a path like '../x.png' raises ValueError."""
    fixture_dir = tmp_path / "owndata_fixture_escape"
    fixture_dir.mkdir()
    
    # Create items.jsonl with path that escapes directory
    items_lines = [
        '{"id": "img1", "path": "img1.png", "mime": "image/png"}',
        '{"id": "img2", "path": "../escape.png", "mime": "image/png"}',
    ]
    (fixture_dir / "items.jsonl").write_text("\n".join(items_lines) + "\n")
    
    # Create minimal other files
    (fixture_dir / "queries.jsonl").write_text('{"qid": "q1", "text": "test", "author": "user", "written_at": "2024-01-01T00:00:00Z"}\n')
    (fixture_dir / "qrels.tsv").write_text("q1\timg1\t1\n")
    
    # Try to load - should raise ValueError
    try:
        load_owndata(fixture_dir)
        assert False, "Expected ValueError when item path escapes directory"
    except ValueError as e:
        assert "resolves outside" in str(e) or "outside dir" in str(e)


def test_duplicate_ids_across_items_and_texts_raises(tmp_path):
    """Test that duplicate ids across items.jsonl and texts.jsonl raises ValueError."""
    fixture_dir = tmp_path / "owndata_fixture_dup"
    fixture_dir.mkdir()
    
    # Create items.jsonl and texts.jsonl with same id
    (fixture_dir / "items.jsonl").write_text('{"id": "img1", "path": "img1.png", "mime": "image/png"}\n')
    (fixture_dir / "texts.jsonl").write_text('{"id": "img1", "text": "Same id as item"}\n')
    
    # Create minimal other files
    (fixture_dir / "queries.jsonl").write_text('{"qid": "q1", "text": "test", "author": "user", "written_at": "2024-01-01T00:00:00Z"}\n')
    (fixture_dir / "qrels.tsv").write_text("q1\timg1\t1\n")
    
    # Try to load - should raise ValueError
    try:
        load_owndata(fixture_dir)
        assert False, "Expected ValueError when duplicate ids across items and texts"
    except ValueError as e:
        assert "duplicate id" in str(e) or "duplicate" in str(e)


def test_duplicate_qids_raises(tmp_path):
    """Test that duplicate qids in queries.jsonl raises ValueError."""
    fixture_dir = tmp_path / "owndata_fixture_dup_qid"
    fixture_dir.mkdir()
    
    # Create queries.jsonl with duplicate qid
    queries_lines = [
        '{"qid": "q1", "text": "First query", "author": "user", "written_at": "2024-01-01T00:00:00Z"}',
        '{"qid": "q1", "text": "Duplicate query", "author": "user2", "written_at": "2024-01-02T00:00:00Z"}',
    ]
    (fixture_dir / "queries.jsonl").write_text("\n".join(queries_lines) + "\n")
    
    # Create minimal other files
    (fixture_dir / "items.jsonl").write_text('{"id": "img1", "path": "img1.png", "mime": "image/png"}\n')
    (fixture_dir / "qrels.tsv").write_text("q1\timg1\t1\n")
    
    # Try to load - should raise ValueError
    try:
        load_owndata(fixture_dir)
        assert False, "Expected ValueError when duplicate qids"
    except ValueError as e:
        assert "duplicate qid" in str(e) or "duplicate" in str(e)


def test_missing_required_key_raises(tmp_path):
    """Test that missing required keys raise ValueError."""
    fixture_dir = tmp_path / "owndata_fixture_missing_key"
    fixture_dir.mkdir()
    
    # Create items.jsonl with missing required key
    (fixture_dir / "items.jsonl").write_text('{"id": "img1", "path": "img1.png"}\n')  # Missing mime
    
    # Create minimal other files  
    (fixture_dir / "queries.jsonl").write_text('{"qid": "q1", "text": "test", "author": "user", "written_at": "2024-01-01T00:00:00Z"}\n')
    (fixture_dir / "qrels.tsv").write_text("q1\timg1\t1\n")
    
    # Try to load - should raise ValueError
    try:
        load_owndata(fixture_dir)
        assert False, "Expected ValueError when missing required key"
    except ValueError as e:
        assert "missing key" in str(e) or "required" in str(e)


def test_invalid_qrels_line_raises(tmp_path):
    """Test that qrels lines with not exactly 3 tab-separated fields raise ValueError."""
    fixture_dir = tmp_path / "owndata_fixture_invalid_qrels"
    fixture_dir.mkdir()
    
    # Create valid files
    (fixture_dir / "items.jsonl").write_text('{"id": "img1", "path": "img1.png", "mime": "image/png"}\n')
    (fixture_dir / "queries.jsonl").write_text('{"qid": "q1", "text": "test", "author": "user", "written_at": "2024-01-01T00:00:00Z"}\n')
    
    # Create qrels.tsv with invalid line (only 2 fields instead of 3)
    (fixture_dir / "qrels.tsv").write_text("q1\timg1\n")  # Missing third field
    
    # Try to load - should raise ValueError
    try:
        load_owndata(fixture_dir)
        assert False, "Expected ValueError when qrels line has invalid format"
    except ValueError as e:
        assert "3 tab-separated fields" in str(e) or "not exactly" in str(e)


def test_qrels_invalid_rel_field_raises(tmp_path):
    """Test that qrels with rel field not equal to '1' raises ValueError."""
    fixture_dir = tmp_path / "owndata_fixture_invalid_rel"
    fixture_dir.mkdir()
    
    # Create valid files
    (fixture_dir / "items.jsonl").write_text('{"id": "img1", "path": "img1.png", "mime": "image/png"}\n')
    (fixture_dir / "queries.jsonl").write_text('{"qid": "q1", "text": "test", "author": "user", "written_at": "2024-01-01T00:00:00Z"}\n')
    
    # Create qrels.tsv with invalid rel field
    (fixture_dir / "qrels.tsv").write_text("q1\timg1\t0\n")  # Should be '1', not '0'
    
    # Try to load - should raise ValueError
    try:
        load_owndata(fixture_dir)
        assert False, "Expected ValueError when qrels rel field is not '1'"
    except ValueError as e:
        assert "rel field not equal to \"1\"" in str(e) or "not equal to" in str(e)


def test_qrels_qid_not_in_queries_raises(tmp_path):
    """Test that qrels with qid not in queries.jsonl raises ValueError."""
    fixture_dir = tmp_path / "owndata_fixture_qid_not_in_queries"
    fixture_dir.mkdir()
    
    # Create valid files
    (fixture_dir / "items.jsonl").write_text('{"id": "img1", "path": "img1.png", "mime": "image/png"}\n')
    (fixture_dir / "queries.jsonl").write_text('{"qid": "q1", "text": "test", "author": "user", "written_at": "2024-01-01T00:00:00Z"}\n')
    
    # Create qrels.tsv with qid not in queries
    (fixture_dir / "qrels.tsv").write_text("q2\timg1\t1\n")  # q2 doesn't exist in queries
    
    # Try to load - should raise ValueError
    try:
        load_owndata(fixture_dir)
        assert False, "Expected ValueError when qrels qid not in queries"
    except ValueError as e:
        assert "qid not in queries" in str(e) or "not in queries" in str(e)


def test_qrels_doc_id_not_in_items_or_texts_raises(tmp_path):
    """Test that qrels with doc_id not in items or texts.jsonl raises ValueError."""
    fixture_dir = tmp_path / "owndata_fixture_doc_id_not_found"
    fixture_dir.mkdir()
    
    # Create valid files
    (fixture_dir / "items.jsonl").write_text('{"id": "img1", "path": "img1.png", "mime": "image/png"}\n')
    (fixture_dir / "queries.jsonl").write_text('{"qid": "q1", "text": "test", "author": "user", "written_at": "2024-01-01T00:00:00Z"}\n')
    
    # Create qrels.tsv with doc_id not in items or texts
    (fixture_dir / "qrels.tsv").write_text("q1\tunknown_doc\t1\n")  # unknown_doc doesn't exist
    
    # Try to load - should raise ValueError
    try:
        load_owndata(fixture_dir)
        assert False, "Expected ValueError when qrels doc_id not in items or texts"
    except ValueError as e:
        assert "doc_id in neither items nor texts" in str(e) or "not in items" in str(e)