"""Tests for benchmarks/e048/retrieval.py (red-first)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the benchmark module importable from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.e048.retrieval import load_manifest, recall_at_k, write_results

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Use json module
import json

@pytest.fixture
def dummy_manifest(tmp_path):
    """Write a 5-item manifest with dummy files."""
    manifest = tmp_path / "manifest.jsonl"
    items = []
    for i in range(5):
        # Create a dummy file with some content
        dummy = tmp_path / f"item{i}.txt"
        dummy.write_text(f"Dummy content {i}")
        items.append({"id": f"item{i}", "path": str(dummy), "captions": [f"Caption for item {i}", f"Another caption for item {i}"]})
    content = '\n'.join(json.dumps(it) for it in items) + '\n'
    manifest.write_text(content)
    return manifest, items

# ---------------------------------------------------------------------------
# Fake embedder for tests
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Deterministic one-hot embedder for testing."""
    
    def __init__(self, dim=10):
        self.dim = dim
        self.name = "fake-embedder"
        self.modality = "image"
    
    def embed_queries(self, texts):
        """One-hot for each caption."""
        # Create one-hot vectors for the 5 items
        # Mapping: each caption of item i -> e_i, one wrong caption -> e_{i+1 mod 5}
        vectors = []
        for text in texts:
            # Extract item index from text if possible
            if "item-" in text:
                idx = int(text.split("item-")[1].split()[0])
                vec = [0.0] * self.dim
                vec[idx % self.dim] = 1.0
            else:
                # Default to first item if pattern not found
                vec = [0.0] * self.dim
                vec[0] = 1.0
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)
    
    def embed_items(self, paths):
        """One-hot for each item."""
        vectors = []
        for path in paths:
            # Extract item index from path if possible
            idx = int(Path(path).stem.split("item")[1])
            vec = [0.0] * self.dim
            vec[idx % self.dim] = 1.0
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)
    
    def provenance(self):
        return {
            "model_id": "fake-model",
            "revision": "1.0",
            "model_files": {"fake.onnx": "a" * 64},
            "runtime": "onnxruntime",
            "runtime_version": "1.16.0",
            "execution_provider": "CPUExecutionProvider",
            "dtype": "float32",
        }

# Lazy import of numpy
import numpy as np

# ---------------------------------------------------------------------------
# Tests for load_manifest
# ---------------------------------------------------------------------------

def test_load_manifest(tmp_path):
    """Test loading a valid manifest."""
    manifest = tmp_path / "manifest.jsonl"
    items = []
    for i in range(3):
        dummy = tmp_path / f"item{i}.txt"
        dummy.write_text(f"content {i}")
        items.append({"id": f"item{i}", "path": str(dummy), "captions": [f"Caption {i}", f"Another caption {i}"]})
    content = '\n'.join(json.dumps(it) for it in items) + '\n'
    manifest.write_text(content)
    
    loaded = load_manifest(manifest)
    
    assert len(loaded) == 3
    assert loaded[0]["id"] == "item0"
    assert loaded[0]["path"] == str(tmp_path / "item0.txt")
    assert loaded[0]["captions"] == ["Caption 0", "Another caption 0"]

def test_load_manifest_rejects_duplicate_id(tmp_path):
    """Test that duplicate IDs cause ValueError."""
    manifest = tmp_path / "manifest.jsonl"
    dummy = tmp_path / "dummy.txt"
    dummy.write_text("content")
    
    items = [
        {"id": "item0", "path": str(dummy), "captions": ["Caption 0"]},
        {"id": "item0", "path": str(dummy), "captions": ["Caption 1"]},  # Duplicate
    ]
    content = '\n'.join(json.dumps(it) for it in items) + '\n'
    manifest.write_text(content)
    
    with pytest.raises(ValueError, match="duplicate id"):
        load_manifest(manifest)

def test_load_manifest_rejects_empty_captions(tmp_path):
    """Test that empty captions cause ValueError."""
    manifest = tmp_path / "manifest.jsonl"
    dummy = tmp_path / "dummy.txt"
    dummy.write_text("content")
    
    items = [
        {"id": "item0", "path": str(dummy), "captions": []},  # Empty captions
    ]
    content = '\n'.join(json.dumps(it) for it in items) + '\n'
    manifest.write_text(content)
    
    with pytest.raises(ValueError, match="empty captions"):
        load_manifest(manifest)

def test_load_manifest_skips_blank_lines(tmp_path):
    """Test that blank lines are skipped."""
    manifest = tmp_path / "manifest.jsonl"
    dummy = tmp_path / "dummy.txt"
    dummy.write_text("content")
    
    item0 = {"id": "item0", "path": str(dummy), "captions": ["Caption 0"]}
    item1 = {"id": "item1", "path": str(dummy), "captions": ["Caption 1"]}
    content = json.dumps(item0) + '\n\n' + json.dumps(item1) + '\n\n'
    manifest.write_text(content)
    
    loaded = load_manifest(manifest)
    assert len(loaded) == 2

def test_load_manifest_rejects_missing_key(tmp_path):
    """Test that missing keys cause ValueError."""
    manifest = tmp_path / "manifest.jsonl"
    dummy = tmp_path / "dummy.txt"
    dummy.write_text("content")
    
    items = [
        {"id": "item0", "path": str(dummy)},  # Missing captions
    ]
    content = '\n'.join(json.dumps(it) for it in items) + '\n'
    manifest.write_text(content)
    
    with pytest.raises(ValueError, match="missing key"):
        load_manifest(manifest)

# ---------------------------------------------------------------------------
# Tests for recall_at_k
# ---------------------------------------------------------------------------

def test_recall_perfect_and_partial():
    """Test perfect and partial recall with fake embedder vectors."""
    # Create fake vectors for 5 items: each item i has vector e_i (one-hot)
    item_vecs = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],  # item0
        [0.0, 1.0, 0.0, 0.0, 0.0],  # item1
        [0.0, 0.0, 1.0, 0.0, 0.0],  # item2
        [0.0, 0.0, 0.0, 1.0, 0.0],  # item3
        [0.0, 0.0, 0.0, 0.0, 1.0],  # item4
    ], dtype=np.float32)
    
    # Query vectors: each caption is a query
    # For each of 5 items, we have 2 captions (10 total queries)
    # Caption i -> item i (perfect match)
    query_vecs = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],  # caption of item0 -> item0
        [1.0, 0.0, 0.0, 0.0, 0.0],  # second caption of item0 -> item0
        [0.0, 1.0, 0.0, 0.0, 0.0],  # caption of item1 -> item1
        [0.0, 1.0, 0.0, 0.0, 0.0],  # second caption of item1 -> item1
        [0.0, 0.0, 1.0, 0.0, 0.0],  # caption of item2 -> item2
        [0.0, 0.0, 1.0, 0.0, 0.0],  # second caption of item2 -> item2
        [0.0, 0.0, 0.0, 1.0, 0.0],  # caption of item3 -> item3
        [0.0, 0.0, 0.0, 1.0, 0.0],  # second caption of item3 -> item3
        [0.0, 0.0, 0.0, 0.0, 1.0],  # caption of item4 -> item4
        [0.0, 0.0, 0.0, 0.0, 1.0],  # second caption of item4 -> item4
    ], dtype=np.float32)
    
    # Create index mapping: item i is at position i in item_vecs
    query_item_index = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    
    # Test perfect recall
    metrics = recall_at_k(query_vecs, item_vecs, query_item_index, ks=(1, 5, 10))
    
    assert metrics["R@1"] == 1.0  # All 10 queries hit in top 1
    assert metrics["R@5"] == 1.0  # All 10 queries hit in top 5
    assert metrics["R@10"] == 1.0  # All 10 queries hit in top 10
    
    # Now test partial: make some queries not match
    partial_query_vecs = query_vecs.copy()
    partial_query_vecs[0] = [0.0, 1.0, 0.0, 0.0, 0.0]  # caption0 now matches item1
    partial_query_vecs[1] = [0.0, 1.0, 0.0, 0.0, 0.0]  # caption1 now matches item1
    
    metrics = recall_at_k(partial_query_vecs, item_vecs, query_item_index, ks=(1, 5, 10))
    
    assert metrics["R@1"] == 0.8  # 8 out of 10 queries hit in top 1 (2 wrong)
    assert metrics["R@5"] == 1.0  # All 10 queries hit in top 5
    assert metrics["R@10"] == 1.0  # All 10 queries hit in top 10


def test_tie_breaks_by_lower_index():
    """Test that ties are broken by lower item index."""
    # Two identical item vectors (item0 and item1 are the same)
    item_vecs = np.array([
        [0.7, 0.7, 0.0],  # item0
        [0.7, 0.7, 0.0],  # item1 (same as item0)
        [0.1, 0.1, 0.1],  # item2
    ], dtype=np.float32)
    
    # Query for item1 (index 1) has identical vector to item0 (index 0)
    query_vecs = np.array([
        [0.7, 0.7, 0.0],  # Matches both item0 and item1
    ], dtype=np.float32)
    
    query_item_index = np.array([1])  # Query is for item1
    
    metrics = recall_at_k(query_vecs, item_vecs, query_item_index, ks=(1, 5, 10))
    
    # With tie-breaking by lower index, item0 (index 0) comes first,
    # so query (looking for item1) will NOT hit at R@1
    assert metrics["R@1"] == 0.0
    assert metrics["R@5"] == 1.0  # Will hit at R@5
    assert metrics["R@10"] == 1.0  # Will hit at R@10

# ---------------------------------------------------------------------------
# Tests for write_results
# ---------------------------------------------------------------------------

def test_write_results_refuses_missing_provenance(tmp_path):
    """Test that write_results refuses when provenance is missing keys."""
    payload = {
        "experiment": "E-048",
        "arm": "test-arm",
        "modality": "image",
        "dataset": "test-dataset",
        "embedder": "fake",
        "host": "test-host",
        "metrics": {"R@1": 1.0, "R@5": 1.0, "R@10": 1.0},
        "n_items": 10,
        "n_queries": 20,
        # Missing provenance keys
        "provenance": {
            "model_id": "fake-model",
            "revision": "1.0",
            "model_files": {"fake.onnx": "a" * 64},
            "runtime": "onnxruntime",
            "runtime_version": "1.16.0",
            # Missing execution_provider
            "dtype": "float32",
        }
    }
    
    out_path = tmp_path / "results.json"
    
    with pytest.raises(ValueError, match="execution_provider"):
        write_results(out_path, payload)
    
    # File should not exist because write failed
    assert not out_path.exists()

def test_write_results_refuses_invalid_sha256(tmp_path):
    """Test that write_results refuses when sha256 is not 64 hex chars."""
    payload = {
        "experiment": "E-048",
        "arm": "test-arm",
        "modality": "image",
        "dataset": "test-dataset",
        "embedder": "fake",
        "host": "test-host",
        "metrics": {"R@1": 1.0, "R@5": 1.0, "R@10": 1.0},
        "n_items": 10,
        "n_queries": 20,
        "provenance": {
            "model_id": "fake-model",
            "revision": "1.0",
            "model_files": {"fake.onnx": "a" * 64},
            "runtime": "onnxruntime",
            "runtime_version": "1.16.0",
            "execution_provider": "CPUExecutionProvider",
            "dtype": "float32",
        }
    }
    
    # Invalid sha256 - not 64 chars
    payload["provenance"]["model_files"]["fake.onnx"] = "a" * 63
    
    out_path = tmp_path / "results.json"
    
    with pytest.raises(ValueError, match="must be 64-char hex"):
        write_results(out_path, payload)
    
    assert not out_path.exists()

def test_write_results_ok(tmp_path):
    """Test that write_results succeeds with valid payload."""
    payload = {
        "experiment": "E-048",
        "arm": "test-arm",
        "modality": "image",
        "dataset": "test-dataset",
        "embedder": "fake",
        "host": "test-host",
        "metrics": {"R@1": 1.0, "R@5": 1.0, "R@10": 1.0},
        "n_items": 10,
        "n_queries": 20,
        "provenance": {
            "model_id": "fake-model",
            "revision": "1.0",
            "model_files": {"fake.onnx": "a" * 64},
            "runtime": "onnxruntime",
            "runtime_version": "1.16.0",
            "execution_provider": "CPUExecutionProvider",
            "dtype": "float32",
        }
    }
    
    out_path = tmp_path / "results" / "subdir" / "results.json"
    
    write_results(out_path, payload)
    
    assert out_path.exists()
    
    # Read and validate the written JSON
    import json
    with open(out_path) as f:
        written = json.load(f)
    
    # Check required fields
    assert "written_utc" in written
    assert "experiment" in written
    assert written["experiment"] == "E-048"
    
    # Check provenance has all required keys
    for key in ("model_id", "revision", "model_files", "runtime", 
                "runtime_version", "execution_provider", "dtype"):
        assert key in written["provenance"]
    
    # Check model_files values are 64-char hex
    for hash_val in written["provenance"]["model_files"].values():
        assert len(hash_val) == 64
        assert all(c in "0123456789abcdef" for c in hash_val)

def test_cli_end_to_end(tmp_path, monkeypatch):
    """Test the CLI end-to-end with fake embedder."""
    # Import sys at the top of the function
    import sys as _sys
    
    # Create a temporary manifest (outside the loop)
    manifest = tmp_path / "manifest.jsonl"
    for i in range(3):
        dummy = tmp_path / f"item{i}.txt"
        dummy.write_text(f"content {i}")
    
    # Write manifest with all items
    items = []
    for i in range(3):
        dummy = tmp_path / f"item{i}.txt"
        items.append({"id": f"item{i}", "path": str(dummy), "captions": [f"Caption {i}"]})
    
    manifest_content = '\n'.join(json.dumps(item) for item in items) + '\n'
    manifest.write_text(manifest_content)
    
    # Save the fake embedder factory to a temporary module
    test_module = tmp_path / "test_fake_embedder.py"
    test_module.write_text("""
import numpy as np
from benchmarks.e048.retrieval import MediaEmbedder

class FakeEmbedder:
    def __init__(self, dim=10):
        self.name = "fake-embedder"
        self.modality = "image"
        self.dim = dim
    
    def embed_queries(self, texts):
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            vec[0] = 1.0
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)
    
    def embed_items(self, paths):
        vectors = []
        for path in paths:
            vec = [0.0] * self.dim
            vec[0] = 1.0
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)
    
    def provenance(self):
        return {
            "model_id": "fake-model",
            "revision": "1.0",
            "model_files": {"fake.onnx": "a" * 64},
            "runtime": "onnxruntime",
            "runtime_version": "1.16.0",
            "execution_provider": "CPUExecutionProvider",
            "dtype": "float32",
        }

def make_fake(**kwargs):
    return FakeEmbedder()
""")
    
    # Add to sys.path so it can be imported
    _sys.path.insert(0, str(tmp_path))
    
    # Import and patch the module
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_fake", test_module)
    test_fake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_fake)
    
    # Patch the import in the retrieval module
    original_import = __import__
    
    def mock_import(name, *args, **kwargs):
        if name == "test_fake":
            return test_fake
        return original_import(name, *args, **kwargs)
    
    monkeypatch.setattr('builtins.__import__', mock_import)
    
    # Run the CLI
    from benchmarks.e048.retrieval import main
    
    out_path = tmp_path / "output.json"
    argv = [
        "--manifest", str(manifest),
        "--embedder", "test_fake:make_fake",
        "--experiment", "E-048",
        "--arm", "test-arm",
        "--dataset", "test-dataset",
        "--host-label", "test-host",
        "--out", str(out_path),
    ]
    
    # This will try to import test_fake, which should be available now
    try:
        main(argv)
    except SystemExit:
        pass  # CLI calls sys.exit()
    
    # Check output
    assert out_path.exists()
    
    # Clean up sys.path
    _sys.path.remove(str(tmp_path))

# For module-level factory in test file (as required by tests)
# We'll need to expose the make_fake function at module level
# Create a temporary module structure for this

def make_fake(**kwargs):
    """Factory for fake embedder (module-level as required by tests)."""
    return FakeEmbedder()

