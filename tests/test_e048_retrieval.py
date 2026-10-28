"""Tests for benchmarks/e048/retrieval.py (red-first)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the benchmark module importable from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.e048.retrieval import load_manifest, recall_at_k, run, write_results, rank_topk

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
# Tests for load_manifest: distractor entries
# ---------------------------------------------------------------------------

def test_load_manifest_distractor_without_captions_key(tmp_path):
    """A distractor entry may omit the captions key and is accepted."""
    manifest = tmp_path / "manifest.jsonl"
    dummy = tmp_path / "d0.txt"
    dummy.write_text("content")

    items = [
        {"id": "d0", "path": str(dummy), "distractor": True},
    ]
    manifest.write_text("\n".join(json.dumps(it) for it in items) + "\n")

    loaded = load_manifest(manifest)
    assert len(loaded) == 1
    assert loaded[0]["distractor"] is True


def test_load_manifest_distractor_with_empty_captions(tmp_path):
    """A distractor entry with an empty captions list is accepted."""
    manifest = tmp_path / "manifest.jsonl"
    dummy = tmp_path / "d0.txt"
    dummy.write_text("content")

    items = [
        {"id": "d0", "path": str(dummy), "distractor": True, "captions": []},
    ]
    manifest.write_text("\n".join(json.dumps(it) for it in items) + "\n")

    loaded = load_manifest(manifest)
    assert len(loaded) == 1
    assert loaded[0]["captions"] == []


def test_load_manifest_distractor_with_captions_raises(tmp_path):
    """A distractor entry carrying non-empty captions is refused."""
    manifest = tmp_path / "manifest.jsonl"
    dummy = tmp_path / "d0.txt"
    dummy.write_text("content")

    items = [
        {"id": "d0", "path": str(dummy), "distractor": True, "captions": ["x"]},
    ]
    manifest.write_text("\n".join(json.dumps(it) for it in items) + "\n")

    with pytest.raises(ValueError, match="distractor entries carry no captions"):
        load_manifest(manifest)


def test_load_manifest_distractor_non_bool_raises(tmp_path):
    """A distractor value that is not a bool is refused."""
    manifest = tmp_path / "manifest.jsonl"
    dummy = tmp_path / "d0.txt"
    dummy.write_text("content")

    items = [
        {"id": "d0", "path": str(dummy), "distractor": "yes"},
    ]
    manifest.write_text("\n".join(json.dumps(it) for it in items) + "\n")

    with pytest.raises(ValueError, match="distractor must be a bool"):
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


# ---------------------------------------------------------------------------
# Fake embedder with fixed lookup vectors for the distractor run test (no ML
# dependencies: works even when torch/transformers/sentence_transformers/
# open_clip are unimportable)
# ---------------------------------------------------------------------------

class DistractorRunEmbedder:
    """Deterministic embedder mapping path-stems and query texts to fixed
    vectors so the distractor-outranks proof is exact."""

    def __init__(self):
        self.dim = 3
        self.name = "distractor-test-embedder"
        self.modality = "image"
        # stem -> vector
        self._item_vecs = {
            "item0": [0.9, 0.1, 0.0],
            "item1": [0.1, 0.9, 0.0],
            "d0": [1.0, 0.0, 0.0],
            "d1": [1.0, 0.0, 0.0],
            "d2": [1.0, 0.0, 0.0],
        }
        # query text -> vector
        self._query_vecs = {
            "q_item0": [1.0, 0.0, 0.0],
            "q_item1": [0.0, 1.0, 0.0],
        }

    def embed_items(self, paths):
        vecs = [self._item_vecs[Path(p).stem] for p in paths]
        return np.array(vecs, dtype=np.float32)

    def embed_queries(self, texts):
        vecs = [self._query_vecs[t] for t in texts]
        return np.array(vecs, dtype=np.float32)

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


# ---------------------------------------------------------------------------
# Tests for run
# ---------------------------------------------------------------------------

def test_run_distractors_compete(tmp_path):
    """2 captioned items + 3 distractors.

    Distractors are embedded as items (n_items counts them, they stay in
    item_paths and compete in ranking) but contribute no queries. A distractor
    whose vector equals a query's vector outranks that query's true item.
    """
    tmp_path.joinpath("item0.txt").write_text("x")
    tmp_path.joinpath("item1.txt").write_text("x")
    tmp_path.joinpath("d0.txt").write_text("x")
    tmp_path.joinpath("d1.txt").write_text("x")
    tmp_path.joinpath("d2.txt").write_text("x")

    items = [
        {"id": "item0", "path": str(tmp_path / "item0.txt"), "captions": ["q_item0"]},
        {"id": "item1", "path": str(tmp_path / "item1.txt"), "captions": ["q_item1"]},
        {"id": "d0", "path": str(tmp_path / "d0.txt"), "distractor": True},
        {"id": "d1", "path": str(tmp_path / "d1.txt"), "distractor": True},
        {"id": "d2", "path": str(tmp_path / "d2.txt"), "distractor": True},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(it) for it in items) + "\n")

    caption_count = sum(len(it["captions"]) for it in items if not it.get("distractor"))

    result = run(manifest, DistractorRunEmbedder())

    assert result["n_items"] == 5
    assert result["n_queries"] == caption_count
    assert result["n_distractors"] == 3
    # d0/d1/d2 share q_item0's vector and outrank the true item0, so Recall@1
    # drops to 0.5 (it would be 1.0 if distractors did not compete in ranking).
    assert result["metrics"]["R@1"] == pytest.approx(0.5)
    assert result["metrics"]["R@5"] == 1.0
    assert result["metrics"]["R@10"] == 1.0



def test_rank_topk_basic():
    """Test rank_topk function returns correct top-K indices and scores."""
    # Create test vectors
    item_vecs = np.array([
        [0.9, 0.1, 0.0],   # item0
        [0.1, 0.8, 0.1],   # item1
        [0.0, 0.1, 0.9],   # item2
        [0.2, 0.2, 0.7],   # item3
    ], dtype=np.float32)
    
    query_vecs = np.array([
        [0.9, 0.0, 0.0],   # query0 matches item0 best
        [0.0, 0.9, 0.0],   # query1 matches item1 best
    ], dtype=np.float32)
    
    # Test rank_topk with k=2
    indices, scores = rank_topk(query_vecs, item_vecs, k=2)
    
    # Check shapes
    assert indices.shape == (2, 2)
    assert scores.shape == (2, 2)
    
    # Check ordering: for query0, item0 should be first, item3 second (after item0)
    # Scores: query0 vs items: [0.81, 0.09, 0.0, 0.18] -> ranked: [0.81, 0.18, 0.09, 0.0]
    # So top2: item0 (score 0.81), item3 (score 0.18)
    np.testing.assert_array_equal(indices[0], [0, 3])
    np.testing.assert_allclose(scores[0], [0.81, 0.18])
    
    # Check query1: scores: [0.09, 0.72, 0.09, 0.18] -> ranked: [0.72, 0.18, 0.09, 0.09]
    np.testing.assert_array_equal(indices[1], [1, 3])
    np.testing.assert_allclose(scores[1], [0.72, 0.18])


def test_rank_topk_tie_breaking():
    """Test that rank_topk breaks ties by lower index."""
    # Two items have identical vectors
    item_vecs = np.array([
        [0.7, 0.7, 0.0],   # item0
        [0.7, 0.7, 0.0],   # item1 (same as item0)
        [0.1, 0.1, 0.1],   # item2
    ], dtype=np.float32)
    
    query_vecs = np.array([
        [0.7, 0.7, 0.0],   # Matches both item0 and item1
    ], dtype=np.float32)
    
    indices, scores = rank_topk(query_vecs, item_vecs, k=3)
    
    # With tie-breaking by lower index, item0 should come first, then item1
    assert indices[0, 0] == 0
    assert indices[0, 1] == 1
    assert indices[0, 2] == 2
    
    # Scores should be identical for item0 and item1 (0.7*0.7 + 0.7*0.7 + 0.0*0.0 = 0.98)
    np.testing.assert_allclose(scores[0, 0], 0.98)
    np.testing.assert_allclose(scores[0, 1], 0.98)


def test_rank_topk_agrees_with_recall():
    """Test that rank_topk agrees with recall_at_k on a random fixture."""
    # Create test data
    item_vecs = np.array([
        [1.0, 0.0, 0.0],  # item0
        [0.0, 1.0, 0.0],  # item1
        [0.0, 0.0, 1.0],  # item2
    ], dtype=np.float32)
    
    query_vecs = np.array([
        [0.9, 0.1, 0.0],  # query0: mainly item0
        [0.0, 0.9, 0.1],  # query1: mainly item1
        [0.0, 0.1, 0.9],  # query2: mainly item2
    ], dtype=np.float32)
    
    query_item_index = np.array([0, 1, 2])  # Each query matches its own item
    
    # Test rank_topk with k=3 (covers all items)
    indices, scores = rank_topk(query_vecs, item_vecs, k=3)
    
    # For each query, check that the true item is in the ranking
    for q_idx in range(len(query_vecs)):
        true_item_idx = query_item_index[q_idx]
        true_rank = np.where(indices[q_idx] == true_item_idx)[0][0]
        
        # true_rank should be less than 3 (since we requested k=3)
        assert true_rank < 3
        
        # Also check with recall_at_k for consistency
        metrics = recall_at_k(query_vecs, item_vecs, query_item_index, ks=(1, 2, 3))
        
        # R@3 should be 1.0 since all items should be in top 3
        assert metrics["R@3"] == 1.0


@pytest.fixture
def dummy_manifest_with_captions(tmp_path):
    """Write a manifest with captioned items for rank_topk tests."""
    manifest = tmp_path / "manifest.jsonl"
    items = []
    for i in range(3):
        dummy = tmp_path / f"item{i}.txt"
        dummy.write_text(f"Dummy content {i}")
        items.append({"id": f"item{i}", "path": str(dummy), "captions": [f"Caption {i}", f"Another caption {i}"]})
    content = "\n".join(json.dumps(it) for it in items) + "\n"
    manifest.write_text(content)
    return manifest, items


def test_run_with_rankings_path(tmp_path):
    """Test that run() writes rankings file when rankings_path is provided."""
    manifest = tmp_path / "manifest.jsonl"
    items = []
    for i in range(3):
        dummy = tmp_path / f"item{i}.txt"
        dummy.write_text(f"content {i}")
        items.append({"id": f"item{i}", "path": str(dummy), "captions": [f"Caption {i}"]})
    content = "\n".join(json.dumps(item) for item in items) + "\n"
    manifest.write_text(content)
    
    rankings_path = tmp_path / "rankings.jsonl"
    
    # Create a simple embedder for testing
    class TestEmbedder:
        def __init__(self):
            self.dim = 3
            self.name = "test-embedder"
            self.modality = "image"
            
        def embed_items(self, paths):
            # Create simple vectors based on path index
            vecs = []
            for path in paths:
                idx = int(Path(path).stem.split("item")[1])
                vec = [0.0] * self.dim
                vec[idx % self.dim] = 1.0
                vecs.append(vec)
            return np.array(vecs, dtype=np.float32)
            
        def embed_queries(self, texts):
            # Simple one-hot encoding
            vecs = []
            for text in texts:
                idx = int(text.split()[1])  # Extract index from "Caption i"
                vec = [0.0] * self.dim
                vec[idx % self.dim] = 1.0
                vecs.append(vec)
            return np.array(vecs, dtype=np.float32)
            
        def provenance(self):
            return {
                "model_id": "test-model",
                "revision": "1.0",
                "model_files": {"test.onnx": "a" * 64},
                "runtime": "test",
                "runtime_version": "1.0",
                "execution_provider": "CPU",
                "dtype": "float32",
            }
    
    embedder = TestEmbedder()
    
    # Run with rankings_path
    result = run(manifest, embedder, rankings_path=str(rankings_path))
    
    # Check that rankings file was created
    assert rankings_path.exists()
    
    # Check file contents
    with open(rankings_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    
    # Should have one line per query
    assert len(lines) == 3  # 3 items * 1 caption each = 3 queries
    
    # Check format of each entry
    for entry in lines:
        assert "q" in entry
        assert "caption" in entry
        assert "true_item" in entry
        assert "true_rank" in entry
        assert "topk" in entry
        
        # Check true_rank is 0-based and less than 10 (we use k=10 in rank_topk)
        assert 0 <= entry["true_rank"] < 10
        
        # Check topk has format [[item_id, score], ...]
        for item_id, score in entry["topk"]:
            assert isinstance(item_id, str)
            assert isinstance(score, (int, float))
            
        # Check that true_item appears in topk
        assert entry["true_item"] in [item_id for item_id, _ in entry["topk"]]
    
    # Check that result contains rankings_path and rankings_sha256
    assert "rankings_path" in result
    assert "rankings_sha256" in result
    assert result["rankings_path"] == str(rankings_path)
    
    # Verify SHA256 matches file
    import hashlib
    sha256_hash = hashlib.sha256()
    with open(rankings_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    expected_sha256 = sha256_hash.hexdigest()
    assert result["rankings_sha256"] == expected_sha256


def test_run_without_rankings_path(tmp_path):
    """Test that run() doesn't write rankings file when rankings_path is None."""
    manifest = tmp_path / "manifest.jsonl"
    items = []
    for i in range(2):
        dummy = tmp_path / f"item{i}.txt"
        dummy.write_text(f"content {i}")
        items.append({"id": f"item{i}", "path": str(dummy), "captions": [f"Caption {i}"]})
    content = "\n".join(json.dumps(item) for item in items) + "\n"
    manifest.write_text(content)
    
    rankings_path = tmp_path / "rankings.jsonl"
    
    # Create a simple embedder
    class TestEmbedder:
        def __init__(self):
            self.dim = 3
            self.name = "test-embedder"
            self.modality = "image"
            
        def embed_items(self, paths):
            vecs = []
            for path in paths:
                idx = int(Path(path).stem.split("item")[1])
                vec = [0.0] * self.dim
                vec[idx % self.dim] = 1.0
                vecs.append(vec)
            return np.array(vecs, dtype=np.float32)
            
        def embed_queries(self, texts):
            vecs = []
            for text in texts:
                idx = int(text.split()[1])
                vec = [0.0] * self.dim
                vec[idx % self.dim] = 1.0
                vecs.append(vec)
            return np.array(vecs, dtype=np.float32)
            
        def provenance(self):
            return {
                "model_id": "test-model",
                "revision": "1.0",
                "model_files": {"test.onnx": "a" * 64},
                "runtime": "test",
                "runtime_version": "1.0",
                "execution_provider": "CPU",
                "dtype": "float32",
            }
    
    embedder = TestEmbedder()
    
    # Run without rankings_path (default None)
    result = run(manifest, embedder)
    
    # Check that rankings file was NOT created
    assert not rankings_path.exists()
    
    # Check that result does NOT contain rankings_path or rankings_sha256
    assert "rankings_path" not in result
    assert "rankings_sha256" not in result
