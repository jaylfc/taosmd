"""Benchmark harness for E-048: text-to-image/audio/video retrieval.

Implements the MediaEmbedder protocol and metrics collection for the
pre-registered experiment E-048 (docs/research-report.md, PR #551).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# MediaEmbedder Protocol (fixed by the lead, implement EXACTLY these names)
# ---------------------------------------------------------------------------

class MediaEmbedder:
    """Protocol for media embedding models."""
    
    name: str          # short label, e.g. "eg2-fp32-image", "openclip-vitb32-laion2b"
    modality: str      # "image" | "audio" | "video"
    
    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Embed query texts.
        
        Returns:
            float32 [N, D] normalized vectors, where N = len(texts)
        """
        raise NotImplementedError
    
    def embed_items(self, paths: list[str]) -> np.ndarray:
        """Embed item files.
        
        Args:
            paths: resolved absolute paths to media files
            
        Returns:
            float32 [N, D] normalized vectors, where N = len(paths)
        """
        raise NotImplementedError
    
    def provenance(self) -> dict:
        """Return model metadata for provenance tracking.
        
        Returns:
            dict with keys: model_id, revision, model_files,
                runtime, runtime_version, execution_provider, dtype
        """
        raise NotImplementedError

# ---------------------------------------------------------------------------
# Retrieval helper functions
# ---------------------------------------------------------------------------

def load_manifest(path) -> list[dict]:
    """Load a JSONL manifest.
    
    Args:
        path: path to JSONL manifest file
        
    Returns:
        list of manifest entries, paths resolved to absolute paths
        
    Raises:
        ValueError: duplicate id, empty captions list, missing key, distractor not bool, distractor with captions
    """
    path = Path(path).resolve()
    entries = []
    seen_ids = set()
    
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip blank lines
                continue
                
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}")
            
            # Validate required keys
            required_keys = {"id", "path"}
            if not required_keys.issubset(entry.keys()):
                raise ValueError("missing key")
            
            # Validate distractor field if present
            if "distractor" in entry and not isinstance(entry["distractor"], bool):
                raise ValueError("distractor must be a bool")

            is_distractor = entry.get("distractor", False)

            if is_distractor:
                # Distractor entries carry no captions
                if "captions" in entry and entry["captions"]:
                    raise ValueError("distractor entries carry no captions")
            else:
                # Validate captions is a non-empty list
                if "captions" not in entry:
                    raise ValueError("missing key")
                if not isinstance(entry["captions"], list) or not entry["captions"]:
                    raise ValueError("empty captions list")
            
            # Check for duplicate IDs
            if entry["id"] in seen_ids:
                raise ValueError("duplicate id")
            seen_ids.add(entry["id"])
            
            # Resolve path to absolute relative to manifest file's directory
            entry["path"] = str((path.parent / entry["path"]).resolve())
            entries.append(entry)
    
    return entries


def recall_at_k(query_vecs, item_vecs, query_item_index, ks=(1, 5, 10)) -> dict:
    """Score and rank queries against items, compute Recall@K.
    
    Args:
        query_vecs: float32 [N_queries, D] normalized query vectors
        item_vecs: float32 [N_items, D] normalized item vectors
        query_item_index: int array of length N_queries, ground truth item index for each query
        ks: tuple of K values for which to compute Recall
        
    Returns:
        dict with keys R@1, R@5, R@10
    """
    scores = query_vecs @ item_vecs.T  # [N_queries, N_items]
    
    # Rank items by descending score, breaking ties by LOWER item index
    ranked_indices = np.argsort(-scores, kind="stable")  # [N_queries, N_items]
    
    # For each query, check if its item is in top-K
    results = {}
    n_queries = len(query_item_index)
    
    for k in ks:
        hits = 0
        for q_idx in range(n_queries):
            # Find position of true item in ranked list
            true_item_idx = query_item_index[q_idx]
            rank = np.where(ranked_indices[q_idx] == true_item_idx)[0][0]
            if rank < k:
                hits += 1
        results[f"R@{k}"] = hits / n_queries if n_queries > 0 else 0.0
    
    return results


def run(manifest_path, embedder) -> dict:
    """Run retrieval benchmark using the provided embedder.
    
    Args:
        manifest_path: path to JSONL manifest
        embedder: MediaEmbedder instance
        
    Returns:
        dict with metrics, n_items, n_queries, n_distractors and timing information
    """
    # Load manifest
    entries = load_manifest(manifest_path)
    
    # Extract items and queries
    item_paths = [entry["path"] for entry in entries]
    captions = []
    query_item_indices = []
    n_distractors = 0
    
    for idx, entry in enumerate(entries):  # Use enumerate to get index directly
        # Distractors are ranked as items but add no queries
        if entry.get("distractor", False):
            n_distractors += 1
            continue
        # Each caption of each item is one query
        for caption in entry["captions"]:
            captions.append(caption)
            query_item_indices.append(idx)  # Use the index from enumerate
    
    query_item_index = np.array(query_item_indices, dtype=np.int64)
    
    # Embed items (batch size 32)
    start_time = time.perf_counter()
    item_vecs = []
    batch = []
    
    for i, path in enumerate(item_paths):
        batch.append(path)
        if len(batch) >= 32 or i == len(item_paths) - 1:
            vecs = embedder.embed_items(batch)
            item_vecs.append(vecs)
            batch = []
    
    item_vecs = np.concatenate(item_vecs, axis=0)
    embed_items_seconds = time.perf_counter() - start_time
    
    # Embed queries (batch size 256)
    start_time = time.perf_counter()
    query_vecs = []
    batch = []
    
    for i, caption in enumerate(captions):
        batch.append(caption)
        if len(batch) >= 256 or i == len(captions) - 1:
            vecs = embedder.embed_queries(batch)
            query_vecs.append(vecs)
            batch = []
    
    query_vecs = np.concatenate(query_vecs, axis=0)
    embed_queries_seconds = time.perf_counter() - start_time
    
    # Validate shapes and normalization
    if item_vecs.ndim != 2 or query_vecs.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got item_vecs.shape={item_vecs.shape}, query_vecs.shape={query_vecs.shape}")
    
    if item_vecs.shape[1] != query_vecs.shape[1]:
        raise ValueError(f"Dimensionality mismatch: item_vecs.shape={item_vecs.shape}, query_vecs.shape={query_vecs.shape}")
    
    if len(item_paths) != item_vecs.shape[0]:
        raise ValueError(f"Item vector count mismatch: {len(item_paths)} items vs {item_vecs.shape[0]} vectors")
    
    if len(captions) != query_vecs.shape[0]:
        raise ValueError(f"Query vector count mismatch: {len(captions)} queries vs {query_vecs.shape[0]} vectors")
    
    # Compute metrics
    metrics = recall_at_k(query_vecs, item_vecs, query_item_index)
    
    return {
        "metrics": metrics,
        "n_items": len(item_paths),
        "n_queries": len(captions),
        "n_distractors": n_distractors,
        "embed_items_seconds": embed_items_seconds,
        "embed_queries_seconds": embed_queries_seconds,
    }

# ---------------------------------------------------------------------------
# Results writing with provenance enforcement
# ---------------------------------------------------------------------------

REQUIRED_PROVENANCE = (
    "model_id", "revision", "model_files", "runtime",
    "runtime_version", "execution_provider", "dtype"
)

def write_results(out_path, payload: dict) -> None:
    """Write results with provenance enforcement.
    
    Args:
        out_path: output file path
        payload: results payload
        
    Raises:
        ValueError: if payload missing required keys or has invalid values
    """
    # Check required top-level keys
    required_keys = {"experiment", "arm", "modality", "dataset", "embedder", "host", "metrics"}
    if not required_keys.issubset(payload.keys()):
        missing = required_keys - set(payload.keys())
        raise ValueError(f"Missing required payload keys: {missing}")
    
    # Check metrics have required R@K values
    metrics = payload["metrics"]
    for k in (1, 5, 10):
        if f"R@{k}" not in metrics:
            raise ValueError(f"metrics missing R@{k}")
    
    # Check n_items and n_queries
    if "n_items" not in payload or "n_queries" not in payload:
        raise ValueError("payload missing n_items or n_queries")
    
    # Check provenance has all required keys
    if "provenance" not in payload:
        raise ValueError("payload missing provenance")
    
    provenance = payload["provenance"]
    missing_provenance = set(REQUIRED_PROVENANCE) - set(provenance.keys())
    if missing_provenance:
        raise ValueError(f"provenance missing required keys: {missing_provenance}")
    
    # Check provenance values are non-empty
    for key in REQUIRED_PROVENANCE:
        if not provenance[key]:
            raise ValueError(f"provenance[{key!r}] is empty")
    
    # Check model_files is a non-empty dict (specific validation)
    if not isinstance(provenance["model_files"], dict) or not provenance["model_files"]:
        raise ValueError("provenance.model_files must be a non-empty dict")
    if not isinstance(provenance["model_files"], dict):
        raise ValueError("provenance.model_files must be a dict")
    
    for filename, sha256 in provenance["model_files"].items():
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"model_files[{filename!r}] must be 64-char hex, got: {sha256!r}")
        if not all(c in "0123456789abcdef" for c in sha256):
            raise ValueError(f"model_files[{filename!r}] must be lowercase hex, got: {sha256!r}")
    
    # All checks passed, prepare final payload
    final_payload = payload.copy()
    final_payload["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    # Ensure output directory exists
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write pretty JSON
    with open(out_path, "w") as f:
        json.dump(final_payload, f, indent=2, sort_keys=True)
        f.write("\n")  # Exactly one trailing newline

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    """Command-line interface for E-048 retrieval harness."""
    if argv is None:
        argv = sys.argv[1:]
    
    import argparse
    parser = argparse.ArgumentParser(
        description="E-048 text-to-media retrieval harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest")
    parser.add_argument("--embedder", required=True, help="Module path to embedder factory")
    parser.add_argument("--experiment", required=True, help="Experiment ID")
    parser.add_argument("--arm", required=True, help="Arm identifier")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--host-label", required=True, help="Host label (free-form)")
    parser.add_argument("--out", required=True, help="Output file path")
    parser.add_argument("--factory-kwargs", default="{}", help="JSON kwargs for embedder factory")
    
    args = parser.parse_args(argv)
    
    # Parse factory kwargs
    try:
        factory_kwargs = json.loads(args.factory_kwargs)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid --factory-kwargs JSON: {e}")
    
    # Import embedder factory
    module_path, factory_name = args.embedder.rsplit(":", 1)
    try:
        module = __import__(module_path, fromlist=[factory_name])
    except ImportError as e:
        raise ValueError(f"Failed to import embedder module {module_path}: {e}")
    
    if not hasattr(module, factory_name):
        raise ValueError(f"Embedder factory {factory_name} not found in module {module_path}")
    
    factory = getattr(module, factory_name)
    
    # Create embedder
    try:
        embedder = factory(**factory_kwargs)
    except Exception as e:
        raise ValueError(f"Failed to create embedder: {e}")
    
    # Run benchmark
    try:
        result = run(args.manifest, embedder)
    except Exception as e:
        raise ValueError(f"Benchmark run failed: {e}")
    
    # Build payload
    payload = {
        "experiment": args.experiment,
        "arm": args.arm,
        "modality": embedder.modality,
        "dataset": args.dataset,
        "embedder": embedder.name,
        "host": args.host_label,
        "metrics": result["metrics"],
        "n_items": result["n_items"],
        "n_queries": result["n_queries"],
        "provenance": embedder.provenance(),
    }
    
    # Write results
    try:
        write_results(args.out, payload)
    except Exception as e:
        raise ValueError(f"Failed to write results: {e}")
    
    # Print metrics as one line
    metrics_str = " ".join(f"{k}={v:.4f}" for k, v in payload["metrics"].items())
    print(metrics_str)


if __name__ == "__main__":
    main()
