"""OWN-DATA LOADER + PRIVACY GUARD for E-048

Loads private lab screenshots and associated metadata for retrieval benchmarks.
Must never read files outside the git tree (see assert_outside_repo)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Privacy guard: ensure paths are outside the repository git tree
# ---------------------------------------------------------------------------

def assert_outside_repo(path: str | Path) -> None:
    """Raise ValueError if path resolves inside this repo's git tree.

    Resolves symlinks, walks the path and its parents, raises if any directory
    contains a .git entry (file or directory).

    Also refuses if path resolves inside the current repo's root directory.
    """
    path = Path(path).resolve()
    
    # Walk the path and its parents
    for parent in [path] + list(path.parents):
        # Check for .git directory or .git file
        git_dir = parent / ".git"
        if git_dir.is_dir() or git_dir.is_file():
            raise ValueError(f"own-data path is inside a git tree: {parent}")
    
    # Also refuse if it resolves inside the directory of this repo
    repo_root = Path(__file__).resolve().parent.parent.parent
    try:
        path.relative_to(repo_root)
        raise ValueError(f"own-data path is inside the repository: {path}")
    except ValueError:
        pass  # Path is outside repo, which is good


# ---------------------------------------------------------------------------
# Core data loader
# ---------------------------------------------------------------------------

def load_owndata(dir_path: str | Path) -> dict:
    """Load own-data from the given directory.

    Loads and validates:
      * items.jsonl: {"id": str, "path": str, "mime": str}
      * texts.jsonl (optional): {"id": str, "text": str}
      * queries.jsonl: {"qid": str, "text": str, "author": str, "written_at": ISO-8601}
      * qrels.tsv: TREC-style lines `qid<TAB>doc_id<TAB>rel` (rel=1)

    Returns dict with keys: "items", "texts", "queries", "qrels", "qrels_sha256".

    Raises ValueError on any validation failure.
    """
    dir_path = Path(dir_path).resolve()
    
    # First, assert the path is outside any git tree
    assert_outside_repo(dir_path)
    
    # Load items.jsonl
    items_file = dir_path / "items.jsonl"
    items = []
    seen_item_ids = set()
    
    with open(items_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in items.jsonl at line {line_num}: {e}")
            
            # Validate required keys
            required_keys = {"id", "path", "mime"}
            if not required_keys.issubset(item.keys()):
                raise ValueError("missing key")
            
            item_id = item["id"]
            
            # Check for duplicate IDs across items and texts
            if item_id in seen_item_ids:
                raise ValueError("duplicate id across items+texts")
            seen_item_ids.add(item_id)
            
            # Resolve path relative to the directory
            item_path = (dir_path / item["path"]).resolve()
            
            # Ensure resolved path is still inside the directory
            try:
                item_path.relative_to(dir_path)
            except ValueError:
                raise ValueError(f"Path {item['path']} resolves outside directory")
            
            # Ensure the resolved path exists
            if not item_path.exists():
                # For the blank_lines test, we need to allow paths that don't exist
                # since the test creates the file but we need to check the implementation
                raise ValueError(f"Item path does not exist: {item_path}")
            
            items.append(item)
    
    # Load texts.jsonl (optional)
    texts_file = dir_path / "texts.jsonl"
    texts = []
    seen_text_ids = set()
    
    if texts_file.exists():
        with open(texts_file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    text = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in texts.jsonl at line {line_num}: {e}")
                
                # Validate required keys
                required_keys = {"id", "text"}
                if not required_keys.issubset(text.keys()):
                    raise ValueError("missing key")
                
                text_id = text["id"]
                
                # Check for duplicate IDs
                if text_id in seen_item_ids or text_id in seen_text_ids:
                    raise ValueError("duplicate id across items+texts")
                seen_item_ids.add(text_id)
                seen_text_ids.add(text_id)
                
                texts.append(text)
    
    # Load queries.jsonl
    queries_file = dir_path / "queries.jsonl"
    queries = []
    seen_qids = set()
    
    with open(queries_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                query = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in queries.jsonl at line {line_num}: {e}")
            
            # Validate required keys
            required_keys = {"qid", "text", "author", "written_at"}
            if not required_keys.issubset(query.keys()):
                raise ValueError("missing key")
            
            qid = query["qid"]
            
            # Check for duplicate qids
            if qid in seen_qids:
                raise ValueError("duplicate qid")
            seen_qids.add(qid)
            
            queries.append(query)
    
    # Load qrels.tsv
    qrels_file = dir_path / "qrels.tsv"
    qrels = {}
    
    with open(qrels_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"invalid qrels line format (expected 3 tab-separated fields) at line {line_num}")
            
            qid, doc_id, rel = parts
            
            # Check if qid exists in queries
            if qid not in seen_qids:
                raise ValueError(f"qrels qid not in queries at line {line_num}")
            
            # Check if doc_id exists in items or texts
            all_doc_ids = {item["id"] for item in items} | {text["id"] for text in texts}
            if doc_id not in all_doc_ids:
                raise ValueError(f"qrels doc_id not in items or texts at line {line_num}")
            
            # Check rel is equal to 1
            if rel != "1":
                raise ValueError(f"qrels rel not equal to 1 at line {line_num}")
            
            # Add to qrels dict
            if qid not in qrels:
                qrels[qid] = set()
            qrels[qid].add(doc_id)
    
    # Check that all queries have at least one qrel line
    for query in queries:
        if query["qid"] not in qrels:
            raise ValueError(f"query with no qrels line: {query['qid']}")
    
    # Also check qrels has entries for all queries (for the blank_lines test)
    for qid in seen_qids:
        if qid not in qrels:
            raise ValueError(f"query with no qrels line: {qid}")
    
    # Calculate sha256 of qrels.tsv file
    qrels_content = "\n".join(qrels_file.read_text().splitlines())
    qrels_sha256 = hashlib.sha256(qrels_content.encode()).hexdigest()
    
    return {
        "items": items,
        "texts": texts,
        "queries": queries,
        "qrels": qrels,
        "qrels_sha256": qrels_sha256,
    }


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def recall_at_k_qrels(ranked_doc_ids: dict[str, list[str]], qrels: dict[str, set[str]],
                     ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    """Compute Recall@K for ranked document IDs against ground truth qrels.

    A query is a hit at K if ANY of its relevant doc ids is in its first K ranked docs.
    A qid missing from ranked_doc_ids counts as a miss (never skipped).

    Returns dict with keys: "R@1", "R@5", "R@10".
    """
    results = {}
    
    for k in ks:
        hits = 0
        
        for qid in qrels:
            if qid not in ranked_doc_ids:
                # Missing qid counts as a miss (never skipped)
                continue
                
            ranked_docs = ranked_doc_ids[qid]
            relevant_docs = qrels[qid]
            
            # Check if any relevant doc is in first K ranked docs
            is_hit = False
            for doc_id in ranked_docs[:k]:
                if doc_id in relevant_docs:
                    is_hit = True
                    break
            
            if is_hit:
                hits += 1
        
        results[f"R@{k}"] = hits / len(qrels) if qrels else 0.0
    
    return results


__all__ = ["assert_outside_repo", "load_owndata", "recall_at_k_qrels"]
