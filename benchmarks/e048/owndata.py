"""Own-data loader and privacy guard for E-048 benchmark.

This module loads private lab screenshots and metadata from a directory
outside the git repository, enforcing strict privacy and validation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def assert_outside_repo(path: str | Path) -> Path:
    """Check that path is not inside a git tree.
    
    Args:
        path: Path to check (file or directory)
        
    Returns:
        Resolved Path object
        
    Raises:
        ValueError: if path is inside a git tree or inside the taosmd project directory
    """
    p = Path(path).resolve(strict=True)
    
    # Check if the path itself or any parent directory contains a .git entry
    check_path = p
    while True:
        git_entry = check_path / ".git"
        if git_entry.exists():
            raise ValueError(f"own-data path is inside a git tree: {check_path}")
        
        # Move up to parent directory
        parent = check_path.parent
        if parent == check_path:  # Reached root
            break
        check_path = parent
    
    # Also check if the path is inside the taosmd project directory
    try:
        # Get the taosmd project root from the current file's location
        taosmd_project_root = Path(__file__).resolve().parents[2]
        # Check if the path is inside the taosmd project
        try:
            p.relative_to(taosmd_project_root)
            raise ValueError(f"own-data path is inside the taosmd project directory: {taosmd_project_root}")
        except ValueError:
            # p is NOT inside taosmd_project_root, which is correct
            pass
    except ImportError:
        # If taosmd is not installed, we can't check this
        pass
    
    return p


def load_owndata(dir_path: str | Path) -> dict:
    """Load own-data from a directory.
    
    Args:
        dir_path: Path to the own-data directory
        
    Returns:
        dict with keys: dir, items, texts, queries, qrels, qrels_sha256
        
    Raises:
        ValueError: on any validation error
    """
    dir_path = Path(dir_path)
    
    # First, assert the path is outside any git tree
    assert_outside_repo(dir_path)
    
    result = {
        "dir": str(dir_path.resolve()),
        "items": [],
        "texts": [],
        "queries": [],
        "qrels": {},
        "qrels_sha256": "",
    }
    
    # Track all IDs for duplicate detection
    all_ids = set()
    
    # Load items.jsonl
    items_path = dir_path / "items.jsonl"
    if items_path.exists():
        with open(items_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip blank lines
                    continue
                    
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in items.jsonl line {line_num}: {e}")
                
                # Validate required keys
                required_keys = {"id", "path", "mime"}
                if not required_keys.issubset(item.keys()):
                    raise ValueError(f"missing key in items.jsonl line {line_num}")
                
                # Check for duplicate id
                if item["id"] in all_ids:
                    raise ValueError(f"duplicate id: {item['id']}")
                all_ids.add(item["id"])
                
                # Resolve path and validate it doesn't escape directory
                item_path = (dir_path / item["path"]).resolve()
                try:
                    item_path.relative_to(dir_path.resolve())
                except ValueError:
                    raise ValueError(f"item path resolves outside directory: {item['path']}")
                
                item["abs_path"] = str(item_path)
                result["items"].append(item)
    
    # Load texts.jsonl (optional)
    texts_path = dir_path / "texts.jsonl"
    if texts_path.exists():
        with open(texts_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip blank lines
                    continue
                    
                try:
                    text = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in texts.jsonl line {line_num}: {e}")
                
                # Validate required keys
                required_keys = {"id", "text"}
                if not required_keys.issubset(text.keys()):
                    raise ValueError(f"missing key in texts.jsonl line {line_num}")
                
                # Check for duplicate id
                if text["id"] in all_ids:
                    raise ValueError(f"duplicate id: {text['id']}")
                all_ids.add(text["id"])
                
                result["texts"].append(text)
    
    # Load queries.jsonl
    queries_path = dir_path / "queries.jsonl"
    if not queries_path.exists():
        raise ValueError("queries.jsonl is required")
        
    with open(queries_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip blank lines
                continue
                
            try:
                query = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in queries.jsonl line {line_num}: {e}")
            
            # Validate required keys
            required_keys = {"qid", "text", "author", "written_at"}
            if not required_keys.issubset(query.keys()):
                raise ValueError(f"missing key in queries.jsonl line {line_num}")
            
            # Check for duplicate qid
            for existing_q in result["queries"]:
                if existing_q["qid"] == query["qid"]:
                    raise ValueError(f"duplicate qid: {query['qid']}")
            
            result["queries"].append(query)
    
    # Load qrels.tsv and compute hash
    qrels_path = dir_path / "qrels.tsv"
    if not qrels_path.exists():
        raise ValueError("qrels.tsv is required")
    
    # Read the entire file for hash computation
    qrels_content = qrels_path.read_text()
    result["qrels_sha256"] = hashlib.sha256(qrels_content.encode()).hexdigest()
    
    # Parse qrels.tsv
    with open(qrels_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip blank lines
                continue
                
            # Validate exactly 3 tab-separated fields
            fields = line.split("\t")
            if len(fields) != 3:
                raise ValueError(f"qrels.tsv line {line_num} must have exactly 3 tab-separated fields")
            
            qid, doc_id, rel = fields
            
            # Validate rel field equals "1"
            if rel != "1":
                raise ValueError(f"qrels.tsv line {line_num} rel field not equal to \"1\": {rel}")
            
            # Validate qid exists in queries
            qid_exists = any(q["qid"] == qid for q in result["queries"])
            if not qid_exists:
                raise ValueError(f"qrels.tsv line {line_num} qid '{qid}' not in queries")
            
            # Validate doc_id exists in items or texts
            doc_id_exists = any(item["id"] == doc_id for item in result["items"]) or \
                          any(text["id"] == doc_id for text in result["texts"])
            if not doc_id_exists:
                raise ValueError("doc_id in neither items nor texts")
            
            # Add to qrels dict
            if qid not in result["qrels"]:
                result["qrels"][qid] = set()
            result["qrels"][qid].add(doc_id)
    
    return result