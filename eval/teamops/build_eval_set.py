"""Build team-ops ground-truth eval set (rulings, ledger, PRs) with frozen 30% holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def parse_rulings(md_text: str) -> List[Dict[str, Any]]:
    items = []
    for line in md_text.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        rest = line[2:]
        if "~" not in rest or "Z" not in rest:
            continue
        try:
            date_end = rest.find("~")
            date = rest[:date_end].strip()
            time_start = date_end + 1
            z_pos = rest.find("Z")
            time = rest[time_start:z_pos].strip()
            after_z = rest[z_pos + 1:].strip()
            if ":" not in after_z:
                continue
            who_part, text_part = after_z.split(":", 1)
            who = who_part.strip()
            text = text_part.strip()
            source_id = f"ruling:{date}T{time}Z:{who}"
            items.append({
                "date": date,
                "who": who,
                "text": text,
                "source_id": source_id,
            })
        except Exception:
            continue
    return items


def parse_ledger(tsv_text: str) -> List[Dict[str, Any]]:
    items = []
    lines = tsv_text.splitlines()
    if not lines:
        return items
    # header = lines[0]
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        # Try tab split first as per spec
        if '\t' in line:
            parts = line.split('\t')
        else:
            # Use shlex to split by spaces but preserve quoted fields
            try:
                parts = shlex.split(line)
            except Exception:
                # If shlex fails, fallback to simple split
                parts = line.split()
        if len(parts) < 7:
            continue
        utc = parts[0].strip()
        seat = parts[1].strip()
        owner = parts[2].strip()
        target = parts[3].strip()
        action = parts[4].strip()
        why = ' '.join(parts[5:-1]) if len(parts) > 6 else (parts[5].strip() if len(parts) > 5 else "")
        undo = parts[-1].strip() if len(parts) >= 7 else ""
        source_id = f"ledger:{utc}:{owner}:{target}"
        items.append({
            "utc": utc,
            "seat": seat,
            "owner": owner,
            "target": target,
            "action": action,
            "why": why,
            "undo": undo,
            "source_id": source_id,
        })
    return items


def parse_prs(json_text: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(json_text)
    except Exception:
        return []
    items = []
    for pr in data:
        if not isinstance(pr, dict):
            continue
        number = pr.get("number")
        title = pr.get("title", "")
        mergedAt = pr.get("mergedAt") or pr.get("merged_at") or pr.get("merged")
        body = pr.get("body", "")
        source_id = f"pr:{number}"
        items.append({
            "number": number,
            "title": title,
            "mergedAt": mergedAt,
            "body": body,
            "source_id": source_id,
        })
    return items


def _extract_subject_key(text: str) -> Optional[str]:
    """Extract a subject key from ruling text.
    Returns the first significant word longer than 3 characters, ignoring common words and withdrawal keywords.
    """
    common = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "will", "would", "shall", "should", "may", "might", "must", "can", "could"}
    withdrawal = {"withdraw", "replace", "supersede", "revoke", "overrule", "annul", "nullify"}
    # Remove punctuation and split
    words = re.findall(r'\b\w+\b', text.lower())
    for w in words:
        if len(w) > 3 and w not in common and w not in withdrawal:
            return w
    return None


def _are_subject_related(t1: str, t2: str) -> bool:
    key1 = _extract_subject_key(t1)
    key2 = _extract_subject_key(t2)
    return key1 is not None and key1 == key2


def make_items(
    rulings: List[Dict[str, Any]],
    ledger: List[Dict[str, Any]],
    prs: List[Dict[str, Any]],
    session_map: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    items = []
    session_map = session_map or []

    session_by_source = {}
    for s in session_map:
        sid = s.get("source_id")
        if sid:
            session_by_source[sid] = {
                "session_id": s.get("session_id"),
                "turn_from": s.get("turn_from"),
                "turn_to": s.get("turn_to"),
            }

    # Process rulings
    for idx, ruling in enumerate(rulings):
        source_id = ruling.get("source_id", f"ruling-{idx}")
        date = ruling.get("date", "2000-01-01")
        who = ruling.get("who", "")
        text = ruling.get("text", "")
        # Question should not contain the gold answer (the text)
        question = f"What was the ruling by {who} on {date}?"
        gold_answer = text
        item = {
            "id": f"r-{idx:05d}",
            "question": question,
            "gold_answer": gold_answer,
            "gold_sources": [source_id] if source_id else [],
            "gold_sessions": [session_by_source[source_id]] if source_id in session_by_source else [],
            "date": date,
            "kind": "ruling",
            "superseded": False,
            "supersedes": [],  # list of source_ids that this ruling supersedes
        }
        items.append(item)

    # Process ledger
    for idx, entry in enumerate(ledger):
        source_id = entry.get("source_id", f"ledger-{idx}")
        utc = entry.get("utc", "")
        owner = entry.get("owner", "")
        target = entry.get("target", "")
        action = entry.get("action", "")
        undo = entry.get("undo", "")
        date = utc.split("T")[0] if utc else "2000-01-01"
        question = f"Who did {action} on {target} and what is the undo?"
        gold_answer = f"{owner} did {action} on {target}; undo: {undo}"
        item = {
            "id": f"l-{idx:05d}",
            "question": question,
            "gold_answer": gold_answer,
            "gold_sources": [source_id] if source_id else [],
            "gold_sessions": [session_by_source[source_id]] if source_id in session_by_source else [],
            "date": date,
            "kind": "ledger",
            "superseded": False,
            "supersedes": [],
        }
        items.append(item)

    # Process PRs
    for idx, pr in enumerate(prs):
        source_id = pr.get("source_id", f"pr-{idx}")
        merged = pr.get("mergedAt")
        number = pr.get("number")
        title = pr.get("title", "")
        date = merged.split("T")[0] if merged else "2000-01-01"
        question = f"What was PR #{number} ({title}) about?"
        gold_answer = pr.get("body", "") or title
        item = {
            "id": f"p-{idx:05d}",
            "question": question,
            "gold_answer": gold_answer,
            "gold_sources": [source_id] if source_id else [],
            "gold_sessions": [session_by_source[source_id]] if source_id in session_by_source else [],
            "date": date,
            "kind": "pr",
            "superseded": False,
            "supersedes": [],
        }
        items.append(item)

    # Detect supersedence for rulings
    ruling_items = [it for it in items if it["kind"] == "ruling"]
    # Build a map from date to list of rulings for efficient lookup? We'll do O(n^2) for simplicity.
    for i, later in enumerate(ruling_items):
        later_text = later["gold_answer"].lower()
        later_date = later["date"]
        # Check if later ruling contains withdrawal/replacement keywords
        if any(k in later_text for k in ["withdraw", "replace", "supersede", "revoke", "overrule", "annul", "nullify"]):
            later_key = _extract_subject_key(later_text)
            if later_key is None:
                continue
            for j, earlier in enumerate(ruling_items):
                if i == j:
                    continue
                earlier_date = earlier["date"]
                if earlier_date > later_date:
                    continue  # only consider earlier rulings
                earlier_key = _extract_subject_key(earlier["gold_answer"].lower())
                if earlier_key is not None and earlier_key == later_key:
                    # Mark earlier as superseded
                    earlier["superseded"] = True
                    # Add later's source_id to earlier's supersedes list? No, later supersedes earlier.
                    # So we add earlier's source_id to later's supersedes list.
                    later["supersedes"].append(earlier["gold_sources"][0])

    # Remove duplicates in supersedes list (though unlikely)
    for item in ruling_items:
        item["supersedes"] = list(dict.fromkeys(item["supersedes"]))

    return items


def split(items: List[Dict[str, Any]], holdout: float = 0.30) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    # Separate base items (rulings, ledger, prs) from derived items (we'll generate derived items later)
    # But note: we don't have derived items yet. We'll generate unanswerable and as_of after splitting base items.
    base_items = [it for it in items if it["kind"] in ("ruling", "ledger", "pr")]
    n_base = len(base_items)
    if n_base == 0:
        # No base items, we still need to handle unanswerable and as_of? But they depend on base items.
        # We'll return empty trains and holdouts and then generate derived items?
        # For simplicity, we'll return empty and let the caller handle.
        return [], [], {"total": 0, "train": 0, "holdout": 0, "holdout_ratio": holdout, "straddling_pairs": []}

    # Sort base items by date, then id
    sorted_base = sorted(base_items, key=lambda x: (x["date"], x["id"]))
    holdout_n = int(round(n_base * holdout))
    if holdout_n < 0:
        holdout_n = 0
    if holdout_n > n_base:
        holdout_n = n_base
    # When holdout_n is 0, we want train_base to be the entire list and holdout_base empty.
    # When holdout_n is n_base, we want train_base empty and holdout_base the entire list.
    holdout_base = sorted_base[n_base - holdout_n:] if holdout_n > 0 else []
    train_base = sorted_base[:n_base - holdout_n]

    # We'll collect straddling pairs: superseded pairs where one is in train and the other in holdout
    straddling_pairs = []
    # We need to know which base items are in which split for quick lookup
    train_base_ids = {item["id"] for item in train_base}
    holdout_base_ids = {item["id"] for item in holdout_base}

    # Check each ruling in base_items for supersedes relations
    for item in base_items:
        if item["kind"] == "ruling":
            for superseded_source in item.get("supersedes", []):
                # Find the item with that source_id in base_items
                superseded_item = None
                for it in base_items:
                    if it.get("gold_sources", []) and it["gold_sources"][0] == superseded_source:
                        superseded_item = it
                        break
                if superseded_item is None:
                    continue
                # Check if they straddle the split
                if (item["id"] in train_base_ids and superseded_item["id"] in holdout_base_ids) or \
                   (item["id"] in holdout_base_ids and superseded_item["id"] in train_base_ids):
                    straddling_pairs.append(sorted([item["id"], superseded_item["id"]]))

    # Generate unanswerable items: ~10% of base items
    unanswerable_count = max(1, int(round(n_base * 0.10))) if n_base > 0 else 0
    unanswerable_items = []
    if unanswerable_count > 0:
        # We'll generate unanswerable items in a deterministic order
        for i in range(unanswerable_count):
            # Assign date: we'll use the date of the i-th base item (round-robin) to have a deterministic date
            base_idx = i % n_base
            date = sorted_base[base_idx]["date"] if n_base > 0 else "2000-01-01"
            question = f"What is the policy on nonexistent_subject_{i:03d}?"
            unanswerable_items.append({
                "id": f"u-{len(items) + i:05d}",
                "question": question,
                "gold_answer": "no record",
                "gold_sources": [],
                "gold_sessions": [],
                "date": date,
                "kind": "unanswerable",
                "superseded": False,
                "supersedes": [],
            })
    # Split unanswerable items by holdout ratio: holdout gets holdout_ratio * count
    unanswerable_holdout_n = int(round(unanswerable_count * holdout))
    if unanswerable_holdout_n < 0:
        unanswerable_holdout_n = 0
    if unanswerable_holdout_n > unanswerable_count:
        unanswerable_holdout_n = unanswerable_count
    unanswerable_holdout = unanswerable_items[:unanswerable_holdout_n]
    unanswerable_train = unanswerable_items[unanswerable_holdout_n:]

    # Generate as_of items: for each superseded pair (later ruling supersedes earlier ruling)
    as_of_items = []
    # We'll iterate over all rulings that have a non-empty supersedes list
    for item in base_items:
        if item["kind"] == "ruling" and item.get("supersedes"):
            for superseded_source in item["supersedes"]:
                # Find the item with that source_id in base_items
                superseded_item = None
                for it in base_items:
                    if it.get("gold_sources", []) and it["gold_sources"][0] == superseded_source:
                        superseded_item = it
                        break
                if superseded_item is None:
                    continue
                # The as_of item's date is the date of the historical ruling (superseded_item)
                date = superseded_item["date"]
                question = f"what was the rule on {date}?"
                # The gold answer is the text of the historical ruling
                gold_answer = superseded_item["gold_answer"]
                # The gold sources is the source_id of the historical ruling
                gold_sources = [superseded_item["gold_sources"][0]]
                as_of_items.append({
                    "id": f"a-{len(items) + len(unanswerable_items) + len(as_of_items):05d}",
                    "question": question,
                    "gold_answer": gold_answer,
                    "gold_sources": gold_sources,
                    "gold_sessions": [],
                    "date": date,
                    "kind": "as_of",
                    "superseded": False,
                    "supersedes": [],
                })
    # Assign as_of items to the same side as the historical ruling (superseded_item)
    as_of_holdout = []
    as_of_train = []
    for a in as_of_items:
        # Find the historical ruling (the one whose gold_sources matches a's gold_sources[0])
        # We know the historical ruling's source_id is a["gold_sources"][0]
        hist_source = a["gold_sources"][0]
        # Find which base item has that source_id
        hist_item = None
        for it in base_items:
            if it.get("gold_sources", []) and it["gold_sources"][0] == hist_source:
                hist_item = it
                break
        if hist_item is None:
            # Should not happen, but if it does, put in train
            as_of_train.append(a)
            continue
        if hist_item["id"] in train_base_ids:
            as_of_train.append(a)
        else:
            as_of_holdout.append(a)

    # Combine
    train = list(train_base) + list(unanswerable_train) + list(as_of_train)
    holdout = list(holdout_base) + list(unanswerable_holdout) + list(as_of_holdout)

    manifest = {
        "total_items": len(items),
        "train": len(train),
        "holdout": len(holdout),
        "holdout_ratio": holdout if n_base > 0 else 0.0,
        "straddling_pairs": straddling_pairs,
    }

    return train, holdout, manifest


def load_text(path: str | Path) -> str:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def write_jsonl(path: str | Path, items: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build team-ops eval set")
    parser.add_argument("--rulings", required=True, help="Path to rulings markdown file")
    parser.add_argument("--ledger", required=True, help="Path to ledger TSV file")
    parser.add_argument("--prs", required=True, help="Path to PRs JSON file")
    parser.add_argument("--output-dir", required=True, help="Directory to write output files")
    parser.add_argument("--holdout", type=float, default=0.30, help="Holdout ratio")
    parser.add_argument("--session-map", default=None, help="Path to session map JSONL file")
    args = parser.parse_args()

    rulings_text = load_text(args.rulings)
    ledger_text = load_text(args.ledger)
    prs_text = load_text(args.prs)

    rulings = parse_rulings(rulings_text)
    ledger = parse_ledger(ledger_text)
    prs = parse_prs(prs_text)

    session_map = []
    if args.session_map and os.path.exists(args.session_map):
        session_map_text = load_text(args.session_map)
        for line in session_map_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                session_map.append(json.loads(line))
            except Exception:
                continue

    items = make_items(rulings, ledger, prs, session_map)

    train, holdout, manifest = split(items, holdout=args.holdout)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(out_dir / "items.jsonl", train)
    write_jsonl(out_dir / "holdout.jsonl", holdout)

    manifest["inputs"] = {
        "rulings": {
            "path": str(Path(args.rulings).resolve()),
            "sha256": sha256_file(args.rulings),
        },
        "ledger": {
            "path": str(Path(args.ledger).resolve()),
            "sha256": sha256_file(args.ledger),
        },
        "prs": {
            "path": str(Path(args.prs).resolve()),
            "sha256": sha256_file(args.prs),
        },
    }
    if args.session_map and os.path.exists(args.session_map):
        manifest["inputs"]["session_map"] = {
            "path": str(Path(args.session_map).resolve()),
            "sha256": sha256_file(args.session_map),
        }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
