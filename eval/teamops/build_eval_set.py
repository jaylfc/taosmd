"""Build team-ops ground-truth eval set (rulings, ledger, PRs) with frozen 30% holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        utc = parts[0]
        seat = parts[1]
        owner = parts[2]
        target = parts[3]
        action = parts[4]
        why = parts[5] if len(parts) > 5 else ""
        undo = parts[6] if len(parts) > 6 else ""
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

def _same_core_subject(t1: str, t2: str) -> bool:
    words1 = set(w for w in t1.lower().split() if len(w) > 3)
    words2 = set(w for w in t2.lower().split() if len(w) > 3)
    common = words1 & words2
    return len(common) > 0


def _are_subject_related(t1: str, t2: str) -> bool:
    return _same_core_subject(t1, t2)

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
        question = f"What was the ruling by {who} on {date}? Context: {text}"
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
        }
        items.append(item)

    # Detect supersedence for rulings
    ruling_items = [it for it in items if it["kind"] == "ruling"]
    for i, later in enumerate(ruling_items):
        later_text = later["gold_answer"].lower()
        later_date = later["date"]
        if any(k in later_text for k in ["withdraw", "replace", "supersede", "revoke", "overrule"]):
            for j, earlier in enumerate(ruling_items):
                if i == j:
                    continue
                earlier_date = earlier["date"]
                if earlier_date > later_date:
                    continue
                earlier_text = earlier["gold_answer"].lower()
                if _are_subject_related(earlier_text, later_text):
                    earlier["superseded"] = True
                    later["supersedes"] = earlier["gold_sources"][0] if earlier["gold_sources"] else earlier["id"]

    return items

def split(items: List[Dict[str, Any]], holdout: float = 0.30) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    sorted_items = sorted(items, key=lambda x: (x["date"], x["id"]))
    n = len(sorted_items)
    if n == 0:
        return [], [], {"total": 0, "train": 0, "holdout": 0, "holdout_ratio": holdout}
    holdout_n = int(round(n * holdout))
    if holdout_n < 0:
        holdout_n = 0
    if holdout_n > n:
        holdout_n = n
    holdout_items = sorted_items[-holdout_n:] if holdout_n > 0 else []
    train_items = sorted_items[:-holdout_n] if holdout_n < n else []

    result_train = list(train_items)
    result_holdout = list(holdout_items)

    unanswerable_count = max(1, int(round(n * 0.10))) if n > 0 else 0
    for i in range(unanswerable_count):
        unans_id = f"u-{len(items) + i:05d}"
        date = sorted_items[i % n]["date"] if n > 0 else "2000-01-01"
        question = f"What is the policy on nonexistent_subject_{i:03d}?"
        item = {
            "id": unans_id,
            "question": question,
            "gold_answer": "no record",
            "gold_sources": [],
            "gold_sessions": [],
            "date": date,
            "kind": "unanswerable",
            "superseded": False,
        }
        result_train.append(item)

    as_of_items = []
    superseded_pairs = []
    for item in items:
        if item.get("supersedes") is not None:
            superseded_pairs.append(item)

    for i, sup_item in enumerate(superseded_pairs):
        as_of_id = f"a-{len(items) + unanswerable_count + i:05d}"
        earlier_date = sup_item["date"]
        date_parts = sup_item["date"].split("-")
        if len(date_parts) == 3:
            try:
                y = int(date_parts[0])
                m = int(date_parts[1])
                d = int(date_parts[2])
                earlier_date = f"{y}-{m:02d}-{max(1, d-1):02d}"
            except Exception:
                earlier_date = sup_item["date"]
        question = f"what was the rule on {earlier_date}?"
        gold_answer = f"rule as of {earlier_date} (referenced by {sup_item.get('supersedes')})"
        item = {
            "id": as_of_id,
            "question": question,
            "gold_answer": gold_answer,
            "gold_sources": [],
            "gold_sessions": [],
            "date": earlier_date,
            "kind": "as_of",
            "superseded": False,
        }
        as_of_items.append(item)

    for a in as_of_items:
        result_train.append(a)

    manifest = {
        "total_items": len(items),
        "train": len(result_train),
        "holdout": len(result_holdout),
        "holdout_ratio": holdout if n > 0 else 0.0,
    }

    return result_train, result_holdout, manifest
