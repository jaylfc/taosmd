"""Conversation-level checkpoint sidecar for the LoCoMo benchmark runner.

Each checkpoint is a newline-delimited JSON file (<out>.ckpt.jsonl) with one
header line followed by one line per completed conversation.  A stable config
hash guards against accidentally resuming a run that used different settings.

File layout::

    {"kind": "header", "config_hash": "<hex16>", "dataset": "<path>"}
    {"kind": "conv", "conv_id": "<id>", "rows": [...]}
    {"kind": "conv", "conv_id": "<id>", "rows": [...]}
    ...

Each line is written with os.fsync so a kill or power loss cannot corrupt a
line that was already acknowledged.  A partial (truncated) FINAL line, which
can happen if the process is killed mid-write, is silently skipped during load.
A malformed line anywhere else raises ValueError to prevent silently mixing
results from incompatible runs.
"""

from __future__ import annotations

import hashlib
import json
import os


# ---------------------------------------------------------------------------
# Config hashing
# ---------------------------------------------------------------------------

_HASH_SKIP = frozenset({"timestamp", "git_sha", "run_id", "failed_qa"})


def config_hash(meta_like: dict) -> str:
    """Return a stable sha256[:16] of the run-defining config.

    Keys in ``_HASH_SKIP`` (``timestamp``, ``git_sha``, ``run_id``,
    ``failed_qa``) are excluded so the hash reflects only the settings that
    determine which results are comparable.  Remaining keys are sorted before
    serialisation so insertion order does not affect the hash.
    """
    filtered = {k: v for k, v in meta_like.items() if k not in _HASH_SKIP}
    payload = json.dumps(filtered, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def write_header(path: str, cfg_hash: str, dataset: str) -> None:
    """Create (or truncate) the sidecar file and write the header line."""
    line = json.dumps({"kind": "header", "config_hash": cfg_hash, "dataset": dataset})
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def append_conversation(path: str, conv_id: str, rows: list[dict]) -> None:
    """Append one conversation record and fsync so a kill cannot lose it."""
    line = json.dumps({"kind": "conv", "conv_id": conv_id, "rows": rows})
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# Load helper
# ---------------------------------------------------------------------------

def load_checkpoint(path: str, expected_hash: str) -> tuple[set[str], list[dict]]:
    """Read the sidecar and return (done_conv_ids, all_rows).

    Raises ValueError if:
    - The first line is not a header, or
    - The header's config_hash differs from ``expected_hash``.

    A malformed or truncated FINAL line is skipped with a warning (crash
    mid-write tolerance).  A malformed line anywhere else raises ValueError.
    """
    with open(path, encoding="utf-8") as fh:
        raw_lines = fh.readlines()

    if not raw_lines:
        raise ValueError("checkpoint file is empty: no header found")

    # Parse header.
    try:
        header = json.loads(raw_lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"checkpoint header is not valid JSON: {exc}") from exc
    if header.get("kind") != "header":
        raise ValueError(
            f"checkpoint first line is not a header (kind={header.get('kind')!r})"
        )
    stored_hash = header.get("config_hash", "")
    if stored_hash != expected_hash:
        raise ValueError(
            f"checkpoint config_hash mismatch: stored={stored_hash!r} "
            f"expected={expected_hash!r}"
        )

    done_ids: set[str] = set()
    all_rows: list[dict] = []

    body_lines = raw_lines[1:]
    for idx, raw in enumerate(body_lines):
        raw = raw.rstrip("\n")
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as exc:
            is_last = idx == len(body_lines) - 1
            if is_last:
                print(
                    f"WARNING: skipping truncated final checkpoint line "
                    f"(crash mid-write): {exc}"
                )
                continue
            raise ValueError(
                f"malformed checkpoint line {idx + 2} (not the last line): {exc}"
            ) from exc
        if rec.get("kind") != "conv":
            raise ValueError(
                f"unexpected checkpoint record kind {rec.get('kind')!r} at line {idx + 2}"
            )
        cid = rec.get("conv_id", "")
        done_ids.add(cid)
        all_rows.extend(rec.get("rows") or [])

    return done_ids, all_rows


# ---------------------------------------------------------------------------
# Pause flag
# ---------------------------------------------------------------------------

def pause_requested(flag_path: str) -> bool:
    """Return True if the pause flag file exists."""
    return os.path.exists(flag_path)
