"""Sync merge utilities for taOSmd account-device-sync.

This module provides content-keyed merge operations for vector_memory and
knowledge_graph stores. Merges are by fact/content, never by row id, because
row ids are device-local (autoincrement / time-hash) and the same fact carries
different ids on different devices.

Bus 7401 contract (verbatim from taOS-dev):
- KG triple: same fact = (entity_id(subject), normalised predicate, entity_id(object))
- SINGULAR_PREDICATES: same subject+predicate, different object, both open -> later valid_from wins
- kg_entities: entity_id is name-derived and deterministic, merge by it
- vector_memory: key = sha256(namespace + normalised text), inserted as new local row
- archive: append-only union, never rewritten
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

from .vector_memory import StoreModeMismatch


def _content_key(text: str, metadata_json: str) -> str:
    """Compute content key: sha256 of json.dumps([agent, project, normalised_text])."""
    try:
        meta = json.loads(metadata_json) if metadata_json else {}
    except (json.JSONDecodeError, TypeError):
        meta = {}
    agent = meta.get("agent") or ""
    project = meta.get("project") or ""
    normalised = " ".join(text.split()).lower()
    return hashlib.sha256(json.dumps([agent, project, normalised]).encode()).hexdigest()


def _get_store_mode(conn: sqlite3.Connection) -> str | None:
    """Get the store mode from store_meta table."""
    row = conn.execute("SELECT value FROM store_meta WHERE key = 'mode'").fetchone()
    return row["value"] if row else None


def _compare_store_modes(local_mode: str | None, peer_mode: str | None) -> None:
    """Compare store modes; raise StoreModeMismatch if they differ on any key present in both."""
    if local_mode is None or peer_mode is None:
        # If either is missing, we can't compare; allow the merge (legacy behavior)
        return

    # Use the same parsing logic as VectorMemory._parse_mode
    def parse_mode(sig: str) -> dict:
        out = {}
        for part in sig.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                out[k] = v
        return out

    local_parsed = parse_mode(local_mode)
    peer_parsed = parse_mode(peer_mode)

    # Check for conflicts on keys present in both
    conflict = {k: v for k, v in peer_parsed.items() if k in local_parsed and local_parsed[k] != v}
    if conflict:
        raise StoreModeMismatch(
            f"vector store mode mismatch: local={local_mode} peer={peer_mode} (conflict on {conflict})"
        )


def _build_local_key_index(local_conn: sqlite3.Connection) -> dict[str, dict]:
    """Build a map of content_key -> local row dict for all local vector_memory rows."""
    index = {}
    rows = local_conn.execute("""
        SELECT id, text, metadata_json, created_at, valid_to
        FROM vector_memory
    """).fetchall()
    for row in rows:
        key = _content_key(row["text"], row["metadata_json"])
        index[key] = {
            "id": row["id"],
            "text": row["text"],
            "metadata_json": row["metadata_json"],
            "created_at": row["created_at"],
            "valid_to": row["valid_to"],
        }
    return index


def merge_vectors(local_db_path: str, peer_db_path: str) -> dict:
    """Merge peer vector_memory into local.

    Args:
        local_db_path: Path to local SQLite database (read-write).
        peer_db_path: Path to peer SQLite database (opened read-only via URI).

    Returns:
        Dict with keys: "added", "deduped", "metadata_updated".

    Content key per row = sha256(json.dumps([agent, project, normalised_text])).
    For each peer row:
      - key absent locally -> INSERT as new local row (fresh autoincrement id)
      - key present -> keep local row; if peer created_at newer, replace metadata_json;
        on EQUAL created_at, lexicographically larger metadata_json wins;
        set local valid_to = peer's when peer's is set.
    Never deletes, never writes a peer id. One transaction, rollback on error.
    """
    # Open local read-write, peer read-only via URI
    local_conn = sqlite3.connect(local_db_path)
    local_conn.row_factory = sqlite3.Row
    peer_uri = f"file:{peer_db_path}?mode=ro"
    peer_conn = sqlite3.connect(peer_uri, uri=True)
    peer_conn.row_factory = sqlite3.Row

    try:
        # Compare store_meta modes before any merge
        local_mode = _get_store_mode(local_conn)
        peer_mode = _get_store_mode(peer_conn)
        _compare_store_modes(local_mode, peer_mode)

        # Begin transaction on local
        local_conn.execute("BEGIN IMMEDIATE")

        added = 0
        deduped = 0
        metadata_updated = 0

        # Build local key index for O(1) lookups
        local_index = _build_local_key_index(local_conn)

        # Fetch all peer rows
        peer_rows = peer_conn.execute("""
            SELECT text, embedding, metadata_json, created_at, valid_to
            FROM vector_memory
        """).fetchall()

        for prow in peer_rows:
            text = prow["text"]
            embedding = prow["embedding"]
            metadata_json = prow["metadata_json"]
            created_at = prow["created_at"]
            valid_to = prow["valid_to"]

            key = _content_key(text, metadata_json)

            match_row = local_index.get(key)

            if match_row is None:
                # Key absent locally -> INSERT as new row (no id column, fresh autoincrement)
                local_conn.execute("""
                    INSERT INTO vector_memory (text, embedding, metadata_json, created_at, valid_to)
                    VALUES (?, ?, ?, ?, ?)
                """, (text, embedding, metadata_json, created_at, valid_to))
                added += 1
                # Update local index so subsequent peer rows with same key see it
                # (though peer shouldn't have duplicate keys, defensive)
                local_index[key] = {"id": None, "text": text, "metadata_json": metadata_json,
                                    "created_at": created_at, "valid_to": valid_to}
            else:
                # Key present -> keep local row, maybe update metadata/valid_to
                deduped += 1

                local_created_at = match_row["created_at"]
                local_metadata_json = match_row["metadata_json"]
                local_valid_to = match_row["valid_to"]

                update_metadata = False
                new_metadata_json = local_metadata_json

                if created_at > local_created_at:
                    update_metadata = True
                    new_metadata_json = metadata_json
                elif created_at == local_created_at:
                    # Tie-break: lexicographically larger metadata_json wins (content tie-break, bus 7404)
                    if metadata_json > local_metadata_json:
                        update_metadata = True
                        new_metadata_json = metadata_json

                new_valid_to = local_valid_to
                if valid_to is not None:
                    new_valid_to = valid_to

                if update_metadata or new_valid_to != local_valid_to:
                    local_conn.execute("""
                        UPDATE vector_memory
                        SET metadata_json = ?, valid_to = ?
                        WHERE id = ?
                    """, (new_metadata_json, new_valid_to, match_row["id"]))
                    if update_metadata and new_metadata_json != local_metadata_json:
                        metadata_updated += 1
                    # Update local index for consistency
                    match_row["metadata_json"] = new_metadata_json
                    match_row["valid_to"] = new_valid_to

        local_conn.commit()
        return {"added": added, "deduped": deduped, "metadata_updated": metadata_updated}

    except BaseException:
        local_conn.rollback()
        raise
    finally:
        local_conn.close()
        peer_conn.close()