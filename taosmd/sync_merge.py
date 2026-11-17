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
import time

from .predicate_vocab import normalise, SINGULAR_PREDICATES
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
    """Get the store mode from store_meta table. Missing table treated as no marker."""
    try:
        row = conn.execute("SELECT value FROM store_meta WHERE key = 'mode'").fetchone()
        return row["value"] if row else None
    except sqlite3.OperationalError:
        # Table does not exist (older store) -> treat as no marker
        return None


def _compare_store_modes(local_conn: sqlite3.Connection, peer_conn: sqlite3.Connection) -> None:
    """Compare store modes; raise StoreModeMismatch if they differ on any key present in both.

    A side with no marker but with rows is treated as legacy dense
    (late_interaction=0, binary_quant=0). A side with no marker and no rows
    imposes no constraint.
    """
    local_mode = _get_store_mode(local_conn)
    peer_mode = _get_store_mode(peer_conn)

    # Helper to check if a store has any rows
    def _has_rows(conn: sqlite3.Connection) -> bool:
        row = conn.execute("SELECT 1 FROM vector_memory LIMIT 1").fetchone()
        return row is not None

    local_has_rows = _has_rows(local_conn)
    peer_has_rows = _has_rows(peer_conn)

    # Resolve local mode: if no marker but has rows, use legacy format
    if local_mode is None:
        if local_has_rows:
            local_mode = "late_interaction=0;binary_quant=0"
        else:
            local_mode = None  # no constraint

    # Resolve peer mode: if no marker but has rows, use legacy format
    if peer_mode is None:
        if peer_has_rows:
            peer_mode = "late_interaction=0;binary_quant=0"
        else:
            peer_mode = None  # no constraint

    # If either resolved to None (empty no-marker store), no comparison needed
    if local_mode is None or peer_mode is None:
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
        _compare_store_modes(local_conn, peer_conn)

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
                if local_valid_to is not None and valid_to is not None:
                    # Both sides have valid_to set -> keep the EARLIER (min) one (convergence, bus 7404)
                    new_valid_to = min(local_valid_to, valid_to)
                elif valid_to is not None:
                    # Only peer has valid_to set -> closure wins (bus 7401)
                    new_valid_to = valid_to
                # else: only local has it or neither -> keep local_valid_to (which may be None)

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


_triple_id_counter = 0


def _new_triple_id(subject_id: str, predicate: str, object_id: str) -> str:
    """Fresh triple id using the same formula as KnowledgeGraph._triple_id.

    Row ids are device-local (bus 7404): peer ids are never reused and a new
    fact gets a fresh local id. time.time_ns() plus a monotonic counter keeps
    inserts within one merge from colliding.
    """
    global _triple_id_counter
    _triple_id_counter += 1
    raw = f"{subject_id}:{predicate}:{object_id}:{time.time_ns()}:{_triple_id_counter}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _parse_source_ids(raw: str | None) -> list[str]:
    """Parse the source_ids JSON column; never raise on malformed input."""
    try:
        data = json.loads(raw) if raw else []
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data]


def _union_source_ids(local_json: str | None, peer_json: str | None) -> str:
    """Sorted JSON union of two source_ids lists (deduped, bus 7401)."""
    merged = set(_parse_source_ids(local_json)) | set(_parse_source_ids(peer_json))
    return json.dumps(sorted(merged))


def _load_local_triples(conn: sqlite3.Connection) -> dict:
    """Index local triples by content key (subject, normalised predicate, object)."""
    index: dict = {}
    rows = conn.execute(
        "SELECT id, subject_id, predicate, object_id, valid_from, valid_to, "
        "confidence, source, source_ids, superseded_by, appeared_count, created_at "
        "FROM kg_triples"
    ).fetchall()
    for row in rows:
        key = (row["subject_id"], normalise(row["predicate"]), row["object_id"])
        index.setdefault(key, []).append(dict(row))
    return index


def _merge_open_pair(conn, local_row: dict, prow: sqlite3.Row, predicate: str) -> None:
    """Both open: keep the local row; valid_from=min, confidence=max, source_ids union, appeared_count=max."""
    new_valid_from = min(local_row["valid_from"], prow["valid_from"])
    new_confidence = max(local_row["confidence"], prow["confidence"])
    new_source_ids = _union_source_ids(local_row["source_ids"], prow["source_ids"])
    new_appeared_count = max(local_row["appeared_count"], prow["appeared_count"])
    conn.execute(
        "UPDATE kg_triples SET valid_from = ?, confidence = ?, source_ids = ?, appeared_count = ? WHERE id = ?",
        (new_valid_from, new_confidence, new_source_ids, new_appeared_count, local_row["id"]),
    )


def _insert_peer_triple(conn: sqlite3.Connection, prow: sqlite3.Row, predicate: str) -> str:
    """Insert a peer triple as a new local row with a fresh id (never a peer id).

    Returns the new local triple id.
    """
    tid = _new_triple_id(prow["subject_id"], predicate, prow["object_id"])
    conn.execute(
        "INSERT INTO kg_triples (id, subject_id, predicate, object_id, valid_from, valid_to, "
        "confidence, source, source_ids, superseded_by, appeared_count, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            tid,
            prow["subject_id"],
            predicate,
            prow["object_id"],
            prow["valid_from"],
            prow["valid_to"],
            prow["confidence"],
            prow["source"],
            prow["source_ids"],
            prow["superseded_by"],  # may be a peer id; remapped in second pass
            prow["appeared_count"],
            prow["created_at"],
        ),
    )
    return tid


def _remap_superseded_by(
    local_conn: sqlite3.Connection, peer_conn: sqlite3.Connection, inserted: list[tuple]
) -> None:
    """Remap superseded_by from peer ids to local ids.

    Args:
        local_conn: Local database connection (read-write).
        peer_conn: Peer database connection (read-only).
        inserted: List of (local_id, peer_superseded_by, subject_id, predicate, object_id, valid_from)
            for each inserted peer triple that had a non-NULL superseded_by.
    """
    for local_id, peer_superseded_by, subject_id, predicate, object_id, valid_from in inserted:
        if peer_superseded_by is None:
            continue
        # Look up the peer's superseding row by its peer id
        peer_row = peer_conn.execute(
            "SELECT subject_id, predicate, object_id, valid_from FROM kg_triples WHERE id = ?",
            (peer_superseded_by,),
        ).fetchone()
        if peer_row is None:
            # Superseding row not found in peer db; leave as NULL
            local_conn.execute(
                "UPDATE kg_triples SET superseded_by = NULL WHERE id = ?", (local_id,)
            )
            continue
        # Find the LOCAL row with the same content key and valid_from
        # Content key = (subject_id, normalised predicate, object_id)
        sup_predicate = normalise(peer_row["predicate"])
        sup_key = (peer_row["subject_id"], sup_predicate, peer_row["object_id"])
        sup_valid_from = peer_row["valid_from"]
        local_row = local_conn.execute(
            "SELECT id FROM kg_triples "
            "WHERE subject_id = ? AND predicate = ? AND object_id = ? AND valid_from = ?",
            (sup_key[0], sup_key[1], sup_key[2], sup_valid_from),
        ).fetchone()
        if local_row is None:
            # No matching local row found; leave as NULL
            local_conn.execute(
                "UPDATE kg_triples SET superseded_by = NULL WHERE id = ?", (local_id,)
            )
        else:
            local_conn.execute(
                "UPDATE kg_triples SET superseded_by = ? WHERE id = ?",
                (local_row["id"], local_id),
            )


def _merge_entities(local_conn: sqlite3.Connection, peer_conn: sqlite3.Connection, counts: dict) -> None:
    """Insert missing entity ids; upgrade local 'unknown' to peer's type (never downgrade)."""
    entity_placeholder = (None, "", "unknown")
    peer_entities = peer_conn.execute(
        "SELECT id, name, type, properties_json, created_at FROM kg_entities"
    ).fetchall()
    for prow in peer_entities:
        eid = prow["id"]
        existing = local_conn.execute(
            "SELECT type FROM kg_entities WHERE id = ?", (eid,)
        ).fetchone()
        if existing is None:
            local_conn.execute(
                "INSERT INTO kg_entities (id, name, type, properties_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (eid, prow["name"], prow["type"], prow["properties_json"], prow["created_at"]),
            )
            counts["entities_added"] += 1
        elif existing["type"] in entity_placeholder and prow["type"] not in entity_placeholder:
            local_conn.execute(
                "UPDATE kg_entities SET type = ? WHERE id = ?", (prow["type"], eid)
            )
            counts["entities_typed"] += 1


def _merge_triples(local_conn: sqlite3.Connection, peer_conn: sqlite3.Connection, counts: dict) -> list:
    """Content-keyed triple merge keyed by (subject_id, normalised predicate, object_id).

    Never deletes; an inserted peer triple always gets a fresh local id.

    Returns a list of (local_id, peer_superseded_by, subject_id, predicate, object_id, valid_from)
    for each inserted peer triple that had a non-NULL superseded_by, for later remapping.
    """
    local_index = _load_local_triples(local_conn)
    peer_triples = peer_conn.execute(
        "SELECT id, subject_id, predicate, object_id, valid_from, valid_to, "
        "confidence, source, source_ids, superseded_by, appeared_count, created_at "
        "FROM kg_triples"
    ).fetchall()

    inserted_with_superseded_by: list[tuple] = []

    for prow in peer_triples:
        predicate = normalise(prow["predicate"])
        key = (prow["subject_id"], predicate, prow["object_id"])
        local_rows = local_index.get(key, [])
        opens = [r for r in local_rows if r["valid_to"] is None]
        closes = [r for r in local_rows if r["valid_to"] is not None]

        if prow["valid_to"] is None:
            # Peer is open.
            if opens:
                # Both open: merge into the local row.
                local_row = opens[0]
                _merge_open_pair(local_conn, local_row, prow, predicate)
                local_row["valid_from"] = min(local_row["valid_from"], prow["valid_from"])
                local_row["confidence"] = max(local_row["confidence"], prow["confidence"])
                local_row["source_ids"] = _union_source_ids(local_row["source_ids"], prow["source_ids"])
                local_row["appeared_count"] = max(local_row["appeared_count"], prow["appeared_count"])
                counts["triples_merged"] += 1
            elif closes and any(c["valid_to"] >= prow["valid_from"] for c in closes):
                # Symmetric: local closed, peer open started at or before local
                # closed -> keep local, do not import.
                continue
            else:
                # Re-assert after closure: keep both rows as separate intervals.
                local_id = _insert_peer_triple(local_conn, prow, predicate)
                counts["triples_added"] += 1
                if prow["superseded_by"] is not None:
                    inserted_with_superseded_by.append(
                        (local_id, prow["superseded_by"], prow["subject_id"], predicate, prow["object_id"], prow["valid_from"])
                    )
        else:
            # Peer closed.
            peer_valid_to = prow["valid_to"]
            if opens:
                local_row = opens[0]
                if local_row["valid_from"] <= peer_valid_to:
                    # Closure wins: close the local open row at peer's valid_to.
                    local_conn.execute(
                        "UPDATE kg_triples SET valid_to = ? WHERE id = ?",
                        (peer_valid_to, local_row["id"]),
                    )
                    local_row["valid_to"] = peer_valid_to
                    counts["triples_closed"] += 1
                else:
                    # Local re-asserted after peer closed: keep both intervals.
                    local_id = _insert_peer_triple(local_conn, prow, predicate)
                    counts["triples_added"] += 1
                    if prow["superseded_by"] is not None:
                        inserted_with_superseded_by.append(
                            (local_id, prow["superseded_by"], prow["subject_id"], predicate, prow["object_id"], prow["valid_from"])
                        )
            elif closes and any(
                c["valid_from"] == prow["valid_from"] and c["valid_to"] == peer_valid_to
                for c in closes
            ):
                # Exact duplicate of an existing closed interval.
                continue
            else:
                # Historical interval the local store lacks.
                local_id = _insert_peer_triple(local_conn, prow, predicate)
                counts["triples_added"] += 1
                if prow["superseded_by"] is not None:
                    inserted_with_superseded_by.append(
                        (local_id, prow["superseded_by"], prow["subject_id"], predicate, prow["object_id"], prow["valid_from"])
                    )

    return inserted_with_superseded_by


def _resolve_singular_predicates(conn: sqlite3.Connection, counts: dict) -> None:
    """For each SINGULAR_PREDICATES subject with more than one open triple,
    the latest valid_from wins (ties: larger object_id, then larger predicate);
    every loser gets valid_to = winner.valid_from and superseded_by = winner.id.
    Nothing is deleted."""
    for predicate in sorted(SINGULAR_PREDICATES):
        subjects = [
            row["subject_id"]
            for row in conn.execute(
                "SELECT subject_id FROM kg_triples WHERE predicate = ? AND valid_to IS NULL "
                "GROUP BY subject_id HAVING COUNT(*) > 1",
                (predicate,),
            ).fetchall()
        ]
        for subject in subjects:
            rows = conn.execute(
                "SELECT id, object_id, valid_from FROM kg_triples "
                "WHERE subject_id = ? AND predicate = ? AND valid_to IS NULL",
                (subject, predicate),
            ).fetchall()
            winner = max(rows, key=lambda r: (r["valid_from"], r["object_id"], predicate))
            for row in rows:
                if row["id"] == winner["id"]:
                    continue
                conn.execute(
                    "UPDATE kg_triples SET valid_to = ?, superseded_by = ? WHERE id = ?",
                    (winner["valid_from"], winner["id"], row["id"]),
                )
                counts["triples_superseded"] += 1


def merge_kg(local_db_path: str, peer_db_path: str) -> dict:
    """Merge the peer knowledge graph into the local store by content key.

    Bus 7401 contract (verbatim): KG triples are keyed by
    (entity_id(subject), normalised predicate, entity_id(object)) -- never by
    row id, because triple ids hash time.time() and vector ids are
    autoincrement, so the same fact carries different ids on different devices
    and the same id names different memories.

    Args:
        local_db_path: Path to the local SQLite database (read-write).
        peer_db_path: Path to the peer SQLite database (opened read-only via
            file:...?mode=ro).

    Returns:
        Counts: {"entities_added", "entities_typed", "triples_added",
        "triples_merged", "triples_closed", "triples_superseded"}.

    Rules:
    - Entities: insert missing ids; when local type is 'unknown' and peer's is
      not, set local type to peer's (a typed value beats "unknown", never
      downgraded).
    - Triples, both open: keep the local row; valid_from = min, confidence =
      max, source_ids = sorted JSON union, appeared_count = max.
    - Peer closed and local open with local.valid_from <= peer.valid_to:
      closure wins (local valid_to = peer.valid_to). Symmetric case keeps
      local and does not import.
    - An open row whose valid_from is greater than the other side's valid_to
      is a re-assert: both rows are kept as separate intervals (the inserted
      peer row gets a fresh id).
    - After the above, for each SINGULAR_PREDICATES subject with more than one
      open triple, the latest valid_from wins; losers get valid_to and
      superseded_by. Nothing is deleted.

    One transaction on the local db; on any exception the transaction is
    rolled back and the exception is re-raised.
    """
    counts = {
        "entities_added": 0,
        "entities_typed": 0,
        "triples_added": 0,
        "triples_merged": 0,
        "triples_closed": 0,
        "triples_superseded": 0,
    }

    local_conn = sqlite3.connect(local_db_path)
    local_conn.row_factory = sqlite3.Row
    peer_uri = f"file:{peer_db_path}?mode=ro"
    peer_conn = sqlite3.connect(peer_uri, uri=True)
    peer_conn.row_factory = sqlite3.Row

    try:
        local_conn.execute("BEGIN IMMEDIATE")
        _merge_entities(local_conn, peer_conn, counts)
        inserted_with_superseded_by = _merge_triples(local_conn, peer_conn, counts)
        _remap_superseded_by(local_conn, peer_conn, inserted_with_superseded_by)
        _resolve_singular_predicates(local_conn, counts)
        local_conn.commit()
    except BaseException:
        local_conn.rollback()
        raise
    finally:
        local_conn.close()
        peer_conn.close()

    return counts
