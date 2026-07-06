"""Dependency-aware task graph for taOSmd (archive-backed projection).

Every mutation is appended to the zero-loss archive before touching the
projection tables, so the full audit trail is free and the tables can be
rebuilt from the archive at any time via :func:`rebuild_from_archive`.

Concept credit: the dependency-graph, ready-queue, and prime ideas come from
beads (github.com/gastownhall/beads); this is an independent minimal
implementation for the taosmd substrate.

Schema lives in ``tasks.db`` under the data dir, following the same pattern
used by ``knowledge-graph.db``, ``vector-memory.db``, etc.

Public API
----------
create_task(title, *, body, project, assignee, priority, depends_on,
            created_by, data_dir) -> dict
list_tasks(*, status, project, assignee, limit, data_dir) -> list[dict]
ready_tasks(*, project, assignee, limit, data_dir) -> list[dict]
prime(*, project, assignee, data_dir) -> {"text": str, "tasks": list[dict]}
update_task(task_id, *, status, assignee, priority, body, data_dir) -> dict
add_edge(from_id, to_id, edge_type, created_by, data_dir) -> dict
remove_edge(from_id, to_id, edge_type, data_dir) -> dict
rebuild_from_archive(data_dir) -> dict
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from . import _db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STATUSES = {"open", "in_progress", "blocked", "closed", "superseded"}
CLOSED_STATUSES = {"closed", "superseded"}
VALID_EDGE_TYPES = {"blocks", "parent", "relates", "duplicates"}

# Approximate token cap for prime briefing (~6000 chars = ~1500 tokens)
PRIME_CHAR_CAP = 6000

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    body        TEXT,
    status      TEXT NOT NULL DEFAULT 'open'
                    CHECK(status IN ('open','in_progress','blocked','closed','superseded')),
    assignee    TEXT,
    project     TEXT,
    priority    INTEGER NOT NULL DEFAULT 0,
    created_ts  REAL NOT NULL,
    updated_ts  REAL NOT NULL,
    closed_ts   REAL,
    created_by  TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS task_edges (
    from_id     TEXT NOT NULL REFERENCES tasks(id),
    to_id       TEXT NOT NULL REFERENCES tasks(id),
    type        TEXT NOT NULL CHECK(type IN ('blocks','parent','relates','duplicates')),
    created_ts  REAL NOT NULL,
    created_by  TEXT NOT NULL,
    removed_ts  REAL
);

CREATE INDEX IF NOT EXISTS idx_tasks_status   ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_project  ON tasks(project);
CREATE INDEX IF NOT EXISTS idx_tasks_assignee ON tasks(assignee);
CREATE INDEX IF NOT EXISTS idx_edges_to_id    ON task_edges(to_id);
CREATE INDEX IF NOT EXISTS idx_edges_from_id  ON task_edges(from_id);

CREATE VIEW IF NOT EXISTS ready_tasks AS
SELECT t.*
FROM tasks t
WHERE t.status = 'open'
  AND NOT EXISTS (
    SELECT 1 FROM task_edges e
    JOIN tasks blocker ON blocker.id = e.from_id
    WHERE e.to_id = t.id
      AND e.type = 'blocks'
      AND e.removed_ts IS NULL
      AND blocker.status NOT IN ('closed', 'superseded')
  )
ORDER BY t.priority DESC, t.created_ts ASC;
"""

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_db_cache: dict[str, sqlite3.Connection] = {}


def _get_db(data_dir: str | None) -> sqlite3.Connection:
    """Open (or return cached) the tasks SQLite connection for ``data_dir``."""
    from . import api as _api  # noqa: PLC0415

    resolved = _api._resolve_data_dir(data_dir)
    conn = _db_cache.get(resolved)
    if conn is not None:
        return conn
    path = Path(resolved)
    path.mkdir(parents=True, exist_ok=True)
    db_path = str(path / "tasks.db")
    conn = _db.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    _db_cache[resolved] = conn
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    # Parse metadata JSON
    try:
        d["metadata"] = json.loads(d.get("metadata") or "{}")
    except (json.JSONDecodeError, TypeError):
        d["metadata"] = {}
    return d


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def _make_task_id(title: str, project: str | None, created_ts: float) -> str:
    """Deterministic content-hash task id."""
    raw = f"{title}|{project or ''}|{created_ts}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"t-{digest}"


# ---------------------------------------------------------------------------
# Archive helper
# ---------------------------------------------------------------------------

async def _archive_task_event(
    action: str,
    payload: dict,
    *,
    created_by: str,
    project: str | None,
    data_dir: str | None,
) -> None:
    """Append a task event to the archive BEFORE touching projection tables."""
    from . import api as _api  # noqa: PLC0415

    stores = await _api._ensure_stores(data_dir)
    archive = stores["archive"]
    await archive.record(
        "task",
        {"action": action, **payload},
        agent_name=created_by,
        project=project,
    )


# ---------------------------------------------------------------------------
# Core mutations
# ---------------------------------------------------------------------------

async def create_task(
    title: str,
    *,
    body: str | None = None,
    project: str | None = None,
    assignee: str | None = None,
    priority: int = 0,
    depends_on: list[str] | None = None,
    created_by: str,
    data_dir: str | None = None,
) -> dict:
    """Create a new task and return the task object.

    ``depends_on`` is an optional list of existing task IDs that must be
    completed before this task is ready; each creates a ``blocks`` edge
    pointing at the new task (blocker -> new task).

    Raises :class:`ValueError` if ``created_by`` is empty or if any ID in
    ``depends_on`` does not exist.
    """
    if not created_by:
        raise ValueError("created_by is required")
    if not title:
        raise ValueError("title is required")

    now = time.time()
    task_id = _make_task_id(title, project, now)

    row: dict[str, Any] = {
        "id": task_id,
        "title": title,
        "body": body,
        "status": "open",
        "assignee": assignee,
        "project": project,
        "priority": int(priority),
        "created_ts": now,
        "updated_ts": now,
        "closed_ts": None,
        "created_by": created_by,
        "metadata": "{}",
    }

    # Archive BEFORE touching projection
    await _archive_task_event(
        "created",
        {k: v for k, v in row.items()},
        created_by=created_by,
        project=project,
        data_dir=data_dir,
    )

    conn = _get_db(data_dir)
    conn.execute(
        """INSERT INTO tasks
           (id, title, body, status, assignee, project, priority,
            created_ts, updated_ts, closed_ts, created_by, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            task_id, title, body, "open", assignee, project, int(priority),
            now, now, None, created_by, "{}",
        ),
    )
    conn.commit()

    # Create blocking edges for depends_on
    if depends_on:
        for blocker_id in depends_on:
            # Verify blocker exists
            blocker = conn.execute(
                "SELECT id FROM tasks WHERE id = ?", (blocker_id,)
            ).fetchone()
            if blocker is None:
                raise ValueError(f"depends_on task not found: {blocker_id!r}")
            await add_edge(blocker_id, task_id, "blocks", created_by, data_dir=data_dir)

    result = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return _row_to_dict(result)


async def list_tasks(
    *,
    status: str | None = None,
    project: str | None = None,
    assignee: str | None = None,
    limit: int = 50,
    data_dir: str | None = None,
) -> list[dict]:
    """Return tasks matching the given filters, newest first."""
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")

    conn = _get_db(data_dir)
    conditions: list[str] = []
    params: list[Any] = []

    if status is not None:
        conditions.append("status = ?")
        params.append(status)
    if project is not None:
        conditions.append("project = ?")
        params.append(project)
    if assignee is not None:
        conditions.append("assignee = ?")
        params.append(assignee)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)
    rows = conn.execute(
        f"SELECT * FROM tasks {where} ORDER BY created_ts DESC LIMIT ?",
        params,
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


async def get_task_projects(
    task_ids: list[str],
    *,
    data_dir: str | None = None,
) -> dict[str, str | None]:
    """Return ``{task_id: project}`` for the ids that exist.

    Missing ids are simply absent from the result. Used by the HTTP
    server's token-binding layer to enforce project scoping on edge
    mutations: ``task_edges`` has no project column, so scoping resolves
    through the tasks table.
    """
    ids = [tid for tid in task_ids if isinstance(tid, str) and tid]
    if not ids:
        return {}
    conn = _get_db(data_dir)
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT id, project FROM tasks WHERE id IN ({placeholders})",  # noqa: S608 - placeholders only
        ids,
    ).fetchall()
    return {row["id"]: row["project"] for row in rows}


async def ready_tasks(
    *,
    project: str | None = None,
    assignee: str | None = None,
    limit: int = 20,
    data_dir: str | None = None,
) -> list[dict]:
    """Return tasks from the ready_tasks view (open, no active blockers)."""
    conn = _get_db(data_dir)
    conditions: list[str] = []
    params: list[Any] = []

    if project is not None:
        conditions.append("project = ?")
        params.append(project)
    if assignee is not None:
        conditions.append("assignee = ?")
        params.append(assignee)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)
    rows = conn.execute(
        f"SELECT * FROM ready_tasks {where} LIMIT ?",
        params,
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


async def prime(
    *,
    project: str | None = None,
    assignee: str | None = None,
    data_dir: str | None = None,
) -> dict:
    """Return a token-budgeted session-bootstrap briefing.

    Returns ``{"text": str, "tasks": list[dict]}``. The text covers four
    sections in priority order: ready, in-progress, blocked, recently-closed
    (last 24 hours). The combined text is capped at ~6000 characters (~1500
    tokens) by truncating lower-priority sections first (recently-closed,
    then blocked, then in-progress) before trimming ready tasks.
    """
    conn = _get_db(data_dir)
    common_conditions: list[str] = []
    common_params: list[Any] = []
    if project is not None:
        common_conditions.append("project = ?")
        common_params.append(project)
    if assignee is not None:
        common_conditions.append("assignee = ?")
        common_params.append(assignee)

    def _where(extra: list[str] | None = None) -> str:
        conds = list(common_conditions)
        if extra:
            conds.extend(extra)
        return f"WHERE {' AND '.join(conds)}" if conds else ""

    # Fetch each section (generous limits; we will truncate in text)
    ready_rows = conn.execute(
        f"SELECT * FROM ready_tasks {_where()} LIMIT 50",
        common_params,
    ).fetchall()

    ip_rows = conn.execute(
        f"SELECT * FROM tasks {_where(['status = ?'])} ORDER BY priority DESC, created_ts ASC LIMIT 20",
        common_params + ["in_progress"],
    ).fetchall()

    # Blocked: status 'open' but has active blocker
    blocked_rows = conn.execute(
        f"""SELECT t.* FROM tasks t
            {_where(['t.status = ?',
                     'EXISTS (SELECT 1 FROM task_edges e '
                     'JOIN tasks b ON b.id = e.from_id '
                     'WHERE e.to_id = t.id AND e.type = ? '
                     'AND e.removed_ts IS NULL '
                     "AND b.status NOT IN (?,?))"
                    ])}
            ORDER BY t.priority DESC, t.created_ts ASC LIMIT 20""",
        common_params + ["open", "blocks", "closed", "superseded"],
    ).fetchall()

    since_24h = time.time() - 86400
    closed_rows = conn.execute(
        f"SELECT * FROM tasks {_where(['status IN (?,?)', 'closed_ts >= ?'])} "
        "ORDER BY closed_ts DESC LIMIT 20",
        common_params + ["closed", "superseded", since_24h],
    ).fetchall()

    # Fetch blocker titles for blocked tasks
    blocker_map: dict[str, str] = {}
    for r in blocked_rows:
        tid = r["id"]
        blocker = conn.execute(
            """SELECT t.title FROM task_edges e
               JOIN tasks t ON t.id = e.from_id
               WHERE e.to_id = ? AND e.type = 'blocks'
                 AND e.removed_ts IS NULL
                 AND t.status NOT IN ('closed','superseded')
               LIMIT 1""",
            (tid,),
        ).fetchone()
        if blocker:
            blocker_map[tid] = blocker["title"]

    # Build text sections
    def _task_line(t: sqlite3.Row) -> str:
        extra = ""
        if t["assignee"]:
            extra += f" @{t['assignee']}"
        if t["priority"]:
            extra += f" p{t['priority']}"
        return f"  [{t['id']}] {t['title']}{extra}"

    sections: list[tuple[str, list[str]]] = []

    if ready_rows:
        lines = [_task_line(t) for t in ready_rows]
        sections.append(("READY", lines))

    if ip_rows:
        lines = [_task_line(t) for t in ip_rows]
        sections.append(("IN PROGRESS", lines))

    if blocked_rows:
        lines = []
        for t in blocked_rows:
            blocker_title = blocker_map.get(t["id"], "unknown")
            lines.append(f"{_task_line(t)}  (blocked by: {blocker_title})")
        sections.append(("BLOCKED", lines))

    if closed_rows:
        lines = [_task_line(t) for t in closed_rows]
        sections.append(("RECENTLY CLOSED (24h)", lines))

    # Assemble text with cap — truncate from the end (lowest priority sections)
    all_tasks_by_section: list[tuple[str, list[sqlite3.Row]]] = [
        ("READY", list(ready_rows)),
        ("IN PROGRESS", list(ip_rows)),
        ("BLOCKED", list(blocked_rows)),
        ("RECENTLY CLOSED (24h)", list(closed_rows)),
    ]

    header = "=== taOSmd Task Briefing ===\n"
    tail = ""
    if project:
        header += f"Project: {project}\n"
    header += "\n"

    # Build final text respecting cap
    text_parts = [header]
    referenced_ids: set[str] = set()
    char_used = len(header)

    for section_name, rows in all_tasks_by_section:
        if not rows:
            continue
        section_header = f"-- {section_name} --\n"
        section_lines: list[str] = []
        if section_name == "BLOCKED":
            for t in rows:
                blocker_title = blocker_map.get(t["id"], "unknown")
                section_lines.append(f"{_task_line(t)}  (blocked by: {blocker_title})\n")
        else:
            for t in rows:
                section_lines.append(f"{_task_line(t)}\n")

        # Check if the whole section fits
        section_text = section_header + "".join(section_lines) + "\n"
        if char_used + len(section_text) <= PRIME_CHAR_CAP:
            text_parts.append(section_text)
            char_used += len(section_text)
            for t in rows:
                referenced_ids.add(t["id"])
        else:
            # Fit as many lines as possible
            remaining = PRIME_CHAR_CAP - char_used - len(section_header) - 1
            if remaining > 20:
                text_parts.append(section_header)
                char_used += len(section_header)
                added = 0
                for i, (t, line) in enumerate(zip(rows, section_lines)):
                    if char_used + len(line) > PRIME_CHAR_CAP:
                        omitted = len(rows) - added
                        text_parts.append(f"  ... ({omitted} more)\n")
                        break
                    text_parts.append(line)
                    char_used += len(line)
                    referenced_ids.add(t["id"])
                    added += 1
            break  # Stop adding sections once we hit the cap

    text = "".join(text_parts)

    # Collect referenced task objects
    all_rows = list(ready_rows) + list(ip_rows) + list(blocked_rows) + list(closed_rows)
    seen: set[str] = set()
    tasks_out: list[dict] = []
    for r in all_rows:
        if r["id"] in referenced_ids and r["id"] not in seen:
            tasks_out.append(_row_to_dict(r))
            seen.add(r["id"])

    return {"text": text, "tasks": tasks_out}


async def update_task(
    task_id: str,
    *,
    status: str | None = None,
    assignee: str | None = None,
    priority: int | None = None,
    body: str | None = None,
    data_dir: str | None = None,
) -> dict:
    """Update task fields and return the updated task.

    Raises :class:`ValueError` if the task does not exist or if ``status``
    is not in the valid set.
    """
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")

    conn = _get_db(data_dir)
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        raise ValueError(f"task not found: {task_id!r}")

    existing = _row_to_dict(row)
    now = time.time()

    updates: dict[str, Any] = {"updated_ts": now}
    if status is not None:
        updates["status"] = status
        if status in CLOSED_STATUSES:
            updates["closed_ts"] = now
    if assignee is not None:
        updates["assignee"] = assignee
    if priority is not None:
        updates["priority"] = int(priority)
    if body is not None:
        updates["body"] = body

    # Archive BEFORE touching projection
    actor = existing.get("created_by") or "unknown"
    project = existing.get("project")
    await _archive_task_event(
        "updated",
        {"id": task_id, **updates},
        created_by=actor,
        project=project,
        data_dir=data_dir,
    )

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    params = list(updates.values()) + [task_id]
    conn.execute(f"UPDATE tasks SET {set_clause} WHERE id = ?", params)
    conn.commit()

    result = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return _row_to_dict(result)


async def add_edge(
    from_id: str,
    to_id: str,
    edge_type: str,
    created_by: str,
    *,
    data_dir: str | None = None,
) -> dict:
    """Add an edge between two tasks and return the edge record.

    Raises :class:`ValueError` if either task does not exist or the edge type
    is invalid.
    """
    if edge_type not in VALID_EDGE_TYPES:
        raise ValueError(f"edge_type must be one of {sorted(VALID_EDGE_TYPES)}")

    conn = _get_db(data_dir)
    for tid, label in ((from_id, "from_id"), (to_id, "to_id")):
        if not conn.execute("SELECT 1 FROM tasks WHERE id = ?", (tid,)).fetchone():
            raise ValueError(f"task not found: {tid!r} ({label})")

    now = time.time()
    edge: dict = {
        "from_id": from_id,
        "to_id": to_id,
        "type": edge_type,
        "created_ts": now,
        "created_by": created_by,
        "removed_ts": None,
    }

    # Fetch project from the to_id task for archive tagging
    to_row = conn.execute("SELECT project FROM tasks WHERE id = ?", (to_id,)).fetchone()
    project = to_row["project"] if to_row else None

    # Archive BEFORE touching projection
    await _archive_task_event(
        "edge_added",
        edge,
        created_by=created_by,
        project=project,
        data_dir=data_dir,
    )

    conn.execute(
        "INSERT INTO task_edges (from_id, to_id, type, created_ts, created_by, removed_ts) "
        "VALUES (?, ?, ?, ?, ?, NULL)",
        (from_id, to_id, edge_type, now, created_by),
    )
    conn.commit()
    return edge


async def remove_edge(
    from_id: str,
    to_id: str,
    edge_type: str,
    *,
    data_dir: str | None = None,
) -> dict:
    """Soft-remove an edge by setting removed_ts (never deletes).

    Returns the edge record with ``removed_ts`` set. Raises :class:`ValueError`
    if the edge does not exist or is already removed.
    """
    if edge_type not in VALID_EDGE_TYPES:
        raise ValueError(f"edge_type must be one of {sorted(VALID_EDGE_TYPES)}")

    conn = _get_db(data_dir)
    row = conn.execute(
        "SELECT rowid, * FROM task_edges "
        "WHERE from_id = ? AND to_id = ? AND type = ? AND removed_ts IS NULL",
        (from_id, to_id, edge_type),
    ).fetchone()
    if row is None:
        raise ValueError(
            f"active edge not found: {from_id!r} --[{edge_type}]--> {to_id!r}"
        )

    now = time.time()

    # Fetch project for archive tagging
    to_task = conn.execute("SELECT project FROM tasks WHERE id = ?", (to_id,)).fetchone()
    project = to_task["project"] if to_task else None
    from_task = conn.execute("SELECT created_by FROM tasks WHERE id = ?", (from_id,)).fetchone()
    actor = from_task["created_by"] if from_task else "unknown"

    edge: dict = {
        "from_id": from_id,
        "to_id": to_id,
        "type": edge_type,
        "created_ts": row["created_ts"],
        "created_by": row["created_by"],
        "removed_ts": now,
    }

    # Archive BEFORE touching projection
    await _archive_task_event(
        "edge_removed",
        edge,
        created_by=actor,
        project=project,
        data_dir=data_dir,
    )

    conn.execute(
        "UPDATE task_edges SET removed_ts = ? "
        "WHERE from_id = ? AND to_id = ? AND type = ? AND removed_ts IS NULL",
        (now, from_id, to_id, edge_type),
    )
    conn.commit()
    return edge


# ---------------------------------------------------------------------------
# Archive rebuild
# ---------------------------------------------------------------------------

async def rebuild_from_archive(data_dir: str | None = None) -> dict:
    """Replay task archive events into fresh projection tables.

    Drops and recreates the ``tasks`` and ``task_edges`` tables (the view
    is recreated automatically), then replays every ``event_type="task"``
    event from the archive in timestamp order.

    Returns ``{"tasks_rebuilt": int, "edges_rebuilt": int}``.
    """
    from . import api as _api  # noqa: PLC0415

    # Drop cached connection so we get a fresh one after schema recreate
    resolved = _api._resolve_data_dir(data_dir)
    old_conn = _db_cache.pop(resolved, None)
    if old_conn is not None:
        try:
            old_conn.close()
        except Exception:  # noqa: BLE001
            pass

    # Rebuild schema on a fresh connection
    path = Path(resolved)
    db_path = str(path / "tasks.db")
    conn = _db.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Drop projection tables (view drops automatically)
    conn.executescript("""
        DROP VIEW  IF EXISTS ready_tasks;
        DROP TABLE IF EXISTS task_edges;
        DROP TABLE IF EXISTS tasks;
    """)
    conn.executescript(_SCHEMA)
    conn.commit()
    _db_cache[resolved] = conn

    # Load archive events
    stores = await _api._ensure_stores(data_dir)
    archive = stores["archive"]
    events = await archive.query(event_type="task", limit=100000)
    # Sort by timestamp ascending for correct replay order
    events.sort(key=lambda e: e.get("timestamp", 0))

    tasks_rebuilt = 0
    edges_rebuilt = 0

    for event in events:
        try:
            data_json = event.get("data_json") or event.get("data") or "{}"
            if isinstance(data_json, str):
                data = json.loads(data_json)
            elif isinstance(data_json, dict):
                data = data_json
            else:
                continue
            action = data.get("action")
            if action == "created":
                conn.execute(
                    """INSERT OR REPLACE INTO tasks
                       (id, title, body, status, assignee, project, priority,
                        created_ts, updated_ts, closed_ts, created_by, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        data.get("id"), data.get("title"), data.get("body"),
                        data.get("status", "open"), data.get("assignee"),
                        data.get("project"), data.get("priority", 0),
                        data.get("created_ts"), data.get("updated_ts"),
                        data.get("closed_ts"), data.get("created_by"), "{}",
                    ),
                )
                tasks_rebuilt += 1
            elif action == "updated":
                task_id = data.get("id")
                if not task_id:
                    continue
                updates = {
                    k: v for k, v in data.items()
                    if k not in ("action", "id") and v is not None
                }
                if not updates:
                    continue
                set_clause = ", ".join(f"{k} = ?" for k in updates)
                params = list(updates.values()) + [task_id]
                conn.execute(
                    f"UPDATE tasks SET {set_clause} WHERE id = ?", params
                )
            elif action == "edge_added":
                conn.execute(
                    "INSERT OR IGNORE INTO task_edges "
                    "(from_id, to_id, type, created_ts, created_by, removed_ts) "
                    "VALUES (?, ?, ?, ?, ?, NULL)",
                    (
                        data.get("from_id"), data.get("to_id"),
                        data.get("type"), data.get("created_ts"),
                        data.get("created_by"),
                    ),
                )
                edges_rebuilt += 1
            elif action == "edge_removed":
                removed_ts = data.get("removed_ts")
                if removed_ts:
                    conn.execute(
                        "UPDATE task_edges SET removed_ts = ? "
                        "WHERE from_id = ? AND to_id = ? AND type = ? "
                        "AND removed_ts IS NULL",
                        (
                            removed_ts, data.get("from_id"),
                            data.get("to_id"), data.get("type"),
                        ),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("rebuild_from_archive: skipped event %s: %s", event.get("id"), exc)

    conn.commit()
    return {"tasks_rebuilt": tasks_rebuilt, "edges_rebuilt": edges_rebuilt}
