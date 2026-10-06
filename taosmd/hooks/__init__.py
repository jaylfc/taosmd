"""Claude Code capture hooks: transcript cursor sync into ingest_batch.

Phase 1: Claude Code only. Reads the per-session transcript from a saved
cursor, keeps user + assistant messages and tool calls/results, and writes
them via ``taosmd.service.ingest_batch`` with stable ids.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

from taosmd._db import connect, run_schema
from taosmd.project import get_project_id


logger = logging.getLogger(__name__)

_CAPTURE_CURSOR_DB = "capture-cursors.db"
_HOOKS_LOG = "hooks.log"
_HOOKS_LOG_DIR = "logs"
_DEFAULT_TOOL_CAP_BYTES = 8 * 1024
_DEFAULT_AGENT = "claude-code"
_DEFAULT_TIMEOUT_S = 5.0

_CURSOR_SCHEMA = """CREATE TABLE IF NOT EXISTS capture_cursors (
    session_id TEXT PRIMARY KEY,
    transcript_path TEXT NOT NULL,
    cwd TEXT NOT NULL,
    project_id TEXT NOT NULL DEFAULT '',
    byte_offset INTEGER NOT NULL DEFAULT 0,
    last_entry_id TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL DEFAULT 0
);"""


def _get_log_path(data_dir: str) -> Path:
    return Path(data_dir) / _HOOKS_LOG_DIR / _HOOKS_LOG


def _log_error(data_dir: str, message: str) -> None:
    try:
        log_path = _get_log_path(data_dir)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")
    except Exception:
        pass


class CaptureCursorStore:
    """Per-session cursor table for capture sync."""

    def __init__(self, data_dir: str) -> None:
        self._data_dir = data_dir
        self._db_path = Path(data_dir) / _CAPTURE_CURSOR_DB
        self._conn = connect(self._db_path)
        run_schema(self._conn, _CURSOR_SCHEMA)

    def get(self, session_id: str):
        row = self._conn.execute(
            "SELECT transcript_path, cwd, project_id, byte_offset, last_entry_id, updated_at "
            "FROM capture_cursors WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "session_id": session_id,
            "transcript_path": row[0],
            "cwd": row[1],
            "project_id": row[2],
            "byte_offset": row[3],
            "last_entry_id": row[4],
            "updated_at": row[5],
        }

    def upsert(
        self,
        session_id: str,
        transcript_path: str,
        cwd: str,
        project_id: str,
        byte_offset: int,
        last_entry_id: str,
        updated_at: float,
    ) -> None:
        self._conn.execute(
            "INSERT INTO capture_cursors "
            "(session_id, transcript_path, cwd, project_id, byte_offset, last_entry_id, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "transcript_path = excluded.transcript_path, "
            "cwd = excluded.cwd, "
            "project_id = excluded.project_id, "
            "byte_offset = excluded.byte_offset, "
            "last_entry_id = excluded.last_entry_id, "
            "updated_at = excluded.updated_at",
            (session_id, transcript_path, cwd, project_id, byte_offset, last_entry_id, updated_at),
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


def _truncate_field(content: str, cap: int) -> tuple[str, bool]:
    if len(content.encode("utf-8")) <= cap:
        return content, False
    return content[:cap], True


def _entry_uuid(entry: dict) -> Optional[str]:
    for key in ("uuid", "id"):
        val = entry.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _entry_ts(entry: dict) -> Optional[float]:
    for key in ("timestamp", "ts", "created_at"):
        val = entry.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


def _parse_transcript_entry(raw: str, tool_cap_bytes: int = _DEFAULT_TOOL_CAP_BYTES) -> Optional[dict]:
    """Parse one JSONL line from a Claude Code transcript.

    Returns a dict with keys ``kind``, ``role``, ``text``, ``metadata``,
    or ``None`` when the line should be skipped.  The ``kind`` field is
    ``"message"``, ``"tool_use"``, ``"tool_result"``, or ``None`` for
    skipped entries.
    """
    try:
        entry = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(entry, dict):
        return None

    entry_type = entry.get("type") or entry.get("kind") or ""

    if entry_type in ("summary", "system", "metadata", "attachment", ""):
        return None

    role: Optional[str] = None
    text = ""
    metadata: dict = {}
    kind = "message"

    if entry_type == "user":
        role = "user"
        msg = entry.get("message") or entry
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, list):
                text = " ".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") != "tool_use"
                )
            else:
                text = str(content)

    elif entry_type == "assistant":
        role = "assistant"
        msg = entry.get("message") or entry
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, list):
                text = " ".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") != "tool_use"
                )
            else:
                text = str(content)

    elif entry_type == "tool_use":
        role = "tool_use"
        kind = "tool_use"
        text = raw
        truncated, was_trunc = _truncate_field(text, tool_cap_bytes)
        metadata["truncated"] = was_trunc
        text = truncated

    elif entry_type == "tool_result":
        role = "tool_result"
        kind = "tool_result"
        text = raw
        truncated, was_trunc = _truncate_field(text, tool_cap_bytes)
        metadata["truncated"] = was_trunc
        text = truncated

    else:
        return None

    text = text.strip()
    if not text:
        return None

    return {"kind": kind, "role": role, "text": text, "metadata": metadata}


def _build_item(
    raw: str,
    session_id: str,
    cwd: str,
    transcript_path: str,
    project_id: str,
    entry: dict,
    byte_offset: int,
    tool_cap_bytes: int = _DEFAULT_TOOL_CAP_BYTES,
) -> Optional[dict]:
    """Build one ingest_batch item from a parsed transcript entry.

    Returns ``None`` for skipped entries.
    """
    parsed = _parse_transcript_entry(raw, tool_cap_bytes=tool_cap_bytes)
    if parsed is None:
        return None

    role = parsed["role"]
    text = parsed["text"]
    meta = dict(parsed["metadata"])

    entry_ts = _entry_ts(entry)
    if entry_ts is not None:
        meta["timestamp"] = entry_ts

    uuid = _entry_uuid(entry)
    if uuid:
        item_id = f"claude-code:{session_id}:{uuid}"
    else:
        item_id = hashlib.sha256(f"{byte_offset}:{raw}".encode()).hexdigest()

    meta.update({
        "source": "hook:claude-code",
        "session_id": session_id,
        "role": role,
        "cwd": cwd,
        "transcript_path": transcript_path,
    })

    return {"text": text, "id": item_id, "metadata": meta}


def _read_complete_lines(raw: bytes, offset: int) -> tuple[list[str], int]:
    """Return (complete_lines, new_offset) from raw bytes starting at offset.

    A trailing incomplete line (no trailing newline) is left for next time.
    """
    if offset < 0:
        offset = 0
    if offset > len(raw):
        offset = len(raw)
    tail = raw[offset:]
    if not tail:
        return [], offset
    text = tail.decode("utf-8", errors="replace")
    if not text.endswith("\n"):
        last_nl = text.rfind("\n")
        if last_nl < 0:
            return [], offset
        lines = text[:last_nl].split("\n")
        return lines, offset + last_nl + 1
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines, offset + len(tail)


async def _ingest_items(items, *, agent: str, data_dir: str) -> dict:
    from taosmd import service as _svc
    return await _svc.ingest_batch(items, agent=agent, data_dir=data_dir)


async def _reconcile_agent(*, agent: str, data_dir: str) -> dict:
    from taosmd import service as _svc
    return await _svc.reconcile(agent=agent, data_dir=data_dir, repair=True)


async def sync_session(
    data_dir: str,
    session_id: str,
    *,
    cursor_store: Optional[CaptureCursorStore] = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    agent: str = _DEFAULT_AGENT,
    tool_cap_bytes: int = _DEFAULT_TOOL_CAP_BYTES,
) -> dict:
    """Sync one Claude Code session's transcript into the archive.

    Returns ``{"ok": bool, "ingested": int, "skipped": int, "timed_out": bool,
    "error": str|None}``.
    """
    store = cursor_store or CaptureCursorStore(data_dir)
    timed_out = False
    had_error: Optional[str] = None
    ingested_count = 0

    try:
        cursor = store.get(session_id)
        if cursor is None:
            cursor = {
                "session_id": session_id,
                "transcript_path": "",
                "cwd": "",
                "project_id": "",
                "byte_offset": 0,
                "last_entry_id": "",
                "updated_at": 0.0,
            }

        transcript_path = cursor["transcript_path"]
        cwd = cursor["cwd"]
        project_id = cursor["project_id"]

        if not transcript_path or not cwd:
            return {
                "ok": True, "ingested": 0, "skipped": 0,
                "timed_out": False, "error": "cursor not initialised",
            }

        if not project_id:
            try:
                project_id = get_project_id(cwd=cwd)
            except Exception:
                project_id = ""

        try:
            raw = Path(transcript_path).read_bytes()
        except OSError as exc:
            return {
                "ok": False, "ingested": 0, "skipped": 0,
                "timed_out": False, "error": str(exc),
            }

        offset = cursor["byte_offset"]
        if len(raw) < offset:
            offset = 0

        lines, new_offset = _read_complete_lines(raw, offset)
        if not lines:
            return {
                "ok": True, "ingested": 0, "skipped": 0,
                "timed_out": False, "error": None,
            }

        items = []
        skipped = 0
        cursor_byte = offset
        for line in lines:
            stripped = line.strip()
            if not stripped:
                cursor_byte += len(line.encode("utf-8")) + 1
                continue
            try:
                entry = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                cursor_byte += len(line.encode("utf-8")) + 1
                skipped += 1
                continue
            if not isinstance(entry, dict):
                cursor_byte += len(line.encode("utf-8")) + 1
                skipped += 1
                continue
            item = _build_item(
                stripped, session_id, cwd, transcript_path, project_id, entry, cursor_byte,
                tool_cap_bytes=tool_cap_bytes,
            )
            cursor_byte += len(line.encode("utf-8")) + 1
            if item is None:
                skipped += 1
                continue
            items.append(item)

        if not items:
            store.upsert(session_id, transcript_path, cwd, project_id, new_offset, "", time.time())
            return {
                "ok": True, "ingested": 0, "skipped": skipped,
                "timed_out": False, "error": None,
            }

        try:
            result = await asyncio.wait_for(
                _ingest_items(items, agent=agent, data_dir=data_dir),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            timed_out = True
            had_error = "timeout"
        except Exception as exc:
            had_error = str(exc)
        else:
            ingested_count = result.get("ingested", 0)
            last_id = ""
            for item in items:
                mid = item.get("id")
                if mid:
                    last_id = mid

            store.upsert(session_id, transcript_path, cwd, project_id, new_offset, last_id, time.time())

        if timed_out:
            try:
                await _reconcile_agent(agent=agent, data_dir=data_dir)
            except Exception as exc:
                _log_error(data_dir, f"reconcile after timeout failed: {exc}")

        return {
            "ok": not bool(had_error),
            "ingested": ingested_count,
            "skipped": skipped,
            "timed_out": timed_out,
            "error": had_error,
        }
    finally:
        if cursor_store is None:
            store.close()


def sync_session_sync(data_dir: str, session_id: str, **kwargs) -> dict:
    """Synchronous wrapper around sync_session."""
    return asyncio.run(sync_session(data_dir, session_id, **kwargs))


def run_hook(data_dir: str, payload: dict) -> int:
    """Handle one hook invocation.

    Reads JSON from *payload* (already parsed), runs sync, always exits 0,
    prints nothing to stdout, logs errors to ``<data dir>/logs/hooks.log``.
    """
    session_id = payload.get("session_id", "")
    transcript_path = payload.get("transcript_path", "")
    cwd = payload.get("cwd", "")
    event_name = payload.get("hook_event_name", "")

    if not session_id or not transcript_path or not cwd:
        _log_error(data_dir, f"hook: missing fields in payload: {payload}")
        return 0

    try:
        store = CaptureCursorStore(data_dir)
        try:
            cursor = store.get(session_id)
            if cursor is None:
                store.upsert(session_id, transcript_path, cwd, "", 0, "", time.time())
        except Exception as exc:
            _log_error(data_dir, f"hook: cursor init failed: {exc}")
        finally:
            store.close()
    except Exception as exc:
        _log_error(data_dir, f"hook: cursor store init failed: {exc}")

    try:
        result = sync_session_sync(data_dir, session_id)
        if result.get("error"):
            _log_error(data_dir, f"hook {event_name}: sync error: {result['error']}")
    except Exception as exc:
        _log_error(data_dir, f"hook {event_name}: unexpected error: {exc}")

    return 0


def sync_all(data_dir: str, *, timeout_s: float = _DEFAULT_TIMEOUT_S, agent: str = _DEFAULT_AGENT) -> dict:
    """Sync every cursor row in the store. Returns aggregate stats."""
    store = CaptureCursorStore(data_dir)
    try:
        rows = store._conn.execute("SELECT session_id FROM capture_cursors").fetchall()
    finally:
        store.close()

    total_ingested = 0
    total_skipped = 0
    errors = 0
    for (sid,) in rows:
        try:
            result = sync_session_sync(
                data_dir, sid,
                timeout_s=timeout_s,
                agent=agent,
            )
            total_ingested += result.get("ingested", 0)
            total_skipped += result.get("skipped", 0)
            if result.get("error"):
                errors += 1
        except Exception as exc:
            _log_error(data_dir, f"sync_all: session {sid}: {exc}")
            errors += 1

    return {
        "sessions": len(rows),
        "ingested": total_ingested,
        "skipped": total_skipped,
        "errors": errors,
    }
