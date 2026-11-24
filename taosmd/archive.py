"""Zero-Loss Archive Layer (taOSmd).

Append-only daily archive of every event in the system. Never modified, never
deleted, never summarised. The raw truth, kept indefinitely.

Events include: agent conversations, tool calls, API interactions, decisions,
errors, system state, and optionally user activity (browsing, reading, app usage).

Storage: data/archive/YYYY/MM/DD.jsonl (one JSON line per event, gzipped after day ends).
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import logging
import os
import sqlite3
import time
import asyncio
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import _db, migrations

logger = logging.getLogger(__name__)


def _redact_leaves(value: Any) -> Any:
    """Recursively redact secrets from all string leaves in a structure.

    Returns a NEW structure with the same shape:
    - str -> redact_secrets(str)[0]
    - dict -> {k: _redact_leaves(v) for k, v in value.items()}
    - list/tuple -> [_redact_leaves(v) for v in value]
    - anything else -> returned as is
    """
    from .secret_filter import redact_secrets

    if isinstance(value, str):
        return redact_secrets(value)[0]
    if isinstance(value, dict):
        return {k: _redact_leaves(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_leaves(v) for v in value]
    return value


# Event types
EVENT_CONVERSATION = "conversation"       # Agent chat message (user or agent)
EVENT_TOOL_CALL = "tool_call"             # Agent tool invocation + result
EVENT_API_CALL = "api_call"               # External API request + response
EVENT_DECISION = "decision"               # Decision point with chosen path
EVENT_ERROR = "error"                     # Error with context and resolution
EVENT_SYSTEM = "system"                   # System state change (agent start/stop, etc.)
EVENT_APP_USAGE = "app_usage"             # User opened/closed/interacted with an app
EVENT_CONTENT_VIEW = "content_view"       # User viewed a post, page, thread, video
EVENT_SEARCH = "search"                   # User searched for something
EVENT_INGEST = "ingest"                   # Content ingested into knowledge base
EVENT_MONITOR = "monitor"                 # Monitoring event (poll, change detected)
EVENT_A2A = "a2a"                         # Agent-to-agent message bus message

ALL_EVENT_TYPES = [
    EVENT_CONVERSATION, EVENT_TOOL_CALL, EVENT_API_CALL, EVENT_DECISION,
    EVENT_ERROR, EVENT_SYSTEM, EVENT_APP_USAGE, EVENT_CONTENT_VIEW,
    EVENT_SEARCH, EVENT_INGEST, EVENT_MONITOR, EVENT_A2A,
]

# User activity events are opt-in
USER_ACTIVITY_EVENTS = {EVENT_APP_USAGE, EVENT_CONTENT_VIEW, EVENT_SEARCH}

# Index schema for fast lookups (the JSONL files are the source of truth)
INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS archive_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,
    agent_name TEXT,
    app_id TEXT,
    project TEXT,
    summary TEXT NOT NULL DEFAULT '',
    file_path TEXT NOT NULL,
    line_number INTEGER NOT NULL,
    data_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_archive_ts ON archive_index(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_archive_type ON archive_index(event_type);
CREATE INDEX IF NOT EXISTS idx_archive_agent ON archive_index(agent_name);
CREATE INDEX IF NOT EXISTS idx_archive_app ON archive_index(app_id);

CREATE TABLE IF NOT EXISTS archive_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS archive_fts USING fts5(
    summary,
    content,
    content_rowid='id',
    tokenize='porter unicode61'
);
"""

A2A_ALARM_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS a2a_alarm_state (
    alarm_key TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    last_fire_ts REAL,
    last_cleared_ts REAL,
    PRIMARY KEY (alarm_key, fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_a2a_alarm_state_key_fp ON a2a_alarm_state(alarm_key, fingerprint);
"""

A2A_IMPORT_DEDUP_SCHEMA = """
CREATE TABLE IF NOT EXISTS a2a_import_dedup (
    key TEXT PRIMARY KEY,
    row_id INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_a2a_import_dedup_key ON a2a_import_dedup(key);
"""

INDEX_SCHEMA = INDEX_SCHEMA + A2A_ALARM_STATE_SCHEMA + A2A_IMPORT_DEDUP_SCHEMA


class ArchiveStore:
    """Append-only event archive with daily JSONL files and SQLite index."""

    def __init__(
        self,
        archive_dir: str | Path = "data/archive",
        index_path: str | Path = "data/archive-index.db",
    ):
        self._archive_dir = Path(archive_dir)
        self._index_path = str(index_path)
        self._conn: sqlite3.Connection | None = None
        self._current_file: Any = None
        self._current_date: str = ""
        self._user_tracking_enabled: bool = False
        # In-memory line counts per JSONL path, avoids re-reading the whole
        # daily file on every write. Seeded from disk the first time a path
        # is seen; reset naturally per path across daily rollover.
        self._line_counts: dict[str, int] = {}
        # Write lock: serialises sequence allocation, the raw JSONL write and
        # the index insert so a record's id is fixed before the line lands.
        self._write_lock = asyncio.Lock()
        # Monotonically increasing record sequence. Seeded on init (see
        # _seed_seq_counter); every increment is durably persisted before the
        # raw write so restarts never reuse a value.
        self._seq_counter: int = 0
        # Legacy id map watermark: highest index id for which we have appended
        # entries to the legacy-id-map. Stored in archive_settings to avoid
        # re-scanning raw files on every init.
        self._legacy_map_watermark: int = 0
        self._warned_files: set[str] = set()

    async def init(self) -> None:
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        Path(self._index_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = _db.connect(self._index_path)
        self._conn.row_factory = sqlite3.Row
        _db.run_schema(self._conn, INDEX_SCHEMA)
        self._conn.commit()
        # Schema versioning and upgrades (subsumes the old hand-rolled
        # `project` back-fill). Must complete before the first SELECT below.
        migrations.migrate(self._conn, "archive_index")
        # Load user tracking preference
        row = self._conn.execute(
            "SELECT value FROM archive_settings WHERE key = 'user_tracking_enabled'"
        ).fetchone()
        self._user_tracking_enabled = bool(row and row["value"] == "true")
        # Seed the record sequence counter so restarts never reuse an id.
        self._seed_seq_counter()
        # Build the legacy id map for index rows whose raw line has no seq
        # (written by older code). Idempotent; only rows above the stored
        # watermark are examined, so a reopen costs one indexed query.
        self._load_legacy_map_watermark()
        try:
            self._extend_legacy_map(only_above_watermark=True)
        except OSError:
            # Read-only directory: rebuild() retries and raises if it still
            # cannot write the map.
            logger.warning("could not extend the legacy id map on open", exc_info=True)

    async def close(self) -> None:
        if self._current_file:
            self._current_file.close()
            self._current_file = None
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    @property
    def user_tracking_enabled(self) -> bool:
        return self._user_tracking_enabled

    async def set_user_tracking(self, enabled: bool) -> None:
        """Enable or disable user activity tracking (opt-in)."""
        self._user_tracking_enabled = enabled
        self._conn.execute(
            """INSERT INTO archive_settings (key, value) VALUES ('user_tracking_enabled', ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            ("true" if enabled else "false",),
        )
        self._conn.commit()

    async def get_a2a_inbox_cursor(self, consumer: str) -> int:
        """Return the last-seen message id for ``consumer``'s inbox, or 0."""
        row = self._conn.execute(
            "SELECT value FROM archive_settings WHERE key = ?",
            (f"a2a_inbox_cursor_{consumer}",),
        ).fetchone()
        return int(row["value"]) if row else 0

    async def set_a2a_inbox_cursor(self, consumer: str, cursor_id: int) -> None:
        """Persist ``consumer``'s inbox cursor to ``cursor_id``."""
        self._conn.execute(
            """INSERT INTO archive_settings (key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (f"a2a_inbox_cursor_{consumer}", str(cursor_id)),
        )
        self._conn.commit()

    async def record_alarm_dedup(
        self, alarm_key: str, fingerprint: str, now: float, cleared_ts: float | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO a2a_alarm_state (alarm_key, fingerprint, last_fire_ts, last_cleared_ts)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(alarm_key, fingerprint) DO UPDATE SET
                   last_fire_ts = excluded.last_fire_ts,
                   last_cleared_ts = excluded.last_cleared_ts""",
            (alarm_key, fingerprint, now, cleared_ts),
        )
        self._conn.commit()

    async def get_alarm_dedup(self, alarm_key: str, fingerprint: str) -> dict | None:
        row = self._conn.execute(
            "SELECT last_fire_ts, last_cleared_ts FROM a2a_alarm_state WHERE alarm_key = ? AND fingerprint = ?",
            (alarm_key, fingerprint),
        ).fetchone()
        if row is None:
            return None
        return {"last_fire_ts": row[0], "last_cleared_ts": row[1]}

    async def clear_alarm_key(self, alarm_key: str, now: float) -> None:
        self._conn.execute(
            "UPDATE a2a_alarm_state SET last_cleared_ts = ? WHERE alarm_key = ?",
            (now, alarm_key),
        )
        self._conn.commit()

    async def record_import_dedup(self, key: str, row_id: int) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO a2a_import_dedup (key, row_id) VALUES (?, ?)",
            (key, row_id),
        )
        self._conn.commit()

    async def get_import_dedup(self, key: str) -> dict | None:
        row = self._conn.execute(
            "SELECT key, row_id FROM a2a_import_dedup WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return {"key": row[0], "row_id": row[1]}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _get_daily_path(self, ts: float) -> Path:
        """Get the JSONL file path for a given timestamp."""
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return self._archive_dir / f"{dt.year}" / f"{dt.month:02d}" / f"{dt.day:02d}.jsonl"

    def _ensure_file(self, ts: float) -> tuple[Any, str]:
        """Ensure the daily JSONL file is open for writing.

        Files are one-per-day (YYYY/MM/DD.jsonl), so the open handle must
        roll over at day boundaries: keying the check on anything coarser
        (e.g. the year/month parent directory) would keep appending to
        yesterday's file after midnight UTC while the index points at
        today's, corrupting file_path/line_number provenance.
        """
        path = self._get_daily_path(ts)
        date_str = str(path)
        if self._current_date != date_str:
            if self._current_file:
                self._current_file.close()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._current_file = open(path, "a", encoding="utf-8")
            self._current_date = date_str
        return self._current_file, str(path)

    async def record(
        self,
        event_type: str,
        data: dict,
        agent_name: str | None = None,
        app_id: str | None = None,
        summary: str = "",
        project: str | None = None,
    ) -> int:
        """Record an event to the archive. Returns the index row ID.

        The id is a record sequence number allocated before the raw write and
        stored in the JSONL line (``seq``), so an index rebuild can restore it.
        Everything after the allocation runs in one SQLite transaction (opened
        with BEGIN IMMEDIATE, which also serialises writers in other processes):

            allocate seq -> write the raw line and flush -> INSERT the index
            row with id = seq -> update the counter -> COMMIT

        On ANY exception the transaction is rolled back and the error is
        re-raised, so the connection is never left inside a transaction and the
        next call works. A raw line may remain without an index row after a
        failure; that is acceptable (raw files are never rewritten) and a later
        ``rebuild()`` indexes it.
        """
        # Skip user activity events if tracking is disabled
        if event_type in USER_ACTIVITY_EVENTS and not self._user_tracking_enabled:
            return -1

        # Deep-copy caller's data so we never mutate it. A bare deepcopy can
        # raise on values that cannot be copied (e.g. threading.Lock() -> "cannot
        # pickle '_thread.lock' object"); on ANY Exception fall back to a
        # JSON-plain snapshot. The caller's dict must still never change, and
        # the JSONL line is byte-identical to master's for the same input
        # (json.dumps(event, default=str)).
        try:
            data = copy.deepcopy(data)
        except Exception:
            data = json.loads(json.dumps(data, default=str))

        # Redact secrets before storage (for JSONL write - same keys as before)
        from .secret_filter import redact_secrets
        summary, _ = redact_secrets(summary)
        for key in ("content", "text", "msg", "query", "body"):
            if key in data and isinstance(data[key], str):
                data[key], _ = redact_secrets(data[key])

        ts = time.time()

        async with self._write_lock:
            if self._conn is None:
                raise RuntimeError("ArchiveStore is not initialised")
            if self._conn.in_transaction:
                # Master's record() committed whatever was pending; keep that
                # rather than failing BEGIN with "within a transaction".
                self._conn.commit()
            self._conn.execute("BEGIN IMMEDIATE")
            seq = 0
            raw_written = False
            try:
                seq = self._allocate_seq()

                event = {
                    "seq": seq,
                    "timestamp": ts,
                    "event_type": event_type,
                    "agent_name": agent_name,
                    "app_id": app_id,
                    "project": project,
                    "summary": summary,
                    "data": data,
                }

                # Write to JSONL
                f, file_path = self._ensure_file(ts)
                line = json.dumps(event, default=str)
                # Append a per-entry SHA-256 content hash so bit-rot and truncated
                # writes are detectable later with verify_entry() / verify_day().
                # The checksum covers the serialised event line (excluding the
                # newline), stored as a separate ``sha256`` key so readers that
                # don't know about checksums can ignore it transparently.
                event["sha256"] = hashlib.sha256(line.encode()).hexdigest()
                line = json.dumps(event, default=str)
                f.write(line + "\n")
                # Once write() returns the line may reach the file even if the
                # flush fails: never hand this seq out again, here or elsewhere.
                raw_written = True
                self._seq_counter = max(self._seq_counter, seq)
                f.flush()

                # Count lines for line_number. Track in memory rather than re-reading
                # the whole daily JSONL on every write: seed from disk the first time
                # we see a path (handles daily rollover -> fresh path), then increment.
                if file_path not in self._line_counts:
                    try:
                        with open(file_path, "r", encoding="utf-8") as existing:
                            self._line_counts[file_path] = sum(1 for _ in existing)
                    except FileNotFoundError:
                        self._line_counts[file_path] = 0
                else:
                    self._line_counts[file_path] += 1
                line_count = self._line_counts[file_path]

                # Fully redacted copy for index and FTS (all string leaves).
                # Build the derived copy from a JSON-plain snapshot so every leaf that
                # becomes a string (including str() of a non-JSON leaf) is scanned by
                # _redact_leaves. A bare _redact_leaves(data) left non-str, non-container
                # leaves unchanged, and the index INSERT serialised them with
                # json.dumps(derived, default=str), so str(obj) landed in
                # archive_index.data_json UNSCANNED.
                derived = _redact_leaves(json.loads(json.dumps(data, default=str)))

                # Index for fast lookup with the assigned id, never lastrowid.
                self._conn.execute(
                    """INSERT INTO archive_index
                       (id, timestamp, event_type, agent_name, app_id, project, summary, file_path, line_number, data_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (seq, ts, event_type, agent_name, app_id, project, summary, file_path, line_count, json.dumps(derived, default=str)),
                )

                # Index in FTS for full-text search
                # Build searchable content from data fields (use fully redacted derived)
                content_parts = [summary]
                for key in ("content", "text", "msg", "query", "tool", "result", "error"):
                    val = derived.get(key)
                    if val and isinstance(val, str):
                        content_parts.append(val)
                content_text = " ".join(content_parts)
                if content_text.strip():
                    self._conn.execute(
                        "INSERT INTO archive_fts (rowid, summary, content) VALUES (?, ?, ?)",
                        (seq, summary, content_text),
                    )

                # Counter and legacy-map watermark move in the same transaction
                # as the row, so they can never disagree with the index.
                self._raise_setting_to("archive_seq_counter", seq)
                self._raise_setting_to(self.LEGACY_MAP_WATERMARK_KEY, seq)

                self._conn.commit()
            except BaseException:
                try:
                    self._conn.rollback()
                except sqlite3.Error:
                    pass
                if raw_written:
                    self._burn_seq(seq)
                raise
            if seq > self._legacy_map_watermark:
                self._legacy_map_watermark = seq
        return seq

    # ------------------------------------------------------------------
    # Record sequence: stable ids across index rebuilds
    # ------------------------------------------------------------------

    def _read_int_setting(self, key: str) -> int:
        row = self._conn.execute(
            "SELECT value FROM archive_settings WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return 0
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return 0

    def _raise_setting_to(self, key: str, value: int) -> None:
        """Set an integer setting to max(current, value). Never lowers it."""
        self._conn.execute(
            """INSERT INTO archive_settings (key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value = CAST(
                   MAX(CAST(archive_settings.value AS INTEGER),
                       CAST(excluded.value AS INTEGER)) AS TEXT)""",
            (key, str(int(value))),
        )

    def _allocate_seq(self) -> int:
        """Next record id: above the persisted counter, MAX(index id) and every
        seq this instance has put on disk. Call inside BEGIN IMMEDIATE.

        MAX(id) matters when older code wrote rows to the same directory (it
        never touches the counter), or when the counter was lost.
        """
        current = self._read_int_setting("archive_seq_counter")
        max_id = self._conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM archive_index"
        ).fetchone()[0]
        return max(current, int(max_id or 0), self._seq_counter) + 1

    def _burn_seq(self, seq: int) -> None:
        """Best effort: after a failure that left a raw line carrying ``seq``,
        persist the counter so no writer (this process or another) reuses it.

        If this also fails (for example the database is unwritable) the orphan
        line stays; rebuild() copes with a repeated seq by giving the later
        line a fresh id.
        """
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            self._raise_setting_to("archive_seq_counter", seq)
            self._conn.commit()
        except Exception:
            try:
                self._conn.rollback()
            except sqlite3.Error:
                pass

    def _seed_seq_counter(self) -> None:
        """Start the in-memory sequence floor above every id already in use.

        Cost on open: O(1) when the counter setting or the index has rows (two
        indexed reads plus a read of the tail of the newest raw file, to catch
        a line whose index row was lost). Only a directory with neither a
        counter nor index rows (a fresh dir, or an index that was deleted)
        scans every raw file, once; the result is persisted so the next open
        does not scan again.
        """
        setting = self._read_int_setting("archive_seq_counter")
        max_id = self._conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM archive_index"
        ).fetchone()[0]
        max_id = int(max_id or 0)
        if setting == 0 and max_id == 0:
            counter = self._max_seq_in_files()
            if counter > 0:
                self._raise_setting_to("archive_seq_counter", counter)
                self._conn.commit()
        else:
            counter = max(setting, max_id, self._max_seq_in_tail())
        # Ids recorded for legacy lines must never be handed out again, even
        # when the index (and the counter stored in it) is gone.
        self._seq_counter = max(counter, self._legacy_map_max())

    def _legacy_map_max(self) -> int:
        """Highest id the legacy-id-map knows, without reading all of it.

        Every extension appends a ``{"max_id": N}`` marker holding the running
        maximum, so the last marker in the file's tail answers it. A map with
        no marker (written before markers existed) is read in full once.
        """
        path = self._legacy_map_path()
        try:
            size = path.stat().st_size
        except OSError:
            return 0
        best = 0
        try:
            with open(path, "rb") as fh:
                fh.seek(max(0, size - 65536))
                tail = fh.read().split(b"\n")
            for raw in reversed(tail):
                try:
                    entry = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(entry, dict) and "max_id" in entry:
                    return max(best, int(entry["max_id"]))
        except (OSError, ValueError, TypeError):
            pass
        try:
            return max(self._load_legacy_map().values(), default=0)
        except OSError:
            return best

    def _max_seq_in_tail(self, max_bytes: int = 4 << 20) -> int:
        """Highest ``seq`` in the last ``max_bytes`` of the newest raw file."""
        newest = None
        for path in self._archive_dir.rglob("*.jsonl*"):
            if not path.name.endswith((".jsonl", ".jsonl.gz")):
                continue
            if newest is None or str(path) > str(newest):
                newest = path
        if newest is None:
            return 0
        try:
            if newest.suffix == ".gz":
                lines = [ln.encode("utf-8") for ln in self._read_archive_lines(newest)]
            else:
                with open(newest, "rb") as fh:
                    fh.seek(0, os.SEEK_END)
                    size = fh.tell()
                    fh.seek(max(0, size - max_bytes))
                    # The first piece may be cut mid-line; a fragment is never
                    # a JSON object, so the parse below skips it.
                    lines = fh.read().split(b"\n")
        except OSError:
            return 0
        best = 0
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                continue
            seq = rec.get("seq") if isinstance(rec, dict) else None
            if isinstance(seq, int) and not isinstance(seq, bool) and seq > best:
                best = seq
        return best

    def _max_seq_in_files(self) -> int:
        """Highest ``seq`` stored in any raw .jsonl or .jsonl.gz file."""
        max_seq = 0
        for path in self._iter_archive_files():
            try:
                lines = self._read_archive_lines(path)
            except OSError:
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                seq = record.get("seq") if isinstance(record, dict) else None
                if isinstance(seq, int) and not isinstance(seq, bool) and seq > max_seq:
                    max_seq = seq
        return max_seq

    def _iter_archive_files(self) -> list[Path]:
        """Every raw archive file (.jsonl and .jsonl.gz), oldest-first by path."""
        files = (
            list(self._archive_dir.rglob("*.jsonl"))
            + list(self._archive_dir.rglob("*.jsonl.gz"))
        )
        # A compress that was interrupted (or whose .gz failed verification)
        # leaves both DD.jsonl and DD.jsonl.gz; the plain file is the source.
        files = [
            p for p in files
            if not (p.name.endswith(".gz") and p.with_suffix("").exists())
        ]
        return sorted(files, key=lambda path: str(path))

    @staticmethod
    def _index_name(path: Path | str) -> str:
        """The name the index uses for a raw file: always the .jsonl form.

        compress_old_files() turns DD.jsonl into DD.jsonl.gz without touching
        index rows, so legacy-map identities and rebuilt rows use the .jsonl
        name regardless of which form is on disk.
        """
        name = str(path)
        return name[:-3] if name.endswith(".gz") else name

    def _read_archive_lines(self, path: Path) -> list[str]:
        """Read all lines of a raw archive file, transparently decompressing .gz."""
        if path.suffix == ".gz":
            # A truncated or corrupt .gz must not stop the store from opening
            # or a rebuild from indexing the lines before the damage.
            lines: list[str] = []
            try:
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        lines.append(line)
            except (OSError, EOFError, zlib.error, UnicodeDecodeError) as exc:
                if str(path) not in self._warned_files:
                    self._warned_files.add(str(path))
                    logger.warning(
                        "damaged archive file %s (%s); using the %d lines before the damage",
                        path, exc.__class__.__name__, len(lines),
                    )
            return lines
        with open(path, "r", encoding="utf-8") as fh:
            return fh.readlines()

    def _archive_file_for(self, file_path: str) -> Path | None:
        """Resolve an index file_path to the raw file on disk (.jsonl or .jsonl.gz)."""
        path = Path(file_path)
        if path.exists():
            return path
        gz_path = path.parent / (path.name + ".gz")
        if gz_path.exists():
            return gz_path
        return None

    # ------------------------------------------------------------------
    # Legacy id map: automatic, idempotent, append-only
    # ------------------------------------------------------------------

    LEGACY_ID_MAP = "legacy-id-map"
    LEGACY_MAP_WATERMARK_KEY = "legacy_id_map_watermark"

    def _legacy_map_path(self) -> Path:
        return self._archive_dir / self.LEGACY_ID_MAP

    def _load_legacy_map(self) -> dict[tuple[str, int, str], int]:
        """Load the legacy-id-map sidecar, identity -> id.

        Later entries override earlier ones; the file is append-only and never
        truncated.
        """
        mapping: dict[tuple[str, int, str], int] = {}
        path = self._legacy_map_path()
        if not path.exists():
            return mapping
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    mapping[
                        (
                            self._index_name(entry["file_path"]),
                            int(entry["line_number"]),
                            entry.get("sha256") or "",
                        )
                    ] = int(entry["id"])
                except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                    continue
        return mapping

    def _load_legacy_map_watermark(self) -> None:
        self._legacy_map_watermark = self._read_int_setting(self.LEGACY_MAP_WATERMARK_KEY)

    def _extend_legacy_map(self, only_above_watermark: bool) -> int:
        """Append map entries for index rows whose raw line has no ``seq``.

        With ``only_above_watermark`` (the init path) only rows above the
        stored watermark are looked at, so a reopen costs one indexed query
        when nothing is new. Otherwise every index row is checked (rebuild and
        freeze), skipping identities already in the map, so repeats add nothing.
        The sidecar is append-only, fsynced, and raw files are only read.
        Returns the number of entries appended.
        """
        if only_above_watermark:
            rows = self._conn.execute(
                "SELECT id, file_path, line_number FROM archive_index WHERE id > ? ORDER BY file_path, line_number",
                (self._legacy_map_watermark,),
            ).fetchall()
            known: dict[tuple[str, int, str], int] = {}
        else:
            rows = self._conn.execute(
                "SELECT id, file_path, line_number FROM archive_index ORDER BY file_path, line_number"
            ).fetchall()
            known = self._load_legacy_map()
        if not rows:
            return 0

        entries: list[str] = []
        cached_path: str | None = None
        cached_lines: list[str] = []
        for row in rows:
            if row["file_path"] != cached_path:
                cached_path = row["file_path"]
                archive_path = self._archive_file_for(cached_path)
                try:
                    cached_lines = self._read_archive_lines(archive_path) if archive_path else []
                except OSError:
                    cached_lines = []
            sha256 = ""
            has_seq = False
            ln = row["line_number"]
            if 1 <= ln <= len(cached_lines):
                try:
                    record = json.loads(cached_lines[ln - 1].strip())
                    sha256 = record.get("sha256") or ""
                    seq = record.get("seq")
                    has_seq = isinstance(seq, int) and not isinstance(seq, bool)
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass
            if has_seq:
                continue
            name = self._index_name(row["file_path"])
            if known.get((name, ln, sha256)) == row["id"]:
                continue
            entries.append(
                json.dumps(
                    {"file_path": name, "line_number": ln, "sha256": sha256, "id": row["id"]},
                    default=str,
                )
            )
        appended = len(entries)
        if entries:
            path = self._legacy_map_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            batch_max = max(r["id"] for r in rows)
            entries.append(json.dumps({"max_id": max(batch_max, self._legacy_map_max())}))
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("\n".join(entries) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        top = max(row["id"] for row in rows)
        if top > self._legacy_map_watermark:
            try:
                self._raise_setting_to(self.LEGACY_MAP_WATERMARK_KEY, top)
                self._conn.commit()
            except BaseException:
                try:
                    self._conn.rollback()
                except sqlite3.Error:
                    pass
                raise
            self._legacy_map_watermark = top
        return appended

    async def freeze_legacy_id_map(self) -> int:
        """Append map entries for every index row that is still unmapped.

        Not needed any more: the map is built automatically on open and again
        at the start of rebuild(). Kept so existing callers keep working.
        Returns entries appended.
        """
        if self._conn is None:
            raise RuntimeError("ArchiveStore is not initialised")
        async with self._write_lock:
            return self._extend_legacy_map(only_above_watermark=False)

    # ------------------------------------------------------------------
    # Index rebuild from the raw files
    # ------------------------------------------------------------------

    async def rebuild(self) -> dict[str, int]:
        """Rebuild ``archive_index`` and ``archive_fts`` from the raw files.

        Row ids come from the ``seq`` stored in each JSONL line, so a rebuild
        never renumbers records. Legacy lines without ``seq`` take the id
        recorded for them in the append-only ``legacy-id-map`` sidecar
        (identity: file path + line number + the line's SHA-256), which the
        store builds by itself (see init). Legacy lines the map does not know,
        and a line whose seq or mapped id is already taken by another line,
        get fresh ids ABOVE every id in use. Raw .jsonl and .jsonl.gz files are
        only ever read, never rewritten.

        It takes the same write lock as record(), and the delete and reinsert
        run in one transaction, so a failure part way leaves the old index
        exactly as it was.

        Returns ``{"indexed": N, "legacy": M}``; ``legacy`` counts lines that
        carry no seq.
        """
        if self._conn is None:
            raise RuntimeError("ArchiveStore is not initialised")
        async with self._write_lock:
            if self._conn.in_transaction:
                self._conn.commit()
            # Make sure every row the old index knows about is in the map
            # before the index is thrown away (no-op when already mapped).
            self._extend_legacy_map(only_above_watermark=False)
            legacy_map = self._load_legacy_map()
            files = self._iter_archive_files()

            def parsed_lines(path: Path):
                for line_number, line in enumerate(self._read_archive_lines(path), start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if isinstance(record, dict):
                        yield line_number, record

            def stored_seq(record: dict) -> int | None:
                seq = record.get("seq")
                if isinstance(seq, int) and not isinstance(seq, bool) and seq > 0:
                    return seq
                return None

            self._conn.execute("BEGIN IMMEDIATE")
            try:
                # Pass 1: the highest id in use anywhere, and which seqs repeat.
                top = max(
                    self._seq_counter,
                    self._read_int_setting("archive_seq_counter"),
                    int(self._conn.execute("SELECT COALESCE(MAX(id), 0) FROM archive_index").fetchone()[0] or 0),
                    max(legacy_map.values(), default=0),
                )
                seen_seqs: set[int] = set()
                contested: set[int] = set(legacy_map.values())  # seqs a mapped legacy line also holds
                repeated: set[int] = set()
                for path in files:
                    for _, record in parsed_lines(path):
                        seq = stored_seq(record)
                        if seq is None:
                            continue
                        top = max(top, seq)
                        if seq in seen_seqs:
                            repeated.add(seq)
                        seen_seqs.add(seq)
                # A seq that two lines claim (an orphan line from a failed write
                # whose seq was reused, or a legacy line mapped to the same id):
                # the line the current index points at keeps it.
                contested = (contested & seen_seqs) | repeated
                index_pos: dict[int, tuple[str, int]] = {}
                for seq in contested:
                    old = self._conn.execute(
                        "SELECT file_path, line_number FROM archive_index WHERE id = ?", (seq,)
                    ).fetchone()
                    if old is not None:
                        index_pos[seq] = (self._index_name(old["file_path"]), old["line_number"])

                self._conn.execute("DELETE FROM archive_index")
                self._conn.execute("DELETE FROM archive_fts")

                # Pass 2: stored seqs first, then mapped legacy ids, then fresh.
                # Legacy lines are deferred so a mapped id can never be taken
                # by an unmapped line that happens to sort earlier.
                assigned: set[int] = set()
                fresh = top
                indexed = legacy = 0
                deferred: list[tuple[str, int, dict]] = []

                def insert(row_id: int, record: dict, file_path: str, line_number: int) -> None:
                    summary = record.get("summary") or ""
                    derived = _redact_leaves(
                        json.loads(json.dumps(record.get("data") or {}, default=str))
                    )
                    if not isinstance(derived, dict):
                        derived = {}
                    self._conn.execute(
                        """INSERT INTO archive_index
                           (id, timestamp, event_type, agent_name, app_id, project, summary, file_path, line_number, data_json)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            row_id,
                            record.get("timestamp") or 0.0,
                            record.get("event_type") or "",
                            record.get("agent_name"),
                            record.get("app_id"),
                            record.get("project"),
                            summary,
                            file_path,
                            line_number,
                            json.dumps(derived, default=str),
                        ),
                    )
                    content_parts = [summary]
                    for key in ("content", "text", "msg", "query", "tool", "result", "error"):
                        val = derived.get(key)
                        if val and isinstance(val, str):
                            content_parts.append(val)
                    content_text = " ".join(content_parts)
                    if content_text.strip():
                        self._conn.execute(
                            "INSERT INTO archive_fts (rowid, summary, content) VALUES (?, ?, ?)",
                            (row_id, summary, content_text),
                        )

                for path in files:
                    file_path = self._index_name(path)
                    for line_number, record in parsed_lines(path):
                        seq = stored_seq(record)
                        if seq is None:
                            legacy += 1
                            deferred.append((file_path, line_number, record))
                            continue
                        owner = index_pos.get(seq)
                        if seq in assigned or (
                            owner is not None and owner != (file_path, line_number)
                        ):
                            # Repeated seq and not the line the index kept.
                            deferred.append((file_path, line_number, record))
                            continue
                        assigned.add(seq)
                        insert(seq, record, file_path, line_number)
                        indexed += 1

                for file_path, line_number, record in deferred:
                    identity = (file_path, line_number, record.get("sha256") or "")
                    row_id = legacy_map.get(identity)
                    if row_id is None or row_id in assigned:
                        fresh += 1
                        row_id = fresh
                    assigned.add(row_id)
                    insert(row_id, record, file_path, line_number)
                    indexed += 1

                self._raise_setting_to("archive_seq_counter", max(top, fresh))
                self._conn.commit()
            except BaseException:
                try:
                    self._conn.rollback()
                except sqlite3.Error:
                    pass
                raise
            self._seq_counter = max(self._seq_counter, top, fresh)
            # Persist the ids just given to unmapped legacy lines so a later
            # rebuild gives them the same ones.
            self._extend_legacy_map(only_above_watermark=False)
            return {"indexed": indexed, "legacy": legacy}

    # ------------------------------------------------------------------
    # Full-text search
    # ------------------------------------------------------------------

    async def search_fts(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across all archived content using FTS5.

        This is the key method for LongMemEval-style recall: it searches
        the raw verbatim text, not just extracted triples.
        """
        try:
            rows = self._conn.execute(
                """SELECT a.*, highlight(archive_fts, 1, '<b>', '</b>') as highlight
                   FROM archive_fts f
                   JOIN archive_index a ON a.id = f.rowid
                   WHERE archive_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (query, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            # Fallback to LIKE search if FTS query syntax fails
            return await self.query(search=query, limit=limit)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    async def query(
        self,
        event_type: str | None = None,
        agent_name: str | None = None,
        app_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """Query the archive index with filters."""
        conditions = []
        params: list = []

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if agent_name:
            conditions.append("agent_name = ?")
            params.append(agent_name)
        if app_id:
            conditions.append("app_id = ?")
            params.append(app_id)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until:
            conditions.append("timestamp <= ?")
            params.append(until)
        if search:
            conditions.append("(summary LIKE ? OR data_json LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])

        rows = self._conn.execute(
            f"SELECT * FROM archive_index {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    async def get_event(self, event_id: int) -> dict | None:
        """Get a single event by index ID, including full data from JSONL."""
        row = self._conn.execute(
            "SELECT * FROM archive_index WHERE id = ?", (event_id,)
        ).fetchone()
        if not row:
            return None
        result = dict(row)
        # Parse stored data_json
        try:
            result["data"] = json.loads(result.get("data_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            result["data"] = {}
        return result

    async def update_event_data_json(self, event_id: int, data: dict) -> None:
        """Update the data_json for an existing index row.

        Used by one-shot migrations that backfill fields (e.g. message kind)
        onto historical archive events. The JSONL source files are never
        modified; only the derived index is updated.
        """
        self._conn.execute(
            "UPDATE archive_index SET data_json = ? WHERE id = ?",
            (json.dumps(data, default=str), event_id),
        )
        self._conn.commit()

    async def count(
        self,
        event_type: str | None = None,
        since: float | None = None,
    ) -> int:
        """Count events with optional filters."""
        conditions = []
        params: list = []
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        row = self._conn.execute(
            f"SELECT COUNT(*) as n FROM archive_index {where}", params
        ).fetchone()
        return row["n"]

    async def daily_summary(self, date: str | None = None) -> dict:
        """Get event counts by type for a given day (YYYY-MM-DD) or today."""
        if date:
            parts = date.split("-")
            dt = datetime(int(parts[0]), int(parts[1]), int(parts[2]), tzinfo=timezone.utc)
        else:
            dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start = dt.timestamp()
        end = start + 86400

        rows = self._conn.execute(
            """SELECT event_type, COUNT(*) as n
               FROM archive_index
               WHERE timestamp >= ? AND timestamp < ?
               GROUP BY event_type""",
            (start, end),
        ).fetchall()
        return {
            "date": dt.strftime("%Y-%m-%d"),
            "events": {r["event_type"]: r["n"] for r in rows},
            "total": sum(r["n"] for r in rows),
        }

    async def daily_counts(self, days: int = 30, agent: str | None = None) -> list[dict]:
        """Per-day archived-event counts for the last ``days`` days, oldest first.

        One grouped query (used by the dashboard growth chart) rather than one
        ``daily_summary`` call per day. ``agent`` scopes the counts to one agent.
        """
        where = "timestamp >= ?"
        params: list = [time.time() - days * 86400]
        if agent is not None:
            where += " AND agent_name = ?"
            params.append(agent)
        rows = self._conn.execute(
            f"""SELECT date(timestamp, 'unixepoch') AS d, COUNT(*) AS n
               FROM archive_index WHERE {where}
               GROUP BY d ORDER BY d""",
            tuple(params),
        ).fetchall()
        return [{"date": r["d"], "count": r["n"]} for r in rows]

    async def recent(self, limit: int = 10, agent: str | None = None) -> list[dict]:
        """The most recent archived events as ``{kind, label, ts}``, newest first."""
        where = ""
        params: list = []
        if agent is not None:
            where = "WHERE agent_name = ?"
            params.append(agent)
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT event_type, timestamp, agent_name FROM archive_index "
            f"{where} ORDER BY timestamp DESC LIMIT ?",
            tuple(params),
        ).fetchall()
        return [
            {
                "kind": r["event_type"],
                "label": f"{r['event_type']} by {r['agent_name'] or 'unknown'}",
                "ts": r["timestamp"],
            }
            for r in rows
        ]

    async def distinct_agents(self) -> int:
        """Number of distinct agents that have archived memories."""
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT agent_name) AS n FROM archive_index "
            "WHERE agent_name IS NOT NULL"
        ).fetchone()
        return int(row["n"])

    async def scoped_total(self, agent: str | None = None) -> int:
        """Total archived events, optionally scoped to one ``agent``."""
        if agent is None:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM archive_index").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM archive_index WHERE agent_name = ?", (agent,)
            ).fetchone()
        return int(row["n"])

    async def top_by(self, column: str, limit: int = 5, agent: str | None = None) -> list[dict]:
        """Top ``{name, count}`` groups by ``column`` (agent_name or project)."""
        if column not in ("agent_name", "project"):
            raise ValueError(f"top_by column must be agent_name or project, got {column!r}")
        where = f"{column} IS NOT NULL"
        params: list = []
        if agent is not None:
            where += " AND agent_name = ?"
            params.append(agent)
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT {column} AS name, COUNT(*) AS n FROM archive_index "
            f"WHERE {where} GROUP BY {column} ORDER BY n DESC LIMIT ?",
            tuple(params),
        ).fetchall()
        return [{"name": r["name"], "count": r["n"]} for r in rows]

    async def list_memories(self, agent: str | None = None, limit: int = 50) -> list[dict]:
        """Recent archived memories for browse: ``{text, agent, kind, ts}`` newest first.

        ``agent`` scopes to one agent (or the reserved ``user`` namespace); ``None``
        returns every namespace. The text comes from the event summary, falling
        back to the recorded payload's text/content/body field.
        """
        where = ""
        params: list = []
        if agent is not None:
            where = "WHERE agent_name = ?"
            params.append(agent)
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT event_type, timestamp, agent_name, summary, data_json "
            f"FROM archive_index {where} ORDER BY timestamp DESC LIMIT ?",
            tuple(params),
        ).fetchall()
        out = []
        for r in rows:
            text = r["summary"] or ""
            if not text:
                try:
                    data = json.loads(r["data_json"] or "{}")
                    text = data.get("text") or data.get("content") or data.get("body") or ""
                except (ValueError, TypeError):
                    text = ""
            out.append({
                "text": text,
                "agent": r["agent_name"],
                "kind": r["event_type"],
                "ts": r["timestamp"],
            })
        return out

    # ------------------------------------------------------------------
    # Integrity verification
    # ------------------------------------------------------------------

    @staticmethod
    def verify_entry(line: str) -> bool:
        """Return True if the JSONL line passes its embedded SHA-256 check.

        Lines without a ``sha256`` key (written before checksums were added)
        are treated as valid; the check is additive and backwards-compatible.
        A missing or blank line is considered invalid.
        """
        line = line.strip()
        if not line:
            return False
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            return False
        stored = record.get("sha256")
        if stored is None:
            # Legacy entry, no checksum, pass through.
            return True
        # Recompute over the line as it was written, i.e. without the sha256
        # key.  Strip sha256 before reserialising to reconstruct the original
        # line that was hashed.
        original = dict(record)
        del original["sha256"]
        original_line = json.dumps(original, default=str)
        expected = hashlib.sha256(original_line.encode()).hexdigest()
        return stored == expected

    async def verify_day(self, date: str) -> dict:
        """Verify checksums for every entry in a day's JSONL file.

        Returns ``{"date": date, "total": N, "ok": M, "bad": K, "legacy": L}``
        where ``legacy`` counts entries without a checksum (pre-checksum rows).
        Does not raise on individual failures; the caller decides how to
        handle them.
        """
        parts = date.split("-")
        path = self._archive_dir / parts[0] / parts[1] / f"{parts[2]}.jsonl"
        gz_path = path.with_suffix(".jsonl.gz")

        lines: list[str] = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        elif gz_path.exists():
            with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
                lines = fh.readlines()

        total = ok = bad = legacy = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                bad += 1
                continue
            if record.get("sha256") is None:
                legacy += 1
                ok += 1
                continue
            if self.verify_entry(line):
                ok += 1
            else:
                bad += 1

        return {"date": date, "total": total, "ok": ok, "bad": bad, "legacy": legacy}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def stats(self) -> dict:
        """Overall archive statistics."""
        total = self._conn.execute("SELECT COUNT(*) as n FROM archive_index").fetchone()["n"]
        oldest = self._conn.execute("SELECT MIN(timestamp) as ts FROM archive_index").fetchone()["ts"]
        newest = self._conn.execute("SELECT MAX(timestamp) as ts FROM archive_index").fetchone()["ts"]

        # Count JSONL files
        file_count = sum(1 for _ in self._archive_dir.rglob("*.jsonl"))
        gz_count = sum(1 for _ in self._archive_dir.rglob("*.jsonl.gz"))

        # Disk usage
        total_bytes = sum(f.stat().st_size for f in self._archive_dir.rglob("*") if f.is_file())

        return {
            "total_events": total,
            "oldest_event": oldest,
            "newest_event": newest,
            "active_files": file_count,
            "compressed_files": gz_count,
            "disk_usage_mb": round(total_bytes / (1024 * 1024), 2),
            "user_tracking_enabled": self._user_tracking_enabled,
        }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def compress_old_files(self, days_old: int = 1) -> int:
        """Gzip JSONL files older than N days. Returns count compressed.

        The raw archive is eternal and lossless, so this is the only code path
        that deletes raw archive data. Every write is verified before the
        source is touched: the gz is written to a temp path, the gzip context
        closes (writing the trailer), then the finished file is fsynced,
        re-opened and decompressed for comparison, then os.replace'd onto the
        final path with a directory fsync (skipped silently where refused).
        A mismatch or any exception leaves the .jsonl in place and logs a
        warning.
        """
        cutoff = time.time() - (days_old * 86400)
        compressed = 0
        for jsonl in self._archive_dir.rglob("*.jsonl"):
            if jsonl.stat().st_mtime < cutoff:
                gz_path = jsonl.with_suffix(".jsonl.gz")
                if gz_path.exists():
                    continue
                tmp_path = jsonl.with_name(jsonl.name + ".gz.tmp")
                try:
                    with open(jsonl, "rb") as f_in:
                        source_bytes = f_in.read()
                    with gzip.open(tmp_path, "wb") as f_out:
                        f_out.write(source_bytes)
                    fd = os.open(tmp_path, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    with gzip.open(tmp_path, "rb") as f_verify:
                        if f_verify.read() != source_bytes:
                            raise ValueError(
                                f"gzip round-trip mismatch for {jsonl}: "
                                "decompressed bytes differ from source"
                            )
                    os.replace(tmp_path, gz_path)
                    dir_fd = None
                    try:
                        dir_fd = os.open(str(jsonl.parent), os.O_RDONLY)
                        os.fsync(dir_fd)
                    except OSError:
                        pass
                    finally:
                        if dir_fd is not None:
                            os.close(dir_fd)
                    jsonl.unlink()
                    compressed += 1
                except Exception:
                    if tmp_path.exists():
                        try:
                            tmp_path.unlink()
                        except OSError:
                            pass
                    logger.warning(
                        "compress_old_files: kept %s, .gz verification failed",
                        jsonl,
                    )
                    continue
        return compressed

    async def export_day(self, date: str) -> list[dict]:
        """Export all events for a specific day as a list of dicts."""
        parts = date.split("-")
        path = self._archive_dir / parts[0] / parts[1] / f"{parts[2]}.jsonl"
        gz_path = path.with_suffix(".jsonl.gz")

        events = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
        elif gz_path.exists():
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
        return events