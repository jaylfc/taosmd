"""Session Catalog — Timeline Directory over Zero-Loss Archives (taOSmd).

A derived index that splits raw archive JSONL files into per-session files,
creates a catalog DB with line pointers, and exposes fast query methods for
temporal lookup, FTS, and context retrieval.

The original archive files are NEVER modified. The catalog and split files are
fully derived — delete and regenerate at any time.

Retrieval path: when the intent classifier detects a temporal/timeline query
("what was I working on Tuesday?"), the catalog is queried first — near-instant
lookup vs expensive vector search.

Session detection: 30-min gap heuristic. If no events for >30 min, a new
session begins. Topics are derived from the first event summary.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from taosmd import prompts

logger = logging.getLogger(__name__)

SESSION_GAP_THRESHOLD = 1800  # 30 minutes

SESSION_CATEGORIES = [
    "coding", "debugging", "research", "planning", "conversation",
    "configuration", "deployment", "testing", "documentation",
    "brainstorming", "review", "maintenance", "other",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    topic TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT 'other',
    archive_file TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    split_file TEXT NOT NULL DEFAULT '',
    turn_count INTEGER NOT NULL DEFAULT 0,
    tier INTEGER NOT NULL DEFAULT 1,
    partial INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sub_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    topic TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    archive_file TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS catalog_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions(date);
CREATE INDEX IF NOT EXISTS idx_sessions_time ON sessions(start_time, end_time);
CREATE INDEX IF NOT EXISTS idx_sessions_topic ON sessions(topic);
CREATE INDEX IF NOT EXISTS idx_sessions_category ON sessions(category);
CREATE INDEX IF NOT EXISTS idx_sub_sessions_parent ON sub_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sub_sessions_time ON sub_sessions(start_time, end_time);

CREATE VIRTUAL TABLE IF NOT EXISTS catalog_fts USING fts5(
    topic,
    description,
    category,
    primary_project,
    primary_topic,
    primary_subtopic,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS card_edges (
    id INTEGER PRIMARY KEY,
    from_card INTEGER NOT NULL,
    to_card INTEGER NOT NULL,
    relation TEXT NOT NULL,
    reason TEXT DEFAULT '',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_card_edges_from ON card_edges(from_card);
"""


def _slugify(text: str, max_len: int = 40) -> str:
    """Convert text to a filename-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    text = text.strip("-")
    return text[:max_len].rstrip("-")


def _format_time(ts: float) -> str:
    """Format a unix timestamp as HH:MM."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M")


def _format_date(ts: float) -> str:
    """Format a unix timestamp as YYYY-MM-DD."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


class SessionCatalog:
    """Session catalog: splitter, index, and query interface over archives."""

    def __init__(
        self,
        db_path: str | Path = "data/session-catalog.db",
        archive_dir: str | Path = "data/archive",
        sessions_dir: str | Path = "data/sessions",
    ):
        self._db_path = str(db_path)
        self._archive_dir = Path(archive_dir)
        self._sessions_dir = Path(sessions_dir)
        self._conn: sqlite3.Connection | None = None

    async def init(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        # Taxonomy columns — safe no-op if already present
        _taxonomy_cols = [
            "ALTER TABLE sessions ADD COLUMN primary_project TEXT DEFAULT ''",
            "ALTER TABLE sessions ADD COLUMN primary_topic TEXT DEFAULT ''",
            "ALTER TABLE sessions ADD COLUMN primary_subtopic TEXT DEFAULT ''",
            "ALTER TABLE sessions ADD COLUMN labels_json TEXT DEFAULT '[]'",
            "ALTER TABLE sessions ADD COLUMN classified_at REAL DEFAULT 0",
        ]
        for _sql in _taxonomy_cols:
            try:
                self._conn.execute(_sql)
            except Exception:
                pass  # column already exists
        # agent_name column — safe no-op if already present
        try:
            self._conn.execute(
                "ALTER TABLE sessions ADD COLUMN agent_name TEXT DEFAULT ''"
            )
        except Exception:
            pass
        try:
            self._conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_sessions_path '
                'ON sessions(primary_project, primary_topic, primary_subtopic)'
            )
        except Exception:
            pass
        self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Archive reading (read-only)
    # ------------------------------------------------------------------

    def _read_archive_file(self, path: Path) -> list[dict]:
        """Read a JSONL or gzipped JSONL archive file, tagging each event with
        its 1-based line number and source file path."""
        events = []
        if path.suffix == ".gz":
            opener = lambda: gzip.open(path, "rt", encoding="utf-8")
        else:
            opener = lambda: open(path, "r", encoding="utf-8")

        with opener() as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    event["_line_num"] = line_num
                    event["_file"] = str(path)
                    events.append(event)
                except json.JSONDecodeError:
                    continue
        return events

    # ------------------------------------------------------------------
    # Splitter
    # ------------------------------------------------------------------

    def split_day(
        self,
        date: str,
        force: bool = False,
        partial: bool = False,
    ) -> dict:
        """Split a day's archive JSONL into per-session files and catalog entries.

        Idempotent: if called again for the same date, old entries are deleted
        and the day is re-split. Use force=True to skip the already-done check.

        Args:
            date: YYYY-MM-DD
            force: always re-split even if entries exist
            partial: mark resulting sessions as partial (day still in progress)

        Returns:
            dict with date, sessions_created, split_files
        """
        if not force:
            existing = self._conn.execute(
                "SELECT COUNT(*) as n FROM sessions WHERE date = ?", (date,)
            ).fetchone()["n"]
            if existing > 0:
                return {"date": date, "sessions_created": existing, "skipped": True}

        # Delete any existing entries for this date (idempotency)
        existing_ids = [
            r["id"] for r in self._conn.execute(
                "SELECT id FROM sessions WHERE date = ?", (date,)
            ).fetchall()
        ]
        if existing_ids:
            placeholders = ",".join("?" * len(existing_ids))
            self._conn.execute(
                f"DELETE FROM sub_sessions WHERE session_id IN ({placeholders})",
                existing_ids,
            )
            self._conn.execute(
                f"DELETE FROM catalog_fts WHERE rowid IN ({placeholders})",
                existing_ids,
            )
            self._conn.execute("DELETE FROM sessions WHERE date = ?", (date,))
            self._conn.commit()

        # Locate archive file
        parts = date.split("-")
        candidates = [
            self._archive_dir / parts[0] / parts[1] / f"{parts[2]}.jsonl",
            self._archive_dir / parts[0] / parts[1] / f"{parts[2]}.jsonl.gz",
        ]
        archive_file = None
        for c in candidates:
            if c.exists():
                archive_file = c
                break

        if not archive_file:
            return {"date": date, "sessions_created": 0, "error": "no archive file"}

        events = self._read_archive_file(archive_file)
        if not events:
            return {"date": date, "sessions_created": 0, "error": "empty archive"}

        # Detect session boundaries using 30-min gap heuristic
        session_groups = self._group_by_gap(events)

        # Write split files and insert catalog entries
        now = time.time()
        split_files = []
        for idx, group in enumerate(session_groups, 1):
            first = group[0]
            last = group[-1]
            start_ts = first.get("timestamp", 0)
            end_ts = last.get("timestamp", start_ts)

            # Derive topic from first event summary
            summaries = [e.get("summary", "") for e in group if e.get("summary")]
            topic = summaries[0][:80] if summaries else "Activity session"

            # Derive agent_name from first event that carries the field
            agent_names = [e.get("agent_name", "") for e in group if e.get("agent_name")]
            session_agent = agent_names[0] if agent_names else ""

            # Build split file path: data/sessions/YYYY/MM/DD/session-NNN-<cat>-<slug>.jsonl
            category = "other"
            slug = _slugify(topic)
            session_num = f"{idx:03d}"
            year, month, day = parts
            split_dir = self._sessions_dir / year / month / day
            split_dir.mkdir(parents=True, exist_ok=True)
            split_filename = f"session-{session_num}-{category}-{slug}.jsonl"
            split_path = split_dir / split_filename

            # Write the split file (strip internal _line_num / _file tags)
            with open(split_path, "w", encoding="utf-8") as sf:
                for e in group:
                    clean = {k: v for k, v in e.items() if not k.startswith("_")}
                    sf.write(json.dumps(clean) + "\n")

            split_files.append(str(split_path))

            # Insert catalog entry
            cursor = self._conn.execute(
                """INSERT INTO sessions
                   (date, start_time, end_time, topic, description, category,
                    archive_file, line_start, line_end, split_file,
                    turn_count, tier, partial, created_at, agent_name)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    date,
                    start_ts,
                    end_ts,
                    topic,
                    f"{len(group)} events",
                    category,
                    str(archive_file),
                    first["_line_num"],
                    last["_line_num"],
                    str(split_path),
                    len(group),
                    1,
                    1 if partial else 0,
                    now,
                    session_agent,
                ),
            )
            session_id = cursor.lastrowid
            self._conn.execute(
                "INSERT INTO catalog_fts (rowid, topic, description, category) VALUES (?, ?, ?, ?)",
                (session_id, topic, f"{len(group)} events", category),
            )

        # Update last split date in meta
        self._conn.execute(
            """INSERT INTO catalog_meta (key, value) VALUES ('last_split_date', ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (date,),
        )
        self._conn.commit()

        return {
            "date": date,
            "sessions_created": len(session_groups),
            "split_files": split_files,
        }

    def _group_by_gap(self, events: list[dict]) -> list[list[dict]]:
        """Split a flat event list into groups separated by >30-min gaps."""
        if not events:
            return []

        groups: list[list[dict]] = []
        current: list[dict] = [events[0]]

        for e in events[1:]:
            ts = e.get("timestamp", 0)
            last_ts = current[-1].get("timestamp", 0)
            if ts - last_ts > SESSION_GAP_THRESHOLD:
                groups.append(current)
                current = [e]
            else:
                current.append(e)

        groups.append(current)
        return groups

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def lookup_date(self, date: str, *, agent_name: str | None = None) -> list[dict]:
        """Get all sessions for a specific date, ordered by start time.

        Args:
            date: YYYY-MM-DD string.
            agent_name: optional agent slug. When set, only sessions whose
                ``agent_name`` column matches are returned. When None (default),
                all sessions for the date are returned.
        """
        if agent_name:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE date = ? AND agent_name = ? ORDER BY start_time",
                (date, agent_name),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE date = ? ORDER BY start_time",
                (date,),
            ).fetchall()
        return [self._format(dict(r)) for r in rows]

    async def lookup_range(self, start_date: str, end_date: str) -> list[dict]:
        """Get sessions within an inclusive date range."""
        rows = self._conn.execute(
            "SELECT * FROM sessions WHERE date >= ? AND date <= ? ORDER BY start_time",
            (start_date, end_date),
        ).fetchall()
        return [self._format(dict(r)) for r in rows]

    async def search_topic(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across session topics and descriptions."""
        try:
            rows = self._conn.execute(
                """SELECT s.* FROM catalog_fts f
                   JOIN sessions s ON s.id = f.rowid
                   WHERE catalog_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (query, limit),
            ).fetchall()
            return [self._format(dict(r)) for r in rows]
        except Exception:
            rows = self._conn.execute(
                """SELECT * FROM sessions
                   WHERE topic LIKE ? OR description LIKE ?
                   ORDER BY start_time DESC LIMIT ?""",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
            return [self._format(dict(r)) for r in rows]

    async def get_session(self, session_id: int) -> dict | None:
        """Get a single session record by id."""
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return self._format(dict(row)) if row else None

    async def get_session_context(
        self,
        session_id: int,
        max_lines: int = 100,
        *,
        agent_name: str | None = None,
    ) -> dict | None:
        """Get session metadata plus the raw event lines from its split file.

        Loads from the split file (fast) rather than seeking through the full
        day archive. Falls back to archive line-range if split file is missing.

        Args:
            session_id: primary key of the session to fetch.
            max_lines:  maximum number of lines to return.
            agent_name: optional agent slug. When set, only lines whose parsed
                JSON ``agent_name`` field matches are included. When None
                (default), all lines are returned unchanged.
        """
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None

        session = self._format(dict(row))
        lines: list[str] = []

        def _accept_line(raw: str) -> bool:
            """Return True if this raw JSONL line passes the agent_name filter."""
            if agent_name is None:
                return True
            try:
                event = json.loads(raw)
                return event.get("agent_name") == agent_name
            except (json.JSONDecodeError, AttributeError):
                return False

        split_path = Path(row["split_file"]) if row["split_file"] else None
        if split_path and split_path.exists():
            with open(split_path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(lines) >= max_lines:
                        break
                    stripped = line.strip()
                    if _accept_line(stripped):
                        lines.append(stripped)
        else:
            # Fallback: read from archive between line_start and line_end
            archive_path = Path(row["archive_file"])
            if archive_path.exists():
                opener = (
                    lambda: gzip.open(archive_path, "rt", encoding="utf-8")
                    if archive_path.suffix == ".gz"
                    else open(archive_path, "r", encoding="utf-8")
                )
                with opener() as f:
                    for i, line in enumerate(f, 1):
                        if i < row["line_start"]:
                            continue
                        if i > row["line_end"]:
                            break
                        if len(lines) >= max_lines:
                            break
                        stripped = line.strip()
                        if _accept_line(stripped):
                            lines.append(stripped)

        session["archive_lines"] = lines
        session["lines_returned"] = len(lines)
        return session

    async def get_sub_sessions(self, session_id: int) -> list[dict]:
        """Get sub-sessions for a parent session."""
        rows = self._conn.execute(
            "SELECT * FROM sub_sessions WHERE session_id = ? ORDER BY start_time",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    async def recent(self, limit: int = 20) -> list[dict]:
        """Get the most recently started sessions."""
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._format(dict(r)) for r in rows]

    async def stats(self) -> dict:
        """Return aggregate statistics about the catalog."""
        total_sessions = self._conn.execute(
            "SELECT COUNT(*) as n FROM sessions"
        ).fetchone()["n"]
        total_sub = self._conn.execute(
            "SELECT COUNT(*) as n FROM sub_sessions"
        ).fetchone()["n"]
        days = self._conn.execute(
            "SELECT COUNT(DISTINCT date) as n FROM sessions"
        ).fetchone()["n"]
        categories = self._conn.execute(
            "SELECT category, COUNT(*) as n FROM sessions GROUP BY category ORDER BY n DESC"
        ).fetchall()
        last_split = self._conn.execute(
            "SELECT value FROM catalog_meta WHERE key = 'last_split_date'"
        ).fetchone()

        return {
            "total_sessions": total_sessions,
            "total_sub_sessions": total_sub,
            "days_cataloged": days,
            "categories": {r["category"]: r["n"] for r in categories},
            "last_split_date": last_split["value"] if last_split else None,
        }

    # ------------------------------------------------------------------
    # LLM Enrichment
    # ------------------------------------------------------------------

    async def enrich_session(
        self,
        session_id: int,
        llm_url: str,
        model: str,
        tier: int = 2,
        *,
        agent_name: str | None = None,
    ) -> dict | None:
        """Enrich a catalog session with LLM-generated topic/description/category.

        Reads the session's split file via get_session_context(), sends the
        content to the Ollama API, then updates the catalog entry and FTS index.

        Falls back to keeping the existing heuristic results if the LLM is
        unreachable or returns an unparseable response.

        Args:
            session_id: primary key of the session to enrich.
            llm_url:    base URL of the Ollama server (e.g. "http://localhost:11434").
            model:      model name to use for generation.
            tier:       tier to set on success (default 2 = LLM-enriched).
            agent_name: optional agent slug to scope enrichment. When set, only
                archive rows belonging to this agent are enriched. When None
                (default), behaves as before — unscoped across all agents.

        Returns:
            Updated session dict, or None if session_id is not found.
        """
        ctx = await self.get_session_context(session_id, agent_name=agent_name)
        if ctx is None:
            return None

        content_lines = ctx.get("archive_lines") or []
        content = "\n".join(content_lines)

        try:
            topic, description, category = await self._llm_enrich(
                content, llm_url, model
            )
            if category not in SESSION_CATEGORIES:
                category = "other"
            await self._update_enrichment(session_id, topic, description, category, tier)
        except Exception as exc:
            logger.warning(
                "LLM enrichment failed for session %s, keeping heuristic: %s",
                session_id,
                exc,
            )

        return await self.get_session(session_id)

    async def _llm_enrich(
        self,
        content: str,
        llm_url: str,
        model: str,
    ) -> tuple[str, str, str]:
        """Send session content to the Ollama API and parse the response.

        Args:
            content:  raw JSONL lines joined as a string.
            llm_url:  base URL of the Ollama server.
            model:    model name for generation.

        Returns:
            (topic, description, category) tuple.

        Raises:
            httpx.HTTPError / Exception on network or parse failure.
        """
        prompt = prompts.session_enrichment_prompt(session_log=content)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{llm_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 150},
                },
            )
            response.raise_for_status()
            text = response.json()["response"]

        # Strip markdown fences if the model wrapped the JSON.
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.DOTALL)
        try:
            data = json.loads(stripped)
            topic = str(data["topic"])
            description = str(data["description"])
            category = str(data["category"]).lower()
            if not topic or not description or not category:
                raise ValueError("empty field in JSON response")
            return topic, description, category
        except (json.JSONDecodeError, KeyError, ValueError):
            # legacy — fallback when JSON parse fails
            return self._parse_enrichment(text)

    def _parse_enrichment(self, text: str) -> tuple[str, str, str]:
        """Parse an LLM response into (topic, description, category).

        legacy — fallback when JSON parse fails (used by _llm_enrich).

        Expected format:
            TOPIC: <topic>
            DESCRIPTION: <description>
            CATEGORY: <category>

        Raises:
            ValueError if any required field is missing.
        """
        topic = description = category = None
        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith("TOPIC:"):
                topic = line[6:].strip()
            elif line.upper().startswith("DESCRIPTION:"):
                description = line[12:].strip()
            elif line.upper().startswith("CATEGORY:"):
                category = line[9:].strip().lower()

        if not topic:
            raise ValueError("LLM response missing TOPIC field")
        if not description:
            raise ValueError("LLM response missing DESCRIPTION field")
        if not category:
            raise ValueError("LLM response missing CATEGORY field")

        return topic, description, category

    async def _update_enrichment(
        self,
        session_id: int,
        topic: str,
        description: str,
        category: str,
        tier: int,
    ) -> None:
        """Update sessions row and FTS index with enriched values.

        Deletes the old FTS entry and inserts a fresh one to keep the index
        consistent with the sessions table.

        Args:
            session_id:  primary key of the session.
            topic:       enriched topic string.
            description: enriched description string.
            category:    enriched category string.
            tier:        tier value to set on the sessions row.
        """
        self._conn.execute(
            """UPDATE sessions
               SET topic = ?, description = ?, category = ?, tier = ?
               WHERE id = ?""",
            (topic, description, category, tier, session_id),
        )
        # Rebuild FTS row: delete old entry, insert fresh one
        self._conn.execute(
            "DELETE FROM catalog_fts WHERE rowid = ?",
            (session_id,),
        )
        self._conn.execute(
            "INSERT INTO catalog_fts (rowid, topic, description, category) VALUES (?, ?, ?, ?)",
            (session_id, topic, description, category),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format(self, session: dict) -> dict:
        """Attach human-readable time strings to a session dict."""
        start = session.get("start_time", 0)
        end = session.get("end_time", 0)
        if start:
            session["start_str"] = _format_time(start)
            session["date_str"] = datetime.fromtimestamp(
                start, tz=timezone.utc
            ).strftime("%A %B %d, %Y")
        if end:
            session["end_str"] = _format_time(end)
        if start and end:
            duration_min = (end - start) / 60
            session["duration_str"] = f"{int(duration_min)}min"
        return session
