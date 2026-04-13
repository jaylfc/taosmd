"""Session Catalog — Timeline Directory over Zero-Loss Archives (taOSmd).

A read-only derived index that an LLM agent processes from raw archive
transcripts/logs, creating structured timelines with:
  - Session boundaries (topic changes detected by LLM)
  - Sub-sessions within sessions
  - Descriptions and categories
  - Line-number pointers back to the source archive files

The original archive files are NEVER modified. The catalog is a separate
database that can be deleted and regenerated at any time.

Retrieval path: when the intent classifier detects a temporal/timeline query
("what was I working on Tuesday?"), the catalog is queried first — near-instant
lookup vs expensive vector search.

Processing: an LLM (Qwen3.5 on GPU, Qwen3-4B on NPU) reads archive JSONL
files and produces structured session records.
"""

from __future__ import annotations

import gzip
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    topic TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT 'general',
    agent_name TEXT,
    archive_file TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    turn_count INTEGER NOT NULL DEFAULT 0,
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
CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent_name);
CREATE INDEX IF NOT EXISTS idx_sub_sessions_parent ON sub_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sub_sessions_time ON sub_sessions(start_time, end_time);

CREATE VIRTUAL TABLE IF NOT EXISTS catalog_fts USING fts5(
    topic,
    description,
    category,
    tokenize='porter unicode61'
);
"""

# Categories for session classification
SESSION_CATEGORIES = [
    "coding", "debugging", "research", "planning", "conversation",
    "configuration", "deployment", "testing", "documentation",
    "brainstorming", "review", "maintenance", "other",
]


class SessionCatalog:
    """Read-only derived index over zero-loss archive files."""

    def __init__(
        self,
        db_path: str | Path = "data/session-catalog.db",
        archive_dir: str | Path = "data/archive",
    ):
        self._db_path = str(db_path)
        self._archive_dir = Path(archive_dir)
        self._conn: sqlite3.Connection | None = None

    async def init(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Archive reading (read-only)
    # ------------------------------------------------------------------

    def _read_archive_file(self, path: Path) -> list[dict]:
        """Read a JSONL or gzipped JSONL archive file."""
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

    def _list_archive_files(self, since_date: str | None = None) -> list[Path]:
        """List all archive JSONL files, optionally filtered by date."""
        files = []
        for ext in ("*.jsonl", "*.jsonl.gz"):
            files.extend(self._archive_dir.rglob(ext))
        files.sort()

        if since_date:
            # Filter to files from this date onwards
            filtered = []
            for f in files:
                # Extract date from path: archive/2026/03/12.jsonl
                try:
                    parts = f.parts
                    year = parts[-3]
                    month = parts[-2]
                    day = f.stem.split(".")[0]
                    file_date = f"{year}-{month}-{day}"
                    if file_date >= since_date:
                        filtered.append(f)
                except (IndexError, ValueError):
                    filtered.append(f)
            return filtered
        return files

    # ------------------------------------------------------------------
    # LLM-powered session detection
    # ------------------------------------------------------------------

    async def _detect_sessions_llm(
        self,
        events: list[dict],
        archive_file: str,
        llm_url: str = "http://localhost:11434",
        model: str = "qwen3.5:4b",
    ) -> list[dict]:
        """Use LLM to detect session boundaries and topics from events."""
        if not events:
            return []

        # Build a compact summary of events for the LLM
        summary_parts = []
        for e in events[:200]:  # Cap to avoid huge prompts
            ts = e.get("timestamp", 0)
            etype = e.get("event_type", "unknown")
            summary = e.get("summary", "")
            agent = e.get("agent_name", "")
            data = e.get("data", {})

            time_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M") if ts else "??:??"
            content = summary or data.get("content", data.get("text", ""))[:100]
            line = f"[{time_str}] {etype}"
            if agent:
                line += f" ({agent})"
            if content:
                line += f": {content}"
            summary_parts.append(line)

        events_text = "\n".join(summary_parts)[:4000]
        first_ts = events[0].get("timestamp", 0)
        date_str = datetime.fromtimestamp(first_ts, tz=timezone.utc).strftime("%A %B %d, %Y") if first_ts else "unknown"

        prompt = f"""Analyze this day's activity log and identify distinct work sessions. A session is a continuous block of related activity on the same topic.

Date: {date_str}
Events:
{events_text}

For each session, respond in this EXACT format (one per line):
SESSION|HH:MM-HH:MM|topic|description|category

Categories: {', '.join(SESSION_CATEGORIES)}

Example:
SESSION|09:15-11:30|taOSmd embedding fixes|Fixed ONNX embedding pipeline, tested on benchmark|coding
SESSION|11:45-12:20|lunch break research|Read about MemPalace memory system on GitHub|research
SESSION|13:00-16:45|benchmark matrix|Ran LongMemEval-S benchmark across 4 configs|testing

Only output SESSION lines, nothing else."""

        try:
            import httpx
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{llm_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.2, "num_predict": 500},
                    },
                )
                if resp.status_code != 200:
                    return self._detect_sessions_heuristic(events, archive_file)

                text = resp.json().get("response", "")
                return self._parse_session_response(text, events, archive_file)

        except Exception as e:
            logger.debug("LLM session detection failed: %s", e)
            return self._detect_sessions_heuristic(events, archive_file)

    def _parse_session_response(
        self, text: str, events: list[dict], archive_file: str,
    ) -> list[dict]:
        """Parse LLM response into session records."""
        sessions = []
        first_ts = events[0].get("timestamp", 0) if events else 0
        date = datetime.fromtimestamp(first_ts, tz=timezone.utc)
        date_str = date.strftime("%Y-%m-%d")

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line.startswith("SESSION|"):
                continue
            parts = line.split("|")
            if len(parts) < 5:
                continue

            time_range = parts[1].strip()
            topic = parts[2].strip()
            description = parts[3].strip()
            category = parts[4].strip().lower()
            if category not in SESSION_CATEGORIES:
                category = "other"

            # Parse time range
            try:
                start_str, end_str = time_range.split("-")
                sh, sm = map(int, start_str.strip().split(":"))
                eh, em = map(int, end_str.strip().split(":"))
                start_time = date.replace(hour=sh, minute=sm, second=0).timestamp()
                end_time = date.replace(hour=eh, minute=em, second=0).timestamp()
            except (ValueError, IndexError):
                continue

            # Find line numbers in archive that correspond to this time range
            line_start = None
            line_end = None
            for e in events:
                ts = e.get("timestamp", 0)
                if ts >= start_time and line_start is None:
                    line_start = e["_line_num"]
                if ts <= end_time:
                    line_end = e["_line_num"]

            if line_start is None:
                line_start = events[0]["_line_num"]
            if line_end is None:
                line_end = events[-1]["_line_num"]

            sessions.append({
                "date": date_str,
                "start_time": start_time,
                "end_time": end_time,
                "topic": topic,
                "description": description,
                "category": category,
                "archive_file": archive_file,
                "line_start": line_start,
                "line_end": line_end,
                "turn_count": sum(
                    1 for e in events
                    if start_time <= e.get("timestamp", 0) <= end_time
                ),
            })

        return sessions if sessions else self._detect_sessions_heuristic(events, archive_file)

    def _detect_sessions_heuristic(
        self, events: list[dict], archive_file: str,
    ) -> list[dict]:
        """Fallback: detect sessions by time gaps (>30 min gap = new session)."""
        if not events:
            return []

        GAP_THRESHOLD = 1800  # 30 minutes
        sessions = []
        current_start = 0
        current_events = []

        for e in events:
            ts = e.get("timestamp", 0)
            if not current_events:
                current_start = ts
                current_events = [e]
                continue

            last_ts = current_events[-1].get("timestamp", 0)
            if ts - last_ts > GAP_THRESHOLD:
                # Close current session
                sessions.append(self._build_heuristic_session(
                    current_events, archive_file
                ))
                current_start = ts
                current_events = [e]
            else:
                current_events.append(e)

        # Close final session
        if current_events:
            sessions.append(self._build_heuristic_session(
                current_events, archive_file
            ))

        return sessions

    def _build_heuristic_session(
        self, events: list[dict], archive_file: str,
    ) -> dict:
        """Build a session record from a group of events (heuristic mode)."""
        first = events[0]
        last = events[-1]
        first_ts = first.get("timestamp", 0)
        date = datetime.fromtimestamp(first_ts, tz=timezone.utc)

        # Derive topic from most common event type and summaries
        summaries = [e.get("summary", "") for e in events if e.get("summary")]
        agents = [e.get("agent_name", "") for e in events if e.get("agent_name")]
        topic = summaries[0][:80] if summaries else "Activity session"
        agent = agents[0] if agents else None

        return {
            "date": date.strftime("%Y-%m-%d"),
            "start_time": first_ts,
            "end_time": last.get("timestamp", first_ts),
            "topic": topic,
            "description": f"{len(events)} events over {len(summaries)} logged activities",
            "category": "other",
            "agent_name": agent,
            "archive_file": archive_file,
            "line_start": first["_line_num"],
            "line_end": last["_line_num"],
            "turn_count": len(events),
        }

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index_day(
        self,
        date: str,
        llm_url: str = "http://localhost:11434",
        model: str = "qwen3.5:4b",
        force: bool = False,
    ) -> dict:
        """Index a single day's archive into the session catalog.

        Args:
            date: YYYY-MM-DD format
            llm_url: Ollama-compatible LLM endpoint
            model: Model for session detection (qwen3.5:4b on GPU, qwen3:4b on NPU)
            force: Re-index even if already cataloged
        """
        # Check if already indexed
        if not force:
            existing = self._conn.execute(
                "SELECT COUNT(*) as n FROM sessions WHERE date = ?", (date,)
            ).fetchone()["n"]
            if existing > 0:
                return {"date": date, "sessions": existing, "skipped": True}

        # Find archive file for this date
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
            return {"date": date, "sessions": 0, "error": "no archive file"}

        # Read events
        events = self._read_archive_file(archive_file)
        if not events:
            return {"date": date, "sessions": 0, "error": "empty archive"}

        # Detect sessions
        detected = await self._detect_sessions_llm(
            events, str(archive_file), llm_url, model
        )

        # Store sessions
        now = time.time()
        stored = 0
        for s in detected:
            cursor = self._conn.execute(
                """INSERT INTO sessions
                   (date, start_time, end_time, topic, description, category,
                    agent_name, archive_file, line_start, line_end, turn_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (s["date"], s["start_time"], s["end_time"], s["topic"],
                 s["description"], s["category"], s.get("agent_name"),
                 s["archive_file"], s["line_start"], s["line_end"],
                 s["turn_count"], now),
            )
            # FTS index
            self._conn.execute(
                "INSERT INTO catalog_fts (rowid, topic, description, category) VALUES (?, ?, ?, ?)",
                (cursor.lastrowid, s["topic"], s["description"], s["category"]),
            )
            stored += 1

        self._conn.commit()

        # Update last indexed date
        self._conn.execute(
            """INSERT INTO catalog_meta (key, value) VALUES ('last_indexed_date', ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (date,),
        )
        self._conn.commit()

        return {"date": date, "sessions": stored}

    async def index_all(
        self,
        since_date: str | None = None,
        llm_url: str = "http://localhost:11434",
        model: str = "qwen3.5:4b",
        force: bool = False,
    ) -> dict:
        """Index all archive files into the session catalog."""
        files = self._list_archive_files(since_date)
        total_sessions = 0
        days_indexed = 0

        for f in files:
            try:
                parts = f.parts
                year = parts[-3]
                month = parts[-2]
                day = f.stem.split(".")[0]
                date = f"{year}-{month}-{day}"
            except (IndexError, ValueError):
                continue

            result = await self.index_day(date, llm_url, model, force)
            if not result.get("skipped") and not result.get("error"):
                total_sessions += result["sessions"]
                days_indexed += 1
                logger.info("Indexed %s: %d sessions", date, result["sessions"])

        return {"days_indexed": days_indexed, "total_sessions": total_sessions}

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def lookup_date(self, date: str) -> list[dict]:
        """Get all sessions for a specific date. The primary timeline query."""
        rows = self._conn.execute(
            "SELECT * FROM sessions WHERE date = ? ORDER BY start_time",
            (date,),
        ).fetchall()
        return [self._format_session(dict(r)) for r in rows]

    async def lookup_range(self, start_date: str, end_date: str) -> list[dict]:
        """Get sessions within a date range."""
        rows = self._conn.execute(
            "SELECT * FROM sessions WHERE date >= ? AND date <= ? ORDER BY start_time",
            (start_date, end_date),
        ).fetchall()
        return [self._format_session(dict(r)) for r in rows]

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
            return [self._format_session(dict(r)) for r in rows]
        except Exception:
            rows = self._conn.execute(
                """SELECT * FROM sessions
                   WHERE topic LIKE ? OR description LIKE ?
                   ORDER BY start_time DESC LIMIT ?""",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
            return [self._format_session(dict(r)) for r in rows]

    async def lookup_category(self, category: str, limit: int = 50) -> list[dict]:
        """Get sessions by category."""
        rows = self._conn.execute(
            "SELECT * FROM sessions WHERE category = ? ORDER BY start_time DESC LIMIT ?",
            (category, limit),
        ).fetchall()
        return [self._format_session(dict(r)) for r in rows]

    async def recent(self, limit: int = 20) -> list[dict]:
        """Get most recent sessions."""
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._format_session(dict(r)) for r in rows]

    async def get_sub_sessions(self, session_id: int) -> list[dict]:
        """Get sub-sessions for a parent session."""
        rows = self._conn.execute(
            "SELECT * FROM sub_sessions WHERE session_id = ? ORDER BY start_time",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Context retrieval (for agents)
    # ------------------------------------------------------------------

    async def get_session_context(
        self,
        session_id: int,
        max_lines: int = 100,
    ) -> dict | None:
        """Get session metadata + raw archive lines for agent context.

        Returns session info plus the actual archive content between
        line_start and line_end, enabling agents to read the raw transcript.
        """
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None

        session = self._format_session(dict(row))
        archive_path = Path(row["archive_file"])

        # Read the specific lines from the archive
        lines = []
        if archive_path.exists():
            if archive_path.suffix == ".gz":
                opener = lambda: gzip.open(archive_path, "rt", encoding="utf-8")
            else:
                opener = lambda: open(archive_path, "r", encoding="utf-8")

            with opener() as f:
                for i, line in enumerate(f, 1):
                    if i < row["line_start"]:
                        continue
                    if i > row["line_end"]:
                        break
                    if len(lines) >= max_lines:
                        break
                    lines.append(line.strip())

        session["archive_lines"] = lines
        session["lines_returned"] = len(lines)
        return session

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_session(self, session: dict) -> dict:
        """Format a session record for output with human-readable times."""
        start = session.get("start_time", 0)
        end = session.get("end_time", 0)
        if start:
            dt = datetime.fromtimestamp(start, tz=timezone.utc)
            session["start_str"] = dt.strftime("%H:%M")
            session["date_str"] = dt.strftime("%A %B %d, %Y")
        if end:
            session["end_str"] = datetime.fromtimestamp(end, tz=timezone.utc).strftime("%H:%M")
        if start and end:
            duration_min = (end - start) / 60
            session["duration_str"] = f"{int(duration_min)}min"
        return session

    # ------------------------------------------------------------------
    # Stats & maintenance
    # ------------------------------------------------------------------

    async def stats(self) -> dict:
        total_sessions = self._conn.execute("SELECT COUNT(*) as n FROM sessions").fetchone()["n"]
        total_sub = self._conn.execute("SELECT COUNT(*) as n FROM sub_sessions").fetchone()["n"]
        days = self._conn.execute("SELECT COUNT(DISTINCT date) as n FROM sessions").fetchone()["n"]
        categories = self._conn.execute(
            "SELECT category, COUNT(*) as n FROM sessions GROUP BY category ORDER BY n DESC"
        ).fetchall()
        last_indexed = self._conn.execute(
            "SELECT value FROM catalog_meta WHERE key = 'last_indexed_date'"
        ).fetchone()

        return {
            "total_sessions": total_sessions,
            "total_sub_sessions": total_sub,
            "days_cataloged": days,
            "categories": {r["category"]: r["n"] for r in categories},
            "last_indexed_date": last_indexed["value"] if last_indexed else None,
        }

    async def rebuild(
        self,
        llm_url: str = "http://localhost:11434",
        model: str = "qwen3.5:4b",
    ) -> dict:
        """Drop and rebuild the entire catalog from archive files."""
        self._conn.execute("DELETE FROM sub_sessions")
        self._conn.execute("DELETE FROM sessions")
        self._conn.execute("DELETE FROM catalog_fts")
        self._conn.execute("DELETE FROM catalog_meta")
        self._conn.commit()
        return await self.index_all(llm_url=llm_url, model=model, force=True)
