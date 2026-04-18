"""Session Crystallization (taOSmd).

Compresses completed conversation sessions into compact "crystal" digests:
narrative summary + key outcomes + lessons learned. Crystals are searchable
and much cheaper to retrieve than raw session logs.

Triggered on session end or manually. Lessons extracted from crystals
feed back into the knowledge graph as reusable insights.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .knowledge_graph import TemporalKnowledgeGraph

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS crystals (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    agent_name TEXT,
    narrative TEXT NOT NULL,
    outcomes TEXT NOT NULL DEFAULT '[]',
    lessons TEXT NOT NULL DEFAULT '[]',
    files_affected TEXT NOT NULL DEFAULT '[]',
    turn_count INTEGER NOT NULL DEFAULT 0,
    duration_seconds REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    catalog_session_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_crystals_session ON crystals(session_id);
CREATE INDEX IF NOT EXISTS idx_crystals_agent ON crystals(agent_name);
CREATE INDEX IF NOT EXISTS idx_crystals_created ON crystals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_crystals_catalog ON crystals(catalog_session_id);

CREATE VIRTUAL TABLE IF NOT EXISTS crystals_fts USING fts5(
    narrative,
    outcomes,
    lessons,
    content='crystals',
    content_rowid='rowid',
    tokenize='porter unicode61'
);
"""


def _crystal_id(session_id: str, agent_name: str | None) -> str:
    raw = f"{session_id}:{agent_name or 'system'}:{time.time()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class CrystalStore:
    """SQLite-backed store for session crystals."""

    def __init__(self, db_path: str | Path = "data/crystals.db"):
        self._db_path = str(db_path)
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

    async def crystallize(
        self,
        session_id: str,
        turns: list[dict],
        agent_name: str | None = None,
        llm_url: str = "http://localhost:11434",
        model: str = "qwen3:4b",
        kg: TemporalKnowledgeGraph | None = None,
    ) -> dict:
        """Crystallize a session into a compact digest.

        Args:
            session_id: Unique session identifier.
            turns: List of {role, content} dicts from the conversation.
            agent_name: Agent that ran this session.
            llm_url: Ollama-compatible LLM endpoint.
            model: Model name for summarisation.
            kg: Optional KG to store extracted lessons as triples.

        Returns:
            Crystal dict with narrative, outcomes, lessons, files_affected.
        """
        if not turns:
            return {"error": "no turns to crystallize"}

        from .agents import is_task_enabled  # noqa: PLC0415

        # Gate: named agents with crystallise disabled return early.
        # Anonymous callers (agent_name is None or "") always run.
        if agent_name and not is_task_enabled(agent_name, "crystallise"):
            logger.debug(
                "CrystalStore.crystallize: crystallise disabled for agent=%r, skipping",
                agent_name,
            )
            return {"skipped": True, "reason": "crystallise task disabled"}

        # Build conversation text (truncate to ~4000 chars for LLM context)
        conv_parts = []
        for turn in turns:
            role = turn.get("role", "unknown")
            content = str(turn.get("content", ""))[:500]
            conv_parts.append(f"{role}: {content}")
        conversation = "\n".join(conv_parts)[:4000]

        start_time = turns[0].get("timestamp", time.time())
        end_time = turns[-1].get("timestamp", time.time())

        # LLM crystallization
        narrative, outcomes, lessons, files = await self._llm_crystallize(
            conversation, llm_url, model
        )

        # Store crystal
        crystal_id = _crystal_id(session_id, agent_name)
        now = time.time()
        self._conn.execute(
            """INSERT OR REPLACE INTO crystals
               (id, session_id, agent_name, narrative, outcomes, lessons,
                files_affected, turn_count, duration_seconds, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (crystal_id, session_id, agent_name, narrative,
             json.dumps(outcomes), json.dumps(lessons), json.dumps(files),
             len(turns), end_time - start_time, now),
        )

        # FTS index
        self._conn.execute(
            "INSERT INTO crystals_fts (rowid, narrative, outcomes, lessons) VALUES (last_insert_rowid(), ?, ?, ?)",
            (narrative, json.dumps(outcomes), json.dumps(lessons)),
        )
        self._conn.commit()

        # Feed lessons back into KG
        if kg and lessons:
            for lesson in lessons:
                try:
                    await kg.add_triple(
                        subject=agent_name or "system",
                        predicate="learned",
                        obj=lesson[:200],
                        source=f"crystal:{crystal_id}",
                        subject_type="agent",
                        object_type="lesson",
                    )
                except Exception as e:
                    logger.debug("Failed to store lesson in KG: %s", e)

        return {
            "id": crystal_id,
            "session_id": session_id,
            "agent_name": agent_name,
            "narrative": narrative,
            "outcomes": outcomes,
            "lessons": lessons,
            "files_affected": files,
            "turn_count": len(turns),
            "duration_seconds": round(end_time - start_time, 1),
        }

    async def _llm_crystallize(
        self,
        conversation: str,
        llm_url: str,
        model: str,
    ) -> tuple[str, list[str], list[str], list[str]]:
        """Use LLM to extract crystal components from conversation."""
        prompt = f"""Summarize this conversation into a crystal digest. Be concise.

CONVERSATION:
{conversation}

Respond in this exact format:
NARRATIVE: <2-3 sentence summary of what happened>
OUTCOMES:
- <key outcome 1>
- <key outcome 2>
LESSONS:
- <lesson learned 1>
- <lesson learned 2>
FILES:
- <file path affected 1>
- <file path affected 2>

If no lessons or files, write "none" for that section."""

        try:
            import httpx
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{llm_url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False,
                          "options": {"temperature": 0.2, "num_predict": 400}},
                )
                if resp.status_code != 200:
                    return self._fallback_crystallize(conversation)

                text = resp.json().get("response", "")
                return self._parse_crystal_response(text, conversation)

        except Exception as e:
            logger.debug("LLM crystallize failed: %s", e)
            return self._fallback_crystallize(conversation)

    def _parse_crystal_response(
        self, text: str, conversation: str,
    ) -> tuple[str, list[str], list[str], list[str]]:
        """Parse the LLM crystal response."""
        narrative = ""
        outcomes: list[str] = []
        lessons: list[str] = []
        files: list[str] = []

        current_section = None
        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("NARRATIVE:"):
                narrative = line.split(":", 1)[1].strip()
                current_section = "narrative"
            elif line.upper().startswith("OUTCOMES:"):
                current_section = "outcomes"
            elif line.upper().startswith("LESSONS:"):
                current_section = "lessons"
            elif line.upper().startswith("FILES:"):
                current_section = "files"
            elif line.startswith("-") and current_section:
                val = line.lstrip("- ").strip()
                if val.lower() == "none":
                    continue
                if current_section == "outcomes":
                    outcomes.append(val)
                elif current_section == "lessons":
                    lessons.append(val)
                elif current_section == "files":
                    files.append(val)
            elif current_section == "narrative" and line:
                narrative += " " + line

        if not narrative:
            narrative = conversation[:200] + "..."

        return narrative, outcomes, lessons, files

    def _fallback_crystallize(
        self, conversation: str,
    ) -> tuple[str, list[str], list[str], list[str]]:
        """Fallback when LLM is unavailable — extract what we can from text."""
        lines = conversation.split("\n")
        narrative = lines[0][:200] + "..." if lines else "Session recorded"
        return narrative, [], [], []

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        """Full-text search across crystals."""
        try:
            rows = self._conn.execute(
                """SELECT c.* FROM crystals_fts f
                   JOIN crystals c ON c.rowid = f.rowid
                   WHERE crystals_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (query, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            rows = self._conn.execute(
                "SELECT * FROM crystals WHERE narrative LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
            return [dict(r) for r in rows]

    async def get_session(self, session_id: str) -> dict | None:
        """Get crystal for a specific session."""
        row = self._conn.execute(
            "SELECT * FROM crystals WHERE session_id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    async def recent(self, limit: int = 20, agent_name: str | None = None) -> list[dict]:
        """Get recent crystals."""
        if agent_name:
            rows = self._conn.execute(
                "SELECT * FROM crystals WHERE agent_name = ? ORDER BY created_at DESC LIMIT ?",
                (agent_name, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM crystals ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    async def stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) as n FROM crystals").fetchone()["n"]
        with_lessons = self._conn.execute(
            "SELECT COUNT(*) as n FROM crystals WHERE lessons != '[]'"
        ).fetchone()["n"]
        return {"total_crystals": total, "with_lessons": with_lessons}
