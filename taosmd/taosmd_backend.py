"""taOSmd Backend Implementation.

Concrete MemoryBackend implementation that wires all taOSmd stores together
and exposes them through the unified backend interface used by the taOS
Memory app.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from . import __version__
from .backend import MemoryBackend

SETTINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

AGENT_CONFIG_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_memory_config (
    agent_name TEXT PRIMARY KEY,
    config_json TEXT NOT NULL DEFAULT '{}'
);
"""

# Default settings values
_DEFAULTS: dict[str, Any] = {
    "default_strategy": "thorough",
    "processing_schedule": "0 2 * * *",
    "enricher_model": "qwen3:4b",
    "crystallize_enabled": True,
    "secret_filter_mode": "redact",
    "retention_hot_threshold": 0.8,
    "retention_warm_threshold": 0.5,
    "retention_cold_threshold": 0.2,
    "archive_compress_days": 7,
}


class TaOSmdBackend(MemoryBackend):
    """Full taOSmd backend with all stores."""

    name = "taosmd"
    version = __version__
    capabilities = [
        "kg",
        "vector",
        "archive",
        "catalog",
        "crystals",
        "pipeline",
        "retention",
        "leases",
        "mesh_sync",
    ]

    def __init__(
        self,
        kg=None,
        vector_memory=None,
        archive=None,
        catalog=None,
        crystals=None,
        insights=None,
        settings_db_path: str | Path = "data/memory-settings.db",
    ) -> None:
        self._kg = kg
        self._vector = vector_memory
        self._archive = archive
        self._catalog = catalog
        self._crystals = crystals
        self._insights = insights
        self._settings_db_path = str(settings_db_path)
        self._conn: sqlite3.Connection | None = None

    async def init(self) -> None:
        """Initialise settings database."""
        Path(self._settings_db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._settings_db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SETTINGS_SCHEMA + AGENT_CONFIG_SCHEMA)
        self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict:
        """Return aggregate stats from all memory stores."""
        result: dict[str, Any] = {}

        if self._kg is not None:
            try:
                result["kg"] = await self._kg.stats()
            except Exception:
                result["kg"] = {}
        else:
            result["kg"] = {}

        if self._vector is not None:
            try:
                result["vector"] = await self._vector.stats()
            except Exception:
                result["vector"] = {}
        else:
            result["vector"] = {}

        if self._archive is not None:
            try:
                result["archive"] = await self._archive.stats()
            except Exception:
                result["archive"] = {}
        else:
            result["archive"] = {}

        if self._catalog is not None:
            try:
                result["catalog"] = await self._catalog.stats()
            except Exception:
                result["catalog"] = {}
        else:
            result["catalog"] = {}

        if self._crystals is not None:
            try:
                result["crystals"] = await self._crystals.stats()
            except Exception:
                result["crystals"] = {}
        else:
            result["crystals"] = {}

        return result

    # ------------------------------------------------------------------
    # Settings schema
    # ------------------------------------------------------------------

    async def get_settings_schema(self) -> dict:
        """Return JSON Schema for taOSmd settings."""
        return {
            "type": "object",
            "properties": {
                "default_strategy": {
                    "type": "string",
                    "enum": ["thorough", "fast", "minimal", "custom"],
                    "description": "Default memory processing strategy for agents.",
                },
                "processing_schedule": {
                    "type": "string",
                    "description": "Cron expression for scheduled memory processing.",
                },
                "enricher_model": {
                    "type": "string",
                    "description": "Model name used for memory enrichment.",
                },
                "crystallize_enabled": {
                    "type": "boolean",
                    "description": "Whether session crystallization is enabled.",
                },
                "secret_filter_mode": {
                    "type": "string",
                    "enum": ["redact", "reject", "warn"],
                    "description": "How to handle detected secrets in memory.",
                },
                "retention_hot_threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Retention score threshold for hot tier.",
                },
                "retention_warm_threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Retention score threshold for warm tier.",
                },
                "retention_cold_threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Retention score threshold for cold tier.",
                },
                "archive_compress_days": {
                    "type": "integer",
                    "description": "Number of days before archive files are compressed.",
                },
            },
        }

    # ------------------------------------------------------------------
    # Settings read/write
    # ------------------------------------------------------------------

    async def get_settings(self) -> dict:
        """Return current settings, falling back to defaults."""
        self._ensure_conn()
        rows = self._conn.execute("SELECT key, value FROM memory_settings").fetchall()
        stored = {r["key"]: json.loads(r["value"]) for r in rows}
        result = dict(_DEFAULTS)
        result.update(stored)
        return result

    async def update_settings(self, settings: dict) -> dict:
        """Persist updated settings and return the merged result."""
        self._ensure_conn()
        for key, value in settings.items():
            self._conn.execute(
                """INSERT INTO memory_settings (key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (key, json.dumps(value)),
            )
        self._conn.commit()
        return await self.get_settings()

    # ------------------------------------------------------------------
    # Agent config read/write
    # ------------------------------------------------------------------

    async def get_agent_config(self, agent_name: str) -> dict:
        """Return memory config for a specific agent."""
        self._ensure_conn()
        row = self._conn.execute(
            "SELECT config_json FROM agent_memory_config WHERE agent_name = ?",
            (agent_name,),
        ).fetchone()
        if row:
            try:
                return json.loads(row["config_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        return {"strategy": "thorough", "layers": []}

    async def update_agent_config(self, agent_name: str, config: dict) -> dict:
        """Update an agent's memory config and return the new config."""
        self._ensure_conn()
        self._conn.execute(
            """INSERT INTO agent_memory_config (agent_name, config_json) VALUES (?, ?)
               ON CONFLICT(agent_name) DO UPDATE SET config_json = excluded.config_json""",
            (agent_name, json.dumps(config)),
        )
        self._conn.commit()
        return await self.get_agent_config(agent_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_conn(self) -> None:
        """Ensure the settings DB connection is open (auto-init if needed)."""
        if self._conn is None:
            Path(self._settings_db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._settings_db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(SETTINGS_SCHEMA + AGENT_CONFIG_SCHEMA)
            self._conn.commit()
