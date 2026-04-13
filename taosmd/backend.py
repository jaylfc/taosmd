"""Memory Backend Interface (taOSmd).

Abstract base class for memory backends. taOSmd implements this natively.
Third-party memory systems (Mem0, MemPalace, etc.) can subclass it to
integrate with the taOS Memory app.

The app queries capabilities to decide which UI tabs to show, and
settings_schema to dynamically render config forms.
"""

from __future__ import annotations
from typing import Any


class MemoryBackend:
    """Base class for memory backends."""

    name: str = "unknown"
    version: str = "0.0.0"
    capabilities: list[str] = []
    # Possible capabilities: "kg", "vector", "archive", "catalog",
    # "crystals", "pipeline", "retention", "leases", "mesh_sync"

    async def get_stats(self) -> dict:
        """Return aggregate stats from all memory stores."""
        raise NotImplementedError

    async def get_settings_schema(self) -> dict:
        """Return JSON Schema describing this backend's configurable settings.

        The Memory app uses this to render settings forms dynamically.
        Example: {"type": "object", "properties": {"strategy": {"type": "string", "enum": ["thorough", "fast", "minimal"]}}}
        """
        return {"type": "object", "properties": {}}

    async def get_settings(self) -> dict:
        """Return current settings values."""
        return {}

    async def update_settings(self, settings: dict) -> dict:
        """Update settings. Returns the new settings."""
        raise NotImplementedError

    async def get_agent_config(self, agent_name: str) -> dict:
        """Return memory config for a specific agent."""
        return {"strategy": "thorough", "layers": []}

    async def update_agent_config(self, agent_name: str, config: dict) -> dict:
        """Update an agent's memory config. Returns the new config."""
        raise NotImplementedError
