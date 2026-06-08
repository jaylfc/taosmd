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

    # ------------------------------------------------------------------
    # Recipes (config profiles) - the SP4 contract seam
    # ------------------------------------------------------------------
    # A recipe is a declared config bundle (retrieval + ingest + generator +
    # librarian) carrying benchmark scores, tier, and pros/cons. The Memory app
    # renders recipes generically from get_recipe_schema() and drives selection
    # via list/get/apply/recommend. Defaults below are graceful no-ops so a
    # backend without recipe support degrades cleanly; taOSmd overrides them.

    async def get_recipe_schema(self) -> dict:
        """Return the JSON Schema for a recipe config bundle.

        Lets a host UI render any recipe generically. Default: empty (this
        backend exposes no recipes).
        """
        return {}

    async def list_recipes(self) -> list[dict]:
        """Return the available recipes, each as a config bundle + metadata."""
        return []

    async def get_recipe(self, recipe_id: str) -> dict | None:
        """Return one recipe by id, or None if unknown."""
        return None

    async def apply_recipe(self, recipe_id: str, *, agent: str | None = None) -> dict:
        """Apply a recipe to ``agent`` (or as the global default when None).

        Returns ``{"applied_recipe_id": str, "recipe": dict}``.
        """
        raise NotImplementedError

    async def recommend(self, device_info: dict | None = None) -> list[dict]:
        """Rank recipes best-first for the given (or locally probed) device.

        ``device_info`` is the ``{host, cluster, aggregate}`` shape; when None,
        the backend falls back to a local hardware probe. Each item is a recipe
        dict with an added ``rationale``.
        """
        return []

    async def create_recipe(self, spec: dict) -> dict:
        """Create a custom recipe from a spec (validated against the schema)."""
        raise NotImplementedError
