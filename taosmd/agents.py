"""Agent registry — formal naming and isolation for multi-agent installs.

Per-agent indexes live at ``data/agent-memory/{name}/index.sqlite`` and
have always existed; agents could just call ``search(agent="x")`` and
the index would appear lazily on first write. That worked for solo
installs but left two real problems:

1. Two agents in the same framework could pick the same name and start
   sharing memory without anyone noticing until a wrong fact surfaced.
2. The agent install message couldn't say "I registered as X" with
   certainty — there was no registration to point at.

The registry is a single ``data/agents.json`` envelope that lists every
agent on this taosmd install. Lazy creation on first write still works
(back-compat) — search/ingest auto-register an agent if it isn't in the
registry yet, with ``display_name == name``. Explicit registration just
gives the install agent a clean success/fail signal and lets the CLI
list/remove agents.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")


class AgentExistsError(ValueError):
    """Raised when registering a name that already exists without clobber=True."""


class AgentNotFoundError(KeyError):
    """Raised when an operation references an agent that isn't registered."""


class InvalidAgentNameError(ValueError):
    """Raised when a name fails the ``NAME_RE`` check."""


@dataclass
class AgentRecord:
    name: str
    display_name: str
    created_at: int
    last_ingest_at: int = 0
    total_chunks: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "last_ingest_at": self.last_ingest_at,
            "total_chunks": self.total_chunks,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentRecord":
        return cls(
            name=data["name"],
            display_name=data.get("display_name", data["name"]),
            created_at=int(data.get("created_at", 0)),
            last_ingest_at=int(data.get("last_ingest_at", 0)),
            total_chunks=int(data.get("total_chunks", 0)),
        )


class AgentRegistry:
    """File-backed registry of agents on this taosmd install.

    Storage is a single JSON envelope rather than SQLite — agent records
    are tiny, writes are rare, and a flat file makes the registry easy
    to back up alongside the rest of ``data/``.
    """

    def __init__(self, data_dir: Path | str = "data"):
        self.data_dir = Path(data_dir)
        self.registry_path = self.data_dir / "agents.json"

    # ----- internal -----------------------------------------------------

    def _read(self) -> dict:
        if not self.registry_path.exists():
            return {"agents": []}
        try:
            return json.loads(self.registry_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {"agents": []}

    def _write(self, data: dict) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: tmp + rename so a crash mid-write can't leave a
        # truncated registry on disk.
        tmp = self.registry_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.registry_path)

    def _agent_dir(self, name: str) -> Path:
        return self.data_dir / "agent-memory" / name

    @staticmethod
    def _validate_name(name: str) -> None:
        if not NAME_RE.match(name):
            raise InvalidAgentNameError(
                f"agent name must match {NAME_RE.pattern!r} "
                "(lowercase, starts with a letter, alnum + - + _ only, max 63 chars)"
            )

    # ----- public API ---------------------------------------------------

    def register_agent(
        self,
        name: str,
        *,
        display_name: str = "",
        clobber: bool = False,
    ) -> dict:
        """Create the agent record + index directory.

        Returns the public agent record. Raises :class:`AgentExistsError`
        if the name is already taken and ``clobber`` is False.
        """
        self._validate_name(name)
        data = self._read()
        existing_idx = next(
            (i for i, a in enumerate(data["agents"]) if a["name"] == name),
            None,
        )
        if existing_idx is not None and not clobber:
            raise AgentExistsError(f"agent {name!r} is already registered")

        record = AgentRecord(
            name=name,
            display_name=display_name or name,
            created_at=int(time.time()),
        )
        if existing_idx is not None:
            data["agents"][existing_idx] = record.to_dict()
        else:
            data["agents"].append(record.to_dict())
        self._write(data)

        # Eagerly create the per-agent dir so back-end stores can write
        # without surprising the install agent with a directory error.
        self._agent_dir(name).mkdir(parents=True, exist_ok=True)

        return record.to_dict()

    def ensure_agent(self, name: str, *, display_name: str = "") -> dict:
        """Auto-register on first use (back-compat with lazy index creation).

        Returns the existing record if already registered, otherwise
        creates a new one. Used by ``search()`` / ``ingest()`` so callers
        that never run ``register_agent()`` still work.
        """
        try:
            return self.get_agent(name)
        except AgentNotFoundError:
            return self.register_agent(name, display_name=display_name)

    def list_agents(self) -> list[dict]:
        return list(self._read()["agents"])

    def agent_exists(self, name: str) -> bool:
        return any(a["name"] == name for a in self._read()["agents"])

    def get_agent(self, name: str) -> dict:
        for a in self._read()["agents"]:
            if a["name"] == name:
                return a
        raise AgentNotFoundError(f"agent {name!r} is not registered")

    def delete_agent(self, name: str, *, drop_data: bool = False) -> None:
        data = self._read()
        before = len(data["agents"])
        data["agents"] = [a for a in data["agents"] if a["name"] != name]
        if len(data["agents"]) == before:
            raise AgentNotFoundError(f"agent {name!r} is not registered")
        self._write(data)

        if drop_data:
            agent_dir = self._agent_dir(name)
            if agent_dir.exists():
                shutil.rmtree(agent_dir)

    def update_stats(
        self,
        name: str,
        *,
        last_ingest_at: int | None = None,
        total_chunks: int | None = None,
    ) -> dict:
        """Patch the per-agent stats. Returns the updated record."""
        data = self._read()
        for a in data["agents"]:
            if a["name"] == name:
                if last_ingest_at is not None:
                    a["last_ingest_at"] = int(last_ingest_at)
                if total_chunks is not None:
                    a["total_chunks"] = int(total_chunks)
                self._write(data)
                return dict(a)
        raise AgentNotFoundError(f"agent {name!r} is not registered")


# ---------------------------------------------------------------------------
# Module-level convenience wrappers around a default registry rooted at
# ./data — matches the rest of taosmd which assumes ./data unless told
# otherwise. Pass a different ``data_dir`` to AgentRegistry directly for
# tests or non-default installs.
# ---------------------------------------------------------------------------

_default_registry: AgentRegistry | None = None


def _registry() -> AgentRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = AgentRegistry()
    return _default_registry


def register_agent(name: str, *, display_name: str = "", clobber: bool = False) -> dict:
    return _registry().register_agent(name, display_name=display_name, clobber=clobber)


def list_agents() -> list[dict]:
    return _registry().list_agents()


def agent_exists(name: str) -> bool:
    return _registry().agent_exists(name)


def get_agent(name: str) -> dict:
    return _registry().get_agent(name)


def delete_agent(name: str, *, drop_data: bool = False) -> None:
    _registry().delete_agent(name, drop_data=drop_data)


def ensure_agent(name: str, *, display_name: str = "") -> dict:
    return _registry().ensure_agent(name, display_name=display_name)


def update_stats(
    name: str,
    *,
    last_ingest_at: int | None = None,
    total_chunks: int | None = None,
) -> dict:
    return _registry().update_stats(name, last_ingest_at=last_ingest_at, total_chunks=total_chunks)


__all__ = [
    "AgentRegistry",
    "AgentRecord",
    "AgentExistsError",
    "AgentNotFoundError",
    "InvalidAgentNameError",
    "register_agent",
    "list_agents",
    "agent_exists",
    "get_agent",
    "delete_agent",
    "ensure_agent",
    "update_stats",
]
