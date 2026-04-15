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

import logging

logger = logging.getLogger(__name__)

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


# Librarian config — controls the LLM enrichment passes for an agent.
# All tasks default ON for new agents on capable installs; the on/off
# switches let users disable enrichment per-agent on resource-constrained
# hardware or when they don't need it for a particular agent.
LIBRARIAN_TASKS = (
    "fact_extraction",
    "preference_extraction",
    "intake_classification",
    "crystallise",
    "reflect",
    "catalog_enrichment",
    "query_expansion",
    "verification",
)

# Fan-out levels map to the integer K passed to each retrieval layer.
# off  — single-result pass (K=1); essentially disables fan-out.
# low  — conservative multi-doc (K=3); safe on Pi NPU with 8K context.
# med  — moderate fan-out (K=10); suits GPU workers or large-context models.
# high — extended reasoning (K=20); use only when context window allows.
FANOUT_LEVELS: dict[str, int] = {"off": 1, "low": 3, "med": 10, "high": 20}

_FANOUT_ORDER = ["off", "low", "med", "high"]


def _default_fanout() -> dict:
    return {
        # Opt-in level — low keeps latency safe on Pi-class hardware by default.
        "default": "low",
        # When True, auto-bumps one tier on workers with GPU + TurboQuant + ≥12GB VRAM.
        "auto_scale": True,
    }


def _default_librarian() -> dict:
    return {
        "enabled": True,
        # Provider:model string (e.g. "ollama:qwen3:4b") or None to use the
        # taosmd install default. Per-agent override lives here so a single
        # install can run different models per agent.
        "model": None,
        "tasks": {t: True for t in LIBRARIAN_TASKS},
        "fanout": _default_fanout(),
    }


def _ensure_fanout(lib: dict) -> dict:
    """Back-fill the fanout block on records that predate it."""
    if "fanout" not in lib:
        lib["fanout"] = _default_fanout()
    else:
        # Ensure both sub-keys are present (partial legacy records).
        fo = lib["fanout"]
        if "default" not in fo:
            fo["default"] = "low"
        if "auto_scale" not in fo:
            fo["auto_scale"] = True
    return lib


@dataclass
class AgentRecord:
    name: str
    display_name: str
    created_at: int
    last_ingest_at: int = 0
    total_chunks: int = 0
    librarian: dict | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "last_ingest_at": self.last_ingest_at,
            "total_chunks": self.total_chunks,
            "librarian": self.librarian or _default_librarian(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentRecord":
        raw_lib = data.get("librarian") or _default_librarian()
        return cls(
            name=data["name"],
            display_name=data.get("display_name", data["name"]),
            created_at=int(data.get("created_at", 0)),
            last_ingest_at=int(data.get("last_ingest_at", 0)),
            total_chunks=int(data.get("total_chunks", 0)),
            librarian=_ensure_fanout(raw_lib),
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
            librarian=_default_librarian(),
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

    # ----- librarian config --------------------------------------------

    def get_librarian(self, name: str) -> dict:
        """Return the librarian config for an agent.

        Older records may have been saved before the librarian field
        existed — those get the default config back so callers don't
        need a back-compat branch. Records saved before the fanout block
        was added get it auto-populated.
        """
        agent = self.get_agent(name)
        lib = agent.get("librarian") or _default_librarian()
        return _ensure_fanout(lib)

    def set_librarian(
        self,
        name: str,
        *,
        enabled: bool | None = None,
        model: str | None = None,
        tasks: dict[str, bool] | None = None,
        clear_model: bool = False,
        fanout: str | None = None,
        fanout_auto_scale: bool | None = None,
    ) -> dict:
        """Patch an agent's librarian config. Returns the updated config.

        - ``enabled``: master on/off switch. When False, every LLM
          enrichment task is skipped regardless of per-task settings.
        - ``model``: provider:model override (e.g. "ollama:qwen3:4b").
          Pass ``clear_model=True`` to revert to the install default.
        - ``tasks``: dict of per-task switches. Keys must come from
          :data:`LIBRARIAN_TASKS`. Unknown keys raise ValueError.
        - ``fanout``: fan-out level — one of ``off | low | med | high``.
          Controls the per-layer top-K used during retrieval.
        - ``fanout_auto_scale``: when True the resolver bumps the fanout
          level one tier up on workers with GPU + TurboQuant + ≥12 GB VRAM.
        """
        if tasks is not None:
            unknown = set(tasks) - set(LIBRARIAN_TASKS)
            if unknown:
                raise ValueError(
                    f"unknown librarian task(s): {sorted(unknown)}. "
                    f"Valid tasks: {LIBRARIAN_TASKS}"
                )
        if fanout is not None and fanout not in FANOUT_LEVELS:
            raise ValueError(
                f"unknown fanout level {fanout!r}. "
                f"Valid levels: {list(FANOUT_LEVELS)}"
            )

        data = self._read()
        for a in data["agents"]:
            if a["name"] == name:
                lib = a.get("librarian") or _default_librarian()
                lib = _ensure_fanout(lib)
                if enabled is not None:
                    lib["enabled"] = bool(enabled)
                if clear_model:
                    lib["model"] = None
                elif model is not None:
                    lib["model"] = model
                if tasks is not None:
                    # Preserve unset tasks; only patch the keys provided.
                    lib_tasks = lib.get("tasks") or {t: True for t in LIBRARIAN_TASKS}
                    lib_tasks.update({k: bool(v) for k, v in tasks.items()})
                    lib["tasks"] = lib_tasks
                if fanout is not None:
                    lib["fanout"]["default"] = fanout
                if fanout_auto_scale is not None:
                    lib["fanout"]["auto_scale"] = bool(fanout_auto_scale)
                a["librarian"] = lib
                self._write(data)
                return lib
        raise AgentNotFoundError(f"agent {name!r} is not registered")

    def is_task_enabled(self, name: str, task: str) -> bool:
        """Convenience check used by enrichment call paths.

        Returns True when both the master enabled flag and the per-task
        flag are on. Unknown agents and unknown tasks return False (a
        wrong name shouldn't accidentally enable everything).
        """
        if task not in LIBRARIAN_TASKS:
            return False
        try:
            lib = self.get_librarian(name)
        except AgentNotFoundError:
            return False
        if not lib.get("enabled", True):
            return False
        return bool(lib.get("tasks", {}).get(task, True))

    def effective_fanout(
        self,
        name: str,
        worker_capabilities: dict | None = None,
    ) -> int:
        """Resolve the retrieval fan-out K for an agent.

        Reads ``librarian.fanout.default`` and returns its integer K.
        If ``librarian.fanout.auto_scale`` is True **and** the supplied
        *worker_capabilities* dict signals a capable GPU worker
        (``gpu_vram_gb >= 12`` **and** ``turboquant == True``), the tier
        is bumped up by one step (low→med, med→high). The ceiling is
        ``high``; calling with a ``high`` baseline + GPU worker stays at
        ``high``.

        Args:
            name: Registered agent name.
            worker_capabilities: Optional dict with keys:
                - ``gpu_vram_gb`` (float/int) — VRAM in GB (default 0)
                - ``turboquant`` (bool) — whether TurboQuant is active (default False)

        Returns:
            Integer K (1, 3, 10, or 20).
        """
        try:
            lib = self.get_librarian(name)
        except AgentNotFoundError:
            return FANOUT_LEVELS["low"]

        fo = lib.get("fanout") or _default_fanout()
        level = fo.get("default", "low")
        if level not in FANOUT_LEVELS:
            level = "low"

        if fo.get("auto_scale", True) and worker_capabilities is not None:
            caps = worker_capabilities
            has_capable_gpu = (
                caps.get("gpu_vram_gb", 0) >= 12
                and caps.get("turboquant", False)
            )
            if has_capable_gpu:
                current_idx = _FANOUT_ORDER.index(level)
                # Bump one tier; clamp at the top.
                bumped_idx = min(current_idx + 1, len(_FANOUT_ORDER) - 1)
                level = _FANOUT_ORDER[bumped_idx]

        return FANOUT_LEVELS[level]



def run_if_enabled(agent_name: str, task: str, fn, *args, fallback=None, **kw):
    """Invoke fn only when the task is enabled for this agent; else return fallback.

    Designed to be the single gate point in orchestrators (catalog_pipeline,
    retrieval). Extractors stay pure — they do not check agent config.

    Usage:
        result = run_if_enabled(agent, fact_extraction, extract_facts, text,
                                 fallback=[])

    Args:
        agent_name: Registered agent name (or "" for anonymous installs).
        task: One of LIBRARIAN_TASKS.
        fn: Callable to invoke when enabled.
        *args: Positional args for fn.
        fallback: Value to return when task is disabled. Defaults to None.
        **kw: Keyword args for fn.

    Returns:
        fn(*args, **kw) when enabled, else fallback.
    """
    if not is_task_enabled(agent_name, task):
        logger.debug(
            "run_if_enabled: task=%r agent=%r disabled", task, agent_name
        )
        return fallback
    return fn(*args, **kw)

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


def get_librarian(name: str) -> dict:
    return _registry().get_librarian(name)


def set_librarian(
    name: str,
    *,
    enabled: bool | None = None,
    model: str | None = None,
    tasks: dict[str, bool] | None = None,
    clear_model: bool = False,
    fanout: str | None = None,
    fanout_auto_scale: bool | None = None,
) -> dict:
    return _registry().set_librarian(
        name,
        enabled=enabled,
        model=model,
        tasks=tasks,
        clear_model=clear_model,
        fanout=fanout,
        fanout_auto_scale=fanout_auto_scale,
    )


def is_task_enabled(name: str, task: str) -> bool:
    return _registry().is_task_enabled(name, task)


def effective_fanout(
    agent_name: str,
    worker_capabilities: dict | None = None,
) -> int:
    """Resolve the retrieval fan-out K for an agent given its worker capabilities.

    Returns the integer K. Reads ``librarian.fanout.default``; if
    ``librarian.fanout.auto_scale`` is True AND the worker reports
    GPU + TurboQuant + ≥12 GB VRAM, bumps up one tier (low→med, med→high).

    Args:
        agent_name: Registered agent name.
        worker_capabilities: Optional dict with keys ``gpu_vram_gb`` and
            ``turboquant``. Pass ``None`` (or omit) for Pi-class workers with
            no GPU information.

    Returns:
        Integer K (1, 3, 10, or 20).
    """
    return _registry().effective_fanout(agent_name, worker_capabilities)


__all__ = [
    "AgentRegistry",
    "AgentRecord",
    "AgentExistsError",
    "AgentNotFoundError",
    "InvalidAgentNameError",
    "LIBRARIAN_TASKS",
    "FANOUT_LEVELS",
    "register_agent",
    "list_agents",
    "agent_exists",
    "get_agent",
    "delete_agent",
    "ensure_agent",
    "update_stats",
    "get_librarian",
    "set_librarian",
    "is_task_enabled",
    "effective_fanout",
    "run_if_enabled",
]
