"""Adapter-agnostic service layer over :mod:`taosmd.api`.

This is the shared core that activation surfaces sit on top of: the local
HTTP/REST server (#85) and the upcoming MCP server (#84) both call these
functions rather than reaching into :mod:`taosmd.api` directly. Keeping the
glue in one place means a single, consistent contract for every adapter and
guarantees their behaviour matches the Python API exactly.

The functions here are deliberately thin — they reuse
:func:`taosmd.api.ingest`, :func:`taosmd.api.search`,
:func:`taosmd.api.list_pending_decisions`, and
:func:`taosmd.api.resolve_pending_decision` (and therefore
``_ensure_stores`` / the stores cache / ``TAOSMD_DATA_DIR`` handling) so
the only thing they add is a uniform, transport-friendly signature:
``(positional, agent=..., data_dir=..., **opts)``.
"""

from __future__ import annotations

from . import api as _api


async def ingest(text, *, agent: str, data_dir=None, **opts) -> dict:
    """Shelve a transcript and embed it for later search.

    Thin wrapper over :func:`taosmd.api.ingest`. ``text`` may be a string,
    a turn dict, or an iterable of either (see the underlying API for the
    accepted shapes). Returns ``{"archived", "agent", "data_dir"}``.
    """
    return await _api.ingest(text, agent=agent, data_dir=data_dir, **opts)


async def search(query: str, *, agent: str, data_dir=None, limit: int = 5, **opts) -> list[dict]:
    """Search memory for passages relevant to ``query``.

    Thin wrapper over :func:`taosmd.api.search`. Returns ranked hits in the
    agent-rules contract shape (``text``/``source``/``timestamp``/
    ``confidence``/``metadata``).
    """
    return await _api.search(query, agent=agent, data_dir=data_dir, limit=limit, **opts)


async def pending_list(*, agent: str | None = None, data_dir=None, limit: int = 20) -> list[dict]:
    """Return unresolved KG-update decisions deferred by the librarian.

    ``agent`` is accepted for adapter symmetry; the pending-decisions queue
    is keyed per data dir (per install), not per agent, so it is not used to
    filter here. Use ``subject=`` on the underlying API if subject-level
    filtering is needed.
    """
    return await _api.list_pending_decisions(limit=limit, data_dir=data_dir)


async def pending_resolve(
    decision_id: str,
    decision: str,
    *,
    note: str = "",
    data_dir=None,
) -> dict:
    """Resolve a pending decision with the user's explicit choice.

    ``decision`` is one of ``accept`` / ``reject`` / ``modify`` (the
    ``action`` argument of :func:`taosmd.api.resolve_pending_decision`).
    Returns ``{ok, applied_kg, resolution}``.
    """
    return await _api.resolve_pending_decision(
        decision_id, action=decision, note=note, data_dir=data_dir,
    )


async def supersede(match: str, *, agent: str | None = None, data_dir=None) -> dict:
    """Soft-supersede vector chunk(s) whose stored text contains ``match``.

    Thin wrapper over :func:`taosmd.api.supersede_vectors`. Used to wire a
    correction into the vector layer by content: matching chunks leave active
    recall while the raw rows + archive entries are retained (zero-loss),
    mirroring how the typed KG invalidates a corrected triple. ``agent`` is
    accepted for adapter symmetry; the vector store is keyed per data dir.
    Returns ``{"superseded": int, "match": str}``.
    """
    count = await _api.supersede_vectors(match, data_dir=data_dir)
    return {"superseded": count, "match": match}


async def stats(*, agent: str, data_dir=None) -> dict:
    """Return lightweight stats for an agent.

    Ensures the stores exist (so a freshly-pointed data dir is initialised),
    then reports the registry record for ``agent`` plus the resolved data
    dir. Shape: ``{"agent", "data_dir", "registered", "created_at",
    "last_ingest_at", "total_chunks"}``. Unknown agents report
    ``registered=False`` with zeroed counters rather than raising, so the
    surface stays forgiving for read-only probes.
    """
    if not agent:
        raise ValueError("agent name is required")
    stores = await _api._ensure_stores(data_dir)

    from .agents import AgentNotFoundError, get_agent  # noqa: PLC0415

    out = {
        "agent": agent,
        "data_dir": stores["data_dir"],
        "registered": False,
        "created_at": 0,
        "last_ingest_at": 0,
        "total_chunks": 0,
    }
    try:
        record = get_agent(agent)
    except AgentNotFoundError:
        return out
    out.update(
        registered=True,
        created_at=record.get("created_at", 0),
        last_ingest_at=record.get("last_ingest_at", 0),
        total_chunks=record.get("total_chunks", 0),
    )
    return out


__all__ = ["ingest", "search", "pending_list", "pending_resolve", "stats", "supersede"]
