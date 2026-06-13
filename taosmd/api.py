"""High-level zero-config API for taOSmd agents.

The two entry points referenced by ``taosmd/docs/agent-rules.md`` (the
per-turn rules block agents copy into their instruction file):

    await taosmd.ingest(transcript, agent="my-agent")
    hits = await taosmd.search("what did we discuss about X?", agent="my-agent")

Stores are auto-discovered from ``~/.taosmd`` (override with ``data_dir=``
or the ``TAOSMD_DATA_DIR`` env var) and lazy-initialised on first call.
The ``config.json`` written by ``python -m taosmd.auto_setup`` is honoured
when present, so an agent that ran the install ritual gets sensible
embed-mode defaults without further wiring.

Ingest writes to the zero-loss archive (verbatim, append-only) and
embeds the same text into vector memory so a later ``search()`` can
reach it. Search returns hits in the agent-rules-contract shape:
``{text, source, timestamp, confidence, metadata}``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = os.path.expanduser("~/.taosmd")
_stores_lock = asyncio.Lock()
_stores_cache: dict[str, dict] = {}


def _resolve_data_dir(data_dir) -> str:
    if data_dir is not None:
        return os.fspath(data_dir)
    env = os.environ.get("TAOSMD_DATA_DIR")
    if env:
        return env
    return _DEFAULT_DATA_DIR


def _load_config(data_dir: str) -> dict:
    cfg_path = Path(data_dir) / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("taosmd: failed to read %s: %s", cfg_path, exc)
        return {}


def _resolve_onnx_path(data_dir: str) -> str | None:
    """Find the MiniLM ONNX model directory.

    Search order: ``$TAOSMD_ONNX_PATH``, ``<data_dir>/models/minilm-onnx``,
    ``$TAOSMD_DIR/models/minilm-onnx`` (set by ``scripts/setup.sh``),
    ``~/taosmd/models/minilm-onnx`` (default install root). Returns None
    when no candidate has a ``model.onnx`` file.
    """
    env = os.environ.get("TAOSMD_ONNX_PATH")
    if env and (Path(env) / "model.onnx").exists():
        return env
    candidates = [
        Path(data_dir) / "models" / "minilm-onnx",
        Path(os.environ.get("TAOSMD_DIR", os.path.expanduser("~/taosmd"))) / "models" / "minilm-onnx",
    ]
    for candidate in candidates:
        if (candidate / "model.onnx").exists():
            return str(candidate)
    return None


async def _ensure_stores(data_dir=None) -> dict:
    """Lazy-init and cache the {archive, vector, kg} stores for a data dir.

    A separate cache entry per resolved data_dir means tests can pass an
    isolated tmpdir without polluting the production cache.
    """
    resolved = _resolve_data_dir(data_dir)
    cached = _stores_cache.get(resolved)
    if cached is not None:
        return cached

    async with _stores_lock:
        cached = _stores_cache.get(resolved)
        if cached is not None:
            return cached

        from taosmd.archive import ArchiveStore
        from taosmd.knowledge_graph import TemporalKnowledgeGraph
        from taosmd.vector_memory import VectorMemory

        path = Path(resolved)
        path.mkdir(parents=True, exist_ok=True)
        config = _load_config(resolved)
        embed_mode = config.get("vector_memory", {}).get("embed_mode", "onnx")
        onnx_path = _resolve_onnx_path(resolved) if embed_mode == "onnx" else ""
        if embed_mode == "onnx" and onnx_path is None:
            logger.warning(
                "taosmd: ONNX model not found under %s/models or $TAOSMD_DIR; "
                "falling back to qmd embed mode (set TAOSMD_ONNX_PATH or run setup.sh).",
                resolved,
            )
            embed_mode = "qmd"

        archive = ArchiveStore(
            archive_dir=str(path / "archive"),
            index_path=str(path / "archive-index.db"),
        )
        await archive.init()
        # Storage-format mode is a per-store (per-data-dir) property, not a
        # per-agent query knob: late-interaction stores per-token matrices and
        # cannot coexist with pooled vectors in one store. It is read from
        # config (seeded from the recommended recipe at setup) so the live
        # store is actually built in the mode the recipe asks for, closing the
        # gap where the lateint recipe was recommended but silently ignored.
        vm_cfg = config.get("vector_memory", {})
        vmem = VectorMemory(
            db_path=str(path / "vector-memory.db"),
            embed_mode=embed_mode,
            onnx_path=onnx_path or "",
            late_interaction=bool(vm_cfg.get("late_interaction", False)),
            colbert_model=vm_cfg.get("colbert_model", "") or "",
        )
        await vmem.init()
        kg = TemporalKnowledgeGraph(db_path=str(path / "knowledge-graph.db"))
        await kg.init()
        from taosmd.claims.store import ClaimStore  # noqa: PLC0415
        claims = ClaimStore(db_path=str(path / "claims.db"))
        await claims.init()

        stores = {
            "archive": archive,
            "vector": vmem,
            "kg": kg,
            "claims": claims,
            "data_dir": str(path),
        }
        _stores_cache[resolved] = stores
        return stores


def _normalize_transcript(transcript) -> list[dict]:
    """Coerce common shapes into a list of structured turn dicts.

    Accepts a string (one turn, role unknown), a single dict (treated as
    one turn), or any iterable of strings/dicts. Strings inside an
    iterable each become a one-turn dict with role ``"unknown"``.
    """
    if isinstance(transcript, str):
        return [{"role": "unknown", "content": transcript}]
    if isinstance(transcript, dict):
        return [dict(transcript)]
    if isinstance(transcript, Iterable):
        out: list[dict] = []
        for item in transcript:
            if isinstance(item, str):
                out.append({"role": "unknown", "content": item})
            elif isinstance(item, dict):
                out.append(dict(item))
            else:
                out.append({"role": "unknown", "content": str(item)})
        return out
    raise TypeError(
        f"transcript must be str, dict, or iterable, got {type(transcript).__name__}"
    )


async def ingest(transcript, *, agent: str, project: str | None = None, data_dir=None) -> dict:
    """Shelve a transcript into the zero-loss archive and embed it for search.

    Per the agent-rules contract: call after every meaningful exchange.
    Verbatim text is preserved in the append-only archive; a copy is
    embedded into vector memory so a later ``search()`` can find it.

    Args:
        transcript: A single message (str), a structured turn dict
            (``{"role", "content", optional "timestamp"}``), or an iterable
            of either.
        agent: Registered agent name. Auto-registered if absent.
        project: Optional project fingerprint for cross-agent memory sharing.
            When set, memories are tagged with this project ID so that
            different agents working on the same project can find each
            other's memories via ``search(..., also_include=[...])``.
            Use ``taosmd.project.get_project_id()`` for automatic detection.
        data_dir: Optional taosmd data dir. Defaults to ``$TAOSMD_DATA_DIR``
            or ``~/.taosmd``.

    Returns:
        ``{"archived": int, "agent": str, "project": str|None, "data_dir": str}``, where ``archived``
        is the count of non-empty turns that were shelved.
    """
    if not agent:
        raise ValueError("agent name is required")
    items = _normalize_transcript(transcript)
    stores = await _ensure_stores(data_dir)

    from taosmd.agents import ensure_agent, update_stats
    ensure_agent(agent)

    archived = 0
    for item in items:
        text = str(item.get("content", "")).strip()
        if not text:
            continue
        span_id = await stores["archive"].record(
            "conversation",
            item,
            agent_name=agent,
            summary=text[:80],
            project=project,
        )
        meta: dict = {"agent": agent}
        if project:
            meta["project"] = project
        if "role" in item:
            meta["role"] = item["role"]
        if "timestamp" in item:
            meta["timestamp"] = item["timestamp"]
        # Provenance: tag the vector row with its archive span so the recall
        # gate can look up the verification status of the claims it backs.
        if isinstance(span_id, int) and span_id >= 0:
            meta["archive_span_id"] = span_id
        await stores["vector"].add(text, metadata=meta)
        # Claims layer (additive): extract facts as claims tagged with the same
        # archive span. Stored unverified; the verify-pass checks them async.
        if "claims" in stores and isinstance(span_id, int) and span_id >= 0:
            from taosmd.claims.extract import claims_from_text  # noqa: PLC0415
            for c in claims_from_text(text, span_id):
                await stores["claims"].add_claim(
                    c["text"], c["archive_span_ids"], c["source_extractor"])
        archived += 1

    update_stats(agent, last_ingest_at=int(time.time()))
    return {"archived": archived, "agent": agent, "project": project, "data_dir": stores["data_dir"]}


async def ingest_batch(
    items,
    *,
    agent: str,
    project: str | None = None,
    data_dir=None,
) -> dict:
    """Bulk-shelve memory chunks with idempotent re-import (#25 contract).

    The migration path for external memory stores (taOS user memory): each
    item carries its own stable ``id`` (the caller's content hash) so the
    whole batch can be re-POSTed safely — items whose ``id`` is already
    stored are counted in ``skipped`` instead of duplicated.

    Args:
        items: List of ``{"text": str, "id": str?, "metadata": dict?}``.
            ``text`` is required per item. ``id`` is the caller's stable
            content hash, preserved as ``source_id`` in the stored metadata
            and used for dedup. ``metadata`` is preserved verbatim (e.g.
            ``collection``/``title`` for taOS user memory).
        agent: Registered agent name (e.g. ``"user-memory"``). Auto-registered
            if absent.
        project: Optional project fingerprint, as in :func:`ingest`.
        data_dir: Optional taosmd data dir (see :func:`ingest`).

    Returns:
        ``{"ingested": int, "skipped": int, "agent": str, "data_dir": str}``.
        ``skipped`` counts duplicate ids (incl. in-batch repeats) and
        empty-text items.

    Raises:
        ValueError: On a structurally invalid request (bad agent, items not
            a list, an item that is not a dict or lacks a string ``text``).
            Validation runs before any write so a 400 never leaves a
            half-applied batch.
    """
    if not agent:
        raise ValueError("agent name is required")
    if not isinstance(items, list):
        raise ValueError("'items' must be a list")
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"items[{i}] must be an object")
        if not isinstance(item.get("text"), str):
            raise ValueError(f"items[{i}].text (string) is required")
        if item.get("id") is not None and not isinstance(item["id"], str):
            raise ValueError(f"items[{i}].id must be a string when provided")
        if item.get("metadata") is not None and not isinstance(item["metadata"], dict):
            raise ValueError(f"items[{i}].metadata must be an object when provided")

    stores = await _ensure_stores(data_dir)

    from taosmd.agents import ensure_agent, update_stats
    ensure_agent(agent)

    seen = stores["vector"].existing_source_ids(agent=agent)
    ingested = 0
    skipped = 0
    for item in items:
        text = item["text"].strip()
        sid = item.get("id")
        if not text or (sid and sid in seen):
            skipped += 1
            continue
        user_md = dict(item.get("metadata") or {})
        if sid:
            user_md["source_id"] = sid
        await stores["archive"].record(
            "conversation",
            {"content": text, "metadata": user_md},
            agent_name=agent,
            summary=text[:80],
            project=project,
        )
        meta: dict = {"agent": agent, "metadata": user_md}
        if project:
            meta["project"] = project
        await stores["vector"].add(text, metadata=meta)
        if sid:
            seen.add(sid)
        ingested += 1

    if ingested:
        update_stats(agent, last_ingest_at=int(time.time()))
    return {
        "ingested": ingested,
        "skipped": skipped,
        "agent": agent,
        "data_dir": stores["data_dir"],
    }


async def _attach_and_gate_claims(hits: list[dict], claim_store, mode: str) -> list[dict]:
    """Attach each hit's backing-claim status (looked up by its source archive
    span) and apply the recall gate. ``mode`` "off" is a no-op passthrough.

    Operates on formatted hits (which carry ``confidence``); the pure gate sorts
    by ``score``, so a transient ``score`` mirror of ``confidence`` is set for
    the gate and stripped afterward, leaving the hit contract clean. The store
    lookup runs only when the gate is on.
    """
    if mode == "off" or not hits:
        return hits
    from taosmd.claims.gate import apply_claims_gate  # noqa: PLC0415
    for h in hits:
        span = (h.get("metadata") or {}).get("archive_span_id")
        spans = [span] if isinstance(span, int) else []
        h["claim_status"] = await claim_store.status_for_spans(spans)
        h["score"] = h.get("confidence", 0.0)
    gated = apply_claims_gate(hits, mode=mode)
    for h in gated:
        h.pop("score", None)
        h.pop("claim_status", None)
    return gated


def _format_hit(hit: dict) -> dict:
    """Reshape a retrieve() hit into the agent-rules contract shape.

    Confidence is taken from the underlying source score (cosine
    similarity for vector hits, KG confidence, etc) rather than the RRF
    rank score, since the agent-rules threshold (``< 0.6``) reads in the
    0-1 similarity range. Timestamp is read from user metadata or the
    underlying row's ``created_at`` / ``timestamp`` field.
    """
    md = hit.get("metadata", {}) or {}
    inner = md.get("metadata") if isinstance(md, dict) else None
    user_md = inner if isinstance(inner, dict) else md

    confidence = (
        md.get("similarity")
        if isinstance(md, dict) and md.get("similarity") is not None
        else hit.get("source_score", 0.0)
    )
    timestamp = (
        user_md.get("timestamp")
        if isinstance(user_md, dict) and user_md.get("timestamp") is not None
        else (md.get("created_at") if isinstance(md, dict) else None)
        or (md.get("timestamp") if isinstance(md, dict) else None)
        or 0
    )

    return {
        "text": hit.get("text", ""),
        "source": hit.get("source", "unknown"),
        "timestamp": timestamp,
        "confidence": round(float(confidence), 4),
        "metadata": user_md if isinstance(user_md, dict) else {},
    }


async def search(
    query: str,
    *,
    agent: str,
    project: str | None = None,
    also_include: list[str] | None = None,
    limit: int = 5,
    mode: str | None = None,
    prefer_verified: str = "off",
    data_dir=None,
) -> list[dict]:
    """Search the librarian's shelves for passages relevant to ``query``.

    Returns ranked hits in the agent-rules contract shape with explicit
    ``source``, ``timestamp``, and ``confidence`` fields. If the top hit
    has ``confidence < 0.6`` per the rules block, agents should treat it
    as "she didn't find anything" rather than invent.

    Args:
        query: The search query.
        agent: Registered agent name. Auto-registered if absent.
        project: Optional project fingerprint. When set, only memories
            tagged with this project are returned (scoped search).
            Use ``taosmd.project.get_project_id()`` for automatic detection.
        also_include: Optional list of additional agent names whose
            memories should be included in the search results (cross-agent
            reads). Only effective when ``project`` is also set.
        limit: Maximum number of hits to return.
        mode: Optional retrieval mode. ``"bm25"`` skips query embedding and
            recipe resolution entirely and returns BM25-only hits (the #25
            user-memory contract: keyword search-as-you-type, sub-300ms).
            Default ``None`` is the full recipe-driven retrieval path.
        data_dir: Optional taosmd data dir (see :func:`ingest`).
    """
    if not agent:
        raise ValueError("agent name is required")
    if mode not in (None, "", "bm25"):
        raise ValueError(f"unsupported search mode: {mode!r} (supported: 'bm25')")
    if not query:
        return []

    stores = await _ensure_stores(data_dir)

    from taosmd.agents import ensure_agent
    from taosmd.retrieval import retrieve as _retrieve
    from taosmd import recipes as _recipes  # noqa: PLC0415
    # Register against the same data_dir the recipe layer resolves against, so
    # resolve_recipe() can read/write this agent's record.
    ensure_agent(agent, data_dir=data_dir)

    # Build the list of agent names to search across.
    # When project + also_include are set, we search the calling agent
    # plus any explicitly included agents within the same project.
    search_agents = [agent]
    if project and also_include:
        for name in also_include:
            if name != agent:
                search_agents.append(name)

    if mode == "bm25":
        # BM25-only path: no embed call, no recipe/reranker resolution. Hits
        # come straight off the vector store's BM25 index in the same
        # contract shape as the full path.
        raw = await stores["vector"].search_bm25(
            query, limit=limit, project=project, search_agents=search_agents
        )
        hits = []
        for h in raw:
            md = dict(h.get("metadata") or {})
            md.setdefault("created_at", h.get("created_at"))
            hits.append(_format_hit({
                "text": h.get("text", ""),
                "source": "vector",
                "source_score": h.get("similarity", 0.0),
                "metadata": md,
            }))
        return await _attach_and_gate_claims(hits, stores.get("claims"), prefer_verified)

    # Resolve the active recipe (per-agent -> global default -> recommend()[0])
    # and map its retrieval knobs onto the retrieve() call below.
    recipe = _recipes.resolve_recipe(agent, data_dir=data_dir)
    rc = recipe.retrieval

    # Build the reranker the recipe asks for; degrade (not block) when absent.
    reranker = None
    degraded = None
    if rc.get("reranker") == "bge-v2-m3":
        from taosmd.cross_encoder import CrossEncoderReranker  # noqa: PLC0415
        ce = CrossEncoderReranker()
        if ce.available:
            reranker = ce
        else:
            # Kick off a progress-reporting download in the background; this
            # one search runs without the reranker rather than blocking.
            _recipes.ensure_reranker_model(block=False)
            degraded = "reranker-downloading"

    # per-agent fanout (recipe-derived via librarian.fanout) governs retrieval breadth; recipe.retrieval.candidate_top_k sets the pre-rerank pool. retrieval.limit is advisory in SP1.
    effective_limit = limit

    raw = await _retrieve(
        query,
        sources={
            "vector": stores["vector"],
            "kg": stores["kg"],
            "archive": stores["archive"],
        },
        strategy=rc.get("strategy", "thorough"),
        agent=agent,
        agent_name=agent,
        limit=effective_limit,
        reranker=reranker,
        adjacent_neighbors=rc.get("adjacent_neighbors", 0),
        fusion=rc.get("fusion", "boost"),
        candidate_top_k=rc.get("candidate_top_k"),
        project=project,
        search_agents=search_agents,
    )
    hits = [_format_hit(hit) for hit in raw]
    if degraded and hits:
        hits[0].setdefault("metadata", {})["recipe_degraded"] = degraded
    return await _attach_and_gate_claims(hits, stores.get("claims"), prefer_verified)


async def list_projects(*, data_dir=None) -> list[dict]:
    """List all projects that have stored memories.

    Returns a list of dicts with project_id, agent count, and last activity.
    Useful for discovery when an agent doesn't know which project it's in.

    Returns:
        ``[{"project_id": str, "agents": [str], "last_ingest": float}]``
    """
    stores = await _ensure_stores(data_dir)

    rows = stores["archive"]._conn.execute(
        "SELECT project, agent_name, MAX(timestamp) as last_ts "
        "FROM archive_index WHERE project IS NOT NULL "
        "GROUP BY project, agent_name ORDER BY last_ts DESC"
    ).fetchall()

    projects: dict[str, dict] = {}
    for row in rows:
        pid = row["project"]
        if pid not in projects:
            projects[pid] = {"project_id": pid, "agents": [], "last_ingest": 0}
        if row["agent_name"] and row["agent_name"] not in projects[pid]["agents"]:
            projects[pid]["agents"].append(row["agent_name"])
        if row["last_ts"] and row["last_ts"] > projects[pid]["last_ingest"]:
            projects[pid]["last_ingest"] = row["last_ts"]

    return list(projects.values())


async def list_shelves(*, project: str, data_dir=None) -> list[dict]:
    """List all agent shelves within a specific project.

    Returns agent names, fact counts, and last activity for the given project.
    Use this to discover what other agents have memories for a project before
    using ``also_include`` in ``search()``.

    Args:
        project: The project fingerprint to query.

    Returns:
        ``[{"agent": str, "facts": int, "last_ingest": float}]``
    """
    stores = await _ensure_stores(data_dir)

    rows = stores["archive"]._conn.execute(
        "SELECT agent_name, COUNT(*) as cnt, MAX(timestamp) as last_ts "
        "FROM archive_index WHERE project = ? "
        "GROUP BY agent_name ORDER BY last_ts DESC",
        (project,),
    ).fetchall()

    return [
        {
            "agent": row["agent_name"] or "unknown",
            "facts": row["cnt"],
            "last_ingest": row["last_ts"],
        }
        for row in rows
    ]


async def list_pending_decisions(
    *, limit: int = 20, subject: str | None = None, data_dir=None,
) -> list[dict]:
    """Return unresolved KG-update decisions deferred by the librarian.

    Per the agent-rules contract: call this at the start of every session
    and surface any pending decisions to the user before answering their
    first non-trivial question. Never silently auto-resolve; the queue
    exists because automatic resolution was the wrong call.
    """
    from pathlib import Path
    from taosmd.pending_decisions import PendingDecisionsStore
    resolved = Path(_resolve_data_dir(data_dir))
    store = PendingDecisionsStore(db_path=resolved / "knowledge-graph.db")
    await store.init()
    try:
        return await store.list_pending(subject=subject, limit=limit)
    finally:
        await store.close()


async def resolve_pending_decision(
    pending_id: str,
    *,
    action: str,
    note: str = "",
    data_dir=None,
) -> dict:
    """Resolve a pending decision with the user's explicit choice.

    ``action`` is one of ``accept`` / ``reject`` / ``modify``. ``accept``
    invokes the matching KG mutation (e.g. invalidate the old triple +
    write the new one for a contradiction). ``reject`` records the
    resolution without touching the KG. ``modify`` records the resolution
    and leaves any KG-side adjustment to the caller (a future agent UI
    can offer free-form edits before re-writing).

    Returns ``{ok: bool, applied_kg: bool, resolution: str}``.
    """
    from pathlib import Path
    from taosmd.pending_decisions import PendingDecisionsStore
    from taosmd.knowledge_graph import TemporalKnowledgeGraph

    if action not in {"accept", "reject", "modify"}:
        raise ValueError(f"action must be accept|reject|modify, got {action!r}")
    resolution = {"accept": "accepted", "reject": "rejected", "modify": "modified"}[action]
    resolved_data_dir = Path(_resolve_data_dir(data_dir))
    kg_path = resolved_data_dir / "knowledge-graph.db"

    store = PendingDecisionsStore(db_path=kg_path)
    await store.init()
    try:
        decision = await store.get(pending_id)
        if decision is None:
            return {"ok": False, "applied_kg": False, "resolution": resolution}

        applied_kg = False
        if resolution == "accepted" and decision["suggested_action"] == "invalidate_old_add_new":
            kg = TemporalKnowledgeGraph(db_path=kg_path)
            await kg.init()
            try:
                # Resolve the old object names before invalidating so we can
                # mirror the correction into the vector layer (soft-supersede
                # the chunk(s) carrying the stale value). Best-effort and
                # zero-loss: raw rows are retained, only hidden from recall.
                stores = await _ensure_stores(data_dir)
                vmem = stores.get("vector")
                old_objects = [
                    row["object_name"]
                    for old_id in decision["old_triple_ids"]
                    if (row := kg._conn.execute(
                        "SELECT o.name AS object_name FROM kg_triples t "
                        "JOIN kg_entities o ON o.id = t.object_id WHERE t.id = ?",
                        (old_id,),
                    ).fetchone()) is not None
                ]
                for old_id in decision["old_triple_ids"]:
                    await kg.invalidate(old_id)
                if vmem is not None:
                    for obj_name in old_objects:
                        try:
                            await vmem.supersede_matching(obj_name)
                        except Exception as e:  # pragma: no cover - defensive
                            logger.debug("vector supersede skipped: %s", e)
                await kg.add_triple(
                    subject=decision["subject"],
                    predicate=decision["predicate"],
                    obj=decision["new_object"],
                    confidence=decision["new_triple_confidence"],
                    source=decision["source"],
                )
                applied_kg = True
            finally:
                await kg.close()

        ok = await store.resolve(pending_id, resolution=resolution, note=note)
        return {"ok": ok, "applied_kg": applied_kg, "resolution": resolution}
    finally:
        await store.close()


async def reconcile(*, agent: str, data_dir=None, repair: bool = True) -> dict:
    """Detect archive turns missing from the vector store and optionally re-add them.

    The append-only archive is the source of truth (Option A). The vector store
    is a derived index. A crash between the archive write and the vector write in
    :func:`ingest` leaves a turn present in the archive but absent from vector
    recall. This function detects and, when ``repair=True``, corrects that gap.

    **How missing is computed**

    Build a ``Counter`` (multiset) of texts from both stores, then subtract the
    vector multiset from the archive multiset. A text that appears twice in the
    archive but only once in the vector store counts as 1 missing copy; a text
    present in the archive but entirely absent from the vector store counts as N
    missing copies where N is its archive count.

    **Supersede guard**

    ``iter_entries(include_superseded=True)`` includes intentionally superseded
    rows in the vector multiset, so a turn whose vector copy was soft-hidden by a
    correction is counted as *present* and is not re-added. Corrected/stale
    content is therefore never resurrected by reconcile.

    **Chunking**

    ``add()`` stores each turn as one row (no splitting). The comparison is done
    at the full-turn level: the archive's ``data.content`` field is compared
    directly against the stored ``text`` column values. Archive rows whose content
    field is empty or whitespace-only are skipped (they were skipped at ingest
    time too, so they can never be in the vector store).

    Args:
        agent: Agent name whose archive + vector entries are reconciled.
        data_dir: Optional taosmd data dir. Defaults to ``$TAOSMD_DATA_DIR`` or
            ``~/.taosmd``.
        repair: When True (default), re-add missing entries to the vector store.
            When False, only report without modifying (dry-run / ``--check``).

    Returns:
        ``{
            "agent": str,
            "archive_turns": int,   # non-empty conversation turns in the archive
            "vector_entries": int,  # entries in the vector store (incl. superseded)
            "missing": int,         # archive turns absent from the vector store
            "readded": int,         # entries re-added (0 when repair=False)
            "checked_ok": bool,     # True when missing == 0
        }``
    """
    import json as _json
    from collections import Counter

    if not agent:
        raise ValueError("agent name is required")

    stores = await _ensure_stores(data_dir)
    archive = stores["archive"]
    vmem = stores["vector"]

    from taosmd.archive import EVENT_CONVERSATION

    # --- Build archive multiset -----------------------------------------
    # Use a large limit; the archive index is an SQLite table so this is fast.
    rows = await archive.query(
        event_type=EVENT_CONVERSATION,
        agent_name=agent,
        limit=1_000_000,
    )
    archive_texts: list[tuple[str, dict]] = []  # (text, row) for re-add
    archive_counter: Counter[str] = Counter()
    for row in rows:
        try:
            data = _json.loads(row.get("data_json", "{}"))
        except (_json.JSONDecodeError, TypeError):
            data = {}
        text = str(data.get("content", "")).strip()
        if not text:
            continue
        archive_counter[text] += 1
        # Keep the first-seen archive row per text value for metadata
        # reconstruction; duplicates will use the same metadata shape.
        archive_texts.append((text, data))

    # --- Build vector multiset (including superseded) -------------------
    vector_counter: Counter[str] = Counter()
    async for text, _meta in vmem.iter_entries(agent=agent, include_superseded=True):
        vector_counter[text] += 1

    # --- Compute missing (count-aware subtraction) ----------------------
    missing_counter: Counter[str] = archive_counter.copy()
    missing_counter.subtract(vector_counter)
    # Discard zero or negative counts (present or over-represented in vector)
    missing: dict[str, int] = {t: c for t, c in missing_counter.items() if c > 0}

    total_missing = sum(missing.values())
    readded = 0

    if repair and missing:
        # Build a lookup of text → archive data for timestamp reconstruction.
        # Only need one representative row per distinct text.
        text_to_data: dict[str, dict] = {}
        for text, data in archive_texts:
            if text not in text_to_data:
                text_to_data[text] = data

        for text, count in missing.items():
            data = text_to_data.get(text, {})
            meta: dict = {"agent": agent}
            # Propagate role and timestamp from the archive entry where available.
            if "role" in data:
                meta["role"] = data["role"]
            if "timestamp" in data:
                meta["timestamp"] = data["timestamp"]
            for _ in range(count):
                added_id = await vmem.add(text, metadata=meta)
                if added_id != -1:
                    readded += 1

    vector_total = sum(vector_counter.values())
    return {
        "agent": agent,
        "archive_turns": sum(archive_counter.values()),
        "vector_entries": vector_total,
        "missing": total_missing,
        "readded": readded,
        "checked_ok": total_missing == 0,
    }


async def supersede_vectors(match: str, *, data_dir=None) -> int:
    """Soft-supersede vector chunk(s) whose stored text contains ``match``.

    The manual counterpart to the automatic KG->vector link: lets a caller
    retire a stale fact from vector recall directly (e.g. when a correction
    arrives outside the KG contradiction path). Zero-loss: the matched rows
    are retained, only stamped ``valid_to`` so ``search()`` skips them.
    Returns the number of rows superseded.
    """
    stores = await _ensure_stores(data_dir)
    return await stores["vector"].supersede_matching(match)


__all__ = [
    "ingest",
    "search",
    "list_projects",
    "list_shelves",
    "list_pending_decisions",
    "resolve_pending_decision",
    "supersede_vectors",
    "reconcile",
]
