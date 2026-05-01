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
        vmem = VectorMemory(
            db_path=str(path / "vector-memory.db"),
            embed_mode=embed_mode,
            onnx_path=onnx_path or "",
        )
        await vmem.init()
        kg = TemporalKnowledgeGraph(db_path=str(path / "knowledge-graph.db"))
        await kg.init()

        stores = {
            "archive": archive,
            "vector": vmem,
            "kg": kg,
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


async def ingest(transcript, *, agent: str, data_dir=None) -> dict:
    """Shelve a transcript into the zero-loss archive and embed it for search.

    Per the agent-rules contract: call after every meaningful exchange.
    Verbatim text is preserved in the append-only archive; a copy is
    embedded into vector memory so a later ``search()`` can find it.

    Args:
        transcript: A single message (str), a structured turn dict
            (``{"role", "content", optional "timestamp"}``), or an iterable
            of either.
        agent: Registered agent name. Auto-registered if absent.
        data_dir: Optional taosmd data dir. Defaults to ``$TAOSMD_DATA_DIR``
            or ``~/.taosmd``.

    Returns:
        ``{"archived": int, "agent": str, "data_dir": str}`` — ``archived``
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
        await stores["archive"].record(
            "conversation",
            item,
            agent_name=agent,
            summary=text[:80],
        )
        meta: dict = {"agent": agent}
        if "role" in item:
            meta["role"] = item["role"]
        if "timestamp" in item:
            meta["timestamp"] = item["timestamp"]
        await stores["vector"].add(text, metadata=meta)
        archived += 1

    update_stats(agent, last_ingest_at=int(time.time()))
    return {"archived": archived, "agent": agent, "data_dir": stores["data_dir"]}


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


async def search(query: str, *, agent: str, limit: int = 5, data_dir=None) -> list[dict]:
    """Search the librarian's shelves for passages relevant to ``query``.

    Returns ranked hits in the agent-rules contract shape with explicit
    ``source``, ``timestamp``, and ``confidence`` fields. If the top hit
    has ``confidence < 0.6`` per the rules block, agents should treat it
    as "she didn't find anything" rather than invent.

    Args:
        query: The search query.
        agent: Registered agent name. Auto-registered if absent.
        limit: Maximum number of hits to return.
        data_dir: Optional taosmd data dir (see :func:`ingest`).
    """
    if not agent:
        raise ValueError("agent name is required")
    if not query:
        return []

    stores = await _ensure_stores(data_dir)

    from taosmd.agents import ensure_agent
    from taosmd.retrieval import retrieve as _retrieve
    ensure_agent(agent)

    raw = await _retrieve(
        query,
        sources={
            "vector": stores["vector"],
            "kg": stores["kg"],
            "archive": stores["archive"],
        },
        agent=agent,
        agent_name=agent,
        limit=limit,
    )
    return [_format_hit(hit) for hit in raw]


__all__ = ["ingest", "search"]
