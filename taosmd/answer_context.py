"""Answer-ready context: dated session blocks, oldest first, inside a token budget.

``search`` returns raw ranked hits. This module turns those hits into the one
string an agent answers from:

* hits come from :func:`taosmd.api.search` itself, so owner, agent, project and
  collection-grant scoping and the fusion are the ones search applies;
* each conversation-turn hit is widened to its neighbouring turns of the same
  session, in their original order;
* turns are grouped into one block per session, each headed by the session's
  own date, and the blocks are laid out oldest first;
* blocks are added in the rank order of their best hit while they fit the token
  budget. A block that does not fit is dropped whole and the next one is tried;
  nothing is ever cut in the middle.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

from . import api as _api

#: Shown to the reader model next to the context. Returned verbatim.
GUIDANCE = (
    "Answer from the dated context. You may combine facts and calculate from "
    "the dates (differences, order, how long ago relative to the current "
    "date). Say you don't know only if a fact you need is missing."
)

def default_limit(budget_tokens: int) -> int:
    """Candidate hits for a budget: more room, more candidate sessions."""
    return max(20, budget_tokens // 250)


_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_DATE_FORMATS = (
    "%Y/%m/%d (%a) %H:%M",  # LongMemEval haystack dates
    "%Y/%m/%d (%a)",
    "%Y-%m-%d (%a)",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
    "%Y-%m-%d",
)
_FACT_SOURCES = ("kg", "kg_expansion")


def estimate_tokens(text: str) -> int:
    """Conservative token estimate: characters / 3.5, rounded up."""
    return math.ceil(len(text) / 3.5)


def default_budget_tokens(
    window_tokens: int,
    *,
    used_tokens: int = 0,
    answer_reserve: int = 512,
    share: float = 0.4,
) -> int:
    """Context budget for one turn, from the model window and what is already used.

    ``min(int(window * share), window - used - answer_reserve)``, floored at 0.

    ``share`` caps evidence at 40% of the window. ``used_tokens`` (system
    prompt, tools and conversation at turn start; about 22k for a typical
    agent) is subtracted first. At a 32,768-token window that leaves about 10k
    tokens of evidence for a non-reasoning reader (``default_budget_tokens(32768,
    used_tokens=22000)`` is 10256).
    """
    return max(
        0,
        min(int(window_tokens * share), window_tokens - used_tokens - answer_reserve),
    )


def _to_datetime(value) -> datetime | None:
    """Best-effort parse of an epoch, ISO string or LongMemEval date, as UTC."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        if value <= 0:
            return None
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                return _to_datetime(float(text))
            except ValueError:
                return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def _fmt_date(dt: datetime) -> str:
    dt = dt.astimezone(timezone.utc)
    return f"{dt.strftime('%Y-%m-%d')} ({_DAYS[dt.weekday()]})"


def _as_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


class _Turn:
    __slots__ = ("role", "text", "index", "explicit_dt", "stored_dt",
                 "session_id", "vector_id", "archive_id")

    def __init__(self, role, text, index, explicit_dt, stored_dt,
                 session_id=None, vector_id=None, archive_id=None):
        self.role = role or "unknown"
        self.text = text
        self.index = index
        self.explicit_dt = explicit_dt  # session_date / timestamp the writer gave
        self.stored_dt = stored_dt  # when the row was stored
        self.session_id = session_id
        self.vector_id = vector_id  # vector row id, when known
        self.archive_id = archive_id  # archive line id, when known

    def ids(self) -> dict:
        return {
            "vector_id": self.vector_id,
            "archive_id": self.archive_id,
            "session_id": self.session_id,
            "turn_index": self.index,
        }

    def line(self) -> str:
        return f"{self.role}: {self.text}"


class _Block:
    kind = "session"

    def __init__(self, rank: int):
        self.rank = rank
        self.turns: list[_Turn] = []
        self.full: list[_Turn] | None = None  # every turn of the session, if known
        self.whole = False  # True once ``turns`` is the whole session
        self.fixed_text: str | None = None  # a block rendered by hand (facts)

    def date(self) -> datetime | None:
        explicit = [t.explicit_dt for t in self.turns if t.explicit_dt is not None]
        if explicit:
            return min(explicit)
        stored = [t.stored_dt for t in self.turns if t.stored_dt is not None]
        return min(stored) if stored else None

    def ids(self, count) -> dict:
        """The items in this block, for the ``included`` / ``dropped`` report."""
        dt = self.date()
        sid = next((t.session_id for t in self.turns if t.session_id is not None), None)
        return {
            "kind": self.kind,
            "session_id": sid,
            "date": _fmt_date(dt) if dt else None,
            "span": "whole" if self.whole else "window",
            "tokens": count(self.render()),
            "turns": [t.ids() for t in self.turns],
        }

    def render(self) -> str:
        if self.fixed_text is not None:
            return self.fixed_text
        dt = self.date()
        head = f"Session {_fmt_date(dt)}" if dt else "Session (date unknown)"
        return "\n".join([head] + [t.line() for t in self.turns])


def _turn_from_row(role, text, meta: dict, created_at, vector_id=None) -> _Turn:
    explicit = _to_datetime(meta.get("session_date")) or _to_datetime(meta.get("timestamp"))
    archive_id = meta.get("archive_span_id")
    sid = meta.get("session_id")
    return _Turn(
        role=meta.get("role"),
        text=text,
        index=_as_int(meta.get("turn_index")),
        explicit_dt=explicit,
        stored_dt=_to_datetime(created_at),
        session_id=sid if isinstance(sid, str) else None,
        vector_id=vector_id,
        archive_id=archive_id if isinstance(archive_id, int) else None,
    )


def _hit_view(hit: dict) -> tuple[str, dict]:
    """A hit's verbatim text and the metadata the layout reads.

    A vector hit already carries both. An archive hit holds an 80 character
    summary as its text and the verbatim turn (with its role and session
    fields) in ``data_json``, and names its agent ``agent_name``; both are
    lifted here so it takes part in session grouping like any other turn.
    """
    md = hit.get("metadata") or {}
    text = hit.get("text", "")
    if hit.get("source") != "archive":
        return text, md
    view: dict = {"agent": md.get("agent_name"), "archive_span_id": md.get("id")}
    try:
        data = json.loads(md.get("data_json") or "")
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict):
        text = str(data.get("content", text))
        view["role"] = data.get("role")
        view["timestamp"] = data.get("timestamp")
        view.update(_api._session_meta(data))
    return text, {k: v for k, v in view.items() if v is not None}


def _turn_from_hit(hit: dict) -> _Turn:
    text, md = _hit_view(hit)
    return _turn_from_row(md.get("role"), text, md, hit.get("timestamp"))


async def answer_context(
    query: str,
    *,
    agent: str,
    data_dir=None,
    budget_tokens: int,
    now=None,
    reasoning_reserve: int = 0,
    answer_reserve: int = 512,
    neighbors: int = 2,
    limit: int | None = None,
    tokenizer=None,
    include_facts: bool = False,
    project: str | None = None,
    also_include: list[str] | None = None,
    mode: str | None = None,
    collections: list[str] | None = None,
    collections_only: bool = False,
    grow: bool = True,
) -> dict:
    """Build the context an agent answers ``query`` from.

    Retrieval, scoping and access are those of :func:`taosmd.api.search`
    (``agent``, ``project``, ``also_include``, ``mode``, ``collections`` and
    ``collections_only`` mean what they mean there). Rows written before
    sessions were recorded have no turn order, so their hit stands alone.

    Selection is coverage before depth. Candidates are the hits (``limit``,
    default ``max(20, budget_tokens // 250)``), grouped by session and ranked by
    each session's best hit. Pass 1 adds, for every candidate session in rank
    order, its hit turns plus ``neighbors`` turns either side, if that fits.
    Pass 2 then goes through the included sessions again in rank order and
    grows each to the whole session (every turn, original order) when the whole
    session fits in what is left. A session that does not fit whole keeps its
    window. ``grow=False`` skips pass 2.

    Layout, one string: ``Current date: <YYYY-MM-DD (Ddd)>``, a blank line, then
    one block per session, oldest first, blocks separated by one blank line.
    Each block is ``Session <YYYY-MM-DD (Ddd)>`` (or ``Session (date unknown)``)
    followed by one ``role: text`` line per turn in original order.

    Budget, in tokens: ``available = budget_tokens - answer_reserve -
    reasoning_reserve - tokens(guidance)``. ``tokenizer`` (``str -> int``) counts
    when given, else ``ceil(chars / 3.5)``. Blocks are added in the rank order of
    their best hit; one that does not fit is dropped whole and the next is tried.
    ``tokens_used`` is the token count of the returned context and never exceeds
    ``budget_available``. When no block fits the context is the empty string.

    ``limit`` is the number of hits fetched.
    ``include_facts`` appends knowledge-graph fact hits as a final ``Known
    facts`` block; they are off by default.

    Returns ``{"context", "guidance", "sessions_included", "sessions_dropped",
    "hits_total", "tokens_used", "budget_available", "expanded", "included",
    "dropped"}``. ``included`` (in context order) and ``dropped`` (in rank order)
    list one entry per block: ``{"kind", "session_id", "date", "turns": [{"vector_id",
    "archive_id", "session_id", "turn_index"}, ...]}``. An id is ``None`` where it
    is not known (a hit that was not widened carries no vector row id).
    """
    if not agent:
        raise ValueError("agent name is required")
    if neighbors < 0:
        raise ValueError("neighbors must be >= 0")
    for name, value in (("budget_tokens", budget_tokens), ("answer_reserve", answer_reserve),
                        ("reasoning_reserve", reasoning_reserve)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if limit is None:
        limit = default_limit(budget_tokens)
    count = tokenizer if tokenizer is not None else estimate_tokens

    if now is None:
        now_dt = datetime.now(timezone.utc)
    else:
        now_dt = _to_datetime(now)
        if now_dt is None:
            raise ValueError(f"cannot read 'now' as a date: {now!r}")

    available = max(0, budget_tokens - answer_reserve - reasoning_reserve - count(GUIDANCE))

    hits = await _api.search(
        query,
        agent=agent,
        project=project,
        also_include=also_include,
        limit=limit,
        mode=mode,
        collections=collections,
        collections_only=collections_only,
        min_candidates=limit,
        data_dir=data_dir,
    )

    result = {
        "context": "",
        "guidance": GUIDANCE,
        "sessions_included": 0,
        "sessions_dropped": 0,
        "hits_total": len(hits),
        "tokens_used": 0,
        "budget_available": available,
        "expanded": False,
        "included": [],
        "dropped": [],
    }
    if not hits:
        return result

    fact_hits = [h for h in hits if h.get("source") in _FACT_SOURCES]
    turn_hits = [(i, h) for i, h in enumerate(hits) if h.get("source") not in _FACT_SOURCES]

    # Neighbours come through the vector store's own scoped loader (same project,
    # TTL and supersede rules as recall). They are grouped by (agent, session),
    # and only the agents that produced a hit are ever read, so a neighbour is
    # never from an agent search did not let the caller read.
    wanted: set[str] = set()
    for _, h in turn_hits:
        md = _hit_view(h)[1]
        if isinstance(md.get("session_id"), str) and _as_int(md.get("turn_index")) is not None:
            wanted.add(md["session_id"])
    sessions: dict[tuple, list[tuple[int, dict]]] = {}
    if wanted:
        stores = await _api._ensure_stores(data_dir)
        for row in stores["vector"].get_session_rows(wanted, project=project):
            meta = row["metadata"]
            ti = _as_int(meta.get("turn_index"))
            if ti is None:
                continue
            sessions.setdefault((meta.get("agent"), meta.get("session_id")), []).append((ti, row))
        for rows in sessions.values():
            rows.sort(key=lambda r: r[0])

    blocks: list[_Block] = []
    by_session: dict[tuple, _Block] = {}
    seen_turns: dict[tuple, set[int]] = {}
    expanded = False
    for rank, hit in turn_hits:
        md = _hit_view(hit)[1]
        sid, ti = md.get("session_id"), _as_int(md.get("turn_index"))
        rows = sessions.get((md.get("agent"), sid)) if ti is not None else None
        pos = next((p for p, (idx, _) in enumerate(rows or []) if idx == ti), None)
        if pos is None:
            block = _Block(rank)
            block.turns.append(_turn_from_hit(hit))
            blocks.append(block)
            continue
        key = (md.get("agent"), sid)
        block = by_session.get(key)
        if block is None:
            block = by_session[key] = _Block(rank)
            seen_turns[key] = set()
            blocks.append(block)
            block.full = [
                _turn_from_row(r["metadata"].get("role"), r["text"], r["metadata"],
                               r["created_at"], vector_id=r["id"])
                for _, r in rows
            ]
        block.rank = min(block.rank, rank)
        for idx, row in rows[max(0, pos - neighbors): pos + neighbors + 1]:
            if idx in seen_turns[key]:
                continue
            seen_turns[key].add(idx)
            block.turns.append(_turn_from_row(
                row["metadata"].get("role"), row["text"], row["metadata"], row["created_at"],
                vector_id=row["id"]))
        block.turns.sort(key=lambda t: t.index)
        expanded = expanded or neighbors > 0

    # A standalone hit whose line is already inside another block adds nothing.
    in_sessions = {t.line() for b in by_session.values() for t in (b.full or b.turns)}
    blocks = [b for b in blocks
              if b in by_session.values() or any(t.line() not in in_sessions for t in b.turns)]

    facts_block = None
    if include_facts and fact_hits:
        facts_block = _Block(len(hits))
        facts_block.kind = "facts"
        facts_block.fixed_text = "\n".join(["Known facts"] + [h.get("text", "") for h in fact_hits])
        blocks.append(facts_block)

    header = f"Current date: {_fmt_date(now_dt)}"

    def fits(candidate: list[_Block]) -> bool:
        return count(_layout(header, candidate, facts_block)) <= available

    # Pass 1, coverage: every candidate session gets its window before any
    # session is grown.
    ranked = sorted(blocks, key=lambda b: b.rank)
    chosen: list[_Block] = []
    dropped: list[_Block] = []
    for block in ranked:
        if fits(chosen + [block]):
            chosen.append(block)
        else:
            dropped.append(block)

    # Pass 2, depth: with what is left, grow sessions to the whole session in
    # rank order. A session that does not fit whole keeps its window.
    if grow:
        for block in chosen:
            if block.full is None or len(block.full) <= len(block.turns):
                continue
            window = block.turns
            block.turns, block.whole = block.full, True
            if not fits(chosen):
                block.turns, block.whole = window, False

    for block in chosen:
        if block.full is not None and len(block.full) == len(block.turns):
            block.whole = True

    if chosen:
        context = _layout(header, chosen, facts_block)
        result["context"] = context
        result["tokens_used"] = count(context)
    result["sessions_included"] = len(chosen)
    result["sessions_dropped"] = len(dropped)
    result["expanded"] = expanded
    result["included"] = [b.ids(count) for b in _in_layout_order(chosen, facts_block)]
    result["dropped"] = [b.ids(count) for b in dropped]
    return result


def _in_layout_order(chosen, facts_block):
    """Session blocks oldest first (undated first), the facts block last."""
    def key(block):
        dt = block.date()
        return (dt is not None, dt or datetime.min.replace(tzinfo=timezone.utc), block.rank)

    sessions = sorted((b for b in chosen if b is not facts_block), key=key)
    return sessions + [b for b in chosen if b is facts_block]


def _layout(header: str, chosen, facts_block) -> str:
    return "\n\n".join([header] + [b.render() for b in _in_layout_order(chosen, facts_block)])
