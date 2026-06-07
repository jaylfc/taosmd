"""Context Window Assembler (taOSmd).

Inspired by MemPalace's L0-L3 layer system and Letta's virtual context memory,
with a core/archival split inspired by agentmemory.

Layers:
  L0: Identity — who is the agent, who is the user (~100 tokens, always loaded)
  L1: Active facts — current KG facts about the user + project (~200 tokens)
      Split into CORE (pinned, always present) and ARCHIVAL (paged by retention score)
  L2: Relevant memories — semantic search results from recent context (~500 tokens)
  L3: Deep recall — archive search, full KG traversal (on-demand, ~1000 tokens)

Core/Archival Split:
  Core memories (pinned=True) always get 30% of the L1 budget.
  Archival memories are scored by retention and paged by score * recency.
  When core exceeds its budget, lowest-scored core items are auto-demoted.

Total context budget is configurable. Each layer has a soft token limit.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .knowledge_graph import TemporalKnowledgeGraph
    from .archive import ArchiveStore
    from .session_catalog import SessionCatalog

logger = logging.getLogger(__name__)

# Rough token estimation. A flat 4 chars/token under-counts dense content
# (code, JSON) and badly under-counts CJK, where each glyph is roughly its
# own token. We keep a cheap, deterministic, dependency-free heuristic:
#   - CJK glyphs count as ~1 token each
#   - punctuation/symbol-heavy text uses fewer chars/token (denser)
#   - plain prose uses the classic ~4 chars/token
CHARS_PER_TOKEN = 4          # prose default (kept public/back-compatible)
CHARS_PER_TOKEN_DENSE = 3    # code / JSON / symbol-heavy text
_DENSE_PUNCT_RATIO = 0.18    # punctuation share above which text is "dense"


def _is_cjk(ch: str) -> bool:
    """True for CJK / Japanese / Korean glyphs (roughly one token each)."""
    cp = ord(ch)
    return (
        0x3000 <= cp <= 0x303F        # CJK symbols and punctuation
        or 0x3040 <= cp <= 0x30FF     # Hiragana + Katakana
        or 0x3400 <= cp <= 0x4DBF     # CJK Ext A
        or 0x4E00 <= cp <= 0x9FFF     # CJK Unified Ideographs
        or 0xF900 <= cp <= 0xFAFF     # CJK Compatibility Ideographs
        or 0xFF00 <= cp <= 0xFFEF     # Halfwidth/Fullwidth forms
        or 0xAC00 <= cp <= 0xD7AF     # Hangul syllables
        or 0x20000 <= cp <= 0x2FA1F   # CJK Ext B-F + supplement
    )


def _chars_per_token(non_cjk_text: str) -> int:
    """Pick chars-per-token for the non-CJK part based on symbol density."""
    if not non_cjk_text:
        return CHARS_PER_TOKEN
    punct = sum(1 for ch in non_cjk_text if not ch.isalnum() and not ch.isspace())
    if punct / len(non_cjk_text) >= _DENSE_PUNCT_RATIO:
        return CHARS_PER_TOKEN_DENSE
    return CHARS_PER_TOKEN


def estimate_tokens(text: str) -> int:
    """Content-aware token count estimate (heuristic, no tokenizer).

    CJK glyphs are counted as ~1 token each; the remaining text is divided
    by a chars-per-token factor that shrinks for punctuation/symbol-heavy
    content (code, JSON) and stays at ~4 for prose.
    """
    if not text:
        return 0
    cjk = 0
    other_chars: list[str] = []
    for ch in text:
        if _is_cjk(ch):
            cjk += 1
        else:
            other_chars.append(ch)
    non_cjk = "".join(other_chars)
    return cjk + len(non_cjk) // _chars_per_token(non_cjk)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens.

    Uses the same content-aware estimate as ``estimate_tokens`` so that a
    truncated string lands near ``max_tokens`` for both prose and dense or
    CJK content. Returns ``text`` unchanged when it already fits.
    """
    if estimate_tokens(text) <= max_tokens:
        return text
    # Walk the string accumulating the estimated token cost char-by-char,
    # stopping just before the budget is exceeded. Single pass, O(n).
    chars_per_token = _chars_per_token("".join(ch for ch in text if not _is_cjk(ch)))
    tokens = 0
    non_cjk_run = 0
    cut = 0
    for i, ch in enumerate(text):
        if _is_cjk(ch):
            tokens += 1
        else:
            non_cjk_run += 1
            if non_cjk_run == chars_per_token:
                tokens += 1
                non_cjk_run = 0
        if tokens > max_tokens:
            break
        cut = i + 1
    if cut <= 0:
        cut = 1
    return text[:cut] + "..."


class ContextAssembler:
    """Assembles agent context from multiple memory layers."""

    def __init__(
        self,
        kg: TemporalKnowledgeGraph | None = None,
        archive: ArchiveStore | None = None,
        qmd_base_url: str = "http://localhost:7832",
        http_client=None,
        catalog: SessionCatalog | None = None,
    ):
        self._kg = kg
        self._archive = archive
        self._qmd_url = qmd_base_url
        self._http = http_client
        self._catalog = catalog

    # ------------------------------------------------------------------
    # L0: Identity
    # ------------------------------------------------------------------

    async def assemble_l0(
        self,
        agent_name: str | None = None,
        user_name: str | None = None,
        system_info: dict | None = None,
        max_tokens: int = 100,
    ) -> str:
        """Always-loaded identity context. ~100 tokens."""
        parts = []

        if agent_name:
            parts.append(f"You are {agent_name}, an AI agent running on taOS.")
        if user_name:
            parts.append(f"Your user is {user_name}.")
        if system_info:
            hw = []
            if system_info.get("cpu"):
                hw.append(system_info["cpu"])
            if system_info.get("npu"):
                hw.append(f"NPU: {system_info['npu']}")
            if system_info.get("ram"):
                hw.append(f"RAM: {system_info['ram']}")
            if hw:
                parts.append(f"Hardware: {', '.join(hw)}.")

        text = " ".join(parts)
        return truncate_to_tokens(text, max_tokens)

    # ------------------------------------------------------------------
    # L1: Active Knowledge Graph Facts (with core/archival split)
    # ------------------------------------------------------------------

    async def assemble_l1(
        self,
        user_name: str | None = None,
        agent_name: str | None = None,
        project: str | None = None,
        max_tokens: int = 200,
        pinned_entities: list[str] | None = None,
    ) -> str:
        """Current facts from the knowledge graph with core/archival split.

        Core facts (from pinned_entities or user/agent) get 30% of budget
        and are always included. Remaining facts are archival, scored by
        importance (hit_rate) * recency and paged within the remaining budget.
        """
        if not self._kg:
            return ""

        core_budget = int(max_tokens * 0.3)
        archival_budget = max_tokens - core_budget

        # Core entities: pinned + user + agent (always loaded)
        core_entities = list(pinned_entities or [])
        if user_name and user_name not in core_entities:
            core_entities.append(user_name)
        if agent_name and agent_name not in core_entities:
            core_entities.append(agent_name)

        # Archival entities: project and anything else
        archival_entities = []
        if project and project not in core_entities:
            archival_entities.append(project)

        # Assemble core facts
        core_parts = []
        for entity in core_entities:
            try:
                results = await self._kg.query_entity(entity, direction="outgoing")
                for r in results[:5]:
                    core_parts.append(f"{entity} {r['predicate']} {r.get('object_name', '?')}")
            except Exception:
                pass

        # Assemble archival facts, scored by importance
        archival_scored = []
        for entity in archival_entities:
            try:
                results = await self._kg.query_entity(entity, direction="outgoing")
                for r in results:
                    importance = r.get("importance", 0)
                    text = f"{entity} {r['predicate']} {r.get('object_name', '?')}"
                    archival_scored.append((importance, text))
            except Exception:
                pass

        # Also gather archival from core entities beyond their top 5
        for entity in core_entities:
            try:
                results = await self._kg.query_entity(entity, direction="outgoing")
                for r in results[5:]:  # Beyond the core top-5
                    importance = r.get("importance", 0)
                    text = f"{entity} {r['predicate']} {r.get('object_name', '?')}"
                    archival_scored.append((importance, text))
            except Exception:
                pass

        # Sort archival by importance descending
        archival_scored.sort(key=lambda x: x[0], reverse=True)

        # Catalog session context for timeline queries
        if self._catalog and project:
            try:
                catalog_results = await self._catalog.search_topic(project, limit=3)
                for r in catalog_results:
                    text = f"[{r.get('date', '')} {r.get('start_str', '')}-{r.get('end_str', '')}] {r['topic']}"
                    if r.get("description"):
                        text += f": {r['description']}"
                    archival_scored.append((0.5, text))
            except Exception:
                pass

        archival_parts = [text for _, text in archival_scored]

        # Build output respecting budgets
        sections = []
        if core_parts:
            core_text = "Core facts:\n" + "\n".join(f"- {p}" for p in core_parts)
            sections.append(truncate_to_tokens(core_text, core_budget))

        if archival_parts:
            arch_text = "Known facts:\n" + "\n".join(f"- {p}" for p in archival_parts)
            sections.append(truncate_to_tokens(arch_text, archival_budget))

        return "\n".join(sections) if sections else ""

    # ------------------------------------------------------------------
    # L2: Relevant Recent Context
    # ------------------------------------------------------------------

    async def assemble_l2(
        self,
        query: str,
        agent_name: str | None = None,
        max_tokens: int = 500,
    ) -> str:
        """Semantic search + recent archive for the current query. ~500 tokens."""
        parts = []

        # Search archive for recent relevant events
        if self._archive:
            try:
                events = await self._archive.query(
                    agent_name=agent_name,
                    search=query,
                    limit=5,
                )
                for e in events:
                    summary = e.get("summary", "")
                    if summary:
                        parts.append(f"[{e['event_type']}] {summary}")
            except Exception:
                pass

        # Search KG for related entities
        if self._kg:
            words = query.split()
            for word in words:
                if len(word) < 3:
                    continue
                try:
                    results = await self._kg.query_entity(word)
                    for r in results[:2]:
                        obj = r.get("object_name") or r.get("subject_name", "")
                        parts.append(f"{word} {r['predicate']} {obj}")
                except Exception:
                    pass

        # Catalog search for timeline-relevant content
        if self._catalog:
            try:
                catalog_results = await self._catalog.search_topic(query, limit=3)
                for r in catalog_results:
                    ctx = await self._catalog.get_session_context(r["id"], max_lines=10)
                    if ctx and ctx.get("content_lines"):
                        summary = " ".join(ctx["content_lines"][:5])[:300]
                        parts.append(f"[session:{r.get('date','')} {r.get('start_str','')}] {summary}")
            except Exception:
                pass

        # QMD semantic search
        if self._http and self._qmd_url:
            try:
                resp = await self._http.post(
                    f"{self._qmd_url}/vsearch",
                    json={"query": query, "limit": 3, "collection": "workspace"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    for r in results:
                        snippet = (r.get("content", "") or "")[:200]
                        if snippet:
                            parts.append(f"[memory] {snippet}")
            except Exception:
                pass

        if not parts:
            return ""

        text = "Relevant context:\n" + "\n".join(f"- {p}" for p in parts)
        return truncate_to_tokens(text, max_tokens)

    # ------------------------------------------------------------------
    # L3: Deep Recall
    # ------------------------------------------------------------------

    async def assemble_l3(
        self,
        query: str,
        agent_name: str | None = None,
        max_tokens: int = 1000,
    ) -> str:
        """Deep search across all memory layers. ~1000 tokens. Expensive."""
        parts = []

        # Full archive search (more results)
        if self._archive:
            try:
                events = await self._archive.query(search=query, limit=10)
                for e in events:
                    data_str = e.get("data_json", "{}")
                    try:
                        data = json.loads(data_str) if isinstance(data_str, str) else data_str
                    except (json.JSONDecodeError, TypeError):
                        data = {}
                    content = data.get("content", data.get("text", data.get("msg", "")))
                    if content:
                        parts.append(f"[archive:{e['event_type']}] {str(content)[:150]}")
                    elif e.get("summary"):
                        parts.append(f"[archive:{e['event_type']}] {e['summary']}")
            except Exception:
                pass

        # KG timeline
        if self._kg:
            try:
                timeline = await self._kg.timeline(limit=10)
                for t in timeline:
                    parts.append(
                        f"[fact] {t.get('subject_name', '?')} {t['predicate']} {t.get('object_name', '?')}"
                    )
            except Exception:
                pass

        # Extended QMD search
        if self._http and self._qmd_url:
            try:
                resp = await self._http.post(
                    f"{self._qmd_url}/vsearch",
                    json={"query": query, "limit": 5, "collection": "workspace"},
                    timeout=15,
                )
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    for r in results:
                        snippet = (r.get("content", "") or "")[:300]
                        if snippet:
                            parts.append(f"[deep-memory] {snippet}")
            except Exception:
                pass

        if not parts:
            return ""

        text = "Deep recall:\n" + "\n".join(f"- {p}" for p in parts)
        return truncate_to_tokens(text, max_tokens)

    # ------------------------------------------------------------------
    # Full assembly
    # ------------------------------------------------------------------

    async def assemble(
        self,
        query: str,
        agent_name: str | None = None,
        user_name: str | None = None,
        project: str | None = None,
        system_info: dict | None = None,
        depth: str = "standard",  # "minimal" | "standard" | "deep" | "auto"
        max_total_tokens: int = 1800,
        retrieval_results: list[dict] | None = None,  # NEW
    ) -> dict:
        """Assemble the full context from all layers.

        When depth="auto", uses intent classification to determine which
        layers to prioritise and how much budget to allocate to each.

        Returns {context: str, layers: {l0, l1, l2, l3}, tokens: int, depth: str, intent: str}
        """
        t0 = time.time()
        layers = {}
        intent = "general"

        # Intent-aware depth selection
        if depth == "auto":
            from .intent_classifier import get_search_strategy
            strategy = get_search_strategy(query)
            intent = strategy["intent"]
            # Adjust token budgets based on intent weights
            kg_budget = int(200 * strategy["kg_weight"])
            archive_budget = int(500 * strategy["archive_weight"])
            qmd_budget = int(500 * strategy["qmd_weight"])
            depth = "deep"  # Always do deep search in auto mode, but weighted
        else:
            kg_budget = 200
            archive_budget = 500
            qmd_budget = 500

        # L0 always loaded
        layers["l0"] = await self.assemble_l0(agent_name, user_name, system_info, max_tokens=100)

        # L1 always loaded (KG facts, budget adjusted by intent)
        layers["l1"] = await self.assemble_l1(user_name, agent_name, project, max_tokens=kg_budget)

        if depth in ("standard", "deep"):
            if retrieval_results is not None:
                # Use pre-retrieved results instead of querying sources
                parts = []
                for r in retrieval_results[:10]:
                    text = r.get("text", "")[:200]
                    source = r.get("source", "unknown")
                    parts.append(f"[{source}] {text}")
                l2_text = "Relevant context:\n" + "\n".join(f"- {p}" for p in parts) if parts else ""
                layers["l2"] = truncate_to_tokens(l2_text, max(archive_budget, qmd_budget))
            else:
                layers["l2"] = await self.assemble_l2(query, agent_name, max_tokens=max(archive_budget, qmd_budget))
        else:
            layers["l2"] = ""

        if depth == "deep":
            layers["l3"] = await self.assemble_l3(query, agent_name, max_tokens=1000)
        else:
            layers["l3"] = ""

        # Combine and respect total budget
        sections = [v for v in layers.values() if v]
        full_context = "\n\n".join(sections)
        full_context = truncate_to_tokens(full_context, max_total_tokens)

        tokens = estimate_tokens(full_context)
        latency_ms = round((time.time() - t0) * 1000, 1)

        return {
            "context": full_context,
            "layers": {k: estimate_tokens(v) for k, v in layers.items()},
            "total_tokens": tokens,
            "depth": depth,
            "intent": intent,
            "latency_ms": latency_ms,
        }
