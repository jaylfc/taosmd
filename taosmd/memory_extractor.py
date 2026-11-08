"""Automatic Memory Extraction (taOSmd).

Extracts structured facts from conversation text and stores them in the
Temporal Knowledge Graph. Runs after each agent conversation turn to
capture knowledge without explicit user action.

Uses lightweight keyword/pattern matching for speed (no LLM call needed).
Can optionally call an LLM for higher-quality extraction when available.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinyagentos.temporal_knowledge_graph import TemporalKnowledgeGraph
    from tinyagentos.archive import ArchiveStore

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Pattern-based fact extraction (no LLM, runs on Pi instantly)
# ------------------------------------------------------------------

# Patterns that indicate relationships
# Subject pattern: anchored more tightly, 1 to 5 words before the verb
_S = r"((?:[\w\-]+\s+){0,4}[\w\-]+)"
# Object pattern: everything after the verb until sentence end
_O = r"([\w][\w\s\-]{1,50}?)"
# Optional article
_ART = r"(?:the |a |an )?"

RELATIONSHIP_PATTERNS = [
    # "X is a Y" / "X is Y"
    (rf"{_S}\s+(?:is|are)\s+{_ART}{_O}(?:\.|,|$)", "is_a"),
    # "X uses Y" / "X runs Y" / "X runs on Y"
    (rf"{_S}\s+(?:uses?|runs?|runs on)\s+{_ART}{_O}(?:\.|,|$)", "uses"),
    # "X prefers Y" / "X prefers running Y"
    (rf"{_S}\s+(?:prefers?|likes?|favou?rs?)\s+(?:running |using )?{_ART}{_O}(?:\.|,|$)", "prefers"),
    # "X created Y" / "X built Y"
    (rf"{_S}\s+(?:created?|built|made|developed|wrote)\s+{_ART}{_O}(?:\.|,|$)", "created"),
    # "X works on Y"
    (rf"{_S}\s+(?:works? on|is working on|working on)\s+{_ART}{_O}(?:\.|,|$)", "works_on"),
    # "X manages Y"
    (rf"{_S}\s+(?:manages?|owns?|maintains?)\s+{_ART}{_O}(?:\.|,|$)", "manages"),
    # "X has Y" / "X includes Y"
    (rf"{_S}\s+(?:has|have|includes?|contains?|features?)\s+{_ART}{_O}(?:\.|,|$)", "has"),
    # "X supports Y"
    (rf"{_S}\s+(?:supports?)\s+{_ART}{_O}(?:\.|,|$)", "supports"),
    # "X moved to Y"
    (rf"{_S}\s+(?:moved? to|switched? to|migrated? to)\s+{_ART}{_O}(?:\.|,|$)", "moved_to"),
    # "X depends on Y"
    (rf"{_S}\s+(?:depends? on|requires?|needs?)\s+{_ART}{_O}(?:\.|,|$)", "depends_on"),
    # "X monitors Y"
    (rf"{_S}\s+(?:monitors?|tracks?|watches?)\s+{_ART}{_O}(?:\.|,|$)", "monitors"),
]

# Words to skip as subjects/objects (too generic)
SKIP_WORDS = {
    "i", "we", "you", "they", "he", "she", "it", "this", "that", "these",
    "those", "the", "a", "an", "some", "all", "any", "each", "every",
    "my", "your", "our", "their", "its", "there", "here", "now", "then",
    "also", "just", "only", "very", "much", "more", "less", "most", "least",
    "really", "actually", "basically", "probably", "maybe", "perhaps",
    "something", "anything", "everything", "nothing", "someone", "anyone",
    "what", "which", "who", "whom", "where", "when", "how", "why",
}

# Minimum word length for entities
MIN_ENTITY_LEN = 2


def _clean_entity(text: str) -> str | None:
    """Clean and validate an extracted entity name."""
    text = text.strip().strip(".,;:!?\"'()[]{}").strip()
    # Truncate at conjunctions/relative pronouns (captures cleaner entities)
    for stop in (" which ", " who ", " that ", " for the ", " rather ", " instead ", " including "):
        idx = text.lower().find(stop)
        if idx > 0:
            text = text[:idx].strip()
    # Strip trailing prepositions
    for prep in (" on", " in", " at", " to", " of", " with", " from", " by"):
        if text.lower().endswith(prep):
            text = text[:len(text) - len(prep)].strip()
    text = text.strip().strip(".,;:!?\"'()[]{}").strip()
    # Skip if too short or a stop word
    if len(text) < MIN_ENTITY_LEN:
        return None
    if text.lower() in SKIP_WORDS:
        return None
    # Skip if all lowercase single word that looks like a verb/adjective
    words = text.split()
    if len(words) == 1 and text.islower() and len(text) < 4:
        return None
    return text


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Handles ". " and newlines."""
    # Split on period+space, newlines, semicolons
    raw = re.split(r'(?<=[.!?])\s+|\n+|;\s*', text)
    return [s.strip() for s in raw if s.strip()]


def _strip_leading_article(text: str) -> str:
    """Remove leading 'The/A/An' from entity text."""
    return re.sub(r'^(?:The|the|A|a|An|an)\s+', '', text).strip()


def extract_facts_from_text(text: str) -> list[dict]:
    """Extract structured facts from free text using pattern matching.

    Splits text into sentences first, then applies patterns to each.
    Returns list of {subject, predicate, object} dicts.
    """
    facts = []
    seen = set()
    sentences = _split_sentences(text)

    for sentence in sentences:
        for pattern, predicate in RELATIONSHIP_PATTERNS:
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                subject = _clean_entity(_strip_leading_article(match.group(1)))
                obj = _clean_entity(_strip_leading_article(match.group(2)))
                if not subject or not obj:
                    continue
                # Deduplicate
                key = (subject.lower(), predicate, obj.lower())
                if key in seen:
                    continue
                seen.add(key)
                facts.append({
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                })

    return facts


# ------------------------------------------------------------------
# LLM-based extraction (higher quality, optional)
# ------------------------------------------------------------------

# (model, llm_url) pairs whose LLM-failure fallback has already been
# WARNING-logged this process. The first failure per pair is loud, repeats
# drop to debug so a dead Ollama doesn't flood the log on every turn.
_llm_fallback_warned: set[tuple[str, str]] = set()

# ------------------------------------------------------------------
# Extraction-method counters (issue #653 item 4)
# ------------------------------------------------------------------
# A silently permanent degrade to the regex path is the most dangerous
# failure mode in the claims field report: a shadow deployment produced
# 10,063/10,063 regex claims with only a debug log to show for it. These
# per-process counters make the llm-vs-regex split inspectable, and the
# consecutive-fallback streak makes a persistently dead LLM path loud
# once instead of once-per-process-then-debug.
# WITNESS: tests/test_memory_extractor_fallback_stats.py::consecutive regex fallback
_extraction_counts: dict[str, int] = {
    "llm": 0,             # LLM extraction produced the facts used
    "regex_fallback": 0,  # LLM failed or returned nothing; regex took over
    "regex_direct": 0,    # regex by design (no llm_url, or task gate off)
}
_consecutive_fallbacks = 0
_llm_inner_fallback = False

# Alert once per streak when this many consecutive turns fell back to regex.
_FALLBACK_ALERT_AFTER = 10


def extraction_stats() -> dict:
    """Snapshot of the per-process extraction-method counters.

    Returns ``{llm, regex_fallback, regex_direct, consecutive_regex_fallbacks}``.
    The persistent counterpart is ``ClaimStore.rate()['by_extractor']``, which
    ``taosmd claims status`` prints across restarts.
    """
    return {
        **_extraction_counts,
        "consecutive_regex_fallbacks": _consecutive_fallbacks,
    }


def _reset_extraction_stats() -> None:
    """Zero the counters (tests; nothing in production resets them)."""
    global _consecutive_fallbacks, _llm_inner_fallback
    for key in _extraction_counts:
        _extraction_counts[key] = 0
    _consecutive_fallbacks = 0
    _llm_inner_fallback = False


def _bump_llm() -> None:
    """One genuine LLM extraction: count it and reset the fallback streak."""
    global _consecutive_fallbacks
    _extraction_counts["llm"] += 1
    _consecutive_fallbacks = 0


def _bump_regex_fallback() -> None:
    """One silent degrade to regex: count it, extend the streak, alert if stuck.

    The per-call WARNING-once-then-debug rule in extract_facts_with_llm covers
    single failures; this streak alert covers the pattern -- a permanently
    dead LLM path -- so it stays loud even after the first-warning window.
    """
    global _consecutive_fallbacks
    _extraction_counts["regex_fallback"] += 1
    _consecutive_fallbacks += 1
    if _consecutive_fallbacks == _FALLBACK_ALERT_AFTER:
        logger.warning(
            "memory extraction: %d consecutive regex fallbacks "
            "(llm=%d, regex_fallback=%d, regex_direct=%d) -- the LLM "
            "extraction path is likely dead or unresolved; check the "
            "memory model config and the LLM endpoint",
            _consecutive_fallbacks,
            _extraction_counts["llm"],
            _extraction_counts["regex_fallback"],
            _extraction_counts["regex_direct"],
        )


async def extract_facts_with_llm(
    text: str,
    llm_url: str,
    http_client,
    *,
    agent_name: str = "default",
    model: str = "default",
) -> list[dict]:
    """Extract facts using an LLM for higher quality. Falls back to pattern matching.

    The prompt comes from :mod:`taosmd.prompts` so every LLM call across
    the library shares the same librarian persona and "never invent"
    rules. Pass ``agent_name`` so the persona binds to the right shelf.

    Every path that ends in the regex fallback bumps the extraction
    counters (see :func:`extraction_stats`); the true-LLM success path
    resets the consecutive-fallback streak. Counting lives here because
    the regex fallback happens inside this function: callers cannot tell
    an LLM answer from a silent degrade by inspecting the return value.
    """
    global _consecutive_fallbacks, _llm_inner_fallback
    if not llm_url:
        return extract_facts_from_text(text)
    _llm_inner_fallback = False

    from .prompts import extraction_prompt
    from .generator_profiles import split_provider  # noqa: PLC0415

    # Resolved model strings may carry a provider prefix ("ollama:qwen3.5:9b").
    # The backend wants the bare model name; the prefixed form is rejected by
    # live Ollama and would silently degrade extraction to regex.
    _, model_name = split_provider(model)
    try:
        resp = await http_client.post(
            f"{llm_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": extraction_prompt(text, agent_name=agent_name)}],
                "temperature": 0,
                "max_tokens": 500,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            # Try to parse JSON from the response
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            facts = json.loads(content)
            if isinstance(facts, list):
                return [f for f in facts if "subject" in f and "predicate" in f and "object" in f]
    except Exception as e:
        key = (model_name, llm_url)
        level = logging.DEBUG if key in _llm_fallback_warned else logging.WARNING
        _llm_fallback_warned.add(key)
        logger.log(
            level,
            "LLM extraction failed (model=%r, url=%s), falling back to regex "
            "pattern extraction: %s",
            model_name, llm_url, e,
        )

    # Fall-through here means: exception, non-200, or unparseable response.
    # Signal the caller so the regex degrade is counted, not mistaken for an
    # LLM success (issue #653 item 4).
    _llm_inner_fallback = True
    return extract_facts_from_text(text)


# ------------------------------------------------------------------
# Auto-extraction pipeline
# ------------------------------------------------------------------

async def process_conversation_turn(
    text: str,
    agent_name: str | None,
    kg: TemporalKnowledgeGraph,
    archive: ArchiveStore | None = None,
    source: str = "conversation",
    llm_url: str = "",
    http_client=None,
    use_llm: bool = True,
    auto_resolve_contradictions: bool = True,
    extraction_model: str = "default",
    pending_store=None,
    defer_below_confidence: float = 0.0,
    default_fact_confidence: float = 1.0,
    vmem=None,
) -> dict:
    """Extract facts from a conversation turn and store in the KG.

    Uses LLM extraction when available (higher quality), falls back to regex.
    Checks for contradictions before storing (Mem0-style ADD/UPDATE logic).

    Safety-net parameters (opt-in):
      - ``pending_store``: a :class:`PendingDecisionsStore` to defer
        low-confidence conflicts to. When None, the legacy auto-resolve
        path runs unchanged.
      - ``defer_below_confidence``: confidence threshold below which a
        contradiction is deferred to the pending queue instead of being
        auto-resolved. Default 0.0 -> never defer (legacy behaviour).
        Suggested value when ``pending_store`` is set: 0.8.
      - ``default_fact_confidence``: confidence assigned to each
        extracted fact when the extractor doesn't supply one. Default
        1.0 (treat agent / user-direct extraction as ground truth).
        Callers that extract from less-certain sources (a nightly
        catalog pass, a summary of a summary) should pass a lower
        value -- e.g. 0.6 for regex on a session summary -- so the
        deferral threshold can actually fire.
      - ``vmem``: an optional :class:`~taosmd.vector_memory.VectorMemory`.
        When provided, an auto-resolved contradiction also soft-supersedes
        the raw vector chunk(s) carrying the old object value, so a corrected
        fact stops resurfacing in vector recall as well as in the KG. When
        None (default), the vector store is left completely untouched.

    Returns ``{facts_extracted, triples_created, contradictions_resolved,
    deferred, deferred_ids, triple_ids, method}``.
    """
    from .agents import is_task_enabled  # noqa: PLC0415

    # Choose extraction method: LLM first, regex fallback
    # Gate: named agents with fact_extraction disabled fall back to regex.
    # Anonymous callers (agent_name is None or "") always run as before.
    method = "regex"
    if agent_name and not is_task_enabled(agent_name, "fact_extraction"):
        logger.debug(
            "process_conversation_turn: fact_extraction disabled for agent=%r, using regex",
            agent_name,
        )
        use_llm = False
    global _consecutive_fallbacks, _llm_inner_fallback
    if use_llm and llm_url and http_client:
        # The memory model is a system-wide setting. When the caller didn't
        # pin a model (unset or the literal "default" sentinel), defer to the
        # global config; if that's also unset, fall back to the prior
        # "default" behaviour so standalone installs are unchanged.
        from .config import resolve_memory_model  # noqa: PLC0415

        resolved_model = extraction_model
        if not resolved_model or resolved_model == "default":
            resolved_model = resolve_memory_model(agent=agent_name or "default") or "default"
        _llm_inner_fallback = False
        facts = await extract_facts_with_llm(
            text, llm_url, http_client,
            agent_name=agent_name or "default",
            model=resolved_model,
        )
        if _llm_inner_fallback:
            # The LLM call itself failed; extract_facts_with_llm already
            # degraded to regex internally. Count the degrade regardless of
            # whether those regex facts came back non-empty -- this is the
            # silent-fallback case the field report couldn't see.
            _bump_regex_fallback()
        elif facts:
            method = "llm"
            _bump_llm()
        else:
            # LLM answered but extracted nothing; regex retries the text.
            facts = extract_facts_from_text(text)
            _bump_regex_fallback()
    else:
        facts = extract_facts_from_text(text)
        _extraction_counts["regex_direct"] += 1

    triple_ids: list[str] = []
    deferred_ids: list[str] = []
    contradictions = 0

    for fact in facts:
        try:
            # Use contradiction-aware insertion for singular predicates.
            # When pending_store is provided AND the fact's confidence is
            # below defer_below_confidence, the new triple is queued for
            # user review instead of auto-resolving against the existing
            # fact; see taosmd/pending_decisions.py.
            fact_confidence = float(fact.get("confidence", default_fact_confidence))
            result = await kg.add_triple_with_contradiction_check(
                subject=fact["subject"],
                predicate=fact["predicate"],
                obj=fact["object"],
                confidence=fact_confidence,
                source=f"{source}:{agent_name or 'user'}",
                auto_resolve=auto_resolve_contradictions,
                pending_store=pending_store,
                defer_below_confidence=defer_below_confidence,
                evidence=text[:500],
                vmem=vmem,
            )
            if result.get("status") == "deferred":
                deferred_ids.append(result["pending_id"])
                continue
            if result.get("triple_id"):
                triple_ids.append(result["triple_id"])
            contradictions += result.get("contradictions_resolved", 0)
        except Exception as e:
            logger.debug("Failed to add triple: %s", e)

    # Archive the extraction event
    if archive and (triple_ids or deferred_ids):
        deferred_note = f", {len(deferred_ids)} deferred for review" if deferred_ids else ""
        await archive.record(
            event_type="memory_extraction",
            data={
                "facts_extracted": len(facts),
                "triples_created": len(triple_ids),
                "contradictions_resolved": contradictions,
                "deferred": len(deferred_ids),
                "deferred_ids": deferred_ids,
                "method": method,
                "source": source,
                "facts": facts,
            },
            agent_name=agent_name,
            summary=(
                f"Extracted {len(facts)} facts ({method}) from {source}, "
                f"{contradictions} contradictions resolved{deferred_note}"
            ),
        )

    return {
        "facts_extracted": len(facts),
        "triples_created": len(triple_ids),
        "contradictions_resolved": contradictions,
        "deferred": len(deferred_ids),
        "deferred_ids": deferred_ids,
        "triple_ids": triple_ids,
        "method": method,
    }
