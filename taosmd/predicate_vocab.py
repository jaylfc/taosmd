"""Closed predicate vocabulary for taOSmd's knowledge graph.

Why: the legacy KG accepts any string as a predicate ("works_on",
"runs_on", "is_a", "decided", "discovered", etc.). Free-form predicates
are easy to write but pathological to query: "who does Alice work for?"
returns nothing if her employment was stored as "works on", "employed by",
"works at", or "joined" depending on the extractor's mood.

This module lifts gbrain's closed-vocab pattern (link-extraction.ts:462-632)
into Python. Three things ship:

  1. ``ALLOWED_PREDICATES``: the closed enum. Adding a predicate to the
     KG that isn't in this set still works (backwards-compat), but the
     write goes through ``validate`` which can log / refuse depending on
     mode.

  2. ``RELATIONSHIP_PATTERNS``: regex set that maps sentence shapes onto
     canonical predicates. Lifted directly from gbrain WORKS_AT_RE etc.
     plus the existing memory_extractor patterns, now ALL emitting
     predicates from the closed vocab.

  3. ``INVERSE_PREDICATES``: gbrain's "direction flag instead of paired
     predicates" pattern. Each predicate has a direction ('forward' or
     'inverse') so a frontmatter like ``key_people: [Pedro]`` on a
     company page knows to write ``pedro works_at stripe``, not
     ``stripe works_at pedro``.

The vocab starter set is gbrain's 12 + a handful from Schema.org Person
properties (knows, parent_of, spouse_of, birth_place) and the existing
SINGULAR_PREDICATES from knowledge_graph.py. Curate this list rather
than letting it grow free-form; that's the whole point.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Closed vocab
# ---------------------------------------------------------------------------

# Predicate categories: useful for prompt-template grouping and for the
# librarian's "what kind of fact is this" framing. Categories are
# documentation only; nothing branches on category at runtime.
PREDICATE_CATEGORIES: dict[str, str] = {
    # Employment / role
    "works_at": "employment",
    "founded": "employment",
    "advises": "employment",
    "partner_of": "employment",
    "led_round": "employment",
    "invested_in": "employment",
    # Personal relations (Schema.org Person)
    "knows": "social",
    "parent_of": "social",
    "spouse_of": "social",
    "sibling_of": "social",
    "member_of": "social",
    # Location
    "lives_in": "location",
    "birth_place": "location",
    "located_in": "location",
    # Activity / preference
    "prefers": "preference",
    "attended": "activity",
    "uses": "activity",
    "uses_model": "activity",
    # Discourse (catch-alls, use sparingly)
    "is_a": "discourse",
    "has": "discourse",
    "owns": "discourse",
    "mentions": "discourse",
    "related_to": "discourse",
    # Legacy from memory_extractor.RELATIONSHIP_PATTERNS, kept so existing
    # extractors don't trip the validator.
    "works_on": "employment",
    "runs_on": "activity",
    "managed_by": "employment",
    "owned_by": "employment",
    "created": "activity",
    "manages": "employment",
    "supports": "activity",
    "moved_to": "location",
    "depends_on": "activity",
    "monitors": "activity",
}

ALLOWED_PREDICATES: frozenset[str] = frozenset(PREDICATE_CATEGORIES)

# Common-typo / synonym map. Normalise BEFORE checking ALLOWED_PREDICATES so
# we don't reject obvious aliases. Keep this tight; the goal is the closed
# vocab, not a synonym explosion.
SYNONYMS: dict[str, str] = {
    "employed_by": "works_at",
    "employee_of": "works_at",
    "work_at": "works_at",
    "works_for": "works_at",
    "co_founded": "founded",
    "founder_of": "founded",
    "invests_in": "invested_in",
    "married_to": "spouse_of",
    "spouse": "spouse_of",
    "lives_at": "lives_in",
    "born_in": "birth_place",
    "born_at": "birth_place",
    "is": "is_a",
    "has_a": "has",
    "uses_the": "uses",
    "preferred": "prefers",
    "likes": "prefers",
}

# Direction policy for bidirectional facts, gbrain's pattern. When the
# librarian sees "Pedro is in the key_people list on Stripe's company page",
# the natural edge is `pedro -> works_at -> stripe`, not the reverse. The
# direction column tells consumers how to read the s/o slots for that
# predicate. Predicates not listed here are treated as "forward" (use s/o
# as written).
INVERSE_DIRECTION: frozenset[str] = frozenset({
    # When the SOURCE is the company / org page, flip s and o so the
    # predicate still reads s = person, o = company.
    "employs",
    "had_round_led_by",
    "advised_by",
})


# ---------------------------------------------------------------------------
# Regex inference: gbrain link-extraction.ts:462-511 + the existing
# memory_extractor.RELATIONSHIP_PATTERNS, all emitting closed-vocab
# predicates. Precedence top-to-bottom: more specific patterns first so
# they win over the discourse catch-alls.
# ---------------------------------------------------------------------------

_S = r"((?:[\w\-]+\s+){0,4}[\w\-]+)"        # 1-5-word subject
_O = r"([\w][\w\s\-]{1,50}?)"               # object phrase, non-greedy
_ART = r"(?:the |a |an )?"
_END = r"(?:\.|,|;|$)"

RELATIONSHIP_PATTERNS: list[tuple[str, str]] = [
    # --- employment / role (gbrain-derived) ---
    (rf"{_S}\s+(?:works? at|joined|employed by|is at|is an employee of)\s+{_ART}{_O}{_END}", "works_at"),
    (rf"{_S}\s+(?:co-?founded|founded|started|launched)\s+{_ART}{_O}{_END}", "founded"),
    (rf"{_S}\s+(?:advises?|advisor (?:to|of))\s+{_ART}{_O}{_END}", "advises"),
    (rf"{_S}\s+(?:invested in|backed|funded|put money in)\s+{_ART}{_O}{_END}", "invested_in"),
    (rf"{_S}\s+(?:led (?:the |a )?round (?:in|for)|led)\s+{_ART}{_O}{_END}", "led_round"),
    (rf"{_S}\s+(?:partner (?:at|of)|partners with)\s+{_ART}{_O}{_END}", "partner_of"),

    # --- activity / events ---
    (rf"{_S}\s+(?:attended|went to|was at|spoke at)\s+{_ART}{_O}{_END}", "attended"),

    # --- social (Schema.org Person) ---
    (rf"{_S}\s+(?:knows|met|introduced to)\s+{_ART}{_O}{_END}", "knows"),
    # Don't include bare "married", too ambiguous with adjective use
    # ("Henrietta is married" with no object). Require an explicit
    # "to <person>" or "spouse of" form.
    (rf"{_S}\s+(?:is married to|spouse of)\s+{_ART}{_O}{_END}", "spouse_of"),
    (rf"{_S}\s+(?:is the parent of|parent of|father of|mother of)\s+{_ART}{_O}{_END}", "parent_of"),
    (rf"{_S}\s+(?:is a member of|member of|joined as a member of)\s+{_ART}{_O}{_END}", "member_of"),

    # --- location ---
    (rf"{_S}\s+(?:lives in|resides in|based in)\s+{_ART}{_O}{_END}", "lives_in"),
    (rf"{_S}\s+(?:was born in|born in)\s+{_ART}{_O}{_END}", "birth_place"),
    (rf"{_S}\s+(?:moved? to|switched? to|migrated? to|relocated to)\s+{_ART}{_O}{_END}", "moved_to"),

    # --- preference / activity (legacy patterns kept, mapped to closed vocab) ---
    (rf"{_S}\s+(?:prefers?|favou?rs?)\s+(?:running |using )?{_ART}{_O}{_END}", "prefers"),
    (rf"{_S}\s+(?:uses?|runs?|runs on)\s+{_ART}{_O}{_END}", "uses"),
    (rf"{_S}\s+(?:works? on|is working on)\s+{_ART}{_O}{_END}", "works_on"),
    (rf"{_S}\s+(?:created?|built|made|developed|wrote)\s+{_ART}{_O}{_END}", "created"),
    (rf"{_S}\s+(?:manages?|maintains?)\s+{_ART}{_O}{_END}", "manages"),
    (rf"{_S}\s+(?:owns?)\s+{_ART}{_O}{_END}", "owns"),
    (rf"{_S}\s+(?:has|have|includes?|contains?|features?)\s+{_ART}{_O}{_END}", "has"),
    (rf"{_S}\s+(?:supports?)\s+{_ART}{_O}{_END}", "supports"),
    (rf"{_S}\s+(?:depends? on|requires?|needs?)\s+{_ART}{_O}{_END}", "depends_on"),
    (rf"{_S}\s+(?:monitors?|tracks?|watches?)\s+{_ART}{_O}{_END}", "monitors"),

    # --- discourse (catch-all, lowest priority) ---
    # Require an explicit article between "is/are" and the object so we
    # don't match copular adjective forms ("is married", "is happy").
    (rf"{_S}\s+(?:is|are)\s+(?:a|an|the)\s+{_O}{_END}", "is_a"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalise(predicate: str) -> str:
    """Lower-case + underscore + synonym-resolve a predicate string.

    Doesn't raise on unknown; that's :func:`validate`'s job. Use this when
    you want a canonical form regardless of vocab membership.
    """
    if not predicate:
        return ""
    key = predicate.strip().lower().replace(" ", "_").replace("-", "_")
    return SYNONYMS.get(key, key)


def is_allowed(predicate: str) -> bool:
    return normalise(predicate) in ALLOWED_PREDICATES


def validate(predicate: str, *, strict: bool = False) -> str:
    """Return the canonical predicate, or raise/log if it isn't in the vocab.

    Strict mode raises ``ValueError`` so callers (typically test code or
    a librarian running in strict-write mode) can refuse the write. The
    default (non-strict) logs a warning and returns the predicate anyway,
    preserving backwards-compat for existing free-form data.
    """
    canon = normalise(predicate)
    if not canon:
        if strict:
            raise ValueError("empty predicate")
        return canon
    if canon not in ALLOWED_PREDICATES:
        msg = (
            f"predicate {predicate!r} (canonical {canon!r}) is not in the "
            "closed vocab; see taosmd/predicate_vocab.py"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
    return canon


def categories() -> dict[str, list[str]]:
    """Group predicates by their category (employment/social/etc)."""
    by_cat: dict[str, list[str]] = {}
    for pred, cat in PREDICATE_CATEGORIES.items():
        by_cat.setdefault(cat, []).append(pred)
    for cat in by_cat:
        by_cat[cat].sort()
    return by_cat


def extract_with_vocab(text: str) -> list[dict]:
    """Regex-extract (subject, predicate, object) triples, vocab-constrained.

    Output predicates are guaranteed to be in ``ALLOWED_PREDICATES``. Same
    interface as ``memory_extractor.extract_facts_from_text``; callers
    can swap one for the other.
    """
    # Local import to keep this module free of taosmd-package import cycles
    # if someone uses the vocab module standalone.
    from .memory_extractor import _clean_entity, _split_sentences, _strip_leading_article

    facts: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for sentence in _split_sentences(text):
        for pattern, predicate in RELATIONSHIP_PATTERNS:
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                subject = _clean_entity(_strip_leading_article(match.group(1)))
                obj = _clean_entity(_strip_leading_article(match.group(2)))
                if not subject or not obj:
                    continue
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


# ---------------------------------------------------------------------------
# Singular predicates: for contradiction detection
# ---------------------------------------------------------------------------

# Predicates where only one active value per subject makes semantic sense.
# Any (subject, predicate, new_object) triple where the same subject already
# has a *different* active object for the same predicate is a contradiction.
# Keep this set a subset of ALLOWED_PREDICATES and curate deliberately;
# adding a predicate here means the detector will flag duplicates for it.
SINGULAR_PREDICATES: frozenset[str] = frozenset({
    "is_a",
    "works_on",
    "lives_in",
    "prefers",
    "uses_model",
    "runs_on",
    "managed_by",
    "owned_by",
    "located_in",
    "works_at",
    "moved_to",
    "birth_place",
})

__all__ = [
    "ALLOWED_PREDICATES",
    "PREDICATE_CATEGORIES",
    "SINGULAR_PREDICATES",
    "SYNONYMS",
    "INVERSE_DIRECTION",
    "RELATIONSHIP_PATTERNS",
    "normalise",
    "is_allowed",
    "validate",
    "categories",
    "extract_with_vocab",
]
