"""Semantic categorisation of memories for the dashboard.

A small fixed taxonomy (the Memoire-style life-and-work buckets) plus a
deterministic, LLM-free keyword classifier. This is the always-available
default and the non-enriched fallback.

# upgrade-path: two refinements layer on top where available. The librarian's
# LLM enrichment can assign a category at ingest (the "enriched" path), and a
# rich knowledge graph can map entity/predicate types into these buckets. Both
# can be stored and counted with a GROUP BY instead of read-time classification.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

# (id, label). Order is the tie-break priority for classification.
CATEGORIES: list[tuple[str, str]] = [
    ("identity", "Identity & Preferences"),
    ("work", "Work & Learning"),
    ("people", "People & Social"),
    ("communication", "Communication & Content"),
    ("sources", "Sources & Access"),
    ("activities", "Activities & History"),
    ("uncategorized", "Uncategorized"),
]

CATEGORY_LABELS: dict[str, str] = {cid: label for cid, label in CATEGORIES}

# Keyword rules, checked in this order; first hit wins. Specific signals
# (identity, work, people) are checked before the broad activities/time words.
_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("identity", ("prefer", "favourite", "favorite", "i like", "i love", "i hate",
                  "dislike", "my name", "birthday", "born ", "pronoun", "dark mode",
                  "light mode", "metric", "imperial", "my age", "i am ")),
    ("work", ("work", "project", " job", "meeting", "deadline", "deploy", "ship",
              "benchmark", "code", "bug", "feature", "study", "learn", "course",
              "research", "exam", "client", "standup")),
    ("people", ("friend", "family", "colleague", " wife", "husband", "partner",
                " son", "daughter", "mother", "father", " team", "manager",
                " met ", "talked to", " boss")),
    ("communication", ("email", "message", "wrote", " post", "article", "video",
                       "document", " note", " sent", "reply", " link", "watched",
                       "read ")),
    ("sources", ("account", "login", "password", "device", " app ", "connected",
                 "api key", " token", "server", "install", "config", "subscription")),
    ("activities", ("went", "visited", " trip", "travel", "booked", "ordered",
                    " ate ", "played", "attended", "flight", "hotel", "yesterday",
                    "today", "last week", "vacation", "holiday")),
]


def classify(text: str | None) -> str:
    """Return the category id for a memory's text (keyword heuristic).

    Deterministic and LLM-free. Returns ``uncategorized`` when nothing matches
    or the text is empty.
    """
    if not text:
        return "uncategorized"
    t = f" {text.lower()} "
    for cid, kws in _RULES:
        if any(kw in t for kw in kws):
            return cid
    return "uncategorized"


def category_counts(texts: Iterable[str | None]) -> list[dict]:
    """Count classified categories across memory texts.

    Returns ``[{name, count}]`` with the human labels, highest count first,
    omitting empty buckets.
    """
    counter: Counter = Counter(classify(t) for t in texts)
    out = [
        {"name": CATEGORY_LABELS[cid], "count": n}
        for cid, n in counter.items()
        if n
    ]
    out.sort(key=lambda x: -x["count"])
    return out
