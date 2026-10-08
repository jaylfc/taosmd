"""Graph Expansion After Vector Search (taOSmd).

After vector search returns initial results, extract entities from those
results and traverse the temporal knowledge graph via BFS to discover
related facts that embedding similarity alone would miss.

This bridges the gap between "what the embedding matched" and "what's
connected to what the embedding matched"; the KG knows relationships
that aren't captured in vector space.
"""

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .knowledge_graph import TemporalKnowledgeGraph

logger = logging.getLogger(__name__)


def _extract_score(result: dict) -> float | None:
    """Extract a valid numeric score from a result dict.

    Priority: result["score"] -> result["source_score"].
    Only accepts int/float that is not bool and not NaN.
    Returns None if no valid score found.
    """
    for key in ("score", "source_score"):
        val = result.get(key)
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            if isinstance(val, float) and math.isnan(val):
                continue
            return float(val)
    return None


def extract_entities_from_text(text: str) -> list[str]:
    """Extract likely entity names from text using heuristics.

    Finds capitalised words/phrases, quoted strings, and known patterns.
    """
    entities: list[str] = []
    seen = set()

    # Quoted strings
    for match in re.finditer(r'"([^"]+)"', text):
        val = match.group(1).strip()
        if len(val) > 1 and val.lower() not in seen:
            entities.append(val)
            seen.add(val.lower())

    # Capitalised multi-word names
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
        val = match.group(1).strip()
        if val.lower() not in seen:
            entities.append(val)
            seen.add(val.lower())

    # Single capitalised words (skip common sentence starters)
    skip = {"The", "This", "That", "These", "Those", "It", "He", "She",
            "They", "We", "You", "My", "Your", "His", "Her", "Its",
            "Our", "Their", "What", "When", "Where", "How", "Why",
            "Who", "Which", "Some", "Any", "All", "No", "Not", "But",
            "And", "Or", "If", "So", "Yet", "For", "Nor", "As", "At",
            "By", "In", "Of", "On", "To", "Up", "An", "Do", "Is", "Was"}
    for word in text.split():
        clean = word.strip(".,!?:;\"'()-[]")
        if (clean and clean[0].isupper() and len(clean) > 1
                and clean not in skip and clean.lower() not in seen):
            entities.append(clean)
            seen.add(clean.lower())

    return entities


async def expand_from_results(
    kg: TemporalKnowledgeGraph,
    vector_results: list[dict],
    max_hops: int = 2,
    max_expanded: int = 10,
    as_of: float | None = None,
    max_seeds: int = 10,
) -> list[dict]:
    """BFS graph traversal starting from entities found in vector results.

    For each entity found in vector search results, query the KG for
    connected triples up to max_hops away. Returns additional context
    not present in the original vector results.

    Args:
        kg: Temporal knowledge graph instance.
        vector_results: Results from VectorMemory.search().
        max_hops: Maximum BFS depth (default 2).
        max_expanded: Maximum number of expanded triples to return.
        as_of: Point-in-time for temporal filtering.
        max_seeds: Maximum number of seed entities to start BFS from (default 10).
                   <= 0 means no seeds (return empty list).

    Returns:
        List of {subject, predicate, object, score, hop} dicts.
    """
    if max_seeds <= 0:
        return []

    # Extract entities from all vector results, tracking per-result entity sets
    # and scores for relevance ranking.
    result_entities: list[set[str]] = []
    result_scores: list[float | None] = []
    all_entities_in_order: list[tuple[str, int]] = []  # (entity, result_index)

    for idx, result in enumerate(vector_results):
        text = result.get("text", "")
        entities = extract_entities_from_text(text)
        entity_set = set(e.lower() for e in entities)
        result_entities.append(entity_set)
        result_scores.append(_extract_score(result))
        for e in entities:
            all_entities_in_order.append((e, idx))

    # Deduplicate while preserving first encounter order
    seen_entities = set()
    unique_entities: list[str] = []
    first_encounter: dict[str, int] = {}  # entity_lower -> first result index
    for e, ridx in all_entities_in_order:
        el = e.lower()
        if el not in seen_entities:
            seen_entities.add(el)
            unique_entities.append(e)
            first_encounter[el] = ridx

    if not unique_entities:
        return []

    # Compute relevance for each unique entity:
    # - max_score: highest valid score among results mentioning this entity
    # - mention_count: number of distinct results mentioning this entity (case-insensitive)
    # - encounter_idx: first result index where entity appears (for tie-break)
    entity_relevance: dict[str, tuple[float | None, int, int]] = {}
    for entity in unique_entities:
        el = entity.lower()
        max_score: float | None = None
        mention_count = 0
        for ridx, (ent_set, score) in enumerate(zip(result_entities, result_scores)):
            if el in ent_set:
                mention_count += 1
                if score is not None:
                    if max_score is None or score > max_score:
                        max_score = score
        entity_relevance[el] = (max_score, mention_count, first_encounter[el])

    # Check if any entity has a valid score
    any_scored = any(v[0] is not None for v in entity_relevance.values())

    if any_scored:
        # Rank by: max_score desc, mention_count desc, encounter_idx asc (earlier wins)
        # Use stable sort with a key that sorts ascending on negated values
        def sort_key(entity: str):
            max_score, mention_count, encounter_idx = entity_relevance[entity.lower()]
            # None scores sort last (treated as -inf)
            score_key = -max_score if max_score is not None else float('inf')
            return (score_key, -mention_count, encounter_idx)

        ranked_entities = sorted(unique_entities, key=sort_key)
    else:
        # No valid scores anywhere: preserve exact encounter order
        ranked_entities = unique_entities

    # Cap seed entities
    seed_entities = ranked_entities[:max_seeds]

    # BFS expansion
    expanded: list[dict] = []
    visited_triples: set[str] = set()
    frontier = [(e, 0) for e in seed_entities]

    while frontier and len(expanded) < max_expanded:
        entity, hop = frontier.pop(0)
        if hop >= max_hops:
            continue

        try:
            triples = await kg.query_entity(
                entity, as_of=as_of, track_access=False
            )
        except Exception:
            continue

        for triple in triples:
            triple_key = f"{triple.get('subject_id', '')}:{triple['predicate']}:{triple.get('object_id', '')}"
            if triple_key in visited_triples:
                continue
            visited_triples.add(triple_key)

            # Score: closer hops are more relevant, weighted by confidence
            confidence = triple.get("confidence", 1.0)
            hop_penalty = 1.0 / (hop + 1)
            score = confidence * hop_penalty

            direction = triple.get("direction", "outgoing")
            if direction == "outgoing":
                subject = entity
                obj = triple.get("object_name", "?")
            else:
                subject = triple.get("subject_name", "?")
                obj = entity

            expanded.append({
                "subject": subject,
                "predicate": triple["predicate"],
                "object": obj,
                "score": round(score, 3),
                "hop": hop,
                "confidence": confidence,
                "current": triple.get("current", True),
            })

            # Add connected entity to frontier for next hop
            next_entity = obj if direction == "outgoing" else subject
            if next_entity != entity and hop + 1 < max_hops:
                frontier.append((next_entity, hop + 1))

    # Sort by score descending
    expanded.sort(key=lambda x: x["score"], reverse=True)
    return expanded[:max_expanded]


def format_expanded_context(expanded: list[dict], max_tokens: int = 200) -> str:
    """Format expanded graph results as context text.

    Args:
        expanded: Results from expand_from_results().
        max_tokens: Approximate token budget (1 token ~ 4 chars).

    Returns:
        Formatted string of related facts.
    """
    if not expanded:
        return ""

    lines = ["Related facts:"]
    chars_budget = max_tokens * 4
    chars_used = len(lines[0])

    for item in expanded:
        line = f"- {item['subject']} {item['predicate']} {item['object']}"
        if not item.get("current", True):
            line += " (historical)"
        if chars_used + len(line) + 1 > chars_budget:
            break
        lines.append(line)
        chars_used += len(line) + 1

    return "\n".join(lines) if len(lines) > 1 else ""
