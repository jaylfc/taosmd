"""Hybrid retrieval scoring utilities.

Adapted from mem0ai/mem0 (mem0/utils/scoring.py, Apache 2.0). Provides:

- ``get_bm25_params``: query-length-adaptive sigmoid (midpoint, steepness)
  pair for normalising raw BM25 scores. Longer queries tend to score
  higher in absolute terms, so the midpoint shifts up.
- ``normalize_bm25``: sigmoid normalisation of an unbounded raw BM25
  score to [0, 1].
- ``score_and_rank``: threshold-gated additive combine of
  ``semantic + bm25 + entity_boost`` with adaptive denominator. Used as
  an alternative to RRF when we want each signal to retain calibrated
  magnitude rather than getting flattened to ranks.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


ENTITY_BOOST_WEIGHT = 0.5


def get_bm25_params(query: str, lemmatized: str | None = None) -> tuple[float, float]:
    """Return (midpoint, steepness) for BM25 sigmoid normalisation."""
    text = lemmatized if lemmatized is not None else query
    num_terms = len(text.split()) if text else 1
    if num_terms <= 3:
        return 5.0, 0.7
    if num_terms <= 6:
        return 7.0, 0.6
    if num_terms <= 9:
        return 9.0, 0.5
    if num_terms <= 15:
        return 10.0, 0.5
    return 12.0, 0.5


def normalize_bm25(raw_score: float, midpoint: float, steepness: float) -> float:
    """Sigmoid-normalise an unbounded BM25 score to [0, 1]."""
    return 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))


def score_and_rank(
    semantic_results: List[Dict[str, Any]],
    bm25_scores: Dict[str, float],
    entity_boosts: Dict[str, float] | None = None,
    threshold: float = 0.0,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Threshold-gated additive ranking.

    Each candidate's combined score:
        combined = (semantic_score + bm25 + entity_boost) / max_possible

    where max_possible adapts to which signals are active so the output
    stays within [0, 1]:

        semantic only            : 1.0
        semantic + bm25          : 2.0
        semantic + bm25 + entity : 2.5
        semantic + entity        : 1.5

    Threshold gates on the semantic score *before* combining; a candidate
    below ``threshold`` is dropped even if BM25/entity would have rescued
    it. This is the "trust the dense recall floor" property — important
    in our regime where BM25 alone can promote keyword-spam.

    ``semantic_results`` items must each have an ``id`` and a ``score``
    key; arbitrary other fields (e.g. ``payload``, ``text``) are passed
    through.
    """
    has_bm25 = bool(bm25_scores)
    entity_boosts = entity_boosts or {}
    has_entity = bool(entity_boosts)

    max_possible = 1.0
    if has_bm25:
        max_possible += 1.0
    if has_entity:
        max_possible += ENTITY_BOOST_WEIGHT

    scored: List[Dict[str, Any]] = []
    for result in semantic_results:
        mem_id = result.get("id")
        if mem_id is None:
            continue

        sem_score = float(result.get("score", 0.0))
        if sem_score < threshold:
            continue

        sid = str(mem_id)
        bm25 = bm25_scores.get(sid, 0.0)
        eb = entity_boosts.get(sid, 0.0)

        raw = sem_score + bm25 + eb
        combined = min(raw / max_possible, 1.0)

        new = dict(result)
        new["score"] = combined
        scored.append(new)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


__all__ = [
    "ENTITY_BOOST_WEIGHT",
    "get_bm25_params",
    "normalize_bm25",
    "score_and_rank",
]
