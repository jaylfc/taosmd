"""Source adapters and result normalisation for taOSmd retrieval pipeline.

Normalises results from vector memory, knowledge graph, session catalog,
archive, and crystals into a common format for ranking and fusion.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalised result schema (for reference):
# {
#     "text": str,           # searchable content
#     "source": str,         # "vector" / "kg" / "catalog" / "archive" / "crystals"
#     "source_id": str,      # original ID from that source
#     "rank": int,           # rank within that source (0-based)
#     "source_score": float, # original score (cosine sim, FTS rank, etc)
#     "metadata": dict,      # source-specific data (full original result)
# }
# ---------------------------------------------------------------------------


def _adapt_vector(results: list[dict]) -> list[dict]:
    """Adapt vector search results to the normalised format.

    Input fields: id, text, similarity, metadata, created_at.
    """
    adapted = []
    for rank, r in enumerate(results):
        adapted.append(
            {
                "text": r["text"],
                "source": "vector",
                "source_id": str(r["id"]),
                "rank": rank,
                "source_score": float(r["similarity"]),
                "metadata": r,
            }
        )
    return adapted


def _adapt_kg(results: list[dict]) -> list[dict]:
    """Adapt knowledge graph query results to the normalised format.

    Input fields: subject_id, predicate, object_id, object_name / subject_name,
    direction, confidence.

    Text format: "{subject} {predicate} {object}".
    For outgoing edges the subject is subject_name; for incoming the object is
    the anchor and subject is the far node.
    """
    adapted = []
    for rank, r in enumerate(results):
        direction = r.get("direction", "outgoing")
        if direction == "outgoing":
            subject = r.get("subject_name", r.get("subject_id", ""))
            obj = r.get("object_name", r.get("object_id", ""))
        else:
            subject = r.get("subject_name", r.get("subject_id", ""))
            obj = r.get("object_name", r.get("object_id", ""))

        predicate = r.get("predicate", "")
        text = f"{subject} {predicate} {obj}"

        adapted.append(
            {
                "text": text,
                "source": "kg",
                "source_id": str(r.get("subject_id", "")),
                "rank": rank,
                "source_score": float(r.get("confidence", 1.0)),
                "metadata": r,
            }
        )
    return adapted


def _adapt_catalog(results: list[dict]) -> list[dict]:
    """Adapt session catalog search results to the normalised format.

    Input fields: id, topic, description, date, start_str, end_str, category.

    Text format: "[{date} {start_str}-{end_str}] {topic}: {description}".
    """
    adapted = []
    for rank, r in enumerate(results):
        date = r.get("date", "")
        start_str = r.get("start_str", "")
        end_str = r.get("end_str", "")
        topic = r.get("topic", "")
        description = r.get("description", "")
        text = f"[{date} {start_str}-{end_str}] {topic}: {description}"

        adapted.append(
            {
                "text": text,
                "source": "catalog",
                "source_id": str(r["id"]),
                "rank": rank,
                "source_score": float(r.get("score", 1.0)),
                "metadata": r,
            }
        )
    return adapted


def _adapt_archive(results: list[dict]) -> list[dict]:
    """Adapt archive FTS results to the normalised format.

    Input fields: id, summary, data_json, event_type, timestamp.

    Text is the summary if present; otherwise falls back to parsed data_json
    content (stringified).
    """
    adapted = []
    for rank, r in enumerate(results):
        summary = r.get("summary", "")
        if summary:
            text = summary
        else:
            raw = r.get("data_json", "")
            if raw:
                try:
                    data = json.loads(raw)
                    text = str(data)
                except (json.JSONDecodeError, TypeError):
                    text = raw
            else:
                text = ""

        adapted.append(
            {
                "text": text,
                "source": "archive",
                "source_id": str(r["id"]),
                "rank": rank,
                "source_score": float(r.get("score", 1.0)),
                "metadata": r,
            }
        )
    return adapted


def _adapt_crystals(results: list[dict]) -> list[dict]:
    """Adapt crystal search results to the normalised format.

    Input fields: id, narrative, session_id, outcomes, lessons.

    Text is the narrative.
    """
    adapted = []
    for rank, r in enumerate(results):
        text = r.get("narrative", "")

        adapted.append(
            {
                "text": text,
                "source": "crystals",
                "source_id": str(r["id"]),
                "rank": rank,
                "source_score": float(r.get("score", 1.0)),
                "metadata": r,
            }
        )
    return adapted


def _rrf_merge(
    ranked_lists: list[list[dict]],
    intent_primary: str | None = None,
    k: int = 60,
    intent_boost: float = 1.5,
) -> list[dict]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Each input list contains normalised result dicts with "text", "source",
    "source_id", "rank", "source_score", and "metadata" fields.

    Args:
        ranked_lists: List of ranked result lists to merge.
        intent_primary: Source name to boost (e.g. "vector"). If set, results
            from that source have their RRF score multiplied by intent_boost.
        k: RRF constant (default 60).
        intent_boost: Multiplier applied to the primary source's scores.

    Returns:
        Single merged list sorted by rrf_score descending, with an added
        "rrf_score" field on each result dict.
    """
    rrf_scores: dict[str, float] = {}
    result_by_key: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for result in ranked_list:
            key = f"{result['source']}:{result['source_id']}"
            score = 1.0 / (k + result["rank"])
            if intent_primary is not None and result["source"] == intent_primary:
                score *= intent_boost
            rrf_scores[key] = rrf_scores.get(key, 0.0) + score
            if key not in result_by_key:
                result_by_key[key] = result

    merged = []
    for key, score in rrf_scores.items():
        entry = dict(result_by_key[key])
        entry["rrf_score"] = score
        merged.append(entry)

    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    logger.debug("_rrf_merge: %d results from %d lists", len(merged), len(ranked_lists))
    return merged


def _deduplicate(results: list[dict], threshold: float = 0.8) -> list[dict]:
    """Remove near-duplicate results by Jaccard word-set similarity.

    Compares each pair of results on the "text" field. When similarity
    >= threshold, the result with the lower rrf_score is dropped. Uses an
    O(n^2) approach; n is expected to be small (<100 results).

    Args:
        results: List of normalised result dicts (should have "rrf_score").
        threshold: Jaccard similarity threshold above which results are
            considered duplicates (default 0.8).

    Returns:
        Filtered list with near-duplicates removed.
    """
    to_remove: set[int] = set()

    for i in range(len(results)):
        if i in to_remove:
            continue
        words_i = set(results[i]["text"].lower().split())
        for j in range(i + 1, len(results)):
            if j in to_remove:
                continue
            words_j = set(results[j]["text"].lower().split())
            union = words_i | words_j
            if not union:
                continue
            jaccard = len(words_i & words_j) / len(union)
            if jaccard >= threshold:
                score_i = results[i].get("rrf_score", 0.0)
                score_j = results[j].get("rrf_score", 0.0)
                to_remove.add(j if score_i >= score_j else i)

    filtered = [r for idx, r in enumerate(results) if idx not in to_remove]
    logger.debug(
        "_deduplicate: kept %d/%d results (threshold=%.2f)",
        len(filtered),
        len(results),
        threshold,
    )
    return filtered
