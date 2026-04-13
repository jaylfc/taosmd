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
