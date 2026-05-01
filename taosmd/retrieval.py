"""Source adapters and result normalisation for taOSmd retrieval pipeline.

Normalises results from vector memory, knowledge graph, session catalog,
archive, and crystals into a common format for ranking and fusion.
"""

from __future__ import annotations

import asyncio
import json
import logging

logger = logging.getLogger(__name__)

from taosmd.agents import run_if_enabled, is_task_enabled  # noqa: E402

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


# ---------------------------------------------------------------------------
# Source query helpers
# ---------------------------------------------------------------------------


async def _query_source(name: str, source: object, query: str, limit: int) -> list[dict]:
    """Query a single source and return adapted results.

    Returns an empty list if the query raises an exception.
    """
    try:
        if name == "vector":
            raw = await source.search(query, limit=limit, hybrid=True)
            return _adapt_vector(raw)
        elif name == "kg":
            words = [w for w in query.split() if len(w) > 2]
            kg_results: list[dict] = []
            for word in words:
                try:
                    rows = await source.query_entity(word)
                    kg_results.extend(rows)
                    if len(kg_results) >= limit:
                        break
                except Exception:
                    pass
            return _adapt_kg(kg_results[:limit])
        elif name == "catalog":
            raw = await source.search_topic(query, limit=limit)
            return _adapt_catalog(raw)
        elif name == "archive":
            raw = await source.search_fts(query, limit=limit)
            return _adapt_archive(raw)
        elif name == "crystals":
            raw = await source.search(query, limit=limit)
            return _adapt_crystals(raw)
        else:
            logger.warning("_query_source: unknown source name %r", name)
            return []
    except Exception as exc:
        logger.warning("_query_source: error querying %r: %s", name, exc)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plan_retrieval(
    query: str,
    agent_name: str = "",
    llm_route_fn=None,
) -> dict:
    """Choose which holdings to search and in what order.

    Hot path: regex intent_classifier decides in microseconds. LLM routing
    fires only when the classifier returns EXPLORATORY, multiple intents tie,
    or ``llm_route_fn`` is provided and the ``query_expansion`` task is enabled.

    Returns a strategy dict compatible with retrieve():
    ``{primary: str, secondary: str | None, extra_holdings: list[str]}``
    """
    from taosmd.intent_classifier import classify_intent, get_search_strategy  # noqa
    from taosmd.intent_classifier import EXPLORATORY  # noqa

    strategy = get_search_strategy(query)
    intent = classify_intent(query)

    use_llm = (
        intent == EXPLORATORY
        or llm_route_fn is not None
    ) and is_task_enabled(agent_name, "query_expansion")

    if use_llm and llm_route_fn is not None:
        try:
            llm_strategy = llm_route_fn(query, agent_name)
            if llm_strategy:
                logger.debug(
                    "plan_retrieval: LLM routing overrides regex for %r", query[:60]
                )
                return llm_strategy
        except Exception as exc:
            logger.warning("plan_retrieval: LLM route failed (%s), using regex", exc)

    return strategy


async def _apply_verification(
    query: str,
    results: list[dict],
    agent_name: str,
    top_k: int = 3,
) -> list[dict]:
    """Adjust top-K result confidences using the verification prompt.

    Fires inside retrieve() after cross-encoder rerank when the agent's
    'verification' task is enabled. Supports/contradicts verdicts shift the
    rrf_score. Results are re-sorted after adjustment.

    Hall-of-records quote for each candidate is taken from metadata.summary
    or metadata.data_json when available.
    """
    if not results:
        return results

    try:
        from taosmd.prompts import verification_prompt  # noqa
    except ImportError:
        return results

    adjusted = list(results)
    for i, hit in enumerate(adjusted[:top_k]):
        hall_quote = (
            hit.get("metadata", {}).get("summary", "")
            or str(hit.get("metadata", {}).get("data_json", ""))[:400]
        )
        if not hall_quote:
            continue
        # Build prompt — actual LLM call is caller's responsibility;
        # we attach the prompt so callers with an llm client can call it.
        hit["_verification_prompt"] = verification_prompt(
            query,
            hit.get("text", ""),
            hall_quote,
            agent_name=agent_name,
        )
        # Optimistic path: callers that pass a verdict back can call
        # apply_verification_verdicts() to adjust scores.

    logger.debug(
        "_apply_verification: tagged %d/%d results with verification prompts",
        min(top_k, len(results)), len(results)
    )
    return adjusted


def apply_verification_verdicts(
    results: list[dict],
    verdicts: list[dict],
) -> list[dict]:
    """Apply pre-computed verification verdicts to result scores.

    Call this after running verification_prompt LLM calls outside retrieval.
    verdicts is a list of {verdict: 'supports|contradicts|silent', index: int}.
    Returns results re-sorted by adjusted rrf_score.
    """
    adjusted = list(results)
    for v in verdicts:
        idx = v.get("index", -1)
        if not (0 <= idx < len(adjusted)):
            continue
        verdict = v.get("verdict", "silent")
        score = adjusted[idx].get("rrf_score", 0.0)
        if verdict == "supports":
            adjusted[idx]["rrf_score"] = score + 0.1
            adjusted[idx]["verification"] = "supports"
        elif verdict == "contradicts":
            adjusted[idx]["rrf_score"] = score - 0.4
            adjusted[idx]["conflict"] = True
            adjusted[idx]["verification"] = "contradicts"
    adjusted.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
    return adjusted


def _user_metadata(hit: dict) -> dict:
    """Return the user-supplied metadata for a hit.

    Normalised hits wrap the underlying source row under "metadata", and the
    vector source row in turn carries its user-supplied metadata under a
    nested "metadata" key. This helper unwraps that case so callers see the
    user dict directly. Falls back to the top-level metadata when there is no
    nesting (non-vector sources).
    """
    meta = hit.get("metadata", {}) or {}
    inner = meta.get("metadata") if isinstance(meta, dict) else None
    return inner if isinstance(inner, dict) else meta


async def _attach_neighbors(
    hits: list[dict],
    sources: dict,
    n: int,
    position_key: str,
    group_key: str | None,
) -> list[dict]:
    """Attach ±n positional neighbours to each hit as a "neighbors" field.

    For each hit, reads ``user_metadata[position_key]`` (and optionally
    ``[group_key]``) and asks the vector source for the rows at
    ``position ± offset`` for offset in 1..n. A row that already appears as a
    primary hit is skipped, and the same row is attached to at most one host
    so the same neighbour is never duplicated across hits.

    Mutates each hit in place. Returns the list. Hits without a
    position-key entry are left unchanged. When the vector source does not
    expose ``get_by_position`` (e.g. an older mock or alternative backend),
    this is a silent no-op.
    """
    if n <= 0:
        return hits
    vmem = sources.get("vector") if sources else None
    if vmem is None or not hasattr(vmem, "get_by_position"):
        return hits

    primary_keys: set[tuple] = set()
    for hit in hits:
        um = _user_metadata(hit)
        pos = um.get(position_key)
        if pos is None:
            continue
        if group_key is not None:
            group = um.get(group_key)
            if group is None:
                # Refuse to populate primary_keys with (None, pos) when the
                # caller requested group filtering — otherwise hits missing
                # the group key would mask same-position neighbours from
                # other groups during dedupe.
                continue
        else:
            group = None
        try:
            primary_keys.add((group, int(pos)))
        except (TypeError, ValueError):
            continue

    seen: set[tuple] = set()
    for hit in hits:
        um = _user_metadata(hit)
        pos = um.get(position_key)
        if pos is None:
            continue
        try:
            pos_int = int(pos)
        except (TypeError, ValueError):
            continue
        if group_key is not None:
            group = um.get(group_key)
            if group is None:
                # Group filtering is requested but this hit has no group;
                # skip rather than silently widening to cross-group lookup
                # (get_by_position drops the group filter when group_value
                # is None).
                continue
        else:
            group = None
        neighbours: list[dict] = []
        for offset in range(-n, n + 1):
            if offset == 0:
                continue
            target = pos_int + offset
            key = (group, target)
            if key in primary_keys or key in seen:
                continue
            try:
                row = await vmem.get_by_position(
                    target,
                    position_key=position_key,
                    group_key=group_key,
                    group_value=group,
                )
            except Exception as exc:
                logger.debug(
                    "_attach_neighbors: get_by_position(%s=%s) failed: %s",
                    position_key, target, exc,
                )
                continue
            if row is None:
                continue
            seen.add(key)
            neighbours.append(row)
        if neighbours:
            hit["neighbors"] = neighbours
    return hits


async def retrieve(
    query: str,
    strategy: str = "thorough",
    memory_layers: list[str] | None = None,
    sources: dict | None = None,
    limit: int = 5,
    reranker: object | None = None,
    agent: str | None = None,
    worker_capabilities: dict | None = None,
    agent_name: str = "",
    verify: bool = False,
    adjacent_neighbors: int = 0,
    position_key: str = "position",
    group_key: str | None = None,
) -> list[dict]:
    """Retrieve relevant results from available memory sources.

    Args:
        query: The search query.
        strategy: One of "thorough", "fast", "minimal", or "custom".
        memory_layers: Source names to use in "custom" mode (e.g. ["vector", "kg"]).
        sources: Dict of initialised store objects keyed by source name.
            Recognised keys: "vector", "kg", "catalog", "archive", "crystals".
            Missing keys mean that source is unavailable.
        limit: Maximum number of results to return. When *agent* is supplied,
            this is overridden by ``effective_fanout(agent, worker_capabilities)``
            so the per-agent fan-out setting controls the per-layer K.
        reranker: Optional CrossEncoderReranker instance. When provided and
            ``reranker.available`` is True, used for thorough/custom reranking.
        agent: Registered agent name. When given, ``effective_fanout`` is
            called to resolve the per-layer K from the agent's librarian
            fanout config and the supplied worker capabilities.
        worker_capabilities: Dict describing the runtime worker, passed through
            to ``effective_fanout``. Recognised keys: ``gpu_vram_gb`` (float)
            and ``turboquant`` (bool). Pass ``None`` for Pi-class workers.
        adjacent_neighbors: When > 0, attach ±N positional neighbours to each
            surviving hit as a ``neighbors`` field. Looks up the vector source
            via ``get_by_position`` using the hit's
            ``metadata[position_key]`` (and optional ``[group_key]``).
            Default 0 (off). Worth +0.089 on LoCoMo same-tier at adj=2 — see
            ``docs/benchmarks.md``.
        position_key: Metadata key holding the integer position used for
            neighbour lookup (default ``"position"``).
        group_key: Optional metadata key constraining neighbours to share the
            same group as their host hit (e.g. ``"session"``). When ``None``
            (default), neighbours can cross group boundaries — matches the
            LoCoMo benchmark's behaviour where ``turn_idx`` is global.

    Returns:
        List of normalised result dicts, sorted by relevance, length <= limit.
    """
    from taosmd.intent_classifier import classify_intent, get_search_strategy  # noqa: PLC0415

    if sources is None:
        sources = {}

    # Resolve effective K from per-agent fanout config when an agent is given.
    if agent is not None:
        from taosmd.agents import effective_fanout as _effective_fanout  # noqa: PLC0415
        limit = _effective_fanout(agent, worker_capabilities)

    fetch_limit = limit * 3

    if strategy == "thorough":
        strategy_info = get_search_strategy(query)
        available = {k: v for k, v in sources.items()}

        tasks = [
            asyncio.create_task(_query_source(name, src, query, fetch_limit))
            for name, src in available.items()
        ]
        results_per_source = await asyncio.gather(*tasks, return_exceptions=True)

        ranked_lists: list[list[dict]] = []
        for res in results_per_source:
            if isinstance(res, Exception):
                ranked_lists.append([])
            else:
                ranked_lists.append(res)

        merged = _rrf_merge(ranked_lists, intent_primary=strategy_info.get("primary"))
        results = _deduplicate(merged)

        if reranker is not None and getattr(reranker, "available", False):
            results = reranker.rerank(query, results, limit)

        if verify and is_task_enabled(agent_name, "verification"):
            results = await _apply_verification(query, results, agent_name)

        results = results[:limit]
        if adjacent_neighbors > 0:
            results = await _attach_neighbors(
                results, sources, adjacent_neighbors, position_key, group_key,
            )
        return results

    elif strategy == "fast":
        strategy_info = get_search_strategy(query)
        primary = strategy_info.get("primary")
        secondary = strategy_info.get("secondary")

        results: list[dict] = []

        if primary and primary in sources:
            results = await _query_source(primary, sources[primary], query, fetch_limit)
        elif sources:
            # Fall back to first available source
            first_name, first_src = next(iter(sources.items()))
            results = await _query_source(first_name, first_src, query, fetch_limit)

        if len(results) < limit and secondary and secondary in sources:
            secondary_results = await _query_source(secondary, sources[secondary], query, fetch_limit)
            merged = _rrf_merge([results, secondary_results])
            results = _deduplicate(merged)

        results = results[:limit]
        if adjacent_neighbors > 0:
            results = await _attach_neighbors(
                results, sources, adjacent_neighbors, position_key, group_key,
            )
        return results

    elif strategy == "minimal":
        strategy_info = get_search_strategy(query)
        primary = strategy_info.get("primary")

        results: list[dict] = []

        if primary and primary in sources:
            results = await _query_source(primary, sources[primary], query, fetch_limit)
        elif sources:
            first_name, first_src = next(iter(sources.items()))
            results = await _query_source(first_name, first_src, query, fetch_limit)

        results = results[:limit]
        if adjacent_neighbors > 0:
            results = await _attach_neighbors(
                results, sources, adjacent_neighbors, position_key, group_key,
            )
        return results

    elif strategy == "custom":
        if memory_layers is None:
            memory_layers = list(sources.keys())

        filtered_sources = {k: v for k, v in sources.items() if k in memory_layers}
        strategy_info = get_search_strategy(query)

        tasks = [
            asyncio.create_task(_query_source(name, src, query, fetch_limit))
            for name, src in filtered_sources.items()
        ]
        results_per_source = await asyncio.gather(*tasks, return_exceptions=True)

        ranked_lists: list[list[dict]] = []
        for res in results_per_source:
            if isinstance(res, Exception):
                ranked_lists.append([])
            else:
                ranked_lists.append(res)

        merged = _rrf_merge(ranked_lists, intent_primary=strategy_info.get("primary"))
        results = _deduplicate(merged)

        if reranker is not None and getattr(reranker, "available", False):
            results = reranker.rerank(query, results, limit)

        results = results[:limit]
        if adjacent_neighbors > 0:
            results = await _attach_neighbors(
                results, sources, adjacent_neighbors, position_key, group_key,
            )
        return results

    else:
        raise ValueError(f"Unknown strategy {strategy!r}. Must be one of: thorough, fast, minimal, custom.")


__all__ = [
    "retrieve",
    "plan_retrieval",
    "apply_verification_verdicts",
]
