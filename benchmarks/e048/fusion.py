"""Rank fusion for E-048 (Reciprocal Rank Fusion)."""

from __future__ import annotations

from typing import Any


def rrf_fuse(
    ranked_lists: dict[str, list[str]],
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    if len(ranked_lists) < 2:
        raise ValueError("at least 2 spaces required")

    if k < 1:
        raise ValueError("k must be >= 1")

    for space_name, ranked_list in ranked_lists.items():
        if not ranked_list:
            raise ValueError(f"empty ranked list for space {space_name}")
        seen = set()
        for doc_id in ranked_list:
            if doc_id in seen:
                raise ValueError(f"duplicate doc id in space {space_name}")
            seen.add(doc_id)

    scores: dict[str, float] = {}

    for _space_name, ranked_list in ranked_lists.items():
        for rank, doc_id in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)

    sorted_results = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    if top_n is not None:
        sorted_results = sorted_results[:top_n]

    return sorted_results


def fuse_queries(
    per_space: dict[str, dict[Any, list[str]]],
    k: int = 60,
    top_n: int | None = 10,
) -> dict[Any, list[str]]:
    all_qids: set[Any] = set()
    for _space_name, qid_to_docs in per_space.items():
        all_qids.update(qid_to_docs.keys())

    result: dict[Any, list[str]] = {}

    for qid in all_qids:
        ranked_lists_for_qid: dict[str, list[str]] = {}
        for space_name, qid_to_docs in per_space.items():
            if qid not in qid_to_docs:
                raise ValueError(f"qid {qid} missing from space {space_name}")
            ranked_lists_for_qid[space_name] = qid_to_docs[qid]

        fused = rrf_fuse(ranked_lists_for_qid, k=k, top_n=top_n)
        result[qid] = [doc_id for doc_id, _score in fused]

    return result
