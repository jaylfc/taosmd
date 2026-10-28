"""QRELs-based Recall@K scorer for E-048.

Computes recall metrics from ranked document IDs and ground truth relevance
judgments (qrels), independent of embedding vectors.
"""

from __future__ import annotations


def recall_at_k_qrels(
    ranked_doc_ids: dict[str, list[str]],
    qrels: dict[str, set[str]],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Recall@K from ranked document IDs and qrels.

    Args:
        ranked_doc_ids: Mapping from query ID to ranked list of document IDs
            (best first). Queries not present are treated as empty rankings.
        qrels: Mapping from query ID to set of relevant document IDs.
            Only queries in qrels are evaluated.
        ks: Tuple of K values for which to compute Recall.

    Returns:
        Dict mapping f"R@{k}" to recall fraction for each k in ks.

    Raises:
        ValueError: If qrels is empty.
    """
    if not qrels:
        raise ValueError("qrels is empty")

    # Initialize hit counters for each K
    hits = {k: 0 for k in ks}
    n_queries = len(qrels)

    # Iterate over qrels keys (not ranked_doc_ids keys)
    for qid, relevant_docs in qrels.items():
        ranked = ranked_doc_ids.get(qid, [])

        # Check each K value
        for k in ks:
            top_k = ranked[:k]
            # Hit if ANY relevant doc is in top K
            if any(doc in top_k for doc in relevant_docs):
                hits[k] += 1

    # Build results with fraction of qrels queries that hit
    return {f"R@{k}": hits[k] / n_queries for k in ks}