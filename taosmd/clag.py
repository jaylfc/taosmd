"""CLAG — cluster-then-retrieve pre-filter for small-model tiers.

Adapted from arXiv:2603.15421 ("CLAG: Adaptive Memory Organization via
Agent-Driven Clustering for SLM Agents"). The paper's thesis matches taosmd's
own tier lesson: small local models are *more* hurt by semantically-plausible-
but-irrelevant retrieved context than frontier models are. CLAG's fix is a
coarse two-stage filter — group the candidate memories into semantic clusters,
keep only the cluster(s) most aligned with the question, and discard the rest
before the generator sees them.

This is a *cheap, dependency-free* adaptation: it runs cosine k-means over the
candidate hit embeddings (numpy only — no extra LLM call, unlike the paper's
SLM cluster selection) and keeps hits from the top-scoring cluster(s). It slots
into the retrieval stage alongside the other post-retrieval filters
(llm-rerank, emem-edu, temporal-boost), BEFORE adjacent-turn injection.

Fail-safe: any error, or too few candidates to cluster meaningfully, returns
the hits unchanged so retrieval never starves the generator. Opt-in via --clag.

NOT YET VALIDATED — prototype pending a LoCoMo bench. Default off.
"""

from __future__ import annotations

import numpy as np


def _cosine_kmeans(X: np.ndarray, k: int, iters: int = 12, seed: int = 0) -> np.ndarray:
    """Cosine k-means over L2-normalised rows ``X`` (n, d). Returns labels (n,).

    Deterministic given ``seed`` (no Math.random equivalent needed at bench
    time — fixed seed keeps runs reproducible).
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.full(n, -1, dtype=np.int64)
    for it in range(iters):
        cn = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        sims = X @ cn.T                      # (n, k) cosine sim to each centroid
        new_labels = sims.argmax(axis=1)
        if it > 0 and np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for j in range(k):
            members = X[labels == j]
            if len(members):
                centroids[j] = members.mean(axis=0)
    return labels


async def cluster_prefilter(
    question: str,
    hits: list[dict],
    vmem,
    *,
    n_clusters: int = 4,
    keep_clusters: int = 1,
    min_hits: int = 8,
    seed: int = 0,
) -> list[dict]:
    """Keep only hits from the cluster(s) most aligned with ``question``.

    Embeds the candidate hit texts + the question (ONNX, cheap), clusters the
    hits with cosine k-means, scores each cluster by mean member-to-question
    cosine, and keeps hits from the top ``keep_clusters`` cluster(s) — in their
    original hit order (preserving the upstream relevance ranking within the
    kept set). Returns ``hits`` unchanged if there are too few candidates or on
    any failure.
    """
    if len(hits) < max(min_hits, n_clusters * 2):
        return hits
    try:
        texts = [h.get("text", "") for h in hits]
        q = np.asarray(await vmem.embed(question, task="search_query"), dtype=np.float32)
        embs = [np.asarray(await vmem.embed(t), dtype=np.float32) for t in texts]
        X = np.vstack(embs)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        qn = q / (np.linalg.norm(q) + 1e-8)

        k = min(n_clusters, len(hits))
        labels = _cosine_kmeans(X, k, seed=seed)

        # Score clusters by mean member-to-question cosine; keep the best ones.
        scores: dict[int, float] = {}
        for j in range(k):
            members = X[labels == j]
            if len(members):
                scores[j] = float((members @ qn).mean())
        if not scores:
            return hits
        keep = set(sorted(scores, key=scores.get, reverse=True)[: max(1, keep_clusters)])
        filtered = [h for h, lab in zip(hits, labels) if lab in keep]
        return filtered if filtered else hits
    except Exception:
        return hits
