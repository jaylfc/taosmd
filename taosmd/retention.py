"""Ebbinghaus Retention Scoring (taOSmd).

Models memory decay over time with reinforcement through repeated access,
inspired by agentmemory's approach. Replaces flat hit_rate scoring with a
psychologically-grounded decay curve.

Formula: score = salience * exp(-lambda * days) + sigma * sum(1/daysSinceAccess)

Tiers:
  hot     (score >= 0.7)  — always available, prioritised in search
  warm    (score >= 0.4)  — available, normal retrieval
  cold    (score >= 0.15) — available but deprioritised
  evictable (score < 0.15) — candidate for removal

Also handles:
  - TTL-based expiry (forgetAfter field)
  - Near-duplicate detection via Jaccard similarity (>0.9 threshold)
"""

from __future__ import annotations

import math
import time

# Decay parameters (tuned for daily granularity)
LAMBDA = 0.01       # Base decay rate — slow, memories last weeks
SIGMA = 0.3         # Reinforcement weight for access history

# Tier thresholds
TIER_HOT = 0.7
TIER_WARM = 0.4
TIER_COLD = 0.15

TIERS = {
    "hot": TIER_HOT,
    "warm": TIER_WARM,
    "cold": TIER_COLD,
    "evictable": 0.0,
}


def retention_score(
    created_at: float,
    access_times: list[float] | None = None,
    salience: float = 1.0,
    now: float | None = None,
) -> float:
    """Compute Ebbinghaus-inspired retention score.

    Args:
        created_at: Unix timestamp when the memory was created.
        access_times: List of unix timestamps when the memory was accessed.
        salience: Base importance weight (0-1). Higher = decays slower.
        now: Current time (defaults to time.time()).

    Returns:
        Retention score (0-1+, clamped to [0, 1]).
    """
    t = now or time.time()
    days_since_creation = max((t - created_at) / 86400, 0.001)

    # Base decay: exponential forgetting curve
    base = salience * math.exp(-LAMBDA * days_since_creation)

    # Reinforcement: each access slows forgetting (recent accesses matter more)
    reinforcement = 0.0
    if access_times:
        for access_t in access_times:
            days_since_access = max((t - access_t) / 86400, 0.001)
            reinforcement += 1.0 / days_since_access

    score = base + SIGMA * reinforcement
    return min(max(score, 0.0), 1.0)


def classify_tier(score: float) -> str:
    """Classify a retention score into a tier."""
    if score >= TIER_HOT:
        return "hot"
    if score >= TIER_WARM:
        return "warm"
    if score >= TIER_COLD:
        return "cold"
    return "evictable"


def is_expired(created_at: float, ttl_seconds: float | None, now: float | None = None) -> bool:
    """Check if a memory has exceeded its TTL."""
    if ttl_seconds is None:
        return False
    t = now or time.time()
    return (t - created_at) > ttl_seconds


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Word-level Jaccard similarity between two texts."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def find_near_duplicates(
    texts: list[str],
    threshold: float = 0.9,
) -> list[tuple[int, int, float]]:
    """Find near-duplicate pairs above the Jaccard threshold.

    Returns list of (index_a, index_b, similarity) where index_a < index_b.
    The older entry (lower index) is typically the one to keep.
    """
    duplicates = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = jaccard_similarity(texts[i], texts[j])
            if sim >= threshold:
                duplicates.append((i, j, sim))
    return duplicates


def score_and_tier(
    created_at: float,
    access_times: list[float] | None = None,
    salience: float = 1.0,
    ttl_seconds: float | None = None,
    now: float | None = None,
) -> dict:
    """Compute retention score, tier, and expiry status in one call.

    Returns {score, tier, expired, days_old}.
    """
    t = now or time.time()
    expired = is_expired(created_at, ttl_seconds, t)
    score = retention_score(created_at, access_times, salience, t)
    tier = classify_tier(score)
    days_old = (t - created_at) / 86400

    return {
        "score": round(score, 4),
        "tier": tier,
        "expired": expired,
        "days_old": round(days_old, 1),
    }
