"""Pure recall gate: filter hits by the verification status of the claims
their source spans back, and front supported claim-backed hits. No I/O.
Each hit carries an optional ``claim_status`` (None for raw non-claim
memories), attached by the recall wiring before this runs.

Modes:
  off              identity (the flag is not set).
  prefer_verified  drop hits whose claim is unsupported/contradicted; keep
                   supported, unverified and raw hits, with supported claim-
                   backed hits moved to the front. Retrieval order is
                   preserved inside each group.
  strict           additionally drop unverified claim-backed hits (fail-closed:
                   only proven-supported claim-backed hits survive; raw hits
                   without a claim are still kept, they were never claims).

Ordering deliberately never re-sorts by ``score``: scores arrive in source-
native units (BM25 source_scores and vector cosine similarities live on
different scales), so a cross-source sort demotes keyword hits below
semantic hits purely from the unit mismatch (observed in a shadow
deployment: a correct rank-2 BM25 hit sank to rank 10). The retrieval
stage already ranked the pool; the gate only filters and fronts.
"""
from __future__ import annotations

_DROP_LENIENT = ("unsupported", "contradicted")
_DROP_STRICT = ("unsupported", "contradicted", "partial", "unverified")


def apply_claims_gate(hits: list[dict], mode: str = "off") -> list[dict]:
    if mode == "off" or not hits:
        return hits
    drop = _DROP_STRICT if mode == "strict" else _DROP_LENIENT
    kept = [h for h in hits if (h.get("claim_status") not in drop)]
    supported = [h for h in kept if h.get("claim_status") == "supported"]
    rest = [h for h in kept if h.get("claim_status") != "supported"]
    return supported + rest
