"""Pure recall gate: reorder/filter hits by the verification status of the
claims their source spans back. No I/O. Each hit carries an optional
``claim_status`` (None for raw non-claim memories), attached by the recall
wiring before this runs.

Modes:
  off              identity (the flag is not set).
  prefer_verified  drop hits whose claim is unsupported/contradicted; boost
                   supported above the rest; keep unverified and raw hits.
  strict           additionally drop unverified claim-backed hits (fail-closed:
                   only proven-supported claim-backed hits survive; raw hits
                   without a claim are still kept, they were never claims).
"""
from __future__ import annotations

_DROP_LENIENT = ("unsupported", "contradicted")
_DROP_STRICT = ("unsupported", "contradicted", "partial", "unverified")
_BOOST = 1.0  # added to a supported claim-backed hit's score so it ranks first


def apply_claims_gate(hits: list[dict], mode: str = "off") -> list[dict]:
    if mode == "off" or not hits:
        return hits
    drop = _DROP_STRICT if mode == "strict" else _DROP_LENIENT
    kept = [h for h in hits if (h.get("claim_status") not in drop)]
    for h in kept:
        if h.get("claim_status") == "supported":
            h["score"] = h.get("score", 0.0) + _BOOST
    kept.sort(key=lambda h: h.get("score", 0.0), reverse=True)
    return kept
