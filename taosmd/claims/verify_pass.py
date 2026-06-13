"""Asynchronous, batched, fail-closed verification pass over unverified claims.

Pulls unverified claims in batches, fetches each claim's cited span texts via
the injected ``fetch_spans`` callable (so the archive dependency is testable),
runs the verifier, and writes the status. A verifier that returns 'unverified'
(its own fail-closed signal) leaves the claim unverified, never promoted. The
pass is idempotent and terminates even when every claim fails: a ``seen`` set
guarantees progress (a claim is attempted at most once per pass).
"""
from __future__ import annotations

from typing import Awaitable, Callable

from taosmd.claims.store import ClaimStore
from taosmd.claims.verifier import Verifier


async def verify_pass(
    store: ClaimStore,
    verifier: Verifier,
    fetch_spans: Callable[[list[int]], Awaitable[list[str]]],
    batch: int = 100,
    now: float | None = None,
) -> int:
    """Verify pending claims. Returns the count whose status was written."""
    done = 0
    seen: set[int] = set()
    while True:
        pending = await store.pull_unverified(limit=batch)
        fresh = [c for c in pending if c["id"] not in seen]
        if not fresh:
            break
        for claim in fresh:
            seen.add(claim["id"])
            spans = await fetch_spans(claim["archive_span_ids"])
            status, model = verifier.verify(claim["text"], spans)
            if status == "unverified":
                # fail-closed: stays unverified, retried in a future pass.
                continue
            await store.set_status(claim["id"], status, verifier_model=model, now=now)
            done += 1
    return done
