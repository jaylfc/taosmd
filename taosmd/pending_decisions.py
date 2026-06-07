"""Pending-decisions queue for low-confidence KG updates.

Conceptual gap this closes: ``TemporalKnowledgeGraph.add_triple_with_contradiction_check``
silently auto-resolves contradictions on singular predicates by invalidating
the old triple and writing the new one. That's correct behaviour when the
new claim is high-confidence (a directly-quoted user statement, say) but
dangerous when it's a lower-confidence inference from a nightly catalog
pass — the librarian might rewrite a real fact the user gave us last week
based on an ambiguous summary from yesterday.

This module is the safety net. The store lives alongside the KG in the
same SQLite file (so it travels with the user's data, no extra DB to
back up). Low-confidence contradictions get deferred here instead of
auto-resolving; the user reviews them via ``taosmd review``.

The store is intentionally small. Three operations matter:

  - ``defer(...)``  -- write a pending decision; called from the KG when
    a contradiction is detected on a confidence-below-threshold update.
  - ``list_pending()`` -- enumerate unresolved decisions for the CLI / agent
    startup check.
  - ``resolve(id, action, note)`` -- mark a decision resolved with one of
    ``accepted``, ``rejected``, ``modified``. The CLI is responsible for
    invoking the matching KG operation (invalidate/keep/etc); this store
    only records the resolution, it doesn't mutate the KG. This separation
    keeps the store re-usable from non-CLI surfaces (an agent UI could
    resolve via the API and trigger its own KG update).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS kg_pending_decisions (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    new_object TEXT NOT NULL,
    new_triple_confidence REAL NOT NULL DEFAULT 1.0,
    old_triple_ids_json TEXT NOT NULL DEFAULT '[]',
    suggested_action TEXT NOT NULL,
    evidence TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    detection_confidence REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    resolved_at REAL,
    resolution TEXT,
    resolution_note TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_pending_unresolved ON kg_pending_decisions(resolved_at);
CREATE INDEX IF NOT EXISTS idx_pending_subject ON kg_pending_decisions(subject);
"""


VALID_KINDS = {"contradiction", "low_confidence_update", "duplicate_candidate"}
VALID_ACTIONS = {"invalidate_old_add_new", "keep_both", "reject_new", "modify"}
VALID_RESOLUTIONS = {"accepted", "rejected", "modified"}


def _pending_id(subject: str, predicate: str, new_object: str, created_at: float) -> str:
    """Stable-ish ID that lets the deferring code idempotently re-queue the
    same conflict without duplicating rows within a short window.

    The created_at is rounded to the day so two pipeline runs on the same
    afternoon dedupe but two different days do not.
    """
    day_bucket = int(created_at // 86400)
    raw = f"{subject}|{predicate}|{new_object}|{day_bucket}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class PendingDecisionsStore:
    """SQLite-backed queue of KG updates that need user confirmation."""

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    async def init(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    async def defer(
        self,
        *,
        kind: str,
        subject: str,
        predicate: str,
        new_object: str,
        old_triple_ids: list[str],
        suggested_action: str,
        evidence: str = "",
        source: str = "",
        new_triple_confidence: float = 1.0,
        detection_confidence: float = 1.0,
    ) -> str:
        """Queue a pending decision. Returns the row id.

        Same-day duplicates collapse onto the existing row (later evidence
        overwrites the evidence field but leaves the id stable) so a noisy
        nightly pipeline doesn't spam the queue.
        """
        if kind not in VALID_KINDS:
            raise ValueError(f"unknown kind {kind!r}; must be one of {VALID_KINDS}")
        if suggested_action not in VALID_ACTIONS:
            raise ValueError(f"unknown suggested_action {suggested_action!r}; must be one of {VALID_ACTIONS}")
        if self._conn is None:
            raise RuntimeError("PendingDecisionsStore not initialized; call init() first")

        now = time.time()
        pid = _pending_id(subject, predicate, new_object, now)
        old_ids_json = json.dumps(list(old_triple_ids))

        self._conn.execute(
            """INSERT INTO kg_pending_decisions
                 (id, kind, subject, predicate, new_object, new_triple_confidence,
                  old_triple_ids_json, suggested_action, evidence, source,
                  detection_confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 evidence = excluded.evidence,
                 source = excluded.source,
                 detection_confidence =
                   MAX(kg_pending_decisions.detection_confidence,
                       excluded.detection_confidence)
               WHERE kg_pending_decisions.resolved_at IS NULL""",
            (pid, kind, subject, predicate, new_object, new_triple_confidence,
             old_ids_json, suggested_action, evidence, source,
             detection_confidence, now),
        )
        self._conn.commit()
        return pid

    async def list_pending(
        self,
        *,
        subject: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return unresolved decisions, newest first."""
        if self._conn is None:
            raise RuntimeError("PendingDecisionsStore not initialized; call init() first")
        if subject is not None:
            rows = self._conn.execute(
                """SELECT * FROM kg_pending_decisions
                   WHERE resolved_at IS NULL AND subject = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (subject, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM kg_pending_decisions
                   WHERE resolved_at IS NULL
                   ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def get(self, pending_id: str) -> dict[str, Any] | None:
        if self._conn is None:
            raise RuntimeError("PendingDecisionsStore not initialized; call init() first")
        row = self._conn.execute(
            "SELECT * FROM kg_pending_decisions WHERE id = ?",
            (pending_id,),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    async def resolve(
        self,
        pending_id: str,
        *,
        resolution: str,
        note: str = "",
    ) -> bool:
        """Mark a pending decision resolved. Returns True on success.

        Note: this store records the resolution but does NOT mutate the KG.
        The caller (CLI or agent) is responsible for invoking the matching
        ``invalidate`` / ``add_triple`` operation against the KG. Keeping
        the store and the KG operation separate means a future agent UI can
        resolve programmatically without rebuilding the CLI workflow.
        """
        if resolution not in VALID_RESOLUTIONS:
            raise ValueError(
                f"unknown resolution {resolution!r}; must be one of {VALID_RESOLUTIONS}"
            )
        if self._conn is None:
            raise RuntimeError("PendingDecisionsStore not initialized; call init() first")
        cur = self._conn.execute(
            """UPDATE kg_pending_decisions
                  SET resolved_at = ?, resolution = ?, resolution_note = ?
                WHERE id = ? AND resolved_at IS NULL""",
            (time.time(), resolution, note, pending_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    async def stats(self) -> dict[str, int]:
        if self._conn is None:
            raise RuntimeError("PendingDecisionsStore not initialized; call init() first")
        total = self._conn.execute(
            "SELECT COUNT(*) AS n FROM kg_pending_decisions"
        ).fetchone()["n"]
        pending = self._conn.execute(
            "SELECT COUNT(*) AS n FROM kg_pending_decisions WHERE resolved_at IS NULL"
        ).fetchone()["n"]
        return {"total": total, "pending": pending, "resolved": total - pending}

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        try:
            d["old_triple_ids"] = json.loads(d.pop("old_triple_ids_json"))
        except (json.JSONDecodeError, KeyError):
            d["old_triple_ids"] = []
        return d
