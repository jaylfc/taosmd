"""Zero-loss sqlite store for verifiable claims.

A claim is an extracted fact plus the archive spans it derives from and a
verification status. Status changes; rows are never deleted (zero-loss). The
store is a compilation artifact, rebuildable from the archive.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from taosmd import _db, migrations

VALID_STATUSES = ("unverified", "supported", "partial", "unsupported", "contradicted")
# What counts as a hallucination for the live rate (checked but not supported).
_HALLUCINATED = ("unsupported", "partial", "contradicted")


def _normalize(text: str) -> str:
    """Canonical form used for claim dedup (issue #653).

    Exact-normalized, not fuzzy: case and whitespace variants of the same
    sentence merge; paraphrases stay distinct rows.
    """
    return " ".join(text.split()).lower()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    norm TEXT,
    archive_span_ids TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'unverified',
    verifier_model TEXT,
    last_checked REAL,
    source_extractor TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);
"""


class ClaimStore:
    def __init__(self, db_path: str | Path = "data/claims.db"):
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    async def init(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = _db.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        _db.run_schema(self._conn, _SCHEMA)
        self._conn.commit()
        migrations.migrate(self._conn, "claims")

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    async def add_claim(self, text: str, archive_span_ids: list[int],
                        source_extractor: str, now: float | None = None) -> int:
        """Insert a claim, or merge into the existing row for the same text.

        Dedup at insert (issue #653): a field report measured 10k rows for 4k
        distinct texts because the same sentence re-extracted every turn got a
        fresh row. Merging ``archive_span_ids`` into the existing row keeps the
        zero-loss property (no row or span is dropped) and stops the corpus
        (and any hallucination-rate metric computed over it) inflating. The
        merged row keeps its original status: a re-add never resets a verdict
        the verify pass already made.
        """
        norm = _normalize(text)
        # The lookup-then-insert/update runs inside a single BEGIN IMMEDIATE
        # transaction so it is atomic across connections/processes: a second
        # writer can't slip a same-norm insert between our miss and our insert
        # (idx_claims_norm is non-unique, so nothing else stops the duplicate).
        # Mirrors knowledge_graph.add_entity; busy_timeout makes the competing
        # writer wait rather than race.
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT id, archive_span_ids FROM claims WHERE norm = ? ORDER BY id LIMIT 1",
                (norm,),
            ).fetchone()
            if row is not None:
                merged = sorted({*json.loads(row["archive_span_ids"]), *archive_span_ids})
                self._conn.execute(
                    "UPDATE claims SET archive_span_ids = ? WHERE id = ?",
                    (json.dumps(merged), row["id"]),
                )
                self._conn.commit()
                return int(row["id"])
            cur = self._conn.execute(
                "INSERT INTO claims (text, norm, archive_span_ids, status, source_extractor, created_at)"
                " VALUES (?, ?, ?, 'unverified', ?, ?)",
                (text, norm, json.dumps(archive_span_ids), source_extractor,
                 now if now is not None else time.time()),
            )
            self._conn.commit()
            return int(cur.lastrowid)
        except BaseException:
            self._conn.rollback()
            raise

    async def get(self, claim_id: int) -> dict | None:
        row = self._conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
        return self._row(row) if row else None

    async def set_status(self, claim_id: int, status: str, verifier_model: str,
                         now: float | None = None) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"unknown claim status: {status!r}")
        self._conn.execute(
            "UPDATE claims SET status = ?, verifier_model = ?, last_checked = ? WHERE id = ?",
            (status, verifier_model, now if now is not None else time.time(), claim_id),
        )
        self._conn.commit()

    async def pull_unverified(self, limit: int = 100) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM claims WHERE status = 'unverified' ORDER BY id LIMIT ?", (limit,)
        ).fetchall()
        return [self._row(r) for r in rows]

    async def status_for_spans(self, span_ids: list[int]) -> str | None:
        """Worst status among claims backed by any of these archive spans.

        Used by the recall gate to judge a hit by the claims its source spans
        back. Worst-wins so one unsupported claim demotes the hit. None when no
        claim references these spans (a raw, non-claim memory)."""
        if not span_ids:
            return None
        rows = self._conn.execute("SELECT archive_span_ids, status FROM claims").fetchall()
        want = set(span_ids)
        order = {s: i for i, s in enumerate(
            ("supported", "unverified", "partial", "contradicted", "unsupported"))}
        worst = None
        for r in rows:
            if want & set(json.loads(r["archive_span_ids"])):
                if worst is None or order[r["status"]] > order[worst]:
                    worst = r["status"]
        return worst

    async def rate(self) -> dict:
        counts: dict = {s: 0 for s in VALID_STATUSES}
        for r in self._conn.execute("SELECT status, COUNT(*) c FROM claims GROUP BY status"):
            counts[r["status"]] = r["c"]
        checked = sum(counts[s] for s in VALID_STATUSES if s != "unverified")
        hall = sum(counts[s] for s in _HALLUCINATED)
        counts["hallucination_rate"] = (hall / checked) if checked else 0.0
        # Extractor breakdown (issue #653 item 4): how each claim row was
        # produced (source_extractor column: regex / llm / unknown). This is
        # per-claim provenance stored with the table; it is not the persistent
        # counterpart of memory_extractor.extraction_stats() -- that counts
        # LLM extraction outcomes per call (llm, regex_fallback,
        # regex_direct) -- and it is not an LLM-health signal, because
        # source_extractor is fixed when the claim row is inserted, not
        # observed per LLM call.
        counts["by_extractor"] = {
            (r["source_extractor"] or "unknown"): r["c"]
            for r in self._conn.execute(
                "SELECT source_extractor, COUNT(*) c FROM claims GROUP BY source_extractor"
            )
        }
        return counts

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        return {
            "id": r["id"], "text": r["text"],
            "archive_span_ids": json.loads(r["archive_span_ids"]),
            "status": r["status"], "verifier_model": r["verifier_model"],
            "last_checked": r["last_checked"], "source_extractor": r["source_extractor"],
            "created_at": r["created_at"],
        }
