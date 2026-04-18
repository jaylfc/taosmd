"""Cross-Memory Insight Synthesis (taOSmd).

Clusters related knowledge graph nodes, then uses an LLM to synthesize
higher-order insights from each cluster. Insights have confidence scores
that reinforce on re-discovery and decay over time.

Think of this as meta-cognition over the memory store: "given everything
I know about X, what patterns or conclusions emerge?"
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .knowledge_graph import TemporalKnowledgeGraph

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS insights (
    id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL UNIQUE,
    cluster_label TEXT NOT NULL,
    insight_text TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    source_triple_ids TEXT NOT NULL DEFAULT '[]',
    reinforcement_count INTEGER NOT NULL DEFAULT 0,
    last_reinforced_at REAL,
    created_at REAL NOT NULL,
    last_decayed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_insights_cluster ON insights(cluster_label);
CREATE INDEX IF NOT EXISTS idx_insights_confidence ON insights(confidence DESC);
"""

# Decay and reinforcement parameters
WEEKLY_DECAY_RATE = 0.05          # Lose 5% confidence per week
REINFORCEMENT_BOOST = 0.1         # Gain when re-discovered
MIN_CONFIDENCE = 0.05             # Floor before eviction
CLUSTER_JACCARD_THRESHOLD = 0.3   # Min similarity to be in same cluster


def _fingerprint(text: str) -> str:
    """Content fingerprint for deduplication."""
    normalised = " ".join(text.lower().split())
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


def _insight_id() -> str:
    return hashlib.sha256(f"{time.time()}".encode()).hexdigest()[:16]


def cluster_entities_by_predicate(
    triples: list[dict],
    min_cluster_size: int = 2,
) -> dict[str, list[dict]]:
    """Group triples into clusters by shared predicates.

    Simple but effective: entities connected by the same predicate type
    are likely related (e.g., all "works_on" triples form a project cluster).
    """
    clusters: dict[str, list[dict]] = {}
    for triple in triples:
        pred = triple.get("predicate", "unknown")
        if pred not in clusters:
            clusters[pred] = []
        clusters[pred].append(triple)

    # Filter to clusters with enough members
    return {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}


def cluster_by_jaccard(
    triples: list[dict],
    threshold: float = CLUSTER_JACCARD_THRESHOLD,
) -> list[list[dict]]:
    """Cluster triples by Jaccard similarity of their text content.

    Fallback for when predicate-based clustering is too coarse.
    """
    if not triples:
        return []

    # Extract text representation for each triple
    texts = []
    for t in triples:
        subject = t.get("subject_name", t.get("subject", ""))
        predicate = t.get("predicate", "")
        obj = t.get("object_name", t.get("object", ""))
        texts.append(f"{subject} {predicate} {obj}".lower().split())

    # Greedy single-linkage clustering
    assigned = [False] * len(triples)
    clusters: list[list[dict]] = []

    for i in range(len(triples)):
        if assigned[i]:
            continue
        cluster = [triples[i]]
        assigned[i] = True

        for j in range(i + 1, len(triples)):
            if assigned[j]:
                continue
            # Jaccard between word sets
            words_i = set(texts[i])
            words_j = set(texts[j])
            if not words_i or not words_j:
                continue
            jaccard = len(words_i & words_j) / len(words_i | words_j)
            if jaccard >= threshold:
                cluster.append(triples[j])
                assigned[j] = True

        if len(cluster) >= 2:
            clusters.append(cluster)

    return clusters


class InsightStore:
    """SQLite-backed store for synthesized insights."""

    def __init__(self, db_path: str | Path = "data/insights.db"):
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

    async def reflect(
        self,
        kg: TemporalKnowledgeGraph,
        entity: str | None = None,
        llm_url: str = "http://localhost:11434",
        model: str = "qwen3:4b",
        agent_name: str = "",
    ) -> list[dict]:
        """Run reflection over KG triples, synthesizing insights.

        If entity is provided, reflects only on that entity's neighbourhood.
        Otherwise reflects on all active triples.
        """
        from .agents import is_task_enabled  # noqa: PLC0415

        # Gate: named agents with reflect disabled return early.
        # Anonymous callers (agent_name is "") always run.
        if agent_name and not is_task_enabled(agent_name, "reflect"):
            logger.debug(
                "InsightStore.reflect: reflect disabled for agent=%r, skipping",
                agent_name,
            )
            return []

        # Gather triples
        if entity:
            triples = await kg.query_entity(entity, track_access=False)
        else:
            triples = await kg.timeline(limit=100)

        if len(triples) < 2:
            return []

        # Cluster by predicate first, fall back to Jaccard
        pred_clusters = cluster_entities_by_predicate(triples)
        if not pred_clusters:
            jaccard_clusters = cluster_by_jaccard(triples)
            pred_clusters = {f"cluster_{i}": c for i, c in enumerate(jaccard_clusters)}

        insights = []
        for label, cluster_triples in pred_clusters.items():
            insight = await self._synthesize_cluster(
                label, cluster_triples, llm_url, model
            )
            if insight:
                insights.append(insight)

        return insights

    async def _synthesize_cluster(
        self,
        label: str,
        triples: list[dict],
        llm_url: str,
        model: str,
    ) -> dict | None:
        """LLM synthesizes one insight from a cluster of triples."""
        # Format triples for the LLM
        facts = []
        triple_ids = []
        for t in triples:
            subject = t.get("subject_name", t.get("subject", "?"))
            predicate = t.get("predicate", "?")
            obj = t.get("object_name", t.get("object", "?"))
            facts.append(f"  {subject} {predicate} {obj}")
            if "id" in t:
                triple_ids.append(t["id"])

        facts_text = "\n".join(facts[:20])  # Cap at 20 facts

        prompt = f"""Given these related facts, synthesize one higher-order insight — a pattern, trend, or conclusion that isn't stated in any single fact but emerges from seeing them together.

Facts ({label}):
{facts_text}

Respond with ONLY the insight in one sentence. Be specific and actionable."""

        try:
            import httpx
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{llm_url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False,
                          "options": {"temperature": 0.4, "num_predict": 100}},
                )
                if resp.status_code != 200:
                    return None

                insight_text = resp.json().get("response", "").strip()
                if not insight_text or len(insight_text) < 10:
                    return None

        except Exception as e:
            logger.debug("LLM reflect failed for cluster %s: %s", label, e)
            return None

        # Check for existing insight with same fingerprint
        fp = _fingerprint(insight_text)
        existing = self._conn.execute(
            "SELECT * FROM insights WHERE fingerprint = ?", (fp,)
        ).fetchone()

        if existing:
            # Reinforce existing insight
            new_conf = min(1.0, existing["confidence"] + REINFORCEMENT_BOOST * (1 - existing["confidence"]))
            self._conn.execute(
                """UPDATE insights SET confidence = ?, reinforcement_count = reinforcement_count + 1,
                   last_reinforced_at = ? WHERE id = ?""",
                (new_conf, time.time(), existing["id"]),
            )
            self._conn.commit()
            return {
                "id": existing["id"],
                "insight": insight_text,
                "confidence": round(new_conf, 3),
                "reinforced": True,
                "cluster": label,
            }

        # New insight
        insight_id = _insight_id()
        now = time.time()
        self._conn.execute(
            """INSERT INTO insights (id, fingerprint, cluster_label, insight_text, confidence,
               source_triple_ids, created_at)
               VALUES (?, ?, ?, ?, 0.5, ?, ?)""",
            (insight_id, fp, label, insight_text, json.dumps(triple_ids), now),
        )
        self._conn.commit()

        return {
            "id": insight_id,
            "insight": insight_text,
            "confidence": 0.5,
            "reinforced": False,
            "cluster": label,
        }

    async def decay_all(self) -> int:
        """Apply weekly confidence decay to all insights. Returns count decayed."""
        now = time.time()
        week_ago = now - 7 * 86400

        rows = self._conn.execute(
            "SELECT id, confidence, last_decayed_at FROM insights WHERE confidence > ?",
            (MIN_CONFIDENCE,),
        ).fetchall()

        decayed = 0
        for row in rows:
            last_decay = row["last_decayed_at"] or row["id"]  # Never decayed
            # Only decay if at least a week since last decay
            if isinstance(last_decay, float) and last_decay > week_ago:
                continue

            new_conf = max(MIN_CONFIDENCE, row["confidence"] * (1 - WEEKLY_DECAY_RATE))
            self._conn.execute(
                "UPDATE insights SET confidence = ?, last_decayed_at = ? WHERE id = ?",
                (new_conf, now, row["id"]),
            )
            decayed += 1

        if decayed:
            self._conn.commit()
        return decayed

    async def evict_stale(self) -> int:
        """Remove insights below minimum confidence. Returns count removed."""
        cursor = self._conn.execute(
            "DELETE FROM insights WHERE confidence <= ?", (MIN_CONFIDENCE,)
        )
        self._conn.commit()
        return cursor.rowcount

    async def get_insights(
        self,
        cluster: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 20,
    ) -> list[dict]:
        """Retrieve insights, optionally filtered."""
        if cluster:
            rows = self._conn.execute(
                """SELECT * FROM insights WHERE cluster_label = ? AND confidence >= ?
                   ORDER BY confidence DESC LIMIT ?""",
                (cluster, min_confidence, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM insights WHERE confidence >= ? ORDER BY confidence DESC LIMIT ?",
                (min_confidence, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    async def stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) as n FROM insights").fetchone()["n"]
        avg_conf = self._conn.execute("SELECT AVG(confidence) as avg FROM insights").fetchone()["avg"]
        return {
            "total_insights": total,
            "avg_confidence": round(avg_conf or 0, 3),
        }
