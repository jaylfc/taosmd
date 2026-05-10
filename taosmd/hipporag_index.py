"""HippoRAG-style retrieval over a per-conversation in-memory KG.

Port of the vector-only PPR variant of HippoRAG (no neural reranker, no
DSPy filter, no synonymy edges by default). One LLM call per session at
ingest extracts (subject, predicate, object, source_turn_ids) triples.
At retrieval time we score the query against pre-embedded facts, pick
the top-k, derive entity-node seed weights from those facts, run
Personalized PageRank over a small undirected graph (entity-entity from
fact co-occurrence + entity-passage from turn membership), and read off
the top-ranked passage nodes.

Reference: arXiv:2405.14831 + /tmp/memresearch/hipporag/src/hipporag/HippoRAG.py
(`graph_search_with_fact_entities` line 1407, `run_ppr` line 1572).

The index is single-conversation — LoCoMo does fresh vmem per conversation,
so HippoIndex follows that pattern rather than maintaining a global graph.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import httpx


_OPENIE_SYSTEM = (
    "Given a conversation session between speakers with numbered turns, your task is to "
    "extract Open Information Extraction triples (subject, predicate, object) capturing all factual "
    "claims, events, and relationships discussed. The triples will populate a knowledge graph used "
    "to answer questions about the conversation later, so completeness matters more than concision.\n"
    "Requirements:\n"
    "1. Each triple is a 3-tuple [subject, predicate, object]. Use lowercase strings; keep entity names "
    "consistent across triples (e.g. always 'sarah' not 'she'/'sarah').\n"
    "2. Avoid pronouns or ambiguous references — resolve them to the canonical entity name.\n"
    "3. Capture temporal context inside the predicate or object when relevant "
    "(e.g. ['sarah', 'visited tokyo on', 'march 2024']).\n"
    "4. Provide source_turn_ids: a list of integer turn IDs the triple was extracted from.\n"
    "5. Triples should collectively cover all substantive content — facts, decisions, plans, "
    "preferences, quantities, dates. Skip pleasantries.\n"
    "6. Multiple triples per turn are fine and expected for information-rich turns.\n"
    'Return JSON: {"triples": [{"subject": "...", "predicate": "...", "object": "...", "source_turn_ids": [n, ...]}, ...]}.'
)


_OPENIE_ONESHOT_INPUT = """Session conversation:
Date: 2:30 pm on 15 March, 2024

Turn 1:
Alice: Hey Bob! How was your trip to Tokyo?

Turn 2:
Bob: It was amazing! I spent 5 days there for the Global AI Innovation Symposium 2024. The conference at Tokyo University was incredible.

Turn 3:
Alice: That sounds exciting!

Turn 4:
Bob: Dr. Yamamoto from Sony AI wants to collaborate on our next project, with a $2 million budget.

Speaker names: Alice, Bob"""


_OPENIE_ONESHOT_OUTPUT = json.dumps({
    "triples": [
        {"subject": "bob", "predicate": "traveled to", "object": "tokyo", "source_turn_ids": [2]},
        {"subject": "bob", "predicate": "spent days in tokyo", "object": "5", "source_turn_ids": [2]},
        {"subject": "bob", "predicate": "attended", "object": "global ai innovation symposium 2024", "source_turn_ids": [2]},
        {"subject": "global ai innovation symposium 2024", "predicate": "held at", "object": "tokyo university", "source_turn_ids": [2]},
        {"subject": "global ai innovation symposium 2024", "predicate": "held in", "object": "march 2024", "source_turn_ids": [2]},
        {"subject": "dr. yamamoto", "predicate": "works at", "object": "sony ai", "source_turn_ids": [4]},
        {"subject": "dr. yamamoto", "predicate": "wants to collaborate with", "object": "bob", "source_turn_ids": [4]},
        {"subject": "bob-sony ai collaboration", "predicate": "has budget of", "object": "$2 million", "source_turn_ids": [4]},
    ]
})


def format_session_for_extraction(turns: list[dict], session_date: str) -> str:
    """Format LoCoMo session turns into the OpenIE prompt input (1-indexed)."""
    lines = [f"Date: {session_date}", ""]
    for i, turn in enumerate(turns, start=1):
        lines.append(f"Turn {i}:")
        lines.append(f"{turn.get('speaker', '')}: {turn.get('text', '')}")
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


async def _ollama_chat_json(
    client: httpx.AsyncClient, ollama_url: str, model: str,
    messages: list[dict], *, timeout: float = 120.0, num_predict: int = 8192,
) -> str:
    resp = await client.post(
        f"{ollama_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": num_predict},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data.get("message") or {}).get("content", "")


async def extract_session_triples(
    session_text: str, speaker_names: list[str], *,
    model: str, ollama_url: str, http_client: httpx.AsyncClient,
    timeout: float = 240.0,
) -> list[dict]:
    """Extract OpenIE triples from one session. Returns list of
    {subject, predicate, object, source_turn_ids}.

    Raises on transport / JSON parse failure; caller decides fallback.
    """
    user_msg = f"Session conversation:\n{session_text}\n\nSpeaker names: {', '.join(speaker_names)}"
    messages = [
        {"role": "system", "content": _OPENIE_SYSTEM},
        {"role": "user", "content": _OPENIE_ONESHOT_INPUT},
        {"role": "assistant", "content": _OPENIE_ONESHOT_OUTPUT},
        {"role": "user", "content": user_msg},
    ]

    raw = await _ollama_chat_json(
        http_client, ollama_url, model, messages,
        timeout=timeout, num_predict=12288,
    )
    parsed = json.loads(raw)
    triples: list[dict] = []
    for entry in parsed.get("triples", []) or []:
        s = (entry.get("subject") or "").strip().lower()
        p = (entry.get("predicate") or "").strip().lower()
        o = (entry.get("object") or "").strip().lower()
        if not s or not p or not o:
            continue
        turn_ids_raw = entry.get("source_turn_ids") or []
        turn_ids: list[int] = []
        for tid in turn_ids_raw:
            try:
                turn_ids.append(int(tid))
            except (TypeError, ValueError):
                continue
        if not turn_ids:
            continue
        triples.append({
            "subject": s, "predicate": p, "object": o, "source_turn_ids": turn_ids,
        })
    return triples


@dataclass
class HippoIndex:
    """In-memory HippoRAG index for one LoCoMo conversation.

    Lifecycle:
      1. add_passage(text, metadata) for each turn — registers the passage node
      2. add_triples(triples, session_first_global_idx, session_turn_count) per session
      3. finalize(embed_fn) — builds igraph + embeds all facts
      4. retrieve(query, query_emb, top_k) — PPR retrieval
    """
    damping: float = 0.5
    linking_top_k: int = 5
    passage_node_weight: float = 0.05

    passages: list[str] = field(default_factory=list)
    passage_metadata: list[dict] = field(default_factory=list)

    facts: list[tuple[str, str, str]] = field(default_factory=list)
    fact_passage_idxs: list[list[int]] = field(default_factory=list)
    entity_to_passage_idxs: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))

    fact_embeddings: Any = None
    graph: Any = None
    node_name_to_idx: dict[str, int] = field(default_factory=dict)
    passage_node_idxs: list[int] = field(default_factory=list)

    def add_passage(self, text: str, metadata: dict) -> int:
        """Register a passage (turn). Returns its passage index."""
        self.passages.append(text)
        self.passage_metadata.append(metadata)
        return len(self.passages) - 1

    def add_triples(
        self, triples: list[dict], session_first_passage_idx: int,
        session_turn_count: int,
    ) -> None:
        """Resolve session-local turn IDs to global passage indices and register
        the facts. Triples whose source_turn_ids don't fall in the session range
        are dropped (extraction hallucination)."""
        for tr in triples:
            local_turn_ids = tr.get("source_turn_ids", [])
            global_idxs: list[int] = []
            for ltid in local_turn_ids:
                gidx = session_first_passage_idx + (ltid - 1)
                if 0 <= ltid - 1 < session_turn_count and gidx < len(self.passages):
                    global_idxs.append(gidx)
            if not global_idxs:
                continue
            triple = (tr["subject"], tr["predicate"], tr["object"])
            self.facts.append(triple)
            self.fact_passage_idxs.append(global_idxs)
            for gidx in global_idxs:
                self.entity_to_passage_idxs[tr["subject"]].add(gidx)
                self.entity_to_passage_idxs[tr["object"]].add(gidx)

    async def finalize(self, embed_fn) -> None:
        """Embed all facts and build the igraph. Call once after ingest."""
        import numpy as np
        try:
            import igraph as ig
        except ImportError as exc:
            raise RuntimeError(
                "HippoIndex requires python-igraph. Install with "
                "`pip install python-igraph`."
            ) from exc

        if not self.facts:
            self.fact_embeddings = np.zeros((0, 1), dtype=np.float32)
            self.graph = ig.Graph()
            return

        fact_texts = [f"{s} {p} {o}" for s, p, o in self.facts]
        fact_vecs = await embed_fn(fact_texts)
        fact_vecs = np.asarray(fact_vecs, dtype=np.float32)
        norms = np.linalg.norm(fact_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.fact_embeddings = fact_vecs / norms

        entity_names = sorted(self.entity_to_passage_idxs.keys())
        entity_node_keys = [f"entity::{e}" for e in entity_names]
        passage_node_keys = [f"passage::{i}" for i in range(len(self.passages))]
        all_node_keys = entity_node_keys + passage_node_keys
        self.node_name_to_idx = {k: i for i, k in enumerate(all_node_keys)}
        self.passage_node_idxs = [
            self.node_name_to_idx[k] for k in passage_node_keys
        ]

        edge_weights: dict[tuple[int, int], float] = defaultdict(float)
        for (s, _p, o), passage_idxs in zip(self.facts, self.fact_passage_idxs):
            si = self.node_name_to_idx.get(f"entity::{s}")
            oi = self.node_name_to_idx.get(f"entity::{o}")
            if si is not None and oi is not None and si != oi:
                key = (min(si, oi), max(si, oi))
                edge_weights[key] += 1.0
            for pidx in passage_idxs:
                pi = self.node_name_to_idx[f"passage::{pidx}"]
                for ent_idx in (si, oi):
                    if ent_idx is None:
                        continue
                    key = (min(ent_idx, pi), max(ent_idx, pi))
                    edge_weights[key] += 1.0

        edges = list(edge_weights.keys())
        weights = [edge_weights[k] for k in edges]

        g = ig.Graph(n=len(all_node_keys), edges=edges, directed=False)
        g.vs["name"] = all_node_keys
        if weights:
            g.es["weight"] = weights
        self.graph = g

    async def retrieve(
        self, query: str, query_emb, top_k: int = 10,
    ) -> list[dict]:
        """PPR retrieval. Returns list of {text, metadata, score} hits."""
        import numpy as np

        if not self.passages:
            return []
        if self.graph is None or self.fact_embeddings is None:
            raise RuntimeError("HippoIndex.finalize() must be called before retrieve()")

        if not self.facts:
            return [
                {"text": self.passages[i], "metadata": self.passage_metadata[i], "score": 0.0}
                for i in range(min(top_k, len(self.passages)))
            ]

        q_vec = np.asarray(query_emb, dtype=np.float32)
        n = np.linalg.norm(q_vec)
        if n > 0:
            q_vec = q_vec / n
        fact_scores = self.fact_embeddings @ q_vec

        top_k_facts_n = min(self.linking_top_k, len(self.facts))
        top_fact_idxs = np.argsort(fact_scores)[-top_k_facts_n:][::-1]

        entity_weights: dict[str, float] = defaultdict(float)
        for fidx in top_fact_idxs:
            score = float(fact_scores[fidx])
            s, _p, o = self.facts[fidx]
            entity_weights[f"entity::{s}"] += score
            entity_weights[f"entity::{o}"] += score

        node_count = len(self.node_name_to_idx)
        reset = np.zeros(node_count, dtype=np.float64)
        for name, w in entity_weights.items():
            idx = self.node_name_to_idx.get(name)
            if idx is not None:
                reset[idx] = max(reset[idx], w)
        for pidx in self.passage_node_idxs:
            reset[pidx] += self.passage_node_weight

        total = reset.sum()
        if total <= 0:
            reset = np.ones(node_count, dtype=np.float64) / node_count
        else:
            reset = reset / total

        try:
            ppr_scores = self.graph.personalized_pagerank(
                damping=self.damping,
                reset=reset.tolist(),
                weights=self.graph.es["weight"] if "weight" in self.graph.edge_attributes() else None,
                directed=False,
                implementation="prpack",
            )
        except Exception:
            ppr_scores = self.graph.personalized_pagerank(
                damping=self.damping,
                reset=reset.tolist(),
                directed=False,
            )

        passage_scores = np.array(
            [ppr_scores[idx] for idx in self.passage_node_idxs], dtype=np.float64,
        )
        sorted_passage_idxs = np.argsort(passage_scores)[::-1]

        hits: list[dict] = []
        for rank, pidx in enumerate(sorted_passage_idxs[:top_k]):
            hits.append({
                "text": self.passages[pidx],
                "metadata": self.passage_metadata[pidx],
                "score": float(passage_scores[pidx]),
            })
        return hits
