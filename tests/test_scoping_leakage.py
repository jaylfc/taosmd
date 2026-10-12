"""Test for agent/project scoping leakage.

Verifies that vector memory search respects agent and project scoping,
and that there is zero cross-agent leakage when scoped correctly.
"""

from __future__ import annotations

import asyncio
import time


from taosmd.vector_memory import VectorMemory


# ---------------------------------------------------------------------------
# Helpers (mirroring tests/test_ttl_filter.py)
# ---------------------------------------------------------------------------


def _fake_embedder(vmem: VectorMemory) -> None:
    """Patch embed() with a deterministic 16-dim bag-of-words hash vector (no ONNX/QMD)."""

    async def _embed(text: str, task: str = "search_document") -> list[float]:
        # Simple bag-of-words hash: split by whitespace and hash each token
        tokens = text.lower().split()
        # Combine token hashes with XOR to get a single hash
        combined_hash = 0
        for token in tokens:
            combined_hash ^= hash(token)
        # Ensure we have a 64-bit value
        combined_hash &= 0xFFFFFFFFFFFFFFFF
        return [((combined_hash >> (i * 3)) & 0xFF) / 255.0 - 0.5 for i in range(16)]

    vmem.embed = _embed  # type: ignore[assignment]


def _make_store(tmp_path) -> VectorMemory:
    vmem = VectorMemory(
        db_path=str(tmp_path / "vec.db"),
        embed_mode="onnx",
    )
    asyncio.run(vmem.init())
    _fake_embedder(vmem)
    return vmem


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_scoping_leakage_pooled_vs_scoped(tmp_path):
    """Pooling leaks across agents; scoping to agent eliminates leakage.

    Also verifies project scoping and that expired rows are never returned.
    """
    vmem = _make_store(tmp_path)
    try:
        # Create 20 near-duplicate sentence pairs: one for Alice, one for Bob.
        base_sentences = [f"This is test sentence number {i}." for i in range(20)]
        # Make near-duplicates by appending agent name to create token difference
        alice_texts = [s + " alice" for s in base_sentences]
        bob_texts   = [s + " bob" for s in base_sentences]

        # Insert Alice's rows.
        alice_ids = []
        for text in alice_texts:
            rid = asyncio.run(vmem.add(text, metadata={"agent": "alice"}))
            alice_ids.append(rid)

        # Insert Bob's rows.
        bob_ids = []
        for text in bob_texts:
            rid = asyncio.run(vmem.add(text, metadata={"agent": "bob"}))
            bob_ids.append(rid)

        # Pooled control: search without scoping should sometimes return the other agent's row.
        pooled_cross_agent_hits = 0
        pooled_queries = 0
        for i, query in enumerate(alice_texts):
            # We query with Alice's text, but we expect that without scoping we might get Bob's row.
            hits = asyncio.run(vmem.search(query, limit=10, hybrid=False, fusion="none"))
            pooled_queries += 1
            for hit in hits:
                if hit["metadata"].get("agent") != "alice":
                    pooled_cross_agent_hits += 1
                    break  # Count at most one cross-agent hit per query (we just need to see leakage)
        pooled_rate = pooled_cross_agent_hits / pooled_queries if pooled_queries else 0
        print(f"Pooled cross-agent rate: {pooled_rate:.2f}")
        # Positive control: we must be able to see leakage at all.
        assert pooled_rate > 0, f"Pooled search should show leakage (rate={pooled_rate}) but got zero"

        # Scoped to Alice: should return zero cross-agent hits.
        scoped_cross_agent_hits = 0
        scoped_queries = 0
        scoped_alice_hits = 0  # count of queries that return Alice's matching row in top 10
        for i, query in enumerate(alice_texts):
            hits = asyncio.run(vmem.search(query, limit=10, hybrid=False, fusion="none", search_agents=["alice"]))
            scoped_queries += 1
            # Check for cross-agent hits.
            for hit in hits:
                if hit["metadata"].get("agent") != "alice":
                    scoped_cross_agent_hits += 1
                    break
            # Check if Alice's own row (alice_ids[i]) is in the top 10.
            hit_ids = {hit["id"] for hit in hits}
            if alice_ids[i] in hit_ids:
                scoped_alice_hits += 1
        scoped_rate = scoped_cross_agent_hits / scoped_queries if scoped_queries else 0
        print(f"Scoped cross-agent rate: {scoped_rate:.2f}")
        assert scoped_cross_agent_hits == 0, f"Scoped search must have zero cross-agent hits (rate={scoped_rate})"
        # Scoping must not empty the result: at least 15 of 20 queries should return Alice's row.
        assert scoped_alice_hits >= 15, f"Expected at least 15 hits for Alice's rows, got {scoped_alice_hits}"

        # Same two assertions through taosmd.retrieval.retrieve
        # We pass our vector store as the only source to isolate the vector memory scoping.
        from taosmd.retrieval import retrieve

        sources = {"vector": vmem}

        # Pooled control via retrieve (should leak)
        pooled_cross_agent_hits_retrieve = 0
        pooled_queries_retrieve = 0
        for i, query in enumerate(alice_texts):
            hits = asyncio.run(retrieve(
                query,
                limit=10,
                agent=None,  # No agent filter -> pooled
                project=None,
                search_agents=None,
                sources=sources,
                fusion="none",
            ))
            pooled_queries_retrieve += 1
            for hit in hits:
                # The hit's metadata is the original row dict, which has a 'metadata' field containing the user metadata.
                if hit.get("metadata", {}).get("metadata", {}).get("agent") != "alice":
                    pooled_cross_agent_hits_retrieve += 1
                    break
        pooled_rate_retrieve = pooled_cross_agent_hits_retrieve / pooled_queries_retrieve if pooled_queries_retrieve else 0
        print(f"Pooled cross-agent rate (retrieve): {pooled_rate_retrieve:.2f}")
        assert pooled_rate_retrieve > 0, f"Pooled retrieve should show leakage (rate={pooled_rate_retrieve}) but got zero"

        # Scoped via retrieve
        scoped_cross_agent_hits_retrieve = 0
        scoped_queries_retrieve = 0
        scoped_alice_hits_retrieve = 0
        for i, query in enumerate(alice_texts):
            hits = asyncio.run(retrieve(
                query,
                limit=10,
                agent=None,
                project=None,
                search_agents=["alice"],
                sources=sources,
                fusion="none",
            ))
            scoped_queries_retrieve += 1
            # Check for cross-agent hits.
            for hit in hits:
                if hit.get("metadata", {}).get("metadata", {}).get("agent") != "alice":
                    scoped_cross_agent_hits_retrieve += 1
                    break
            # Check if Alice's own row (alice_ids[i]) is in the top 10.
            hit_ids = {hit.get("metadata", {}).get("id") for hit in hits}
            if alice_ids[i] in hit_ids:
                scoped_alice_hits_retrieve += 1
        scoped_rate_retrieve = scoped_cross_agent_hits_retrieve / scoped_queries_retrieve if scoped_queries_retrieve else 0
        print(f"Scoped cross-agent rate (retrieve): {scoped_rate_retrieve:.2f}")
        assert scoped_cross_agent_hits_retrieve == 0, f"Scoped retrieve must have zero cross-agent hits (rate={scoped_rate_retrieve})"
        assert scoped_alice_hits_retrieve >= 15, f"Expected at least 15 hits for Alice's rows via retrieve, got {scoped_alice_hits_retrieve}"

        # Project scoping: 10 rows with project p1, 10 with project p2; search with project=p1 returns no p2 rows.
        # We'll reuse the same VectorMemory? Better to create a new one to avoid interference.
        # But we can just add more rows to the same store and then test.
        # Let's add 10 rows for project p1 (agent alice) and 10 for project p2 (agent alice).
        p1_texts = [f"Project p1 sentence {i}." for i in range(10)]
        p2_texts = [f"Project p2 sentence {i}." for i in range(10)]
        for text in p1_texts:
            asyncio.run(vmem.add(text, metadata={"agent": "alice", "project": "p1"}))
        for text in p2_texts:
            asyncio.run(vmem.add(text, metadata={"agent": "alice", "project": "p2"}))

        # Search with project=p1 should return zero p2 rows.
        project_cross_hits = 0
        for i, query in enumerate(p1_texts):
            hits = asyncio.run(vmem.search(query, limit=10, hybrid=False, fusion="none", project="p1"))
            for hit in hits:
                if hit["metadata"].get("project") == "p2":
                    project_cross_hits += 1
                    break
        assert project_cross_hits == 0, f"Project scoping leaked: found {project_cross_hits} p2 rows when querying p1"

        # Expired rows: set valid_to to a past timestamp
        # We'll test that expired rows (via valid_to) are never returned under either scope.
        past = time.time() - 3600
        expired_id = asyncio.run(vmem.add("Expired row", metadata={"agent": "alice"}))
        active_id  = asyncio.run(vmem.add("Active row",  metadata={"agent": "alice"}))
        # Make the expired row expired by setting valid_to to past
        asyncio.run(vmem.supersede(expired_id, ended_at=past))

        # Search for "Expired row" should not return it, even without scoping.
        hits = asyncio.run(vmem.search("Expired row", limit=10, hybrid=False, fusion="none"))
        assert all(hit["id"] != expired_id for hit in hits), "Expired row must not be returned in pooled search"
        # Search with scoping to alice should also not return it.
        hits = asyncio.run(vmem.search("Expired row", limit=10, hybrid=False, fusion="none", search_agents=["alice"]))
        assert all(hit["id"] != expired_id for hit in hits), "Expired row must not be returned in scoped search"
        # Active row should be returned.
        hits = asyncio.run(vmem.search("Active row", limit=10, hybrid=False, fusion="none", search_agents=["alice"]))
        assert any(hit["id"] == active_id for hit in hits), "Active row must be returned in scoped search"

    finally:
        asyncio.run(vmem.close())
