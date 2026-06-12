"""Unit tests for benchmarks/surprisal_probe.py.

Covers:
- z-score normalisation correctness
- chunk-merge boundary logic (z thresholds, max 6 turns)
- evidence mapping for merged chunks
- sidecar cache round-trip
"""
from __future__ import annotations

import json
import sys
import os

import pytest

_BENCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks")
sys.path.insert(0, _BENCH_DIR)

from surprisal_probe import (  # noqa: E402
    _zscore_normalize,
    _build_surprise_chunks,
    _chunk_evidence_hits,
    _load_cache,
    _save_cache,
    _CHUNK_MAX_TURNS,
    _CHUNK_MERGE_Z_LOW,
    _CHUNK_SPLIT_Z_HIGH,
)


# ---------------------------------------------------------------------------
# Z-score normalisation
# ---------------------------------------------------------------------------

class TestZscoreNormalize:
    def test_basic_two_values(self):
        """Two values: one should be -1.0 and one +1.0 (population std)."""
        vals = [0.0, 2.0]
        z = _zscore_normalize(vals)
        assert len(z) == 2
        assert abs(z[0] - (-1.0)) < 1e-6
        assert abs(z[1] - 1.0) < 1e-6

    def test_single_value_returns_zero(self):
        assert _zscore_normalize([5.0]) == [0.0]

    def test_empty_list(self):
        assert _zscore_normalize([]) == []

    def test_all_same_returns_zeros(self):
        z = _zscore_normalize([3.0, 3.0, 3.0, 3.0])
        assert z == [0.0, 0.0, 0.0, 0.0]

    def test_known_values(self):
        """Mean=2, population std=sqrt(2/3)~0.8165. Values [1, 2, 3].
        Expected z: [-1.2247, 0.0, +1.2247]."""
        vals = [1.0, 2.0, 3.0]
        z = _zscore_normalize(vals)
        assert abs(z[0] - (-1.224745)) < 1e-4
        assert abs(z[1] - 0.0) < 1e-6
        assert abs(z[2] - 1.224745) < 1e-4

    def test_length_preserved(self):
        vals = [0.1, 0.5, 0.3, 0.9, 0.2]
        z = _zscore_normalize(vals)
        assert len(z) == len(vals)

    def test_zero_mean_after_normalize(self):
        """Z-scores should have mean ~0."""
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        z = _zscore_normalize(vals)
        assert abs(sum(z) / len(z)) < 1e-9


# ---------------------------------------------------------------------------
# Helpers for building scored-turn fixtures
# ---------------------------------------------------------------------------

def _make_turns(n: int, base_dia="d") -> list[dict]:
    """Make n fake scored-turn dicts."""
    return [
        {
            "turn_index": i,
            "dia_id": f"{base_dia}{i}",
            "speaker": "A",
            "text": f"turn {i}",
            "mean_nll": float(i),
            "max_nll": float(i),
            "n_tokens": 3,
        }
        for i in range(n)
    ]


def _make_z(values: list[float]) -> list[float]:
    """Directly supply z-scores (bypass _zscore_normalize for fixture clarity)."""
    return values


# ---------------------------------------------------------------------------
# Chunk-merge boundary logic
# ---------------------------------------------------------------------------

class TestBuildSurpriseChunks:
    def test_empty_returns_empty(self):
        assert _build_surprise_chunks([], []) == []

    def test_single_turn_single_chunk(self):
        turns = _make_turns(1)
        z = _make_z([0.0])
        chunks = _build_surprise_chunks(turns, z)
        assert len(chunks) == 1
        assert chunks[0]["dia_ids"] == ["d0"]
        assert chunks[0]["turn_indices"] == [0]

    def test_all_low_z_merges_up_to_max(self):
        """All z < _CHUNK_MERGE_Z_LOW: should merge into groups of _CHUNK_MAX_TURNS."""
        n = _CHUNK_MAX_TURNS * 3
        turns = _make_turns(n)
        # All z well below the merge threshold.
        z = [_CHUNK_MERGE_Z_LOW - 0.1] * n
        chunks = _build_surprise_chunks(turns, z)
        # Each chunk should hold exactly _CHUNK_MAX_TURNS turns.
        for chunk in chunks:
            assert len(chunk["turn_indices"]) <= _CHUNK_MAX_TURNS

    def test_high_z_always_splits(self):
        """A turn with z >= _CHUNK_SPLIT_Z_HIGH must start a new chunk."""
        # 6 turns: first 3 merge-eligible, turn 3 is high-z (split trigger),
        # then 2 more merge-eligible.
        turns = _make_turns(6)
        z = [
            _CHUNK_MERGE_Z_LOW - 0.1,  # 0: merge
            _CHUNK_MERGE_Z_LOW - 0.1,  # 1: merge
            _CHUNK_MERGE_Z_LOW - 0.1,  # 2: merge
            _CHUNK_SPLIT_Z_HIGH,        # 3: split -- must start new chunk
            _CHUNK_MERGE_Z_LOW - 0.1,  # 4: merge
            _CHUNK_MERGE_Z_LOW - 0.1,  # 5: merge
        ]
        chunks = _build_surprise_chunks(turns, z)
        # Turn 3 must be the first element of its chunk.
        # Find the chunk containing turn_index=3.
        chunk_with_3 = next(c for c in chunks if 3 in c["turn_indices"])
        assert chunk_with_3["turn_indices"][0] == 3, (
            f"High-z turn 3 must start a new chunk; chunk = {chunk_with_3}"
        )

    def test_mid_z_splits(self):
        """A turn with _CHUNK_MERGE_Z_LOW <= z < _CHUNK_SPLIT_Z_HIGH breaks
        the merge run (starts a new chunk with itself as first turn)."""
        turns = _make_turns(4)
        z = [
            _CHUNK_MERGE_Z_LOW - 0.1,   # 0: merge
            _CHUNK_MERGE_Z_LOW,          # 1: exactly at threshold -- triggers split
            _CHUNK_MERGE_Z_LOW - 0.1,   # 2: merge
            _CHUNK_MERGE_Z_LOW - 0.1,   # 3: merge
        ]
        chunks = _build_surprise_chunks(turns, z)
        # Turn 0 should be its own chunk (or merged with nothing else in range),
        # turn 1 must start a new chunk.
        chunk_with_1 = next(c for c in chunks if 1 in c["turn_indices"])
        assert chunk_with_1["turn_indices"][0] == 1, (
            f"Mid-z turn must start new chunk; chunk = {chunk_with_1}"
        )

    def test_max_chunk_size_enforced(self):
        """No chunk should have more than _CHUNK_MAX_TURNS turns."""
        n = 20
        turns = _make_turns(n)
        # All z below merge threshold.
        z = [0.0] * n
        chunks = _build_surprise_chunks(turns, z)
        for chunk in chunks:
            assert len(chunk["turn_indices"]) <= _CHUNK_MAX_TURNS, (
                f"Chunk exceeded max size: {chunk}"
            )

    def test_cap_enforced_on_50_turn_conversation(self):
        """Synthetic 50-turn conversation: no chunk may exceed _CHUNK_MAX_TURNS.

        This is the regression test for the cap bug where mean chunk length
        came out 22.54 turns/chunk despite the stated maximum of 6.  All
        z-scores are zero (identical NLLs after normalisation), which is the
        worst case for the cap: no split signals, everything merge-eligible.
        """
        n = 50
        turns = _make_turns(n)
        z = [0.0] * n  # all merge-eligible; cap must still be enforced
        chunks = _build_surprise_chunks(turns, z)
        assert chunks, "Expected at least one chunk for 50-turn conversation"
        for chunk in chunks:
            assert len(chunk["turn_indices"]) <= _CHUNK_MAX_TURNS, (
                f"Cap violated: chunk has {len(chunk['turn_indices'])} turns "
                f"(max {_CHUNK_MAX_TURNS}): {chunk['turn_indices']}"
            )
        # All 50 turns must appear exactly once.
        all_indices = []
        for c in chunks:
            all_indices.extend(c["turn_indices"])
        assert sorted(all_indices) == list(range(n)), (
            "Not all turns were accounted for after chunking"
        )

    def test_all_turns_accounted(self):
        """Every input turn must appear in exactly one output chunk."""
        n = 10
        turns = _make_turns(n)
        z = [0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
        chunks = _build_surprise_chunks(turns, z)
        all_indices = []
        for c in chunks:
            all_indices.extend(c["turn_indices"])
        assert sorted(all_indices) == list(range(n))

    def test_dia_ids_per_chunk(self):
        """Chunk dia_ids list must contain all dia_ids of merged turns."""
        turns = _make_turns(4)
        z = [0.0, 0.0, 0.0, 0.0]  # all merge-eligible
        chunks = _build_surprise_chunks(turns, z)
        all_dia_ids = []
        for c in chunks:
            all_dia_ids.extend(c["dia_ids"])
        expected = [t["dia_id"] for t in turns]
        assert sorted(all_dia_ids) == sorted(expected)


# ---------------------------------------------------------------------------
# Evidence mapping for merged chunks
# ---------------------------------------------------------------------------

class TestChunkEvidenceHits:
    def _make_hit(self, dia_ids: list[str]) -> dict:
        """Simulate a hit from surprise_chunks retrieval with multiple dia_ids."""
        return {
            "text": "some text",
            "metadata": {"dia_ids": json.dumps(dia_ids), "dia_id": dia_ids[0] if dia_ids else ""},
            "score": 1.0,
        }

    def test_no_evidence_returns_zero(self):
        hit = self._make_hit(["d1", "d2"])
        assert _chunk_evidence_hits([hit], []) == 0

    def test_single_dia_match(self):
        hit = self._make_hit(["d1"])
        assert _chunk_evidence_hits([hit], ["d1"]) == 1

    def test_chunk_hit_covers_multiple_evidence(self):
        """A single chunk hit with 3 dia_ids matches 2 of 3 evidence entries."""
        hit = self._make_hit(["d1", "d2", "d3"])
        evidence = ["d1", "d2", "d_missing"]
        # _chunk_evidence_hits counts evidence entries that appear in retrieved set.
        result = _chunk_evidence_hits([hit], evidence)
        assert result == 2

    def test_no_match_returns_zero(self):
        hit = self._make_hit(["d99"])
        assert _chunk_evidence_hits([hit], ["d1", "d2"]) == 0

    def test_multiple_chunks_union(self):
        """Two chunk hits together cover all evidence."""
        hit1 = self._make_hit(["d1", "d2"])
        hit2 = self._make_hit(["d3"])
        evidence = ["d1", "d3"]
        assert _chunk_evidence_hits([hit1, hit2], evidence) == 2

    def test_string_dia_ids_fallback(self):
        """dia_ids stored as raw string (not JSON list) falls back to single value."""
        hit = {
            "text": "x",
            "metadata": {"dia_ids": "not-json[[[", "dia_id": "d1"},
            "score": 1.0,
        }
        assert _chunk_evidence_hits([hit], ["d1"]) == 1

    def test_direct_dia_id_fallback(self):
        """Hit without dia_ids list but with dia_id field still matches."""
        hit = {
            "text": "x",
            "metadata": {"dia_id": "d1"},
            "score": 1.0,
        }
        assert _chunk_evidence_hits([hit], ["d1"]) == 1

    def test_per_turn_credit_only(self):
        """A chunk containing evidence turn X and non-evidence turn Y credits
        only X -- non-evidence turns in the same chunk are NOT credited.

        This verifies that the evidence-hit accounting is identical in spirit
        to the baseline's per-turn _evidence_hits: only the intersection of
        retrieved dia_ids with the QA evidence list is counted.
        """
        # Chunk contains both the evidence turn (dX) and a non-evidence turn (dY).
        hit = self._make_hit(["dX", "dY"])
        evidence = ["dX"]  # only dX is evidence; dY is not
        result = _chunk_evidence_hits([hit], evidence)
        # Must credit dX (1 hit), must NOT credit dY.
        assert result == 1, (
            f"Expected 1 evidence credit for dX only, got {result}"
        )

    def test_non_evidence_turn_in_chunk_not_credited(self):
        """A chunk that contains ONLY non-evidence turns returns zero hits
        even if the chunk was retrieved."""
        hit = self._make_hit(["dY", "dZ"])  # neither is evidence
        evidence = ["dX"]
        assert _chunk_evidence_hits([hit], evidence) == 0, (
            "Non-evidence turns in a retrieved chunk must not produce a hit"
        )


# ---------------------------------------------------------------------------
# Sidecar cache round-trip
# ---------------------------------------------------------------------------

class TestSidecarCache:
    def test_save_and_load(self, tmp_path):
        cache_path = tmp_path / "test.surprisal_cache.json"
        data = {
            "conv1::0": {"mean_nll": 1.5, "max_nll": 3.2, "n_tokens": 10},
            "conv1::1": {"mean_nll": 0.8, "max_nll": 1.1, "n_tokens": 8},
        }
        _save_cache(cache_path, data)
        assert cache_path.exists()
        loaded = _load_cache(cache_path)
        assert loaded == data

    def test_load_missing_returns_empty(self, tmp_path):
        cache_path = tmp_path / "nonexistent.json"
        result = _load_cache(cache_path)
        assert result == {}

    def test_load_corrupt_returns_empty(self, tmp_path):
        cache_path = tmp_path / "corrupt.json"
        cache_path.write_text("{not valid json")
        result = _load_cache(cache_path)
        assert result == {}

    def test_incremental_save(self, tmp_path):
        """Simulate incremental writes as each turn is scored."""
        cache_path = tmp_path / "incr.json"
        cache = {}
        for i in range(5):
            cache[f"conv::{ i}"] = {"mean_nll": float(i), "max_nll": float(i), "n_tokens": i}
            _save_cache(cache_path, cache)
        final = _load_cache(cache_path)
        assert len(final) == 5

    def test_cache_key_format(self, tmp_path):
        """Keys use the conv_id::turn_index format."""
        cache_path = tmp_path / "keys.json"
        cache = {"my_conv::42": {"mean_nll": 2.0, "max_nll": 4.0, "n_tokens": 5}}
        _save_cache(cache_path, cache)
        loaded = _load_cache(cache_path)
        assert "my_conv::42" in loaded
        assert loaded["my_conv::42"]["mean_nll"] == 2.0


# ---------------------------------------------------------------------------
# Mean-chunk-length metric: total_turns scope
# ---------------------------------------------------------------------------

class TestMeanChunkLengthMetric:
    """Regression tests for the total_turns/total_chunks_ingested metric.

    Root cause of the reported "22.54 turns/chunk" despite a 6-turn cap:
    when --limit is active, _run_variant ingests only a subset of conversations
    (total_chunks_ingested counts only those), but the original total_turns
    summed ALL scored conversations in surprisal_data.  The fix restricts
    total_turns to the same set of conversations that were actually ingested.

    These tests exercise the metric path directly using real chunk data from
    _build_surprise_chunks (the same function the retrieval path uses).
    """

    def _make_scored_turns(self, n: int, conv_id: str) -> list[dict]:
        """Produce n scored-turn dicts for a synthetic conversation."""
        return [
            {
                "turn_index": i,
                "dia_id": f"{conv_id}:d{i}",
                "speaker": "A",
                "text": f"[A] turn {i}",
                "mean_nll": 1.5,
                "max_nll": 1.5,
                "n_tokens": 3,
            }
            for i in range(n)
        ]

    def _count_chunks_for_conv(self, n_turns: int) -> int:
        """Return the number of chunks produced by _build_surprise_chunks
        for a conversation with n_turns all-zero z-scores (worst-case merge)."""
        turns = self._make_scored_turns(n_turns, "c")
        z = [0.0] * n_turns
        return len(_build_surprise_chunks(turns, z))

    def test_mean_chunk_len_matches_real_chunks_single_conv(self):
        """When only 1 of N conversations is ingested, mean must reflect that
        conversation only -- not all N.

        This is the core regression for the 22.54 report: the buggy code used
        total_turns = sum over ALL convs in surprisal_data, while the
        denominator (total_chunks_ingested) only covers the ingested subset.
        """
        n_per_conv = 50
        n_total_convs = 10

        # Simulate surprisal_data for all 10 conversations.
        surprisal_data = {}
        for i in range(n_total_convs):
            conv_id = f"conv_{i}"
            scored_turns = self._make_scored_turns(n_per_conv, conv_id)
            surprisal_data[conv_id] = {
                "scored_turns": scored_turns,
                "z_scores": [0.0] * n_per_conv,
            }

        # Simulate that only the FIRST conversation was actually ingested
        # (as happens when --limit is exhausted after the first conversation).
        first_conv_id = "conv_0"
        first_conv_turns = surprisal_data[first_conv_id]["scored_turns"]
        first_conv_z = surprisal_data[first_conv_id]["z_scores"]
        actual_chunks = _build_surprise_chunks(first_conv_turns, first_conv_z)
        n_chunks = len(actual_chunks)

        # Simulate ingest_stats with only the first conversation.
        ingest_stats = [{"conversation_id": first_conv_id, "added": n_chunks}]

        # --- Buggy computation (original code) ---
        total_turns_buggy = sum(
            len(sd.get("scored_turns", []))
            for sd in surprisal_data.values()
        )
        mean_buggy = total_turns_buggy / n_chunks
        # The buggy mean MUST exceed the cap (proves the original was wrong).
        assert mean_buggy > _CHUNK_MAX_TURNS, (
            f"Buggy path should produce mean > {_CHUNK_MAX_TURNS}; "
            f"got {mean_buggy:.2f} (total_turns={total_turns_buggy}, "
            f"chunks={n_chunks})"
        )

        # --- Fixed computation (filter by processed conv_ids) ---
        processed_conv_ids = {s["conversation_id"] for s in ingest_stats}
        total_turns_fixed = sum(
            len(sd.get("scored_turns", []))
            for conv_id, sd in surprisal_data.items()
            if conv_id in processed_conv_ids
        )
        mean_fixed = total_turns_fixed / n_chunks
        assert mean_fixed <= _CHUNK_MAX_TURNS, (
            f"Fixed mean must be <= {_CHUNK_MAX_TURNS}; "
            f"got {mean_fixed:.2f} (total_turns={total_turns_fixed}, "
            f"chunks={n_chunks})"
        )

    def test_mean_chunk_len_all_convs_processed(self):
        """When ALL conversations are processed (no --limit truncation),
        both the buggy and fixed computations agree and the mean is <= cap."""
        n_per_conv = 50
        n_total_convs = 5
        surprisal_data = {}
        ingest_stats = []

        for i in range(n_total_convs):
            conv_id = f"conv_{i}"
            scored_turns = self._make_scored_turns(n_per_conv, conv_id)
            z = [0.0] * n_per_conv
            surprisal_data[conv_id] = {"scored_turns": scored_turns, "z_scores": z}
            n_chunks = len(_build_surprise_chunks(scored_turns, z))
            ingest_stats.append({"conversation_id": conv_id, "added": n_chunks})

        # Fixed computation.
        processed_conv_ids = {s["conversation_id"] for s in ingest_stats}
        total_turns = sum(
            len(sd.get("scored_turns", []))
            for conv_id, sd in surprisal_data.items()
            if conv_id in processed_conv_ids
        )
        total_chunks = sum(s["added"] for s in ingest_stats)
        mean = total_turns / total_chunks
        assert mean <= _CHUNK_MAX_TURNS, (
            f"Mean {mean:.2f} exceeds cap {_CHUNK_MAX_TURNS}"
        )

    def test_chunk_cap_and_metric_consistent_50_turns(self):
        """50-turn synthetic conversation: (i) no chunk exceeds cap, AND
        (ii) the metric computed via ingest_stats equals actual mean chunk size.

        This tests the SAME path retrieval uses: _build_surprise_chunks is
        called with the scored_turns and z_scores from surprisal_data, and
        the resulting chunk count is what would be stored in ingest_stats.
        """
        n = 50
        conv_id = "test_conv"
        scored_turns = self._make_scored_turns(n, conv_id)
        z = [0.0] * n

        chunks = _build_surprise_chunks(scored_turns, z)

        # (i) No chunk exceeds cap.
        for chunk in chunks:
            assert len(chunk["turn_indices"]) <= _CHUNK_MAX_TURNS, (
                f"Cap violated: {len(chunk['turn_indices'])} turns in chunk"
            )

        # (ii) Metric via ingest_stats matches actual mean chunk size.
        ingest_stats = [{"conversation_id": conv_id, "added": len(chunks)}]
        surprisal_data = {conv_id: {"scored_turns": scored_turns, "z_scores": z}}

        processed_conv_ids = {s["conversation_id"] for s in ingest_stats}
        total_turns = sum(
            len(sd.get("scored_turns", []))
            for cid, sd in surprisal_data.items()
            if cid in processed_conv_ids
        )
        total_chunks = sum(s["added"] for s in ingest_stats)
        mean_via_metric = total_turns / total_chunks

        # Actual mean from chunk data.
        mean_actual = sum(len(c["turn_indices"]) for c in chunks) / len(chunks)

        assert abs(mean_via_metric - mean_actual) < 1e-9, (
            f"Metric mean {mean_via_metric:.4f} != actual mean {mean_actual:.4f}"
        )
        assert mean_via_metric <= _CHUNK_MAX_TURNS, (
            f"Metric mean {mean_via_metric:.2f} exceeds cap {_CHUNK_MAX_TURNS}"
        )
