"""Tests for the MemX-inspired low-confidence rejection gate (Lever 1).

MemX rule R1: abstain BEFORE generation when retrieval confidence is low.
Concretely, if the lexical (FTS / BM25 / keyword) recall set is EMPTY AND the
maximum dense cosine similarity across the retrieved candidates is below a
threshold tau, return an empty (abstain) retrieval result so the generator
says it does not know rather than fabricating.

The gate is opt-in and env-gated (TAOSMD_REJECT_GATE, TAOSMD_REJECT_TAU),
DEFAULT-OFF. When off, retrieve() behaviour is byte-for-byte unchanged.
"""

from __future__ import annotations

import asyncio

from taosmd.retrieval import apply_reject_gate, retrieve


# ---------------------------------------------------------------------------
# Mock sources that let us drive the gate's two inputs independently:
#   - whether any LEXICAL source returns a hit (archive / kg / catalog)
#   - the maximum DENSE cosine similarity from the vector source
# ---------------------------------------------------------------------------


class VectorWithSim:
    """Vector source returning a single hit with a controllable similarity."""

    def __init__(self, similarity: float):
        self._similarity = similarity

    async def search(self, query, limit=5, hybrid=True, fusion="rrf",
                     project=None, search_agents=None):
        return [
            {"id": 1, "text": "candidate memory", "similarity": self._similarity,
             "metadata": {}, "created_at": 1744531200.0},
        ]


class EmptyLexical:
    """Lexical source (archive-shaped) that never returns a hit."""

    async def search_fts(self, query, limit=20):
        return []


class HitLexical:
    """Lexical source (archive-shaped) that always returns a hit."""

    async def search_fts(self, query, limit=20):
        return [{"id": 9, "summary": "a keyword match", "data_json": "{}",
                 "event_type": "conversation", "timestamp": 1744531200.0}]


def _norm(source: str, source_score: float, text: str = "x") -> dict:
    """Build a minimal normalised result dict for the pure-helper tests."""
    return {
        "text": text,
        "source": source,
        "source_id": "1",
        "rank": 0,
        "source_score": source_score,
        "metadata": {},
        "rrf_score": 0.01,
    }


# ---------------------------------------------------------------------------
# Pure-helper tests: apply_reject_gate()
# ---------------------------------------------------------------------------


def test_gate_abstains_empty_lexical_and_low_dense():
    """(a) empty keyword set AND max dense sim < tau -> abstain (empty)."""
    results = [_norm("vector", 0.40)]  # only a dense hit, sim below tau
    out = apply_reject_gate(results, tau=0.50)
    assert out == []


def test_gate_passes_when_lexical_hit_present():
    """(b) a keyword hit present -> pass through even if dense sim is low."""
    results = [_norm("archive", 1.0), _norm("vector", 0.10)]
    out = apply_reject_gate(results, tau=0.50)
    assert out == results


def test_gate_passes_when_dense_sim_at_or_above_tau():
    """(c) max dense sim >= tau -> pass through (no lexical hit needed)."""
    results = [_norm("vector", 0.50)]  # exactly at tau
    out = apply_reject_gate(results, tau=0.50)
    assert out == results

    results_high = [_norm("vector", 0.92)]
    assert apply_reject_gate(results_high, tau=0.50) == results_high


def test_gate_abstains_with_no_vector_results():
    """No dense candidates at all means max dense sim is 0.0 -> abstain."""
    out = apply_reject_gate([], tau=0.50)
    assert out == []


def test_gate_treats_kg_and_catalog_as_lexical():
    """kg / catalog keyword hits also count as the lexical recall set."""
    for lexical_source in ("kg", "catalog", "archive"):
        results = [_norm(lexical_source, 1.0), _norm("vector", 0.05)]
        out = apply_reject_gate(results, tau=0.50)
        assert out == results, f"{lexical_source} should keep results"


def test_gate_accepts_raw_vector_rows_with_similarity():
    """BEAM-shaped raw vmem.search rows ({text, similarity, metadata}, no
    'source' key) are read as dense candidates via their 'similarity'."""
    low = [{"text": "c", "similarity": 0.30, "metadata": {}}]
    assert apply_reject_gate(low, tau=0.50) == []

    high = [{"text": "c", "similarity": 0.80, "metadata": {}}]
    assert apply_reject_gate(high, tau=0.50) == high


# ---------------------------------------------------------------------------
# Integration tests through retrieve(): env-gated, default-off
# ---------------------------------------------------------------------------


def test_retrieve_gate_off_is_baseline(monkeypatch):
    """(d) gate OFF (env unset) -> identical to baseline, never abstains."""
    monkeypatch.delenv("TAOSMD_REJECT_GATE", raising=False)
    monkeypatch.delenv("TAOSMD_REJECT_TAU", raising=False)

    sources = {"vector": VectorWithSim(0.10), "archive": EmptyLexical()}
    baseline = asyncio.run(retrieve(
        query="something unanswerable", strategy="thorough",
        sources=sources, limit=5,
    ))
    # Low dense sim + empty lexical, but the gate is OFF, so we still get a hit.
    assert len(baseline) > 0


def test_retrieve_gate_on_abstains_on_low_confidence(monkeypatch):
    """Gate ON: empty lexical + low dense sim -> empty (abstain)."""
    monkeypatch.setenv("TAOSMD_REJECT_GATE", "1")
    monkeypatch.setenv("TAOSMD_REJECT_TAU", "0.50")

    sources = {"vector": VectorWithSim(0.10), "archive": EmptyLexical()}
    out = asyncio.run(retrieve(
        query="something unanswerable", strategy="thorough",
        sources=sources, limit=5,
    ))
    assert out == []


def test_retrieve_gate_on_passes_with_lexical_hit(monkeypatch):
    """Gate ON: a lexical hit present -> pass through even at low dense sim."""
    monkeypatch.setenv("TAOSMD_REJECT_GATE", "1")
    monkeypatch.setenv("TAOSMD_REJECT_TAU", "0.50")

    sources = {"vector": VectorWithSim(0.10), "archive": HitLexical()}
    out = asyncio.run(retrieve(
        query="answerable via keyword", strategy="thorough",
        sources=sources, limit=5,
    ))
    assert len(out) > 0


def test_retrieve_gate_on_passes_with_high_dense_sim(monkeypatch):
    """Gate ON: high dense sim -> pass through even with empty lexical set."""
    monkeypatch.setenv("TAOSMD_REJECT_GATE", "1")
    monkeypatch.setenv("TAOSMD_REJECT_TAU", "0.50")

    sources = {"vector": VectorWithSim(0.92), "archive": EmptyLexical()}
    out = asyncio.run(retrieve(
        query="answerable via dense match", strategy="thorough",
        sources=sources, limit=5,
    ))
    assert len(out) > 0


def test_retrieve_gate_default_tau_is_half(monkeypatch):
    """When TAOSMD_REJECT_TAU is unset, the default tau is 0.50."""
    monkeypatch.setenv("TAOSMD_REJECT_GATE", "1")
    monkeypatch.delenv("TAOSMD_REJECT_TAU", raising=False)

    # sim 0.49 is just under the 0.50 default -> abstain
    sources = {"vector": VectorWithSim(0.49), "archive": EmptyLexical()}
    out = asyncio.run(retrieve(
        query="just under default tau", strategy="thorough",
        sources=sources, limit=5,
    ))
    assert out == []
