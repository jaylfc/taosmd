"""Tests for the opt-in late-interaction (MaxSim) scoring path in VectorMemory.

Late interaction is a ColBERT-style retrieval probe: instead of pooling the
MiniLM token vectors into one and scoring by cosine, it stores the full
per-token matrix and scores by mean_q(max_t(q_t . d_t)). It must be OFF by
default, must be mutually exclusive with binary_quant, and when enabled it must
rank an exact-match document first and prefer docs whose tokens cover the query.

The tests use a deterministic monkeypatched embed_tokens (fixed per-word unit
vectors) so they never touch the real ONNX model or any network.
"""

from __future__ import annotations

import asyncio

import pytest

from taosmd.vector_memory import _EMBED_DIM, VectorMemory

# A tiny fixed vocabulary -> orthonormal basis index. Each word maps to a unit
# vector (one-hot) in the real 384-dim embedding space, so a dot product of two
# token vectors is 1.0 iff they are the same word and 0.0 otherwise. That makes
# MaxSim well-defined and exact: MaxSim_d = (fraction of query words that appear
# in doc d). Vectors are _EMBED_DIM-wide because the storage/load path reshapes
# the persisted float16 blob to (-1, _EMBED_DIM).
_VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon", "the", "cat", "sat",
          "on", "mat", "quantum", "field", "theory", "grocery", "eggs", "milk"]
_INDEX = {w: i for i, w in enumerate(_VOCAB)}


def _unit_vec(word: str) -> list[float]:
    """Return a fixed 384-dim unit vector for a word (one-hot in the toy vocab)."""
    v = [0.0] * _EMBED_DIM
    v[_INDEX.get(word.lower(), 0)] = 1.0
    return v


def _deterministic_embedder(vmem: VectorMemory) -> None:
    """Patch embed()/embed_tokens() with deterministic word-level unit vectors.

    No ONNX, no network. embed() returns a pooled vector (mean of token vectors)
    purely so the search() guard `if not query_emb` passes; the late-interaction
    branch computes its own MaxSim from embed_tokens().
    """

    async def _fake_embed_tokens(text: str, task: str = "search_document") -> list[list[float]]:
        words = [w for w in text.lower().split() if w]
        return [_unit_vec(w) for w in words] or [[0.0] * _EMBED_DIM]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        toks = await _fake_embed_tokens(text, task)
        # Mean-pool the token vectors so we hand back a non-empty pooled vector.
        cols = list(zip(*toks))
        return [sum(c) / len(toks) for c in cols]

    vmem.embed_tokens = _fake_embed_tokens  # type: ignore[assignment]
    vmem.embed = _fake_embed  # type: ignore[assignment]


def _make_store(tmp_path, *, late_interaction: bool) -> VectorMemory:
    """Init a VectorMemory with a deterministic embedder and the given mode."""
    vmem = VectorMemory(
        db_path=str(tmp_path / "vec.db"),
        embed_mode="onnx",  # falls through; we patch embed()/embed_tokens() directly
        late_interaction=late_interaction,
    )
    asyncio.run(vmem.init())
    _deterministic_embedder(vmem)
    return vmem


def test_late_interaction_defaults_off(tmp_path):
    """late_interaction is opt-in: default construction leaves it disabled."""
    vmem = VectorMemory(db_path=str(tmp_path / "vec.db"))
    assert vmem.late_interaction is False


def test_late_interaction_is_constructor_settable(tmp_path):
    """The constructor flag enables late-interaction mode."""
    vmem = VectorMemory(db_path=str(tmp_path / "vec.db"), late_interaction=True)
    assert vmem.late_interaction is True


def test_binary_quant_and_late_interaction_are_mutually_exclusive(tmp_path):
    """Enabling both modes at once is a constructor error."""
    with pytest.raises(ValueError):
        VectorMemory(
            db_path=str(tmp_path / "vec.db"),
            binary_quant=True,
            late_interaction=True,
        )


def test_late_interaction_search_ranks_exact_match_first(tmp_path):
    """Search returns ranked hits with MaxSim scores; exact match ranks first."""
    vmem = _make_store(tmp_path, late_interaction=True)
    try:
        for text in ("the cat sat on the mat", "quantum field theory", "grocery eggs milk"):
            asyncio.run(vmem.add(text))
        hits = asyncio.run(vmem.search("the cat sat on the mat", limit=3, hybrid=False, fusion="none"))
        assert hits, "late-interaction search returned no hits"
        scores = [h["similarity"] for h in hits if "similarity" in h]
        assert scores, "hits carried no similarity field"
        # MaxSim of cosines is bounded in [0, 1] for these unit token vectors.
        assert all(0.0 <= s <= 1.0 for s in scores), f"scores out of [0,1]: {scores}"
        # The exact-match document should rank first with a perfect MaxSim.
        assert hits[0]["text"] == "the cat sat on the mat"
        assert hits[0]["similarity"] == pytest.approx(1.0, abs=1e-4)
    finally:
        asyncio.run(vmem.close())


def test_late_interaction_prefers_token_coverage(tmp_path):
    """A doc whose tokens include the query tokens scores >= one that doesn't."""
    vmem = _make_store(tmp_path, late_interaction=True)
    try:
        asyncio.run(vmem.add("alpha beta gamma"))   # covers the query fully
        asyncio.run(vmem.add("delta epsilon milk"))  # covers none of the query
        hits = asyncio.run(vmem.search("alpha beta", limit=2, hybrid=False, fusion="none"))
        by_text = {h["text"]: h["similarity"] for h in hits}
        assert by_text["alpha beta gamma"] >= by_text["delta epsilon milk"]
        # Full coverage of both query tokens => MaxSim 1.0.
        assert by_text["alpha beta gamma"] == pytest.approx(1.0, abs=1e-4)
    finally:
        asyncio.run(vmem.close())
