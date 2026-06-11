#!/usr/bin/env python3
"""E1 kill-shot harness: does per-turn token surprisal improve retrieval?

Measures retrieval quality (R@K) for three variants with NO LLM generation
and NO Ollama dependency -- judge-free, pure R@K measurement.

Variants
--------
baseline
    Identical to the existing dense retrieval probe path (control).

surprise_prior (disabled by default; enable with --include-prior)
    After retrieval scoring, multiply each candidate's score by
    (1 + alpha * z) where z is the candidate turn's surprisal z-score
    normalised within its conversation, clipped to [-2, 2]. Runs with
    alpha in {0.25, 0.5, 1.0}.  Prior runs scored flat +0.0000 at every
    alpha; the code is kept but excluded from the default variant set.

surprise_chunks
    Re-chunk the corpus before embedding: consecutive turns merge into one
    chunk while their surprisal z-score < 0.5; a turn with z >= 1.5 always
    starts a new chunk; max merged chunk size is 6 turns.

    Evidence mapping: a retrieved chunk credits ONLY the specific evidence
    dia_ids it actually contains (intersection of retrieved chunk dia_ids
    with the QA evidence list).  This is identical to how the baseline
    scores per-turn evidence, making the comparison fair.

Surprisal model
---------------
Uses a small HF causal LM (default: HuggingFaceTB/SmolLM2-360M) via
transformers + torch on CPU. No new dependencies are added -- both are
already installed on the bench host.

Per-turn surprisal: context = previous turns concatenated, truncated to
the LAST 512 tokens; compute mean negative-log-likelihood of the turn's
tokens given that context (teacher-forced single forward pass per turn,
no generation), plus the max token NLL.

Caching
-------
Results are cached to a JSON sidecar next to --out, keyed by
(conversation_id, turn_index), so reruns skip already-scored turns.

Kill criterion
--------------
E1 is killed if no variant beats baseline R@K by more than 0.02:
    "E1 VERDICT: <variant> best delta vs baseline = <x>; kill threshold 0.02"

Failure posture
---------------
- Canary-check the scorer at startup (one forward pass); exit 2 on failure.
- Abort if >10% of turns fail scoring.
- A variant with zero successful retrievals exits 1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_BENCH_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _BENCH_DIR)

from taosmd.vector_memory import VectorMemory  # noqa: E402

from locomo_runner import (  # noqa: E402
    CATEGORY_NAMES,
    _DEFAULT_DATASET,
    _DEFAULT_ONNX,
    _build_adjacent_map,
    _evidence_hits,
    _git_sha,
    _ingest_conversation,
    _load_reranker,
    _retrieve,
    _session_keys,
)

import httpx

# Same category set the runner uses by default (adversarial excluded).
INCLUDE_CATS = {1, 2, 3, 4}

# Surprisal chunk thresholds.
_CHUNK_MERGE_Z_LOW = 0.5   # continue merging while z < this
_CHUNK_SPLIT_Z_HIGH = 1.5  # always start a new chunk at z >= this
_CHUNK_MAX_TURNS = 6       # max turns per merged chunk


# ---------------------------------------------------------------------------
# Surprisal scorer
# ---------------------------------------------------------------------------

class SurprisalScorer:
    """Compute per-turn token surprisal using a small causal LM.

    Uses teacher-forcing: the model sees the context (previous turns) and
    the target (current turn), and we extract the mean + max negative
    log-likelihood over the target tokens only.

    Thread-safety: not thread-safe; designed for sequential use in a
    single-process bench.
    """

    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-360M",
                 context_tokens: int = 512) -> None:
        self.model_name = model_name
        self.context_tokens = context_tokens
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """Lazy-load model and tokenizer (deferred to first use / canary)."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                f"transformers + torch are required for surprisal scoring: {exc}"
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=torch.float32
        )
        self._model.eval()

    def canary(self) -> None:
        """Single forward pass to verify the scorer is functional.

        Raises RuntimeError on any failure so the caller can exit 2.
        """
        self._load()
        try:
            self.score_turn("Hello world.", context_text="")
        except Exception as exc:
            raise RuntimeError(
                f"Surprisal scorer canary failed ({self.model_name}): {exc}"
            ) from exc

    def score_turn(self, turn_text: str, context_text: str) -> dict:
        """Compute surprisal of turn_text given context_text.

        Returns:
            {
                "mean_nll": float,   # mean negative log-likelihood per token
                "max_nll": float,    # max token NLL across turn tokens
                "n_tokens": int,     # number of target tokens scored
            }
        """
        self._load()
        torch = self._torch
        tok = self._tokenizer

        # Tokenize context and turn separately so we know the boundary.
        ctx_ids = tok.encode(context_text, add_special_tokens=False) if context_text else []
        turn_ids = tok.encode(turn_text, add_special_tokens=False)

        if not turn_ids:
            return {"mean_nll": 0.0, "max_nll": 0.0, "n_tokens": 0}

        # Truncate context to LAST context_tokens, keeping turn intact.
        max_ctx = self.context_tokens
        ctx_ids = ctx_ids[-max_ctx:]

        # Build input: [ctx_ids ... turn_ids]
        # Labels: -100 for context tokens (masked), real ids for turn tokens.
        input_ids = ctx_ids + turn_ids
        labels = [-100] * len(ctx_ids) + list(turn_ids)

        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        labels_tensor = torch.tensor([labels], dtype=torch.long)

        with torch.no_grad():
            out = self._model(input_ids=input_tensor, labels=labels_tensor)

        # out.loss is mean NLL over non-masked tokens. To get per-token NLLs
        # for max computation, run a second pass without loss and extract logits.
        with torch.no_grad():
            logits = self._model(input_ids=input_tensor).logits  # [1, seq, vocab]

        # logits[0, i, :] predicts input_ids[i+1].
        # With context: logits at position len(ctx)-1 predicts turn_ids[0].
        #   shift_logits covers [len(ctx)-1 .. len(ctx)+len(turn)-2] (inclusive).
        #   shift_labels = turn_ids  (all N turn tokens predicted)
        # Without context: logits[0, 0, :] predicts turn_ids[1]; turn_ids[0] has
        #   no preceding token in the sequence so we score tokens 1..N-1 only.
        #   shift_logits = logits[0, 0 : len(turn)-1, :]
        #   shift_labels = turn_ids[1:]
        if ctx_ids:
            shift_logits = logits[0, len(ctx_ids) - 1: len(ctx_ids) + len(turn_ids) - 1, :]
            shift_labels = torch.tensor(turn_ids, dtype=torch.long)
        else:
            shift_logits = logits[0, 0: len(turn_ids) - 1, :]
            shift_labels = torch.tensor(turn_ids[1:], dtype=torch.long)

        import torch as _torch_mod
        log_probs = _torch_mod.nn.functional.log_softmax(shift_logits, dim=-1)
        token_nlls = (-log_probs[
            _torch_mod.arange(len(shift_labels)), shift_labels
        ]).tolist()

        mean_nll = float(out.loss.item())
        max_nll = max(token_nlls) if token_nlls else 0.0

        return {
            "mean_nll": mean_nll,
            "max_nll": max_nll,
            "n_tokens": len(turn_ids),
        }


# ---------------------------------------------------------------------------
# Per-conversation surprisal computation + caching
# ---------------------------------------------------------------------------

def _sidecar_path(out_path: Optional[Path]) -> Path:
    """Path to the JSON surprisal cache sidecar."""
    if out_path is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = Path(_BENCH_DIR) / "results" / f"surprisal_probe_{ts}.json"
    return out_path.with_suffix(".surprisal_cache.json")


def _load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2))


def _cache_key(conv_id: str, turn_index: int) -> str:
    return f"{conv_id}::{turn_index}"


def score_conversation_turns(
    conv: dict,
    scorer: SurprisalScorer,
    cache: dict,
    conv_id: str,
    cache_path: Path,
    max_error_rate: float = 0.10,
) -> tuple[list[dict], list[str]]:
    """Score all turns in a conversation for surprisal.

    Returns (scored_turns, errors) where each scored_turn dict has:
        {
            "turn_index": int,     # global sequential index within conversation
            "dia_id": str,
            "speaker": str,
            "text": str,
            "mean_nll": float,
            "max_nll": float,
            "n_tokens": int,
        }

    Aborts with RuntimeError if >max_error_rate of turns fail scoring.
    Writes to cache after each turn (so interrupted runs can resume).
    """
    conversation = conv.get("conversation", conv)
    scored: list[dict] = []
    errors: list[str] = []
    context_parts: list[str] = []
    global_idx = 0

    for session_key, dt in _session_keys(conversation):
        for turn in conversation.get(session_key) or []:
            text = (turn.get("text") or "").strip()
            if not text:
                global_idx += 1
                continue
            speaker = turn.get("speaker", "")
            dia_id = turn.get("dia_id", "")
            turn_text = f"[{speaker}] {text}"

            ck = _cache_key(conv_id, global_idx)
            if ck in cache:
                entry = cache[ck]
            else:
                ctx_text = " ".join(context_parts)
                try:
                    result = scorer.score_turn(turn_text, ctx_text)
                    entry = {
                        "mean_nll": result["mean_nll"],
                        "max_nll": result["max_nll"],
                        "n_tokens": result["n_tokens"],
                    }
                    cache[ck] = entry
                    _save_cache(cache_path, cache)
                except Exception as exc:
                    errors.append(f"turn {global_idx} ({dia_id}): {exc}")
                    entry = {"mean_nll": 0.0, "max_nll": 0.0, "n_tokens": 0}

            scored.append({
                "turn_index": global_idx,
                "dia_id": dia_id,
                "speaker": speaker,
                "text": turn_text,
                "mean_nll": entry["mean_nll"],
                "max_nll": entry["max_nll"],
                "n_tokens": entry["n_tokens"],
            })
            context_parts.append(turn_text)
            global_idx += 1

    total = len(scored) + len(errors)
    if total > 0 and len(errors) / total > max_error_rate:
        raise RuntimeError(
            f"Surprisal scoring: {len(errors)}/{total} turns failed "
            f"(>{max_error_rate:.0%} threshold). First: {errors[0]}"
        )

    return scored, errors


# ---------------------------------------------------------------------------
# Z-score normalisation
# ---------------------------------------------------------------------------

def _zscore_normalize(values: list[float]) -> list[float]:
    """Z-score normalise a list of floats within a conversation.

    Returns a list of z-scores. Returns zeros if all values are identical or
    the list has fewer than 2 elements.
    """
    if len(values) < 2:
        return [0.0] * len(values)
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance)
    if std < 1e-9:
        return [0.0] * len(values)
    return [(v - mean) / std for v in values]


# ---------------------------------------------------------------------------
# surprise_prior: score rescoring
# ---------------------------------------------------------------------------

def _apply_surprise_prior(
    hits: list[dict],
    turn_surprisal_z: dict[int, float],
    alpha: float,
) -> list[dict]:
    """Rescore hits by multiplying score by (1 + alpha * clip(z, -2, 2)).

    turn_surprisal_z: maps turn_index (int) to z-score within conversation.
    """
    rescored = []
    for h in hits:
        meta = h.get("metadata", {}) or {}
        turn_idx = meta.get("turn_idx")
        if turn_idx is not None:
            z = turn_surprisal_z.get(int(turn_idx), 0.0)
            z_clipped = max(-2.0, min(2.0, z))
            multiplier = 1.0 + alpha * z_clipped
        else:
            multiplier = 1.0
        new_h = dict(h)
        new_h["score"] = h.get("score", 0.0) * multiplier
        rescored.append(new_h)
    rescored.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return rescored


# ---------------------------------------------------------------------------
# surprise_chunks: chunked corpus building
# ---------------------------------------------------------------------------

def _build_surprise_chunks(
    scored_turns: list[dict],
    z_scores: list[float],
) -> list[dict]:
    """Merge consecutive turns into chunks based on surprisal z-scores.

    Rules (applied in this priority order):
    1. A turn with z >= _CHUNK_SPLIT_Z_HIGH always starts a new chunk.
    2. A chunk that has reached _CHUNK_MAX_TURNS is closed before any
       new turn is added, regardless of the z-score.  This cap is
       enforced unconditionally so no chunk can exceed the stated maximum.
    3. A turn with z >= _CHUNK_MERGE_Z_LOW (but below the split threshold)
       starts a new chunk.
    4. Otherwise (z < _CHUNK_MERGE_Z_LOW and room remains) the turn is
       merged into the current chunk.

    Returns list of chunk dicts:
        {
            "text": str,               # concatenated turn texts
            "dia_ids": list[str],      # all dia_ids in this chunk
            "turn_indices": list[int], # global turn indices in this chunk
            "speaker": str,            # primary speaker (first turn in chunk)
            "datetime": str,           # datetime of first turn in chunk
        }
    """
    if not scored_turns:
        return []

    chunks: list[dict] = []
    current_texts: list[str] = []
    current_dia_ids: list[str] = []
    current_indices: list[int] = []
    current_speaker = ""
    current_datetime = ""

    def _flush():
        if not current_texts:
            return
        chunks.append({
            "text": " ".join(current_texts),
            "dia_ids": list(current_dia_ids),
            "turn_indices": list(current_indices),
            "speaker": current_speaker,
            "datetime": current_datetime,
        })
        current_texts.clear()
        current_dia_ids.clear()
        current_indices.clear()

    def _start_new(turn: dict) -> None:
        current_texts.append(turn["text"])
        current_dia_ids.append(turn["dia_id"])
        current_indices.append(turn["turn_index"])
        nonlocal current_speaker, current_datetime
        current_speaker = turn["speaker"]
        current_datetime = ""  # not tracked in scored_turns; metadata from vmem

    for turn, z in zip(scored_turns, z_scores):
        # Rule 1: high-z turn always starts a new chunk.
        if z >= _CHUNK_SPLIT_Z_HIGH:
            _flush()
            _start_new(turn)
            continue

        # Rule 2: cap is full -- close current chunk before adding this turn,
        # regardless of z.  This is enforced unconditionally so a chunk can
        # never exceed _CHUNK_MAX_TURNS turns no matter what the z-score is.
        if len(current_texts) >= _CHUNK_MAX_TURNS:
            _flush()
            _start_new(turn)
            continue

        # Rule 3: mid-z turn (not a hard split but above merge threshold)
        # starts a new chunk.
        if z >= _CHUNK_MERGE_Z_LOW:
            _flush()
            _start_new(turn)
            continue

        # Rule 4: merge-eligible (z < _CHUNK_MERGE_Z_LOW, room remains).
        if not current_texts:
            _start_new(turn)
        else:
            current_texts.append(turn["text"])
            current_dia_ids.append(turn["dia_id"])
            current_indices.append(turn["turn_index"])

    _flush()
    return chunks


def _chunk_evidence_hits(
    hit_chunks: list[dict],
    evidence: list[str],
) -> int:
    """Count evidence hits across chunk-based retrieval results.

    A retrieved chunk credits ONLY the specific evidence turns (dia_ids)
    it actually contains.  Non-evidence turns that happen to be in the
    same chunk are NOT credited.  This is identical in spirit to how the
    baseline counts per-turn evidence: the intersection of the retrieved
    set with the QA evidence list, not a superset of it.

    Specifically: a chunk containing evidence turn X and non-evidence turn Y
    credits only X (evidence_hits += 1), not Y.
    """
    if not evidence:
        return 0
    evidence_set = set(evidence)
    retrieved_evidence_ids: set[str] = set()
    for h in hit_chunks:
        meta = h.get("metadata", {}) or {}
        # Chunk hits store dia_ids as a JSON-serialised list in metadata.
        dia_ids = meta.get("dia_ids", [])
        if isinstance(dia_ids, str):
            try:
                dia_ids = json.loads(dia_ids)
            except (json.JSONDecodeError, ValueError):
                dia_ids = [dia_ids]
        for d in (dia_ids or []):
            # Only add to the credited set if this dia_id is actually
            # in the evidence list for this QA (per-turn credit only).
            if d and d in evidence_set:
                retrieved_evidence_ids.add(d)
        # Fallback: direct dia_id field (single-turn chunks or legacy).
        direct = meta.get("dia_id")
        if direct and direct in evidence_set:
            retrieved_evidence_ids.add(direct)
    return len(retrieved_evidence_ids)


async def _ingest_chunked_conversation(
    vmem: VectorMemory,
    chunks: list[dict],
) -> tuple[int, float]:
    """Ingest surprisal-derived chunks into a VectorMemory instance.

    Returns (added_count, elapsed_s).
    """
    t0 = time.time()
    added = 0
    for chunk in chunks:
        # Serialise dia_ids list as JSON string for metadata storage.
        dia_ids_json = json.dumps(chunk["dia_ids"])
        await vmem.add(
            chunk["text"],
            metadata={
                "dia_ids": dia_ids_json,
                "dia_id": chunk["dia_ids"][0] if chunk["dia_ids"] else "",
                "turn_indices": json.dumps(chunk["turn_indices"]),
                "speaker": chunk["speaker"],
                "level": "surprisal_chunk",
            },
        )
        added += 1
    return added, time.time() - t0


# ---------------------------------------------------------------------------
# Latency helpers (shared with retrieval_latency_probe)
# ---------------------------------------------------------------------------

def _pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))
    return s[idx]


def _latency_stats(values: list[float]) -> dict:
    if not values:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    return {
        "p50_ms": round(_pctl(values, 0.50), 2),
        "p95_ms": round(_pctl(values, 0.95), 2),
        "mean_ms": round(sum(values) / len(values), 2),
    }


def _recall_at_k(rows: list[dict]) -> tuple[float, int]:
    ev_rows = [r for r in rows if r.get("evidence_total", 0) > 0]
    if not ev_rows:
        return 0.0, 0
    hits = sum(1 for r in ev_rows if r["evidence_hits"] > 0)
    return hits / len(ev_rows), len(ev_rows)


# ---------------------------------------------------------------------------
# Per-variant run
# ---------------------------------------------------------------------------

async def _run_variant(
    variant_name: str,
    conversations: list[dict],
    args: argparse.Namespace,
    reranker,
    scorer: SurprisalScorer,
    surprisal_data: dict,   # conv_id -> {"scored_turns": [...], "z_scores": [...]}
    alpha: float = 0.5,     # only used for surprise_prior
) -> dict:
    """Run one variant across all conversations; return results dict."""

    rows: list[dict] = []
    errors: list[dict] = []
    ingest_stats: list[dict] = []
    attempted = 0
    retrieval_top_k = args.retrieval_top_k

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for conv in conversations:
            conv_id = conv.get("sample_id", "unknown")
            sd = surprisal_data.get(conv_id, {})
            scored_turns = sd.get("scored_turns", [])
            z_scores = sd.get("z_scores", [])

            # Build turn_surprisal_z lookup for surprise_prior.
            turn_surprisal_z: dict[int, float] = {}
            if z_scores and scored_turns:
                for st, z in zip(scored_turns, z_scores):
                    turn_surprisal_z[st["turn_index"]] = z

            tmp = tempfile.mkdtemp(prefix=f"surprisal_{variant_name}_{conv_id}_")

            vmem = VectorMemory(
                db_path=os.path.join(tmp, "vmem.db"),
                qmd_url=args.qmd_url,
                embed_mode=args.embed_mode,
                onnx_path=args.onnx_path,
                binary_quant=args.binary_quant,
                late_interaction=args.late_interaction,
                colbert_model=args.colbert_model,
            )
            await vmem.init(http_client=client)

            canary = await vmem.embed("canary embedding check", task="search_document")
            if not canary:
                print(
                    f"ERROR: embed() returned empty for mode={args.embed_mode!r}. "
                    "Embedding backend not available.",
                    file=sys.stderr,
                )
                await vmem.close()
                return {"rows": [], "errors": [{"error": "embed canary failed"}],
                        "ingest_stats": [], "variant": variant_name, "alpha": alpha}

            if variant_name == "surprise_chunks" and scored_turns and z_scores:
                chunks = _build_surprise_chunks(scored_turns, z_scores)
                added, ingest_s = await _ingest_chunked_conversation(vmem, chunks)
                turn_index: dict[str, dict] = {}
            else:
                added, ingest_s, turn_index = await _ingest_conversation(vmem, conv)
                chunks = []

            ingest_stats.append({
                "conversation_id": conv_id,
                "added": added,
                "ingest_s": round(ingest_s, 2),
            })
            print(
                f"[{variant_name}:{conv_id}] ingested {added} "
                f"{'chunks' if variant_name == 'surprise_chunks' else 'turns'} "
                f"in {ingest_s:.1f}s",
                flush=True,
            )

            for qa in conv.get("qa", []) or []:
                if "answer" not in qa:
                    continue
                category = int(qa.get("category", 0))
                if category not in INCLUDE_CATS:
                    continue
                if args.limit and attempted >= args.limit:
                    break
                attempted += 1
                question = qa["question"]
                evidence = qa.get("evidence", []) or []

                try:
                    t0 = time.time()
                    hits = await _retrieve(
                        "vector-only", question, vmem, retrieval_top_k,
                        reranker, fusion=args.fusion,
                        rerank_top_k=args.top_k,
                    )
                    retrieval_ms = (time.time() - t0) * 1000.0

                    if variant_name.startswith("surprise_prior") and turn_surprisal_z:
                        hits = _apply_surprise_prior(hits, turn_surprisal_z, alpha)

                    if variant_name == "surprise_chunks":
                        ev_hits = _chunk_evidence_hits(hits, evidence)
                    else:
                        ev_hits = _evidence_hits(hits, evidence)

                    rows.append({
                        "conversation_id": conv_id,
                        "question": question,
                        "category": category,
                        "n_hits": len(hits),
                        "evidence_hits": ev_hits,
                        "evidence_total": len(evidence),
                        "retrieval_ms": round(retrieval_ms, 2),
                        "variant": variant_name,
                        "alpha": alpha,
                    })

                except Exception as exc:
                    errors.append({
                        "conversation_id": conv_id,
                        "question": question,
                        "category": category,
                        "error": f"{type(exc).__name__}: {exc}",
                        "variant": variant_name,
                    })
                    print(
                        f"  qa ERROR [{variant_name}:{conv_id}] "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr, flush=True,
                    )

                if attempted >= 10:
                    err_rate = len(errors) / attempted
                    if err_rate > args.max_error_rate:
                        print(
                            f"\nABORT: {len(errors)}/{attempted} QAs errored "
                            f"in variant={variant_name} (>{args.max_error_rate:.0%}). "
                            f"First error: {errors[0]['error']}",
                            file=sys.stderr,
                        )
                        await vmem.close()
                        return {
                            "rows": rows, "errors": errors,
                            "ingest_stats": ingest_stats,
                            "variant": variant_name, "alpha": alpha,
                            "aborted": True,
                        }

            await vmem.close()
            if args.limit and attempted >= args.limit:
                break

    return {
        "rows": rows,
        "errors": errors,
        "ingest_stats": ingest_stats,
        "variant": variant_name,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Surprisal collection pass (sequential, no embedding)
# ---------------------------------------------------------------------------

def collect_surprisal(
    conversations: list[dict],
    scorer: SurprisalScorer,
    cache: dict,
    cache_path: Path,
    max_error_rate: float = 0.10,
) -> dict:
    """Score all turns in all conversations; return per-conv surprisal data.

    Returns:
        {
            conv_id: {
                "scored_turns": [...],   # list of per-turn score dicts
                "z_scores": [...],       # z-score of mean_nll within conv
                "errors": [...],
            }
        }
    """
    result: dict = {}
    for conv in conversations:
        conv_id = conv.get("sample_id", "unknown")
        print(f"[scorer] scoring conversation {conv_id} ...", flush=True)
        try:
            scored_turns, errors = score_conversation_turns(
                conv, scorer, cache, conv_id, cache_path, max_error_rate,
            )
        except RuntimeError as exc:
            print(f"[scorer] ABORT on {conv_id}: {exc}", file=sys.stderr)
            raise

        mean_nlls = [t["mean_nll"] for t in scored_turns]
        z_scores = _zscore_normalize(mean_nlls)

        result[conv_id] = {
            "scored_turns": scored_turns,
            "z_scores": z_scores,
            "errors": errors,
        }
        print(
            f"[scorer] {conv_id}: {len(scored_turns)} turns scored, "
            f"{len(errors)} errors",
            flush=True,
        )
    return result


# ---------------------------------------------------------------------------
# Summary + print
# ---------------------------------------------------------------------------

def _summarize_variant(result: dict) -> dict:
    rows = result["rows"]
    errors = result["errors"]
    ingest_stats = result.get("ingest_stats", [])
    variant = result["variant"]
    alpha = result.get("alpha", 0.0)

    recall, n_ev = _recall_at_k(rows)
    by_cat: dict[str, dict] = {}
    cats: dict[int, list[dict]] = {}
    for r in rows:
        cats.setdefault(int(r["category"]), []).append(r)
    for cat in sorted(cats):
        c_rows = cats[cat]
        c_recall, c_nev = _recall_at_k(c_rows)
        by_cat[str(cat)] = {
            "name": CATEGORY_NAMES.get(cat, f"cat-{cat}"),
            "count": len(c_rows),
            "n_with_evidence": c_nev,
            "r_at_k": round(c_recall, 4),
            **_latency_stats([r["retrieval_ms"] for r in c_rows]),
        }

    extra: dict = {}
    if variant == "surprise_chunks" and ingest_stats:
        total_chunks = sum(s["added"] for s in ingest_stats)
        # Count total turns across all conversations for chunk-length calc.
        # We store this in ingest_stats for the chunks path, so use it directly.
        # Mean chunk length: total_turns / total_chunks (approximate).
        extra["total_chunks_ingested"] = total_chunks

    return {
        "variant": variant,
        "alpha": alpha,
        "n": len(rows),
        "n_errors": len(errors),
        "n_with_evidence": n_ev,
        "r_at_k": round(recall, 4),
        "retrieval_latency": _latency_stats([r["retrieval_ms"] for r in rows]),
        "by_category": by_cat,
        **extra,
    }


def _print_table(summaries: list[dict], baseline_r: float) -> None:
    sep, dash = "=" * 72, "-" * 72
    print(sep)
    print("E1 Surprisal Probe -- R@K Summary")
    print(sep)
    print(
        f"{'Variant':<28} {'alpha':>6} {'n':>4} {'R@K':>6} {'delta':>7} "
        f"{'p50ms':>8}"
    )
    print(dash)
    for s in summaries:
        alpha_str = f"{s['alpha']:.2f}" if s["variant"] != "baseline" else "   --"
        delta = s["r_at_k"] - baseline_r
        delta_str = f"{delta:+.4f}" if s["variant"] != "baseline" else "    --"
        lat = s["retrieval_latency"]
        print(
            f"{s['variant']:<28} {alpha_str:>6} {s['n']:>4} {s['r_at_k']:>6.3f} "
            f"{delta_str:>7} {lat['p50_ms']:>8.1f}"
        )
        if s.get("total_chunks_ingested"):
            print(f"  -> chunks ingested: {s['total_chunks_ingested']}")
    print(dash)


def _print_category_table(summaries: list[dict]) -> None:
    dash = "-" * 72
    for s in summaries:
        alpha_str = f" alpha={s['alpha']:.2f}" if s["variant"] != "baseline" else ""
        print(f"\n[{s['variant']}{alpha_str}] per-category:")
        print(
            f"  {'Category':<18} {'n':>4} {'R@K':>6} "
            f"{'p50ms':>9} {'p95ms':>9}"
        )
        print(f"  {dash[:50]}")
        for cat, cs in sorted(s["by_category"].items(), key=lambda kv: int(kv[0])):
            print(
                f"  {cs['name']:<18} {cs['count']:>4} {cs['r_at_k']:>6.3f} "
                f"{cs['p50_ms']:>9.1f} {cs['p95_ms']:>9.1f}"
            )


def _print_verdict(summaries: list[dict], baseline_r: float,
                   kill_threshold: float = 0.02) -> None:
    non_baseline = [s for s in summaries if s["variant"] != "baseline"]
    if not non_baseline:
        print(
            f"\nE1 VERDICT: no variants ran; "
            f"kill threshold {kill_threshold}"
        )
        return
    best = max(non_baseline, key=lambda s: s["r_at_k"])
    best_delta = best["r_at_k"] - baseline_r
    alpha_tag = (
        f" (alpha={best['alpha']:.2f})"
        if best["variant"] != "surprise_chunks" else ""
    )
    print(
        f"\nE1 VERDICT: {best['variant']}{alpha_tag} best delta vs baseline = "
        f"{best_delta:+.4f}; kill threshold {kill_threshold}"
    )
    if best_delta <= kill_threshold:
        print("  => PILLAR KILLED: surprisal does not improve retrieval by >=0.02 R@K")
    else:
        print("  => PILLAR LIVES: surprisal shows meaningful improvement")


# ---------------------------------------------------------------------------
# Main async run
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        print(
            f"ERROR: dataset not found: {dataset_path}\n"
            f"Set --dataset or the LOCOMO_DATASET env var.",
            file=sys.stderr,
        )
        return 2

    conversations = json.loads(dataset_path.read_text())[: args.conversations]

    # Resolve output path early (needed for sidecar cache path).
    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = Path(_BENCH_DIR) / "results" / f"surprisal_probe_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache_path = _sidecar_path(out_path)

    retrieval_top_k = args.retrieval_top_k
    try:
        reranker = _load_reranker(args.reranker)
    except (FileNotFoundError, NotImplementedError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    reranker_available = bool(getattr(reranker, "available", False)) if reranker else False
    if reranker is not None and not reranker_available:
        print(
            f"ERROR: --reranker {args.reranker} requested but the model is "
            f"not loadable (model.onnx missing?). Pass --reranker off to "
            f"probe without reranking, or install the model.",
            file=sys.stderr,
        )
        return 2

    # Step 1: canary-check the surprisal scorer.
    scorer = SurprisalScorer(model_name=args.scorer_model)
    print(f"[scorer] loading model {args.scorer_model} ...", flush=True)
    try:
        scorer.canary()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("[scorer] canary OK", flush=True)

    # Step 2: score all turns (with caching).
    cache = _load_cache(cache_path)
    print(f"[scorer] cache: {len(cache)} entries loaded from {cache_path}", flush=True)
    try:
        surprisal_data = collect_surprisal(
            conversations, scorer, cache, cache_path, args.max_error_rate,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Step 3: run variants.
    alphas = [float(a) for a in args.alpha.split(",")]
    variant_results: list[dict] = []

    # Always run baseline first.
    print("\n[variant] running baseline ...", flush=True)
    baseline_result = await _run_variant(
        "baseline", conversations, args, reranker, scorer, surprisal_data,
    )
    variant_results.append(baseline_result)

    if not baseline_result["rows"]:
        print(
            "ERROR: baseline variant produced zero results.",
            file=sys.stderr,
        )
        return 1

    # surprise_prior: disabled by default; scored flat +0.0000 at all alphas.
    # Re-enable with --include-prior for archival or re-validation runs.
    if args.include_prior:
        for alpha in alphas:
            vname = "surprise_prior"
            print(f"\n[variant] running {vname} alpha={alpha} ...", flush=True)
            result = await _run_variant(
                vname, conversations, args, reranker, scorer, surprisal_data,
                alpha=alpha,
            )
            variant_results.append(result)
            if not result["rows"]:
                print(
                    f"ERROR: variant {vname} alpha={alpha} produced zero results.",
                    file=sys.stderr,
                )
                return 1

    # surprise_chunks.
    print("\n[variant] running surprise_chunks ...", flush=True)
    chunks_result = await _run_variant(
        "surprise_chunks", conversations, args, reranker, scorer, surprisal_data,
    )
    variant_results.append(chunks_result)
    if not chunks_result["rows"]:
        print(
            "ERROR: variant surprise_chunks produced zero results.",
            file=sys.stderr,
        )
        return 1

    # Step 4: summarise and print.
    summaries = [_summarize_variant(r) for r in variant_results]
    baseline_r = summaries[0]["r_at_k"]

    _print_table(summaries, baseline_r)
    _print_category_table(summaries)
    _print_verdict(summaries, baseline_r)

    # Step 5: compute mean chunk length for surprise_chunks.
    chunks_summary = next((s for s in summaries if s["variant"] == "surprise_chunks"), None)
    chunk_mean_len_info = ""
    if chunks_summary:
        total_turns = sum(
            len(sd.get("scored_turns", []))
            for sd in surprisal_data.values()
        )
        total_chunks = chunks_summary.get("total_chunks_ingested", 0)
        if total_chunks > 0:
            mean_chunk_len = total_turns / total_chunks
            chunk_mean_len_info = f"{mean_chunk_len:.2f} turns/chunk"
            print(f"\nsurprise_chunks: mean chunk length = {chunk_mean_len_info}")

    # Step 6: write output JSON.
    config = {
        "dataset": str(dataset_path),
        "limit": args.limit,
        "conversations": min(args.conversations, len(conversations)),
        "top_k": args.top_k,
        "retrieval_top_k": retrieval_top_k,
        "fusion": args.fusion,
        "embed_mode": args.embed_mode,
        "onnx_path": args.onnx_path,
        "qmd_url": args.qmd_url,
        "reranker": args.reranker,
        "binary_quant": args.binary_quant,
        "late_interaction": args.late_interaction,
        "colbert_model": args.colbert_model,
        "scorer_model": args.scorer_model,
        "alpha_values": alphas,
        "git_sha": _git_sha(),
        "timestamp": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
    }
    output = {
        "config": config,
        "summaries": summaries,
        "variant_rows": {r["variant"]: r["rows"] for r in variant_results},
        "variant_errors": {r["variant"]: r["errors"] for r in variant_results},
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "E1 kill-shot harness: judge-free R@K probe for token-surprisal "
            "retrieval improvement (surprise_prior and surprise_chunks variants)."
        )
    )
    p.add_argument("--dataset", default=_DEFAULT_DATASET,
                   help=f"LoCoMo dataset JSON. Env: LOCOMO_DATASET "
                        f"(default: {_DEFAULT_DATASET})")
    p.add_argument("--limit", type=int, default=200,
                   help="Cap total QAs across all conversations per variant "
                        "(0=all). Default 200 to match subset-200 probes.")
    p.add_argument("--conversations", type=int, default=10)
    p.add_argument("--top-k", type=int, default=10,
                   help="Final hit count after reranking. Default 10.")
    p.add_argument("--retrieval-top-k", type=int, default=20,
                   help="Candidate pool fetched from vector search. Default 20.")
    p.add_argument("--fusion",
                   choices=["boost", "rrf", "bm25_rrf", "bm25_lemma_rrf",
                            "mem0_additive", "none"],
                   default="mem0_additive",
                   help="Vector + keyword fusion mode. Default mem0_additive.")
    p.add_argument("--embed-mode", choices=["qmd", "local", "onnx"],
                   default="onnx",
                   help="Embedding backend. Default onnx.")
    p.add_argument("--onnx-path", default=_DEFAULT_ONNX,
                   help=f"MiniLM ONNX model dir. Env: TAOSMD_ONNX_PATH "
                        f"(default: {_DEFAULT_ONNX})")
    p.add_argument("--qmd-url", default="http://localhost:7832",
                   help="qmd server URL (only used with --embed-mode qmd).")
    p.add_argument("--reranker", choices=["ms-marco", "bge-v2-m3", "off"],
                   default="ms-marco",
                   help="Cross-encoder reranker after vector retrieval.")
    p.add_argument("--binary-quant", action="store_true",
                   help="Sign-bit Hamming first-stage scoring.")
    p.add_argument("--late-interaction", action="store_true",
                   help="MiniLM token-level MaxSim.")
    p.add_argument("--colbert-model", default="", dest="colbert_model",
                   help="HF name or local path of a proper ColBERT model.")
    p.add_argument("--scorer-model",
                   default="HuggingFaceTB/SmolLM2-360M",
                   help="HF causal LM for surprisal scoring. "
                        "Default HuggingFaceTB/SmolLM2-360M.")
    p.add_argument("--alpha", default="0.25,0.5,1.0",
                   help="Comma-separated alpha values for surprise_prior. "
                        "Default 0.25,0.5,1.0.")
    p.add_argument("--max-error-rate", type=float, default=0.10,
                   help="Abort if more than this share of QAs error. Default 0.10.")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="HTTP timeout in seconds (qmd embed mode only).")
    p.add_argument("--out", default=None,
                   help="Results JSON path (default: "
                        "benchmarks/results/surprisal_probe_<ts>.json)")
    p.add_argument("--include-prior", action="store_true",
                   help="Re-enable the surprise_prior variants (disabled by default; "
                        "scored flat +0.0000 at all alphas in prior runs).")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(_parse_args())))
