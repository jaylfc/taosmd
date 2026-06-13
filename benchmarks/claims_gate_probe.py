#!/usr/bin/env python3
"""E-009: does the claims gate improve answers without tanking recall?

Pre-registered experiment (docs/research-report.md rev 1.16). On a LoCoMo
slice it ingests each conversation, extracts claims with archive-span
provenance, verifies them with a cross-family local LLM, then answers every
eligible question THREE times -- gate ``off`` (baseline), ``prefer_verified``,
and ``strict`` -- judged externally. It reports, per mode:

  * judged accuracy (external YES/NO judge, same as the LoCoMo runner)
  * mean R@K (evidence overlap by ``dia_id`` on the gated hits)
  * served-hallucination rate (share of answered QAs whose generation context
    still contained a hit whose backing claim is unsupported/contradicted)

plus the live claim-status distribution and the measured extraction
hallucination rate (F-009 as a standing number).

Faithfulness. Every claims-layer component is the SHIPPED code:
``claims_from_text`` (the regex extractor + provenance), ``ClaimStore``,
``LocalEntailmentVerifier``, ``verify_pass``, and the pure ``apply_claims_gate``.
The only harness glue is binding provenance to the runner's per-turn rows: each
ingested turn gets a monotonic integer span id (its ``turn_idx``), claims are
stored against that id, and at gate time each hit's ``claim_status`` is looked
up with the real ``ClaimStore.status_for_spans([turn_idx])`` -- exactly the
production path's ``archive_span_id`` semantics. Retrieval, context building,
generation, judging, and R@K reuse the LoCoMo runner's helpers verbatim, so the
baseline (gate off) reproduces the standard runner's measurement.

Kill criterion (verbatim, pre-registered): the claims gate ships default-on
only if it reduces the served-hallucination rate by a meaningful margin AND does
not drop judged accuracy or R@K by more than 0.02. If it trades accuracy for
purity, it ships default-OFF as an opt-in integrity mode with its measured trade
documented; it is never silently enabled.

Offline plumbing check (no Ollama, no judge -- proves the wiring):

    python benchmarks/claims_gate_probe.py --self-test

Real run (Ollama serving generator + verifier + judge):

    python benchmarks/claims_gate_probe.py \
        --dataset benchmarks/data/locomo10.json --conversations 10 \
        --per-conv-limit 20 --model qwen3:4b \
        --verifier-model qwen3:4b-instruct-2507 --judge-model qwen3:14b \
        --embed-mode onnx --out benchmarks/results/e009.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from taosmd.vector_memory import VectorMemory  # noqa: E402
from taosmd.claims.store import ClaimStore  # noqa: E402
from taosmd.claims.extract import claims_from_text  # noqa: E402
from taosmd.claims.gate import apply_claims_gate  # noqa: E402
from taosmd.claims.verifier import LocalEntailmentVerifier, FakeVerifier  # noqa: E402
from taosmd.claims.verify_pass import verify_pass  # noqa: E402

# Reuse the LoCoMo runner's proven helpers so the gate-off baseline reproduces
# the standard runner's measurement exactly.
from locomo_runner import (  # noqa: E402
    ANSWER_PROMPT,
    _build_context,
    _evidence_hits,
    _generate,
    _judge,
    _load_reranker,
    _retrieve,
    _session_keys,
)

MODES = ("off", "prefer_verified", "strict")
_HALLUCINATED = ("unsupported", "contradicted")


# --------------------------------------------------------------------------- #
# Ingest: bind each turn to an integer span id, extract claims against it.
# --------------------------------------------------------------------------- #
async def _ingest(
    vmem: VectorMemory, store: ClaimStore, conv: dict
) -> tuple[int, int, dict[int, str]]:
    """Ingest turns into ``vmem`` and claims into ``store``.

    Returns ``(turns, claims, span_texts)`` where ``span_texts`` maps each
    turn's integer span id to its text (the verify-pass's ``fetch_spans``
    source). The span id is the turn's ``turn_idx`` -- the same key the recall
    gate looks claims up by.
    """
    conversation = conv.get("conversation", conv)
    span_texts: dict[int, str] = {}
    turns = 0
    claims = 0
    span_id = 0
    for session_key, dt in _session_keys(conversation):
        for turn in conversation.get(session_key) or []:
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            speaker = turn.get("speaker", "")
            dia_id = turn.get("dia_id", "")
            full = f"[{speaker}] {text}"
            await vmem.add(
                full,
                metadata={
                    "dia_id": dia_id,
                    "session": session_key,
                    "datetime": dt,
                    "speaker": speaker,
                    "turn_idx": span_id,
                    "level": "turn",
                },
            )
            span_texts[span_id] = full
            # Real extractor + provenance binding (production code path).
            for c in claims_from_text(full, span_id):
                await store.add_claim(
                    c["text"], c["archive_span_ids"], c["source_extractor"]
                )
                claims += 1
            span_id += 1
            turns += 1
    return turns, claims, span_texts


# --------------------------------------------------------------------------- #
# Gate application: the real status lookup + the real pure gate.
# --------------------------------------------------------------------------- #
async def _gate(hits: list[dict], store: ClaimStore, mode: str) -> list[dict]:
    """Attach each hit's backing-claim status and apply the real recall gate.

    Mirrors ``taosmd.api._attach_and_gate_claims``: set a transient ``score``
    mirror for the pure gate, look up ``claim_status`` via the real store, run
    ``apply_claims_gate``. Hits keep ``claim_status`` here (the caller needs it
    to score served-hallucination); the production wiring strips it because the
    contract shape does not carry it.
    """
    for h in hits:
        turn_idx = (h.get("metadata") or {}).get("turn_idx")
        spans = [turn_idx] if isinstance(turn_idx, int) else []
        h["claim_status"] = await store.status_for_spans(spans)
        h.setdefault("score", h.get("score", 0.0))
    return apply_claims_gate(hits, mode=mode)


def _served_hallucination(hits: list[dict]) -> bool:
    """True if any hit in the generation context is backed by an unsupported or
    contradicted claim -- the exposure the gate exists to remove."""
    return any(h.get("claim_status") in _HALLUCINATED for h in hits)


# --------------------------------------------------------------------------- #
# Per-QA: retrieve once, answer under each mode.
# --------------------------------------------------------------------------- #
async def _process_qa(
    qa: dict,
    vmem: VectorMemory,
    store: ClaimStore,
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    reranker: object | None,
) -> dict | None:
    if "answer" not in qa:
        return None
    question = qa["question"]
    reference = str(qa["answer"])
    category = int(qa.get("category", 0))
    evidence = qa.get("evidence", []) or []

    # One retrieval; the gate operates post-retrieval, the same place
    # production applies it. ``_retrieve`` fetches a candidate pool of
    # ``retrieval_top_k`` (the reranker, when present, narrows it to
    # ``top_k``); we then narrow to ``top_k`` BEFORE gating so every mode --
    # including the off baseline -- gates exactly the production-shaped input
    # that ``search(limit=top_k)`` hands to ``_attach_and_gate_claims``. The
    # gate may then drop hits, so gate-on contexts can be smaller; that is the
    # real behavior under test, and there is no post-gate slice (production
    # does not re-slice after gating).
    raw = await _retrieve(
        args.strategy,
        question,
        vmem,
        args.retrieval_top_k,
        reranker=reranker,
        fusion=args.fusion,
        rerank_top_k=args.top_k,
    )
    raw = raw[: args.top_k]

    per_mode: dict[str, dict] = {}
    for mode in MODES:
        hits = [
            {"text": h.get("text", ""), "metadata": dict(h.get("metadata") or {}),
             "score": h.get("score", 0.0)}
            for h in raw
        ]
        context_hits = await _gate(hits, store, mode)
        context = _build_context(context_hits, context_format=args.context_format)
        try:
            predicted = await _generate(
                args.llm_backend, client, args.ollama_url, args.model,
                ANSWER_PROMPT.format(context=context, question=question),
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001 - record, never crash the sweep
            print(f"[e009] generate failed ({mode}): {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            predicted = ""
        judged = await _judge(
            client, args.ollama_url, args.judge_model,
            question, reference, predicted, backend=args.llm_backend,
        )
        per_mode[mode] = {
            "predicted": predicted,
            "judge": judged,
            "recall_at_k": _evidence_hits(context_hits, evidence),
            "evidence_total": len(evidence),
            "served_hallucination": _served_hallucination(context_hits),
            "kept": len(context_hits),
        }

    return {
        "question": question,
        "reference": reference,
        "category": category,
        "modes": per_mode,
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _aggregate(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    n = len(rows)
    for mode in MODES:
        if not n:
            out[mode] = {"n": 0, "judge": 0.0, "recall_rate": 0.0,
                         "served_hallucination_rate": 0.0}
            continue
        ms = [r["modes"][mode] for r in rows]
        judge = sum(m["judge"] for m in ms) / n
        # R@K exactly as the LoCoMo runner defines it (_summary): over QAs that
        # carry evidence, the fraction that retrieved at least one evidence
        # span. Same number the kill criterion is registered against.
        hit_rows = [m for m in ms if m["evidence_total"] > 0]
        recall = (sum(1 for m in hit_rows if m["recall_at_k"] > 0) / len(hit_rows)) if hit_rows else 0.0
        sh = sum(1 for m in ms if m["served_hallucination"]) / n
        out[mode] = {
            "n": n,
            "judge": round(judge, 4),
            "recall_rate": round(recall, 4),
            "served_hallucination_rate": round(sh, 4),
        }
    return out


def _verdict(agg: dict) -> dict:
    """Score each gated mode against the off baseline by the pre-registered
    criterion. ``meets_necessary_conditions`` encodes the two HARD, measurable
    bounds (judged accuracy and R@K each drop by <= 0.02) plus a non-zero
    served-hallucination reduction. The registered "meaningful margin" on that
    reduction is a qualitative human call applied at review against the raw
    ``served_hallucination_drop`` printed below -- the gate is never flipped
    silently, so the harness reports the numbers and does not self-authorize."""
    base = agg["off"]
    out = {}
    for mode in ("prefer_verified", "strict"):
        m = agg[mode]
        sh_drop = base["served_hallucination_rate"] - m["served_hallucination_rate"]
        judge_drop = base["judge"] - m["judge"]
        recall_drop = base["recall_rate"] - m["recall_rate"]
        meets_necessary_conditions = (
            sh_drop > 0.0
            and judge_drop <= 0.02
            and recall_drop <= 0.02
        )
        out[mode] = {
            "served_hallucination_drop": round(sh_drop, 4),
            "judge_drop": round(judge_drop, 4),
            "recall_drop": round(recall_drop, 4),
            "meets_necessary_conditions": meets_necessary_conditions,
        }
    return out


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
async def run(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    conversations = json.loads(dataset_path.read_text())[: args.conversations]

    include_cats = {1, 2, 3, 4}
    if args.include_adversarial:
        include_cats.add(5)

    try:
        reranker = _load_reranker(args.reranker)
    except (FileNotFoundError, NotImplementedError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    claim_total = 0
    # The verifier's verify() is synchronous (it blocks on a request inside the
    # already-async verify-pass), so it needs a sync httpx.Client -- the same
    # construction the `taosmd verify` CLI uses. The async client serves the
    # generator and judge.
    sync_client = httpx.Client(timeout=httpx.Timeout(args.verifier_timeout))
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        verifier = LocalEntailmentVerifier(
            sync_client, args.ollama_url, args.verifier_model,
            timeout=args.verifier_timeout,
        )
        for ci, conv in enumerate(conversations):
            conv_id = conv.get("sample_id", f"conv_{ci}")
            with tempfile.TemporaryDirectory() as tmp:
                vmem = VectorMemory(
                    db_path=os.path.join(tmp, "vmem.db"),
                    qmd_url=args.qmd_url,
                    embed_mode=args.embed_mode,
                    onnx_path=args.onnx_path,
                )
                await vmem.init(http_client=client)
                store = ClaimStore(db_path=os.path.join(tmp, "claims.db"))
                await store.init()

                turns, claims, span_texts = await _ingest(vmem, store, conv)
                claim_total += claims

                async def fetch_spans(ids: list[int]) -> list[str]:
                    return [span_texts[i] for i in ids if i in span_texts]

                verified = await verify_pass(store, verifier, fetch_spans,
                                             batch=args.verify_batch)
                rate = await store.rate()
                print(
                    f"[{conv_id}] {turns} turns, {claims} claims, "
                    f"{verified} verified, hall_rate={rate['hallucination_rate']:.3f}",
                    flush=True,
                )

                eligible: list[dict] = []
                for qa in conv.get("qa", []) or []:
                    if int(qa.get("category", 0)) not in include_cats:
                        continue
                    if args.per_conv_limit and len(eligible) >= args.per_conv_limit:
                        break
                    eligible.append(qa)

                for qa in eligible:
                    row = await _process_qa(qa, vmem, store, client, args, reranker)
                    if row is not None:
                        rows.append(row)

                await store.close()
                await vmem.close()
    sync_client.close()

    agg = _aggregate(rows)
    verdict = _verdict(agg)
    report = {
        "experiment": "E-009",
        "dataset": str(dataset_path),
        "conversations": len(conversations),
        "qas": len(rows),
        "claims_extracted": claim_total,
        "config": {
            "model": args.model,
            "verifier_model": args.verifier_model,
            "judge_model": args.judge_model,
            "embed_mode": args.embed_mode,
            "top_k": args.top_k,
            "retrieval_top_k": args.retrieval_top_k,
            "reranker": args.reranker,
            "fusion": args.fusion,
            "strategy": args.strategy,
        },
        "aggregate": agg,
        "verdict": verdict,
    }

    print("\n=== E-009 claims-gate probe ===")
    print(f"QAs: {len(rows)}   claims extracted: {claim_total}")
    hdr = f"{'mode':<16}{'judge':>8}{'recall':>8}{'served_hall':>13}"
    print(hdr)
    for mode in MODES:
        a = agg[mode]
        print(f"{mode:<16}{a['judge']:>8.3f}{a['recall_rate']:>8.3f}"
              f"{a['served_hallucination_rate']:>13.3f}")
    print("\nverdict vs gate-off (necessary conditions; 'meaningful margin' is a human call):")
    for mode, v in verdict.items():
        print(f"  {mode}: served_hall_drop={v['served_hallucination_drop']:+.3f}  "
              f"judge_drop={v['judge_drop']:+.3f}  recall_drop={v['recall_drop']:+.3f}  "
              f"=> meets_necessary_conditions={v['meets_necessary_conditions']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# Self-test: prove the plumbing offline (no Ollama, no judge).
# --------------------------------------------------------------------------- #
async def self_test() -> int:
    """Drive ingest -> verify -> gate end to end with a FakeVerifier and no
    network. Asserts the gate actually demotes unsupported-backed hits.
    """
    with tempfile.TemporaryDirectory() as tmp:
        store = ClaimStore(db_path=os.path.join(tmp, "claims.db"))
        await store.init()
        # Two spans, one supported claim, one we will mark unsupported.
        span_texts = {0: "[A] Alice lives in Paris.", 1: "[B] Bob hates olives."}
        cid0 = await store.add_claim("Alice lives Paris", [0], "regex")
        cid1 = await store.add_claim("Bob loves olives", [1], "regex")

        # Scripted verifier: claim text -> verdict.
        verifier = FakeVerifier({
            "Alice lives Paris": "supported",
            "Bob loves olives": "unsupported",
        })

        async def fetch_spans(ids):
            return [span_texts[i] for i in ids if i in span_texts]

        await verify_pass(store, verifier, fetch_spans)
        assert (await store.get(cid0))["status"] == "supported"
        assert (await store.get(cid1))["status"] == "unsupported"

        hits = [
            {"text": span_texts[0], "metadata": {"turn_idx": 0, "dia_id": "D0"}, "score": 0.5},
            {"text": span_texts[1], "metadata": {"turn_idx": 1, "dia_id": "D1"}, "score": 0.9},
        ]

        off = await _gate([dict(h, metadata=dict(h["metadata"])) for h in hits], store, "off")
        assert len(off) == 2, "off must be a passthrough"
        assert _served_hallucination(off) is True, "off exposes the unsupported hit"

        pref = await _gate([dict(h, metadata=dict(h["metadata"])) for h in hits], store, "prefer_verified")
        assert [h["metadata"]["turn_idx"] for h in pref] == [0], "prefer_verified drops the unsupported hit"
        assert _served_hallucination(pref) is False
        assert pref[0]["score"] > 0.9, "supported hit is boosted above the raw 0.9"

        strict = await _gate([dict(h, metadata=dict(h["metadata"])) for h in hits], store, "strict")
        assert [h["metadata"]["turn_idx"] for h in strict] == [0], "strict keeps only supported"

        rate = await store.rate()
        assert abs(rate["hallucination_rate"] - 0.5) < 1e-9, "1 of 2 checked is unsupported"
        await store.close()
    print("self-test OK: ingest -> verify -> gate demotes unsupported, boosts supported")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="E-009 claims-gate probe (LoCoMo)")
    p.add_argument("--self-test", action="store_true",
                   help="run the offline plumbing check and exit (no network)")
    p.add_argument("--dataset", default="benchmarks/data/locomo10.json")
    p.add_argument("--conversations", type=int, default=10)
    p.add_argument("--per-conv-limit", type=int, default=0,
                   help="cap QAs per conversation (0 = all eligible)")
    p.add_argument("--include-adversarial", action="store_true",
                   help="include category-5 (adversarial) QAs")
    p.add_argument("--model", default="qwen3:4b", help="generator model")
    p.add_argument("--verifier-model", default="qwen3:4b-instruct-2507",
                   help="cross-family entailment verifier (F-009 method)")
    p.add_argument("--judge-model", default="qwen3:14b", help="external answer judge")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--llm-backend", choices=["ollama", "llama-server"], default="ollama")
    p.add_argument("--qmd-url", default="http://localhost:7832")
    p.add_argument("--embed-mode", choices=["qmd", "local", "onnx"], default="onnx")
    p.add_argument("--onnx-path", default="models/all-MiniLM-L6-v2.onnx")
    p.add_argument("--top-k", type=int, default=10, help="final context size")
    p.add_argument("--retrieval-top-k", type=int, default=None,
                   help="candidate pool fetched before rerank/gate "
                        "(default: = top-k; only differs with a reranker)")
    p.add_argument("--reranker", choices=["ms-marco", "bge-v2-m3", "off"], default="off")
    p.add_argument("--fusion", default="boost")
    p.add_argument("--strategy", default="vector-only")
    p.add_argument("--context-format", choices=["plain", "session_date", "both"],
                   default="plain")
    p.add_argument("--verify-batch", type=int, default=100)
    p.add_argument("--verifier-timeout", type=float, default=60.0)
    p.add_argument("--out", default="")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.self_test:
        return asyncio.run(self_test())
    if args.retrieval_top_k is None:
        args.retrieval_top_k = args.top_k
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
