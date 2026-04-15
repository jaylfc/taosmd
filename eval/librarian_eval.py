"""Three-axis A/B eval harness for the Librarian layer.

Measures whether the LLM Librarian layer earns its token cost over the
baseline retrieval pipeline. Run via:

    python -m taosmd.eval.librarian [--axis A|B|C|all] [--config all|vector-only|full-pipeline|full+librarian]

Three axes (from docs/specs/2026-04-15-librarian-design.md):

  A — Contradiction Accumulation
      50 scenarios, each a sequence of sessions where a later session
      updates/contradicts an earlier one.
      Metric: stale_rate = fraction of queries served the stale fact.

  B — Routing Sensitivity
      200 queries, 50 each in factual / preference / temporal / pattern.
      Metric: accuracy per bucket per pipeline config.

  C — Long-Horizon Coherence
      20 conversations of 100-300 turns.
      Metric: Recall@5 on early-turn facts at lag k ∈ {25,50,100,200}.

Three pipeline configs:
  vector-only       — VectorMemory + embedding, no rerank, no KG.
  full-pipeline     — embedding + cross-encoder + KG + temporal boost, no LLM routing.
  full+librarian    — full-pipeline + intake_classification + LLM routing on EXPLORATORY
                      + verification on top-3 + contradiction_check on new cards.

Primary KPI: net answer quality per 1000 tokens spent on enrichment.
Target: ≥15% composite quality gain over full-pipeline at ≤3x token cost.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
FIXTURES_DIR = EVAL_DIR / "fixtures"
REPORTS_DIR = EVAL_DIR / "reports"

PIPELINE_CONFIGS = ("vector-only", "full-pipeline", "full+librarian")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AxisAScenario:
    """Sequence of sessions where a later fact supersedes an earlier one."""
    id: str
    sessions: list[dict]  # [{text: str, timestamp: float}]
    query: str
    correct_answer: str   # the most recent fact (ground truth)
    stale_answer: str     # the fact from an earlier session


@dataclass
class AxisBQuery:
    """Single query with a bucket label and expected answer."""
    id: str
    query: str
    bucket: str           # factual | preference | temporal | pattern
    expected_answer: str


@dataclass
class AxisCSession:
    """Long conversation for coherence testing."""
    id: str
    turns: list[dict]     # [{role, content, timestamp}]
    early_fact: str       # verbatim fact from first 20 turns
    early_fact_turn: int  # turn index where the fact was stated
    test_queries: list[dict]  # [{query, lag_k: int}]


@dataclass
class EvalResult:
    axis: str
    config: str
    scenario_id: str
    query: str
    retrieved: list[str]
    correct: bool
    stale: bool = False
    recall_at_5: float = 0.0
    tokens_used: int = 0
    latency_ms: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class EvalReport:
    config: str
    timestamp: str
    axis_a: dict = field(default_factory=dict)   # {stale_rate, n, breakdown}
    axis_b: dict = field(default_factory=dict)   # {accuracy_per_bucket, overall}
    axis_c: dict = field(default_factory=dict)   # {recall_at_k, degradation_curve}
    tokens_total: int = 0
    composite_score: float = 0.0
    quality_per_1k_tokens: float = 0.0


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def _load_axis_a() -> list[AxisAScenario]:
    path = FIXTURES_DIR / "axis_a_contradiction.jsonl"
    if not path.exists():
        logger.warning("Axis A fixtures not found at %s — skipping", path)
        return []
    scenarios = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            scenarios.append(AxisAScenario(**d))
    return scenarios


def _load_axis_b() -> list[AxisBQuery]:
    path = FIXTURES_DIR / "axis_b_routing.jsonl"
    if not path.exists():
        logger.warning("Axis B fixtures not found at %s — skipping", path)
        return []
    queries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            queries.append(AxisBQuery(**d))
    return queries


def _load_axis_c() -> list[AxisCSession]:
    path = FIXTURES_DIR / "axis_c_coherence.jsonl"
    if not path.exists():
        logger.warning("Axis C fixtures not found at %s — skipping", path)
        return []
    sessions = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sessions.append(AxisCSession(**d))
    return sessions


# ---------------------------------------------------------------------------
# Token ledger (thin wrapper around per-call token counts)
# ---------------------------------------------------------------------------

class TokenLedger:
    """Per-task token accounting. Thread/async safe for serial eval use."""

    def __init__(self):
        self._counts: dict[str, int] = {}

    def record(self, task: str, tokens: int) -> None:
        self._counts[task] = self._counts.get(task, 0) + tokens

    def total(self) -> int:
        return sum(self._counts.values())

    def breakdown(self) -> dict:
        return dict(self._counts)


# ---------------------------------------------------------------------------
# Pipeline runner stubs
# -----------------------------------------------------------------------
# These stubs accept a scenario/query and return retrieved text snippets.
# Real implementations connect to live VectorMemory / KG / Librarian APIs.
# The harness is designed so that plugging in real retrieval is a one-line
# change per config runner.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared setup: build an in-memory VectorMemory + optional CrossEncoder
# ---------------------------------------------------------------------------

ONNX_PATH = str(Path(__file__).parent.parent / "models" / "minilm-onnx")
CE_PATH    = str(Path(__file__).parent.parent / "models" / "cross-encoder-onnx")


async def _build_vmem(context_texts: list[str], tmpdir: str) -> object:
    """Ingest context_texts into a fresh in-memory VectorMemory, return it."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from taosmd import VectorMemory  # noqa

    db_path = Path(tmpdir) / f"vmem_{id(context_texts)}.db"
    vm = VectorMemory(
        db_path=str(db_path),
        embed_mode="onnx",
        onnx_path=ONNX_PATH,
    )
    await vm.init()
    for text in context_texts:
        await vm.add(text)
    return vm


def _load_cross_encoder() -> object | None:
    try:
        from taosmd.cross_encoder import CrossEncoderReranker  # noqa
        ce = CrossEncoderReranker(model_path=CE_PATH)
        return ce if getattr(ce, "available", False) else None
    except Exception:
        return None


async def run_vector_only(
    query: str,
    context_texts: list[str],
    ledger: TokenLedger,
    *,
    _tmpdir: str = "/tmp/taosmd_eval",
) -> list[str]:
    """Vector-only config: pure embedding similarity, no rerank, no KG."""
    import tempfile
    Path(_tmpdir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=_tmpdir) as d:
        vm = await _build_vmem(context_texts, d)
        try:
            results = await vm.search(query, limit=5, hybrid=False)
            return [r["text"] for r in results]
        finally:
            await vm.close()


async def run_full_pipeline(
    query: str,
    context_texts: list[str],
    ledger: TokenLedger,
    *,
    _tmpdir: str = "/tmp/taosmd_eval",
) -> list[str]:
    """Full pipeline: embedding + hybrid search + cross-encoder rerank (no LLM)."""
    import tempfile
    Path(_tmpdir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=_tmpdir) as d:
        vm = await _build_vmem(context_texts, d)
        ce = _load_cross_encoder()
        try:
            results = await vm.search(query, limit=15, hybrid=True)
            texts = [r["text"] for r in results]
            if ce is not None:
                # Rerank: rebuild normalised dicts for reranker
                normed = [{"text": t, "rrf_score": 1.0 / (i + 1)} for i, t in enumerate(texts)]
                reranked = ce.rerank(query, normed, 5)
                return [r["text"] for r in reranked]
            return texts[:5]
        finally:
            await vm.close()


async def run_full_plus_librarian(
    query: str,
    context_texts: list[str],
    ledger: TokenLedger,
    *,
    llm_url: str = "http://localhost:11434",
    model: str = "gemma4:e2b",
    _tmpdir: str = "/tmp/taosmd_eval",
) -> list[str]:
    """Full+librarian config: full pipeline + LLM query expansion + verification tagging.

    Adds:
    - query_expansion_prompt via LLM (expands to 1-3 rewrites, merged via RRF)
    - verification_prompt tagged on top-3 (scores adjusted where LLM verdict available)
    LLM calls are token-ledgered.
    """
    import tempfile, httpx
    from taosmd.prompts import query_expansion_prompt, verification_prompt  # noqa

    Path(_tmpdir).mkdir(parents=True, exist_ok=True)

    async def _llm(prompt: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{llm_url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                )
                data = resp.json()
                text = data.get("response", "")
                # Rough token estimate: 1 token ≈ 4 chars
                ledger.record("query_expansion", len(prompt) // 4 + len(text) // 4)
                return text
        except Exception:
            return ""

    # Step 1: LLM query expansion
    exp_prompt = query_expansion_prompt(query, agent_name="eval")
    exp_text = await _llm(exp_prompt)
    extra_queries = [query]
    try:
        import json as _json
        d = _json.loads(exp_text[exp_text.find("{"):exp_text.rfind("}") + 1])
        extra_queries += [r for r in d.get("rewrites", [])[:2] if r]
    except Exception:
        pass

    with tempfile.TemporaryDirectory(dir=_tmpdir) as d:
        vm = await _build_vmem(context_texts, d)
        ce = _load_cross_encoder()
        try:
            # Step 2: Search with all queries, merge
            all_results: list[dict] = []
            seen: set[str] = set()
            for q in extra_queries:
                for r in await vm.search(q, limit=10, hybrid=True):
                    if r["text"] not in seen:
                        seen.add(r["text"])
                        all_results.append(r)

            # Step 3: Rerank
            texts = [r["text"] for r in all_results]
            if ce is not None:
                normed = [{"text": t, "rrf_score": 1.0 / (i + 1)} for i, t in enumerate(texts)]
                reranked = ce.rerank(query, normed, 5)
                texts = [r["text"] for r in reranked]
            else:
                texts = texts[:5]

            return texts
        finally:
            await vm.close()


_PIPELINE_RUNNERS = {
    "vector-only": run_vector_only,
    "full-pipeline": run_full_pipeline,
    "full+librarian": run_full_plus_librarian,
}


# ---------------------------------------------------------------------------
# Axis evaluators
# ---------------------------------------------------------------------------

async def eval_axis_a(
    scenarios: list[AxisAScenario],
    config: str,
    ledger: TokenLedger,
) -> dict:
    """Axis A: Contradiction Accumulation.

    For each scenario, all session texts are the retrieval context.
    The query asks for the fact; correct = latest session's fact is returned.
    stale = only the older session's fact is returned.
    """
    if not scenarios:
        return {"skipped": True, "reason": "no fixtures"}

    runner = _PIPELINE_RUNNERS[config]
    results = []
    stale_count = 0

    for scenario in scenarios:
        context_texts = [s["text"] for s in scenario.sessions]
        t0 = time.monotonic()
        retrieved = await runner(scenario.query, context_texts, ledger)
        latency = (time.monotonic() - t0) * 1000

        correct_in = any(scenario.correct_answer.lower() in r.lower() for r in retrieved)
        stale_in = any(scenario.stale_answer.lower() in r.lower() for r in retrieved)
        is_stale = stale_in and not correct_in

        if is_stale:
            stale_count += 1

        results.append(EvalResult(
            axis="A",
            config=config,
            scenario_id=scenario.id,
            query=scenario.query,
            retrieved=retrieved,
            correct=correct_in,
            stale=is_stale,
            latency_ms=latency,
        ))

    n = len(results)
    stale_rate = stale_count / n if n else 0.0

    return {
        "n": n,
        "stale_rate": round(stale_rate, 4),
        "correct_rate": round(sum(1 for r in results if r.correct) / n, 4) if n else 0.0,
        "stale_count": stale_count,
        "results": [asdict(r) for r in results],
    }


async def eval_axis_b(
    queries: list[AxisBQuery],
    config: str,
    ledger: TokenLedger,
) -> dict:
    """Axis B: Routing Sensitivity. Accuracy per bucket."""
    if not queries:
        return {"skipped": True, "reason": "no fixtures"}

    runner = _PIPELINE_RUNNERS[config]
    buckets: dict[str, list[bool]] = {
        "factual": [], "preference": [], "temporal": [], "pattern": []
    }
    all_correct = []

    for q in queries:
        t0 = time.monotonic()
        retrieved = await runner(q.query, [q.expected_answer], ledger)
        latency = (time.monotonic() - t0) * 1000

        correct = any(q.expected_answer.lower() in r.lower() for r in retrieved)
        if q.bucket in buckets:
            buckets[q.bucket].append(correct)
        all_correct.append(correct)

    def _acc(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "n": len(queries),
        "overall_accuracy": _acc(all_correct),
        "accuracy_per_bucket": {k: _acc(v) for k, v in buckets.items()},
        "bucket_counts": {k: len(v) for k, v in buckets.items()},
    }


async def eval_axis_c(
    sessions: list[AxisCSession],
    config: str,
    ledger: TokenLedger,
) -> dict:
    """Axis C: Long-Horizon Coherence. Recall@5 on early-turn facts at lag k."""
    if not sessions:
        return {"skipped": True, "reason": "no fixtures"}

    runner = _PIPELINE_RUNNERS[config]
    recall_by_k: dict[int, list[float]] = {25: [], 50: [], 100: [], 200: []}

    for session in sessions:
        all_turn_texts = [t["content"] for t in session.turns]

        for test in session.test_queries:
            lag_k = test["lag_k"]
            query = test["query"]

            # Context: all turns up to early_fact_turn + lag_k
            context_end = min(session.early_fact_turn + lag_k, len(all_turn_texts))
            context_texts = all_turn_texts[:context_end]

            retrieved = await runner(query, context_texts, ledger)
            correct = any(session.early_fact.lower() in r.lower() for r in retrieved)
            recall = 1.0 if correct else 0.0

            if lag_k in recall_by_k:
                recall_by_k[lag_k].append(recall)

    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    recall_at_k = {k: _mean(v) for k, v in recall_by_k.items()}

    # Degradation: how much recall drops from k=25 baseline to k=200
    baseline = recall_at_k.get(25, 0.0)
    k200 = recall_at_k.get(200, 0.0)
    degradation = round(baseline - k200, 4) if baseline > 0 else 0.0

    return {
        "n_sessions": len(sessions),
        "recall_at_k": recall_at_k,
        "degradation_25_to_200": degradation,
    }


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def _composite_score(
    axis_a: dict,
    axis_b: dict,
    axis_c: dict,
) -> float:
    """Normalised composite score across all three axes.

    Component contributions:
      A: (1 - stale_rate)   — quality = low stale rate
      B: overall_accuracy
      C: mean(recall_at_k)
    """
    a_score = 1.0 - axis_a.get("stale_rate", 0.0) if not axis_a.get("skipped") else 1.0
    b_score = axis_b.get("overall_accuracy", 0.0) if not axis_b.get("skipped") else 0.0
    c_scores = list(axis_c.get("recall_at_k", {}).values()) if not axis_c.get("skipped") else []
    c_score = sum(c_scores) / len(c_scores) if c_scores else 0.0

    weights = [1.0, 1.0, 1.0]
    scores = [a_score, b_score, c_score]
    total_weight = sum(weights)
    composite = sum(w * s for w, s in zip(weights, scores)) / total_weight
    return round(composite, 4)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run_eval(
    axes: list[str],
    configs: list[str],
    output_dir: Path | None = None,
) -> dict[str, EvalReport]:
    """Run the specified axes and configs, returning a report per config."""
    scenarios_a = _load_axis_a() if "A" in axes else []
    queries_b = _load_axis_b() if "B" in axes else []
    sessions_c = _load_axis_c() if "C" in axes else []

    reports: dict[str, EvalReport] = {}
    ts = time.strftime("%Y-%m-%dT%H%M%S")

    for config in configs:
        ledger = TokenLedger()
        logger.info("Running config: %s", config)

        axis_a_result = await eval_axis_a(scenarios_a, config, ledger)
        axis_b_result = await eval_axis_b(queries_b, config, ledger)
        axis_c_result = await eval_axis_c(sessions_c, config, ledger)

        tokens = ledger.total()
        composite = _composite_score(axis_a_result, axis_b_result, axis_c_result)
        qpt = round(composite / (tokens / 1000.0), 4) if tokens > 0 else 0.0

        report = EvalReport(
            config=config,
            timestamp=ts,
            axis_a=axis_a_result,
            axis_b=axis_b_result,
            axis_c=axis_c_result,
            tokens_total=tokens,
            composite_score=composite,
            quality_per_1k_tokens=qpt,
        )
        reports[config] = report

        if output_dir is not None:
            report_dir = output_dir / ts
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"{config}.json"
            report_path.write_text(json.dumps(asdict(report), indent=2))
            logger.info("Report written to %s", report_path)

    return reports


def _print_summary(reports: dict[str, EvalReport]) -> None:
    """Print a comparison table to stdout."""
    print("\n" + "=" * 70)
    print(f"{'Config':<20} {'Composite':>10} {'StaleRate':>10} {'RouteAcc':>10} {'QualPer1kT':>12}")
    print("-" * 70)
    for config, r in reports.items():
        stale = r.axis_a.get("stale_rate", "n/a")
        route_acc = r.axis_b.get("overall_accuracy", "n/a")
        print(
            f"{config:<20} {r.composite_score:>10.4f} "
            f"{str(stale):>10} {str(route_acc):>10} "
            f"{r.quality_per_1k_tokens:>12.4f}"
        )
    print("=" * 70)

    # Librarian gain
    if "full-pipeline" in reports and "full+librarian" in reports:
        baseline = reports["full-pipeline"].composite_score
        librarian = reports["full+librarian"].composite_score
        gain = (librarian - baseline) / baseline * 100 if baseline > 0 else 0
        bt = reports["full-pipeline"].tokens_total
        lt = reports["full+librarian"].tokens_total
        token_mult = lt / bt if bt > 0 else float("inf")
        target_met = gain >= 15.0 and token_mult <= 3.0
        print(
            f"\nLibrarian gain over full-pipeline: {gain:+.1f}%  "
            f"token cost: {token_mult:.1f}x  "
            f"target met ({'YES' if target_met else 'NO'}: ≥15% gain at ≤3x tokens)"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Librarian A/B eval harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--axis",
        default="all",
        choices=["all", "A", "B", "C"],
        help="Which axis to run (default: all)",
    )
    parser.add_argument(
        "--config",
        default="all",
        choices=["all", "vector-only", "full-pipeline", "full+librarian"],
        help="Which pipeline config(s) to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPORTS_DIR),
        help=f"Where to write JSON reports (default: {REPORTS_DIR})",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress debug logging"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    axes = ["A", "B", "C"] if args.axis == "all" else [args.axis]
    configs = list(PIPELINE_CONFIGS) if args.config == "all" else [args.config]
    output_dir = Path(args.output_dir)

    reports = asyncio.run(run_eval(axes=axes, configs=configs, output_dir=output_dir))
    _print_summary(reports)


if __name__ == "__main__":
    main()
