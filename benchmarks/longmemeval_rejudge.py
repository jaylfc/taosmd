#!/usr/bin/env python3
"""Re-judge LongMemEval results with different judge modes.

Reads a results file written with TAOSMD_LME_SAVE_CONTEXT=1 and re-scores
every row under each requested mode by calling the same score_answer_llm
from the runner. Outputs per-mode accuracy overall and by question_type,
plus an agreement block.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

import httpx


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "benchmarks" / "longmemeval_runner.py"

# Allow injection of a pre-loaded runner module for testing
_runner_mod_override = None
# Exposed for tests that set runner_mod directly
runner_mod = None


def _load_runner():
    spec = importlib.util.spec_from_file_location("longmemeval_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _get_runner_mod():
    global _runner_mod_override, runner_mod
    if _runner_mod_override is not None:
        return _runner_mod_override
    if runner_mod is not None:
        return runner_mod
    return _load_runner()


def set_runner_mod_for_testing(mod):
    """Inject a runner module for testing. Call with None to reset."""
    global _runner_mod_override, runner_mod
    _runner_mod_override = mod
    runner_mod = mod


async def _score_with_mode(runner_mod, client, row, mode: str) -> bool:
    """Score a single row using the specified judge mode."""
    # Use the provided runner_mod (which may be a test-injected stub) so
    # monkeypatches in tests apply. Set the env var for this call so the
    # runner reads it at runtime.
    os.environ["TAOSMD_JUDGE_MODE"] = mode
    try:
        return await runner_mod.score_answer_llm(
            client,
            row.get("predicted", ""),  # The predicted answer
            row.get("answer", ""),      # The gold/reference answer
            row["question"],
            row.get("context") if mode == "evidence" else None,
        )
    finally:
        os.environ.pop("TAOSMD_JUDGE_MODE", None)


def _compute_accuracy(results: list[dict], question_type: str | None = None) -> float:
    """Compute accuracy for a list of results, optionally filtered by type."""
    if question_type:
        filtered = [r for r in results if r["question_type"] == question_type]
    else:
        filtered = results
    if not filtered:
        return 0.0
    correct = sum(1 for r in filtered if r["correct"])
    return correct / len(filtered) * 100


def _compute_agreement(ref_results: list[dict], ev_results: list[dict]) -> dict:
    """Compute agreement between reference and evidence mode results."""
    total = len(ref_results)
    agree = 0
    ref_correct_ev_incorrect = 0
    ref_incorrect_ev_correct = 0

    for r, e in zip(ref_results, ev_results):
        if r["correct"] == e["correct"]:
            agree += 1
        elif r["correct"] and not e["correct"]:
            ref_correct_ev_incorrect += 1
        elif not r["correct"] and e["correct"]:
            ref_incorrect_ev_correct += 1

    return {
        "total": total,
        "agree": agree,
        "disagree": total - agree,
        "reference_correct_evidence_incorrect": ref_correct_ev_incorrect,
        "reference_incorrect_evidence_correct": ref_incorrect_ev_correct,
    }


def _print_table(modes_data: dict):
    """Print a 3-column table: mode, accuracy, n."""
    print("\n  Mode         Accuracy     N")
    print("  " + "-" * 30)
    for mode, data in modes_data.items():
        print(f"  {mode:<12} {data['accuracy']:>6.1f}%  {data['n']}")


async def main_async(args):
    # Load results file
    with open(args.results) as f:
        doc = json.load(f)

    results = doc.get("results", [])
    if not results:
        print("ERROR: No results in file", file=sys.stderr)
        sys.exit(1)

    # Verify all rows have context
    missing_context = [r for r in results if "context" not in r]
    if missing_context:
        print(
            f"ERROR: {len(missing_context)} row(s) missing 'context' key. "
            "Re-judge requires TAOSMD_LME_SAVE_CONTEXT=1.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Load runner module
    runner_mod = _get_runner_mod()

    # Create HTTP client for judge
    ollama_url = args.ollama_url or getattr(runner_mod, "REMOTE_LLM_URL", "http://localhost:11434")
    judge_model = args.judge_model or getattr(runner_mod, "JUDGE_MODEL", "qwen2.5:3b")
    runner_mod.REMOTE_LLM_URL = ollama_url
    runner_mod.JUDGE_MODEL = judge_model

    client = httpx.AsyncClient(timeout=30)

    try:
        modes = args.modes.split(",")
        modes_data = {}
        mode_results = {}

        for mode in modes:
            print(f"  Re-judging in {mode} mode...")
            mode_correct = []
            for row in results:
                correct = await _score_with_mode(runner_mod, client, row, mode)
                mode_correct.append(correct)

            # Build results with correctness for this mode
            mode_results[mode] = [
                {**row, "correct": correct} for row, correct in zip(results, mode_correct)
            ]

            # Overall accuracy
            n = len(results)
            correct_count = sum(mode_correct)
            accuracy = correct_count / n * 100 if n > 0 else 0.0

            # By question type
            by_type = {}
            for qtype in set(r["question_type"] for r in results):
                type_results = [r for r in mode_results[mode] if r["question_type"] == qtype]
                type_correct = sum(1 for r in type_results if r["correct"])
                type_total = len(type_results)
                by_type[qtype] = {
                    "n": type_total,
                    "correct": type_correct,
                    "accuracy": type_correct / type_total * 100 if type_total > 0 else 0.0,
                }

            modes_data[mode] = {
                "accuracy": accuracy,
                "n": n,
                "correct": correct_count,
                "by_type": by_type,
            }

        # Agreement between modes (only if exactly 2 modes)
        agreement = {}
        if len(modes) == 2:
            m1, m2 = modes[0], modes[1]
            agreement = _compute_agreement(mode_results[m1], mode_results[m2])

        # Print table
        _print_table(modes_data)

        # Write output
        out_path = Path(args.results).with_suffix(".rejudge.json")
        output_doc = {
            "source_file": args.results,
            "judge_model": judge_model,
            "modes": modes_data,
            "agreement": agreement,
        }
        with open(out_path, "w") as f:
            json.dump(output_doc, f, indent=2)
        print(f"\n  Re-judge results -> {out_path}")

    finally:
        await client.aclose()


def main():
    parser = argparse.ArgumentParser(description="Re-judge LongMemEval results with different judge modes")
    parser.add_argument("--results", required=True, help="Path to results JSON file (must have context)")
    parser.add_argument("--judge-model", required=True, help="Ollama model tag for judging")
    parser.add_argument("--ollama-url", default=None, help="Ollama base URL (default: from runner)")
    parser.add_argument(
        "--modes",
        default="reference,evidence",
        help="Comma-separated list of judge modes to run (default: reference,evidence)",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
