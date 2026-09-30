#!/usr/bin/env python3
"""Re-judge LongMemEval result rows with a second judge.

Reads a result JSON written by ``benchmarks/longmemeval_runner.py`` and
re-scores every row that has a non-empty ``answer`` using the runner's own
``score_answer_llm`` and idk-phrase short-circuit (imported, never copied).
Rows with an empty answer score ``False`` without a judge call.

Writes the updated rows with two new keys:
- ``judge_rejudged`` (bool): True when a judge call was made, False when the
  row was scored by the idk-phrase / empty-answer rule.
- ``judge_model`` (str): the model used for rejudging (``--judge-model``).

Usage:
    python benchmarks/longmemeval_rescore.py <in.json> --judge-model M [--out path]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.longmemeval_runner import score_answer_llm  # noqa: E402
import httpx as _httpx  # noqa: E402


_IDK_PHRASES = (
    "i don't know",
    "i do not know",
    "i'm sorry",
    "not in the context",
    "does not contain",
    "no information",
)


def _is_idk(answer: str) -> bool:
    return bool(answer and any(p in answer.lower() for p in _IDK_PHRASES))


async def _judge_row(client: _httpx.AsyncClient, row: dict, judge_model: str) -> dict:
    """Return updated row with judge_rejudged and judge_model set.

    Empty or idk-phrase answers score False without a judge call (matching the
    runner's own idk short-circuit). Non-empty, non-idk answers are sent to
    score_answer_llm (imported, never copied).
    """
    answer = row.get("answer", "")
    if not answer or _is_idk(answer):
        row["correct"] = False
        row["judge_rejudged"] = False
        row["judge_model"] = judge_model
        return row

    gold = row.get("gold", "")
    question = row.get("question", "")
    try:
        correct = await score_answer_llm(client, answer, gold, question, judge_model=judge_model)
    except Exception:
        correct = False
    row["correct"] = bool(correct)
    row["judge_rejudged"] = True
    row["judge_model"] = judge_model
    return row


async def run_rescore(in_path: str, judge_model: str, out_path: str | None) -> None:
    with open(in_path) as f:
        doc = json.load(f)

    results = doc.get("results", [])
    if not results:
        print(f"No results found in {in_path}", file=sys.stderr)
        sys.exit(1)

    client = _httpx.AsyncClient(timeout=30)
    try:
        updated = []
        for row in results:
            updated.append(await _judge_row(client, row, judge_model))

        doc["results"] = updated
        doc["judge_model"] = judge_model

        if not out_path:
            root, ext = os.path.splitext(in_path)
            out_path = f"{root}_rescored{ext}"

        with open(out_path, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"Rescored results written to {out_path}")
    finally:
        await client.aclose()

    n = len(updated)
    n_correct = sum(1 for r in updated if r.get("correct"))
    print(f"\nOverall: {n_correct}/{n} ({n_correct / n * 100:.1f}%)")

    by_type: dict[str, dict[str, int]] = {}
    for r in updated:
        qtype = r.get("question_type", "unknown")
        if qtype not in by_type:
            by_type[qtype] = {"correct": 0, "total": 0}
        by_type[qtype]["total"] += 1
        if r.get("correct"):
            by_type[qtype]["correct"] += 1

    print("\nBy question type:")
    for qtype, data in sorted(by_type.items()):
        pct = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0.0
        print(f"  {qtype:30s} {data['correct']:3d}/{data['total']:<3d} ({pct:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-judge LongMemEval result rows with a second judge."
    )
    parser.add_argument("in_json", help="Path to a result JSON from longmemeval_runner.py")
    parser.add_argument("--judge-model", required=True, help="Ollama model for rejudging")
    parser.add_argument("--out", default=None, help="Output path (default: <in>_rescored.json)")
    args = parser.parse_args()
    asyncio.run(run_rescore(args.in_json, args.judge_model, args.out))


if __name__ == "__main__":
    main()
