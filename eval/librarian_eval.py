# Copyright (c) 2023-present, Jaylfc and contributors.
# Licensed under the MIT License.

"""Librarian evaluation script."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


from eval.teamops.build_eval_set import make_items, split, parse_rulings, parse_ledger, parse_prs


@dataclass
class Query:
    query: str
    expected_answer: str
    context: str


@dataclass
class Result:
    query: str
    retrieved: list[str]
    latency: float
    correct: bool


def librarian_eval(
    rulings_path: str | Path,
    ledger_path: str | Path,
    prs_path: str | Path,
    session_map_path: str | Path | None,
    queries: list[Query],
    holdout: float = 0.30,
) -> list[Result]:
    """Run the librarian evaluation.

    Args:
        rulings_path: Path to rulings markdown file.
        ledger_path: Path to ledger TSV file.
        prs_path: Path to PRs JSON file.
        session_map_path: Path to session map JSONL file.
        queries: List of queries to evaluate.
        holdout: Holdout ratio.

    Returns:
        List of results.
    """
    # Load data
    rulings_text = Path(rulings_path).read_text(encoding="utf-8")
    ledger_text = Path(ledger_path).read_text(encoding="utf-8")
    prs_text = Path(prs_path).read_text(encoding="utf-8")
    session_map = []
    if session_map_path and Path(session_map_path).exists():
        session_map_text = Path(session_map_path).read_text(encoding="utf-8")
        for line in session_map_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                session_map.append(json.loads(line))
            except Exception:
                pass

    # Build eval set
    rulings = make_items(parse_rulings(rulings_text), [], [], [])
    ledger = make_items([], parse_ledger(ledger_text), [], [])
    prs = make_items([], [], parse_prs(prs_text), [])
    items = rulings + ledger + prs

    # Split into train and holdout
    train, holdout_items, manifest = split(items, holdout=holdout)

    # For simplicity, we evaluate on the holdout set.
    # In a real implementation, we would use the train set to build an index and evaluate on the holdout set.
    # Here, we just do a simple string match for demonstration.
    results = []
    for q in queries:
        t0 = time.monotonic()
        retrieved = []
        # Simple string match: check if the query text appears in the text of any holdout item.
        for item in holdout_items:
            if q.query.lower() in item["gold_answer"].lower():
                retrieved.append(item["gold_answer"])
        latency = (time.monotonic() - t0) * 1000
        correct = any(q.expected_answer.lower() in r.lower() for r in retrieved)
        results.append(
            Result(
                query=q.query,
                retrieved=retrieved,
                latency=latency,
                correct=correct,
            )
        )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run librarian evaluation")
    parser.add_argument("--rulings", required=True, help="Path to rulings markdown file")
    parser.add_argument("--ledger", required=True, help="Path to ledger TSV file")
    parser.add_argument("--prs", required=True, help="Path to PRs JSON file")
    parser.add_argument(
        "--session-map", default=None, help="Path to session map JSONL file (optional)"
    )
    parser.add_argument(
        "--queries",
        required=True,
        help="Path to queries JSONL file, each line a JSON object with query and expected_answer",
    )
    parser.add_argument(
        "--holdout", type=float, default=0.30, help="Holdout ratio (default: 0.30)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write results JSONL file",
    )
    args = parser.parse_args()

    # Load queries
    queries = []
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                query_obj = json.loads(line)
                queries.append(
                    Query(
                        query=query_obj["query"],
                        expected_answer=query_obj["expected_answer"],
                        context="",  # context is not used in this simple implementation
                    )
                )
            except Exception:
                pass

    # Run evaluation
    results = librarian_eval(
        rulings_path=args.rulings,
        ledger_path=args.ledger,
        prs_path=args.prs,
        session_map_path=args.session_map,
        queries=queries,
        holdout=args.holdout,
    )

    # Write results
    with open(args.output, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(asdict(result)) + "\n")
