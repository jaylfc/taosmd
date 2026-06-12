#!/usr/bin/env python3
"""E-005 Stage 1: applicability scan for the temporal date-range lever.

Counts how many LoCoMo questions contain a date-range expression the
lever can parse (taosmd/temporal.py). The pre-registered gate in
docs/research-report.md keys on the temporal category (2): under 10
percent applicability means LoCoMo cannot measure the lever and no
LoCoMo claim is made in either direction.

Usage: python3 benchmarks/temporal_applicability_scan.py [path/to/locomo10.json]
"""

from __future__ import annotations

import json
import os
import sys

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_BENCH_DIR)
sys.path.insert(0, _REPO_ROOT)

from taosmd.temporal import (  # noqa: E402
    extract_temporal_expression,
    parse_temporal_expression,
)

_DEFAULT_DATASET = os.path.join(
    _REPO_ROOT, "data", "locomo", "data", "locomo10.json"
)

INCLUDE_CATS = {1, 2, 3, 4}


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_DATASET
    with open(path) as f:
        data = json.load(f)

    total = 0
    by_cat: dict[int, int] = {}
    hits_by_cat: dict[int, int] = {}
    for conv in data:
        for qa in conv.get("qa", []):
            cat = qa.get("category")
            if cat not in INCLUDE_CATS:
                continue
            total += 1
            by_cat[cat] = by_cat.get(cat, 0) + 1
            expr = extract_temporal_expression(qa.get("question", ""))
            if expr and parse_temporal_expression(expr):
                hits_by_cat[cat] = hits_by_cat.get(cat, 0) + 1

    print(f"total questions (cats 1-4): {total}")
    for cat in sorted(by_cat):
        h = hits_by_cat.get(cat, 0)
        pct = 100 * h / by_cat[cat]
        print(f"  cat {cat}: {h}/{by_cat[cat]} applicable ({pct:.1f}%)")
    allh = sum(hits_by_cat.values())
    print(f"  overall: {allh}/{total} ({100 * allh / total:.1f}%)")
    temporal_pct = 100 * hits_by_cat.get(2, 0) / max(1, by_cat.get(2, 0))
    verdict = "MEASURABLE" if temporal_pct >= 10.0 else "NOT MEASURABLE on LoCoMo"
    print(f"E-005 STAGE 1 VERDICT: temporal-category applicability "
          f"{temporal_pct:.1f}%; gate 10%; {verdict}")


if __name__ == "__main__":
    main()
