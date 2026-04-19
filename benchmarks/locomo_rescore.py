#!/usr/bin/env python3
"""
locomo_rescore.py — date-tolerant re-judging of LoCoMo Temporal QAs.

For Category 2 (Temporal), recomputes judge using month/year-only matching so
date-format differences don't count as wrong answers.

Also supports online re-judging via an external Ollama model (--rejudge-model)
so scores from runs with different generator models become apples-to-apples.

Usage:
    python3 benchmarks/locomo_rescore.py path/to/results.json
    python3 benchmarks/locomo_rescore.py path/to/results.json --rejudge-model qwen3.5:9b
    python3 benchmarks/locomo_rescore.py --self-test
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Import JUDGE_PROMPT from runner (same directory).
# Fall back to an inline copy if the import fails (e.g. missing deps).
# ---------------------------------------------------------------------------

def _load_judge_prompt():
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    try:
        from locomo_runner import JUDGE_PROMPT  # noqa: PLC0415
        return JUDGE_PROMPT
    except Exception:
        # Inline fallback — kept in sync with locomo_runner.py
        return (
            "You are grading a predicted answer against a reference answer.\n"
            "Reply with a single token: YES if the predicted answer matches the reference\n"
            "(same facts, minor wording differences are fine), otherwise NO.\n\n"
            "Question: {question}\n"
            "Reference: {reference}\n"
            "Predicted: {predicted}\n\n"
            "Reply YES or NO only."
        )

JUDGE_PROMPT = _load_judge_prompt()

# ---------------------------------------------------------------------------
# Month name lookup
# ---------------------------------------------------------------------------

MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# Regex fragments
_MON = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
_D   = r"\d{1,2}"
_Y   = r"\d{4}"

PATTERNS = [
    # ISO-ish: 2023-05-07
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "iso"),
    # "7 May 2023" or "07 May 2023"
    (re.compile(r"\b(" + _D + r")\s+(" + _MON + r")\s+(" + _Y + r")\b", re.IGNORECASE), "dmy"),
    # "May 7, 2023" or "May 7 2023"
    (re.compile(r"\b(" + _MON + r")\s+(" + _D + r"),?\s+(" + _Y + r")\b", re.IGNORECASE), "mdy"),
    # "May 2023"
    (re.compile(r"\b(" + _MON + r")\s+(" + _Y + r")\b", re.IGNORECASE), "my"),
    # bare year — match last so more-specific patterns win
    (re.compile(r"\b(" + _Y + r")\b"), "y"),
]


def extract_dates(text: str) -> list:
    """
    Return list of (year, month, day) tuples extracted from text.
    month and day may be None if not present in the pattern match.
    Year is always an int; month/day are ints or None.
    """
    if not text:
        return []

    dates = []
    seen_spans = []

    def overlaps(start, end):
        for s, e in seen_spans:
            if start < e and end > s:
                return True
        return False

    for pattern, fmt in PATTERNS:
        for m in pattern.finditer(text):
            if overlaps(m.start(), m.end()):
                continue
            try:
                if fmt == "iso":
                    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
                elif fmt == "dmy":
                    day = int(m.group(1))
                    month = MONTH_NAMES.get(m.group(2).lower())
                    year = int(m.group(3))
                    if month is None:
                        continue
                elif fmt == "mdy":
                    month = MONTH_NAMES.get(m.group(1).lower())
                    day = int(m.group(2))
                    year = int(m.group(3))
                    if month is None:
                        continue
                elif fmt == "my":
                    month = MONTH_NAMES.get(m.group(1).lower())
                    year = int(m.group(2))
                    day = None
                    if month is None:
                        continue
                elif fmt == "y":
                    year = int(m.group(1))
                    # Only accept plausible calendar years
                    if year < 1900 or year > 2100:
                        continue
                    month = None
                    day = None
                else:
                    continue

                dates.append((year, month, day))
                seen_spans.append((m.start(), m.end()))
            except (ValueError, IndexError):
                continue

    return dates


def _ym_match(ref_dates, pred_dates):
    """True if any (year, month) pair matches between ref and pred."""
    for ry, rm, _rd in ref_dates:
        if rm is None:
            continue
        for py, pm, _pd in pred_dates:
            if pm is None:
                continue
            if ry == py and rm == pm:
                return True
    return False


def _ymd_match(ref_dates, pred_dates):
    """True if any (year, month, day) triple matches where all three are defined."""
    for ry, rm, rd in ref_dates:
        if rm is None or rd is None:
            continue
        for py, pm, pd in pred_dates:
            if pm is None or pd is None:
                continue
            if ry == py and rm == pm and rd == pd:
                return True
    return False


def rescore_record(record: dict) -> dict:
    """Return copy of record with judge_tolerant and judge_tolerant_strict added."""
    rec = dict(record)
    category = record.get("category")
    orig_judge = float(record.get("judge", 0.0))

    if category != 2:
        rec["judge_tolerant"] = orig_judge
        rec["judge_tolerant_strict"] = orig_judge
        return rec

    ref_text  = str(record.get("reference", "") or "")
    pred_text = str(record.get("predicted", "") or "")

    ref_dates  = extract_dates(ref_text)
    pred_dates = extract_dates(pred_text)

    # If we can't parse any date from the reference, fall back to original.
    if not ref_dates:
        rec["judge_tolerant"] = orig_judge
        rec["judge_tolerant_strict"] = orig_judge
        return rec

    rec["judge_tolerant"]        = 1.0 if _ym_match(ref_dates, pred_dates) else 0.0
    rec["judge_tolerant_strict"] = 1.0 if _ymd_match(ref_dates, pred_dates) else 0.0
    return rec


# ---------------------------------------------------------------------------
# Online re-judging via Ollama
# ---------------------------------------------------------------------------

def parse_judge_reply(reply: str) -> float:
    """Map YES/NO reply to 1.0/0.0. Anything else → 0.0."""
    stripped = reply.strip().lower()
    if stripped.startswith("yes"):
        return 1.0
    return 0.0


async def _call_ollama_judge(
    client: httpx.AsyncClient,
    ollama_url: str,
    model: str,
    record: dict,
    timeout: float = 240.0,
) -> float | None:
    """Call Ollama with JUDGE_PROMPT for one record. Returns 1.0/0.0 or None on error."""
    prompt = JUDGE_PROMPT.format(
        question=record.get("question", ""),
        reference=record.get("reference", ""),
        predicted=record.get("predicted", ""),
    )
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = await client.post(f"{ollama_url}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        reply = data.get("response", "")
        return parse_judge_reply(reply)
    except Exception as exc:
        # Some exceptions (e.g. httpx.ReadTimeout) have empty str() — use repr + type name.
        print(
            f"  [rejudge] error {type(exc).__name__} for (question="
            f"{record.get('question','')[:40]!r}): {exc!r}",
            file=sys.stderr,
        )
        return None


async def rejudge_all(
    records: list,
    ollama_url: str,
    model: str,
    concurrency: int,
    timeout: float = 240.0,
) -> list:
    """
    Add judge_rejudged to each record. Returns the updated records list.

    judge_rejudged is 1.0 (YES-prefixed reply), 0.0 (anything else), or None
    (HTTP error / parse failure / timeout). parse_judge_reply treats any
    non-YES reply uniformly as 0.0.
    """
    sem = asyncio.Semaphore(concurrency)
    results = list(records)  # shallow copy list

    async def _judge_one(idx: int, rec: dict):
        async with sem:
            val = await _call_ollama_judge(client, ollama_url, model, rec, timeout=timeout)
        results[idx] = dict(rec)
        results[idx]["judge_rejudged"] = val

    limits = httpx.Limits(max_connections=concurrency + 2, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [asyncio.create_task(_judge_one(i, r)) for i, r in enumerate(records)]
        done = 0
        for coro in asyncio.as_completed(tasks):
            await coro
            done += 1
            if done % 25 == 0:
                print(f"  rejudge progress: {done}/{len(records)}")

    print(f"  rejudge progress: {len(records)}/{len(records)}")
    return results


# ---------------------------------------------------------------------------
# Scorecard printing
# ---------------------------------------------------------------------------

CAT_LABELS = {
    1: "Single-hop",
    2: "Temporal  ",
    3: "Multi-hop ",
    4: "Open-dom  ",
    5: "Adversarial",
}


def build_scorecard(results: list, with_rejudge: bool = False) -> dict:
    """Aggregate per-category and overall stats."""
    cats = {}
    for rec in results:
        cat = rec.get("category")
        if cat not in cats:
            cats[cat] = {"count": 0, "orig": 0.0, "tol": 0.0, "f1": 0.0, "rejudge": 0.0, "rejudge_n": 0}
        cats[cat]["count"] += 1
        cats[cat]["orig"] += float(rec.get("judge", 0.0))
        cats[cat]["tol"]  += float(rec.get("judge_tolerant", rec.get("judge", 0.0)))
        cats[cat]["f1"]   += float(rec.get("f1", 0.0))
        if with_rejudge:
            val = rec.get("judge_rejudged")
            if val is not None:
                cats[cat]["rejudge"] += float(val)
                cats[cat]["rejudge_n"] += 1

    overall = {"count": 0, "orig": 0.0, "tol": 0.0, "f1": 0.0, "rejudge": 0.0, "rejudge_n": 0}
    for v in cats.values():
        overall["count"]    += v["count"]
        overall["orig"]     += v["orig"]
        overall["tol"]      += v["tol"]
        overall["f1"]       += v["f1"]
        overall["rejudge"]  += v["rejudge"]
        overall["rejudge_n"] += v["rejudge_n"]

    return {"by_cat": cats, "overall": overall}


def print_scorecard(filename: str, scorecard: dict, rejudge_model: str | None = None):
    w = 80
    print("=" * w)
    header = f"LoCoMo Rescore — {filename}"
    if rejudge_model:
        header += f"  (rejudge model: {rejudge_model})"
    print(header)
    print("=" * w)

    if rejudge_model:
        print(f"{'Category':<20} {'Count':>6}  {'F1':>6}  {'Orig-Judge':>10}  {'Tol-Judge':>9}  {'Rejudge':>7}  {'Delta-Rejudge':>13}")
    else:
        print(f"{'Category':<20} {'Count':>6}  {'F1':>6}  {'Orig-Judge':>10}  {'Tol-Judge':>9}  {'Delta-Tol':>9}")
    print("-" * w)

    by_cat = scorecard["by_cat"]
    for cat_id in sorted(by_cat.keys()):
        v = by_cat[cat_id]
        n = v["count"]
        if n == 0:
            continue
        label = CAT_LABELS.get(cat_id, f"Cat-{cat_id}   ")
        orig  = v["orig"] / n
        tol   = v["tol"]  / n
        f1    = v["f1"]   / n
        delta_tol = tol - orig

        if rejudge_model:
            rn = v["rejudge_n"]
            rj = (v["rejudge"] / rn) if rn > 0 else float("nan")
            delta_rj = rj - orig
            sign_rj  = "+" if delta_rj >= 0 else ""
            print(
                f"{label} ({cat_id})  {n:>6}  {f1:.3f}        {orig:.2f}       {tol:.2f}     {rj:.2f}      {sign_rj}{delta_rj:.2f}"
            )
        else:
            sign = "+" if delta_tol >= 0 else ""
            print(f"{label} ({cat_id})  {n:>6}  {f1:.3f}        {orig:.2f}       {tol:.2f}   {sign}{delta_tol:.2f}")

    print("-" * w)
    ov = scorecard["overall"]
    n  = ov["count"]
    if n:
        orig  = ov["orig"] / n
        tol   = ov["tol"]  / n
        f1    = ov["f1"]   / n
        delta_tol = tol - orig

        if rejudge_model:
            rn = ov["rejudge_n"]
            rj = (ov["rejudge"] / rn) if rn > 0 else float("nan")
            delta_rj = rj - orig
            sign_rj  = "+" if delta_rj >= 0 else ""
            print(
                f"{'Overall':<20} {n:>6}  {f1:.3f}        {orig:.2f}       {tol:.2f}     {rj:.2f}      {sign_rj}{delta_rj:.2f}"
            )
        else:
            sign = "+" if delta_tol >= 0 else ""
            print(f"{'Overall':<20} {n:>6}  {f1:.3f}        {orig:.2f}       {tol:.2f}   {sign}{delta_tol:.2f}")

    print("=" * w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(path: Path, rejudge_model: str | None, ollama_url: str, concurrency: int, timeout: float = 240.0):
    with path.open() as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print(f"No results found in {path}", file=sys.stderr)
        sys.exit(1)

    rescored = [rescore_record(r) for r in results]

    if rejudge_model:
        print(f"Re-judging {len(rescored)} items with model '{rejudge_model}' @ {ollama_url} ...")
        rescored = asyncio.run(
            rejudge_all(rescored, ollama_url, rejudge_model, concurrency, timeout=timeout)
        )

    scorecard = build_scorecard(rescored, with_rejudge=bool(rejudge_model))
    print_scorecard(path.name, scorecard, rejudge_model=rejudge_model)

    out_path = path.parent / (path.stem + ".rescored.json")
    sidecar = dict(data)
    sidecar["results"] = rescored
    with out_path.open("w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"\nSidecar written: {out_path}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    failures = 0

    # --- offline date-tolerant tests ---
    cases = [
        # (ref, pred, expected_tolerant, expected_strict, description)
        ("7 May 2023",  "May 8, 2023",  1.0, 0.0, "same month/year, diff day → tol=1 strict=0"),
        ("June 2023",   "this month",   0.0, 0.0, "no date in pred → tol=0"),
        ("June 2023",   "June 2023",    1.0, 0.0, "exact month/year, no day → tol=1 strict=0"),
        ("2023-05-07",  "May 7, 2023",  1.0, 1.0, "ISO vs MDY same date → tol=1 strict=1"),
        ("7 May 2023",  "7 May 2023",   1.0, 1.0, "exact match → tol=1 strict=1"),
        ("June 2023",   "July 2023",    0.0, 0.0, "different month → tol=0"),
        ("no date here","also none",    None, None, "no ref dates → falls back to orig judge"),
    ]

    print("Self-test (offline date-tolerant):")
    print("-" * 60)
    for ref, pred, exp_tol, exp_strict, desc in cases:
        orig_judge = 0.5  # sentinel to detect fallback
        record = {
            "category": 2,
            "reference": ref,
            "predicted": pred,
            "judge": orig_judge,
        }
        out = rescore_record(record)
        tol    = out["judge_tolerant"]
        strict = out["judge_tolerant_strict"]

        if exp_tol is None:
            ok = (tol == orig_judge and strict == orig_judge)
        else:
            ok = (tol == exp_tol and strict == exp_strict)

        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"  [{status}] {desc}")
        if not ok:
            print(f"         expected tol={exp_tol} strict={exp_strict}, got tol={tol} strict={strict}")

    print("-" * 60)
    # non-category-2 passthrough
    rec = {"category": 1, "reference": "May 2023", "predicted": "May 2023", "judge": 0.75}
    out = rescore_record(rec)
    ok = out["judge_tolerant"] == 0.75
    print(f"  [{'PASS' if ok else 'FAIL'}] cat=1 passes through unchanged (judge_tolerant == orig)")
    if not ok:
        failures += 1

    # --- rejudge parser tests ---
    print()
    print("Self-test (rejudge YES/NO parser):")
    print("-" * 60)
    parser_cases = [
        ("YES",       1.0, "uppercase YES → 1.0"),
        ("yes",       1.0, "lowercase yes → 1.0"),
        ("Yes",       1.0, "mixed case Yes → 1.0"),
        ("YES\n",     1.0, "YES with newline → 1.0"),
        ("YES extra", 1.0, "YES with trailing text → 1.0"),
        ("NO",        0.0, "uppercase NO → 0.0"),
        ("no",        0.0, "lowercase no → 0.0"),
        ("maybe",     0.0, "unparseable → 0.0"),
        ("",          0.0, "empty → 0.0"),
    ]
    for reply, expected, desc in parser_cases:
        got = parse_judge_reply(reply)
        ok = got == expected
        if not ok:
            failures += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
        if not ok:
            print(f"         expected {expected}, got {got}")

    # --- scorecard assembly with rejudge column ---
    print()
    print("Self-test (scorecard assembly with rejudge column):")
    print("-" * 60)
    fake_results = [
        {"category": 1, "judge": 1.0, "judge_tolerant": 1.0, "judge_rejudged": 1.0, "f1": 0.8},
        {"category": 1, "judge": 0.0, "judge_tolerant": 0.0, "judge_rejudged": 0.0, "f1": 0.2},
        {"category": 2, "judge": 0.0, "judge_tolerant": 1.0, "judge_rejudged": 0.0, "f1": 0.4},
        {"category": 2, "judge": 1.0, "judge_tolerant": 1.0, "judge_rejudged": 1.0, "f1": 0.9},
    ]
    sc = build_scorecard(fake_results, with_rejudge=True)
    # cat 1: orig=0.5, tol=0.5, rejudge=0.5
    c1 = sc["by_cat"][1]
    ok1 = c1["orig"] / c1["count"] == 0.5 and c1["rejudge"] / c1["rejudge_n"] == 0.5
    print(f"  [{'PASS' if ok1 else 'FAIL'}] cat=1 scorecard orig/rejudge both 0.5")
    if not ok1:
        failures += 1
    # cat 2: orig=0.5, tol=1.0, rejudge=0.5
    c2 = sc["by_cat"][2]
    ok2 = c2["tol"] / c2["count"] == 1.0 and c2["rejudge"] / c2["rejudge_n"] == 0.5
    print(f"  [{'PASS' if ok2 else 'FAIL'}] cat=2 scorecard tol=1.0 rejudge=0.5")
    if not ok2:
        failures += 1

    # print a sample scorecard to visually verify formatting and check F1 column present
    import io, contextlib
    print()
    print("Sample scorecard output (with rejudge column):")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_scorecard("test_results.json", sc, rejudge_model="qwen3.5:9b")
    scorecard_text = buf.getvalue()
    print(scorecard_text, end="")
    ok_f1 = "F1" in scorecard_text
    print(f"  [{'PASS' if ok_f1 else 'FAIL'}] F1 column present in scorecard output")
    if not ok_f1:
        failures += 1

    print()
    print("Sample scorecard output (offline / no rejudge):")
    sc_offline = build_scorecard(fake_results, with_rejudge=False)
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        print_scorecard("test_results.json", sc_offline)
    scorecard_text2 = buf2.getvalue()
    print(scorecard_text2, end="")
    ok_f1_offline = "F1" in scorecard_text2
    print(f"  [{'PASS' if ok_f1_offline else 'FAIL'}] F1 column present in offline scorecard output")
    if not ok_f1_offline:
        failures += 1

    print()
    print("-" * 60)
    if failures == 0:
        print("All tests passed.")
    else:
        print(f"{failures} test(s) FAILED.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _default_ollama = os.environ.get("TAOSMD_OLLAMA_URL", "http://localhost:11434")

    parser = argparse.ArgumentParser(description="Re-judge LoCoMo results with date-tolerant scoring.")
    parser.add_argument("results_json", nargs="?", help="Path to results JSON file")
    parser.add_argument("--self-test", action="store_true", help="Run built-in self-tests and exit")
    parser.add_argument(
        "--rejudge-model",
        metavar="MODEL",
        default=None,
        help="Ollama model to use for external re-judging (e.g. qwen3.5:9b)",
    )
    parser.add_argument(
        "--ollama-url",
        metavar="URL",
        default=_default_ollama,
        help=f"Ollama base URL (default: {_default_ollama}, env: TAOSMD_OLLAMA_URL)",
    )
    parser.add_argument(
        "--concurrency",
        metavar="N",
        type=int,
        default=4,
        help="Number of concurrent Ollama requests for rejudging (default: 4)",
    )
    parser.add_argument(
        "--timeout",
        metavar="SECONDS",
        type=float,
        default=240.0,
        help="Per-call HTTP timeout for rejudge Ollama calls (default: 240s). "
             "qwen3.5:9b can occasionally take >60s on a single call under "
             "Ollama NUM_PARALLEL>1 contention.",
    )
    args = parser.parse_args()

    if args.self_test:
        self_test()
    elif args.results_json:
        run(
            Path(args.results_json),
            rejudge_model=args.rejudge_model,
            ollama_url=args.ollama_url,
            concurrency=args.concurrency,
            timeout=args.timeout,
        )
    else:
        parser.print_help()
        sys.exit(1)
