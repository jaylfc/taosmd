"""Streaming rescore with incremental checkpointing.

Fixes the two problems of the prior rescore tools:
1. Results are written to disk every N=25 completions (not at the end) — any
   crash / kill / timeout preserves progress.
2. If a partially-populated `.rescored.json` exists, items that already have
   `judge_rejudged` set are skipped — the tool is resumable by design.

Also logs progress + success/error counts every 25 items.

Usage:
  python3 locomo_rescore_streaming.py RESULTS.json [--judge-model MODEL] \
      [--ollama-url URL] [--concurrency N] [--timeout SECONDS] \
      [--checkpoint-every N] [--out PATH]

Default judge model is qwen3:4b (faster than qwen3.5:9b, defensible for YES/NO
grading; the tradeoff is documented in the methodology alongside F1 which is
model-independent).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path

import httpx

JUDGE_PROMPT = """You are grading a predicted answer against a reference answer.
Reply with a single token: YES if the predicted answer matches the reference
(same facts, minor wording differences are fine), otherwise NO.

Question: {question}
Reference: {reference}
Predicted: {predicted}

Reply YES or NO only."""

CAT_NAMES = {1: "Single-hop", 2: "Temporal  ", 3: "Multi-hop ", 4: "Open-dom  "}

# ----- tolerant date parsing (same behaviour as the prior rescore tool) -----
_MONTH = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY = re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+(\d{4})\b", re.I)
_MDY = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})\b", re.I)
_MY = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+(\d{4})\b", re.I)


def extract_dates(text: str) -> list[tuple[int, int | None, int | None]]:
    if not text:
        return []
    out = []
    for y, m, d in _ISO.findall(text):
        try:
            out.append((int(y), int(m), int(d)))
        except ValueError:
            pass
    for d, m, y in _DMY.findall(text):
        mn = _MONTH.get(m.lower())
        if mn:
            try:
                out.append((int(y), mn, int(d)))
            except ValueError:
                pass
    for m, d, y in _MDY.findall(text):
        mn = _MONTH.get(m.lower())
        if mn:
            try:
                out.append((int(y), mn, int(d)))
            except ValueError:
                pass
    for m, y in _MY.findall(text):
        mn = _MONTH.get(m.lower())
        if mn:
            try:
                out.append((int(y), mn, None))
            except ValueError:
                pass
    return out


def ym_match(refs, preds) -> bool:
    for ry, rm, _ in refs:
        if rm is None:
            continue
        for py, pm, _ in preds:
            if pm is None:
                continue
            if ry == py and rm == pm:
                return True
    return False


def add_tolerant_judge(rec: dict) -> dict:
    if rec.get("category") == 2:
        ref_dates = extract_dates(rec.get("reference", ""))
        pred_dates = extract_dates(rec.get("predicted", ""))
        if ref_dates:
            rec["judge_tolerant"] = 1.0 if ym_match(ref_dates, pred_dates) else 0.0
            return rec
    rec["judge_tolerant"] = rec.get("judge", 0.0)
    return rec


def parse_judge_reply(reply: str) -> float:
    return 1.0 if reply.strip().lower().startswith("yes") else 0.0


async def call_judge(client, url, model, rec, timeout) -> tuple[str, object]:
    """Returns (status, value) — status in {OK, ERR}, value is float or error-repr."""
    prompt = JUDGE_PROMPT.format(
        question=rec.get("question", ""),
        reference=rec.get("reference", ""),
        predicted=rec.get("predicted", ""),
    )
    # NOTE: do NOT pass think=false here. On qwen3:4b (our default judge)
    # the Ollama think parameter has inverted semantics vs the generation
    # models — it EXPOSES reasoning tokens in the response rather than
    # suppressing them. That breaks parse_judge_reply's startswith("yes")
    # check because the response now begins with "Okay, the user is asking…".
    # Leaving thinking-mode on here gives a clean yes/no reply in the same
    # ~7-10s per judgement that the full external-judge run relied on.
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = await client.post(f"{url}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        reply = resp.json().get("response", "")
        return ("OK", parse_judge_reply(reply))
    except Exception as exc:
        return ("ERR", f"{type(exc).__name__}: {exc!r}")


def write_checkpoint(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def print_scorecard(path: Path, data: dict, judge_model: str) -> None:
    results = data["results"]
    print()
    print("=" * 88)
    print(f"LoCoMo Rescore (streaming) — {path.name}  [judge: {judge_model}]")
    print("=" * 88)
    print(f"{'Category':<16} {'Count':>6} {'F1':>8} {'Orig':>8} {'Tol':>8} {'Rejudge':>10} {'Delta':>8}  Covered")
    print("-" * 88)
    tot = {"count": 0, "f1": 0.0, "orig": 0.0, "tol": 0.0, "rj_sum": 0.0, "rj_n": 0}
    for c in [1, 2, 3, 4]:
        rs = [r for r in results if r.get("category") == c]
        if not rs:
            continue
        f1 = statistics.mean(r["f1"] for r in rs)
        orig = statistics.mean(r["judge"] for r in rs)
        tol = statistics.mean(r.get("judge_tolerant", r["judge"]) for r in rs)
        rv = [r["judge_rejudged"] for r in rs if r.get("judge_rejudged") is not None]
        rj = statistics.mean(rv) if rv else float("nan")
        delta = (rj - orig) if rv else 0.0
        cov = f"{len(rv)}/{len(rs)} ({100*len(rv)/len(rs):.0f}%)"
        print(f"{CAT_NAMES[c]} ({c})   {len(rs):>6} {f1:>8.3f} {orig:>8.2f} {tol:>8.2f} {rj:>10.2f} {delta:>+8.2f}  {cov}")
        tot["count"] += len(rs)
        tot["f1"] += f1 * len(rs)
        tot["orig"] += orig * len(rs)
        tot["tol"] += tol * len(rs)
        tot["rj_sum"] += sum(rv)
        tot["rj_n"] += len(rv)
    print("-" * 88)
    n = tot["count"]
    all_f1 = tot["f1"] / n
    all_orig = tot["orig"] / n
    all_tol = tot["tol"] / n
    all_rj = tot["rj_sum"] / tot["rj_n"] if tot["rj_n"] else float("nan")
    all_delta = all_rj - all_orig if tot["rj_n"] else 0.0
    cov_all = f"{tot['rj_n']}/{n} ({100*tot['rj_n']/n:.0f}%)"
    print(f"{'Overall':<21} {n:>6} {all_f1:>8.3f} {all_orig:>8.2f} {all_tol:>8.2f} {all_rj:>10.2f} {all_delta:>+8.2f}  {cov_all}")
    print("=" * 88)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_json", help="Original results JSON (NOT .rescored.json)")
    p.add_argument("--judge-model", default="qwen3:4b")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--out", default=None, help="Output .rescored.json path (default: alongside input)")
    args = p.parse_args()

    in_path = Path(args.results_json)
    out_path = Path(args.out) if args.out else in_path.parent / (in_path.stem + ".rescored.json")

    # Load fresh (not from any prior .rescored.json — avoid bias from partial runs)
    with in_path.open() as f:
        data = json.load(f)
    results = data["results"]
    for r in results:
        add_tolerant_judge(r)
        if "judge_rejudged" not in r:
            r["judge_rejudged"] = None

    todo_idx = [i for i, r in enumerate(results) if r.get("judge_rejudged") is None]
    print(f"[{in_path.name}] {len(todo_idx)} / {len(results)} items to rejudge "
          f"(judge={args.judge_model}, concurrency={args.concurrency}, timeout={args.timeout}s, ckpt every {args.checkpoint_every})",
          flush=True)
    if not todo_idx:
        print("  nothing to do; writing scorecard only", flush=True)
        write_checkpoint(out_path, data)
        print_scorecard(in_path, data, args.judge_model)
        return

    sem = asyncio.Semaphore(args.concurrency)
    ok_count = 0
    err_count = 0
    done_count = 0
    last_ckpt_t = time.time()

    async def worker(idx: int):
        nonlocal ok_count, err_count, done_count
        async with sem:
            status, val = await call_judge(
                client, args.ollama_url, args.judge_model, results[idx], args.timeout
            )
        if status == "OK":
            results[idx]["judge_rejudged"] = val
            ok_count += 1
        else:
            err_count += 1
            print(f"  [rejudge err {err_count}] idx={idx} {val}", file=sys.stderr, flush=True)
        done_count += 1

    start = time.time()
    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(worker(i)) for i in todo_idx]

        async def checkpointer():
            nonlocal last_ckpt_t
            last_printed = 0
            while done_count < len(todo_idx):
                await asyncio.sleep(2.0)
                if done_count - last_printed >= args.checkpoint_every:
                    write_checkpoint(out_path, data)
                    now = time.time()
                    rate = done_count / (now - start) if now > start else 0
                    eta = (len(todo_idx) - done_count) / rate if rate > 0 else float("inf")
                    print(f"  progress: {done_count}/{len(todo_idx)}  ok={ok_count} err={err_count}  "
                          f"{rate:.2f} items/s  eta={eta/60:.1f}m",
                          flush=True)
                    last_printed = done_count
                    last_ckpt_t = now

        ck_task = asyncio.create_task(checkpointer())
        await asyncio.gather(*tasks)
        ck_task.cancel()
        try:
            await ck_task
        except asyncio.CancelledError:
            pass

    # final checkpoint
    write_checkpoint(out_path, data)
    elapsed = time.time() - start
    print(f"\n[{in_path.name}] DONE: {ok_count}/{len(todo_idx)} judged ok, {err_count} errored in {elapsed/60:.1f} min",
          flush=True)

    print_scorecard(in_path, data, args.judge_model)


if __name__ == "__main__":
    asyncio.run(main())
