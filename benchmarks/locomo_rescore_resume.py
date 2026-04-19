"""Resume rescore: retry only records where judge_rejudged is None.

Uses a longer per-call timeout to handle qwen3.5:9b slow responses. Writes back
to the same .rescored.json in place, then recomputes and prints the scorecard.
"""
import argparse
import asyncio
import json
import statistics
import sys
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


def parse_reply(reply: str) -> float:
    return 1.0 if reply.strip().lower().startswith("yes") else 0.0


async def judge_one(client, url, model, rec, timeout, sem):
    async with sem:
        prompt = JUDGE_PROMPT.format(
            question=rec.get("question", ""),
            reference=rec.get("reference", ""),
            predicted=rec.get("predicted", ""),
        )
        payload = {"model": model, "prompt": prompt, "stream": False}
        try:
            resp = await client.post(f"{url}/api/generate", json=payload, timeout=timeout)
            resp.raise_for_status()
            return parse_reply(resp.json().get("response", ""))
        except Exception as exc:
            print(f"  retry err {type(exc).__name__}: {exc!r}", file=sys.stderr)
            return None


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("rescored_json")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--model", default="qwen3.5:9b")
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--concurrency", type=int, default=3)
    args = p.parse_args()

    path = Path(args.rescored_json)
    data = json.loads(path.read_text())
    results = data["results"]

    todo = [r for r in results if r.get("judge_rejudged") is None]
    print(f"[{path.name}] {len(todo)} items to retry (of {len(results)} total)")
    if not todo:
        print("  nothing to do")
        return

    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as client:
        results_list = await asyncio.gather(*[
            judge_one(client, args.ollama_url, args.model, r, args.timeout, sem)
            for r in todo
        ])
    for rec, new_judge in zip(todo, results_list):
        if new_judge is not None:
            rec["judge_rejudged"] = new_judge

    path.write_text(json.dumps(data, indent=2))
    still_none = sum(1 for r in results if r.get("judge_rejudged") is None)
    print(f"  wrote {path.name}; still-errored after retry: {still_none}/{len(results)}")

    # Recompute and print scorecard
    print()
    print("=" * 80)
    print(f"LoCoMo Rescore (resumed) — {path.name}")
    print("=" * 80)
    header = f"{'Category':<16} {'Count':>6} {'F1':>8} {'Orig':>8} {'Rejudge':>10} {'Delta':>8}"
    print(header)
    print("-" * 80)
    tot_count = 0
    tot_f1 = 0.0
    tot_orig = 0.0
    tot_rej_sum = 0.0
    tot_rej_n = 0
    for c in [1, 2, 3, 4]:
        rs = [r for r in results if r["category"] == c]
        if not rs:
            continue
        f1 = statistics.mean(r["f1"] for r in rs)
        orig = statistics.mean(r["judge"] for r in rs)
        rv = [r["judge_rejudged"] for r in rs if r.get("judge_rejudged") is not None]
        rejudge = statistics.mean(rv) if rv else 0.0
        delta = rejudge - orig
        name = CAT_NAMES[c]
        print(f"{name} ({c})   {len(rs):>6} {f1:>8.3f} {orig:>8.2f} {rejudge:>10.2f} {delta:>+8.2f}  (n={len(rv)})")
        tot_count += len(rs)
        tot_f1 += f1 * len(rs)
        tot_orig += orig * len(rs)
        tot_rej_sum += sum(rv)
        tot_rej_n += len(rv)
    print("-" * 80)
    all_f1 = tot_f1 / tot_count
    all_orig = tot_orig / tot_count
    all_rej = tot_rej_sum / tot_rej_n if tot_rej_n else 0.0
    print(f"{'Overall':<21} {tot_count:>6} {all_f1:>8.3f} {all_orig:>8.2f} {all_rej:>10.2f} {all_rej-all_orig:>+8.2f}  (n={tot_rej_n})")
    print("=" * 80)


asyncio.run(main())
