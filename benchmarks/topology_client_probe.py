#!/usr/bin/env python3
"""Topology client probe for taOSmd Block T.

Drives a running taOSmd HTTP service from another machine and records
end-to-end latency, ingest throughput, and system-wide CPU busy fraction.

The `system_cpu_busy_fraction` field reports the system-wide CPU busy fraction
over the ingest+query window (user+nice+system+irq+softirq+steal over total),
NOT the probe process's own CPU usage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_BENCH_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _BENCH_DIR)

from benchmarks.bench_provenance import collect
from locomo_runner import _DEFAULT_DATASET, _session_keys  # noqa: E402

# Module-level clock for testability. Tests monkeypatch this with a fake
# that returns a scripted sequence.
_now = time.perf_counter


def _pctl(values: list[float], q: float) -> float:
    """Nearest-rank percentile. q in [0, 1]."""
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


def _parse_proc_stat(text: str) -> tuple[int, int]:
    """Parse /proc/stat first line. Returns (busy_ticks, total_ticks)."""
    for line in text.splitlines():
        if line.startswith("cpu "):
            parts = line.split()
            # user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice
            vals = list(map(int, parts[1:]))
            # busy = user + nice + system + irq + softirq + steal
            busy = vals[0] + vals[1] + vals[2] + vals[5] + vals[6] + (vals[7] if len(vals) > 7 else 0)
            total = sum(vals)
            return busy, total
    return 0, 0


def _read_cpu_snapshot() -> tuple[int, int] | None:
    """Read /proc/stat and return (busy, total) or None if unreadable."""
    try:
        with open("/proc/stat", "r") as f:
            return _parse_proc_stat(f.read())
    except Exception:
        return None


def _cpu_busy_fraction(before: tuple[int, int] | None, after: tuple[int, int] | None) -> float | None:
    if before is None or after is None:
        return None
    busy_before, total_before = before
    busy_after, total_after = after
    busy_delta = busy_after - busy_before
    total_delta = total_after - total_before
    if total_delta <= 0:
        return 0.0
    return busy_delta / total_delta


def _run_ping(host: str, count: int) -> dict | None:
    """Run ping -c <count> <host>. Return dict with p50_ms, p95_ms, raw_ms, or None on failure."""
    try:
        result = subprocess.run(
            ["ping", "-c", str(count), host],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "p50_ms": None, "p95_ms": None, "raw_ms": None}

    if result.returncode != 0:
        return {"error": result.stderr.strip() or "ping failed", "p50_ms": None, "p95_ms": None, "raw_ms": None}

    # Parse lines like: "64 bytes from 192.168.1.1: icmp_seq=1 ttl=64 time=1.23 ms"
    times = []
    for line in result.stdout.splitlines():
        if "time=" in line:
            try:
                t_str = line.split("time=")[1].split()[0]
                times.append(float(t_str))
            except (IndexError, ValueError):
                continue

    if not times:
        return {"error": "no ping times parsed", "p50_ms": None, "p95_ms": None, "raw_ms": None}

    return {
        "p50_ms": round(_pctl(times, 0.50), 2),
        "p95_ms": round(_pctl(times, 0.95), 2),
        "raw_ms": [round(t, 2) for t in times],
    }


async def _get_version(client: httpx.AsyncClient, base_url: str) -> dict:
    resp = await client.get(f"{base_url}/version")
    if resp.status_code != 200:
        raise RuntimeError(f"/version returned {resp.status_code}: {resp.text}")
    return resp.json()


async def _ingest_batch(
    client: httpx.AsyncClient,
    base_url: str,
    items: list[dict],
    agent: str,
    token: str | None,
) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = await client.post(
        f"{base_url}/ingest/batch",
        json={"items": items, "agent": agent},
        headers=headers,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"/ingest/batch returned {resp.status_code}: {resp.text}")
    return resp.json()


async def _search(
    client: httpx.AsyncClient,
    base_url: str,
    query: str,
    agent: str,
    limit: int,
    token: str | None,
) -> tuple[list[dict], int]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = await client.post(
        f"{base_url}/search",
        json={"query": query, "agent": agent, "limit": limit},
        headers=headers,
    )
    # INTENTIONAL BUG FOR RED-FIRST: read "results" instead of "hits"
    data = resp.json()
    hits = data.get("results", [])
    return hits, resp.status_code


async def run(args: argparse.Namespace) -> int:
    started_at = datetime.now(timezone.utc).isoformat()

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        return 2

    with dataset_path.open() as f:
        conversations = json.load(f)

    if args.conversation_index >= len(conversations):
        print(f"ERROR: conversation index {args.conversation_index} out of range (0-{len(conversations)-1})", file=sys.stderr)
        return 2

    conv = conversations[args.conversation_index]

    # Read token from file if provided
    token = None
    if args.token_file:
        token_path = Path(args.token_file).expanduser()
        if not token_path.exists():
            print(f"ERROR: token file not found: {token_path}", file=sys.stderr)
            return 2
        token = token_path.read_text().strip()

    parsed = urlparse(args.base_url)
    host = parsed.hostname or "localhost"

    # a. GET /version
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            version = await _get_version(client, args.base_url)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        # b. Ping
        ping_result = _run_ping(host, args.pings)

        # c. Ingest
        cpu_before = _read_cpu_snapshot()
        ingest_start = _now()

        session_keys_list = _session_keys(conv.get("conversation", conv))
        all_turns = []
        for session_key, _dt in session_keys_list:
            for turn in conv.get(session_key) or []:
                text = (turn.get("text") or "").strip()
                if text:
                    all_turns.append(text)

        turns_sent = 0
        vector_failures_total = 0
        batch_size = 50

        for i in range(0, len(all_turns), batch_size):
            batch = all_turns[i:i + batch_size]
            items = [{"text": t} for t in batch]
            try:
                result = await _ingest_batch(client, args.base_url, items, args.agent, token)
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
            turns_sent += len(batch)
            vector_failures_total += result.get("vector_failures", 0)

        ingest_end = _now()
        ingest_seconds = ingest_end - ingest_start
        turns_per_sec = turns_sent / ingest_seconds if ingest_seconds > 0 else 0.0

        # d. Queries
        qa_list = conv.get("qa", []) or []
        eligible_qa = [qa for qa in qa_list if int(qa.get("category", 0)) != 5 and "answer" in qa]
        selected_qa = eligible_qa[:args.queries]

        query_results = []
        query_times = []
        error_count = 0

        for qa in selected_qa:
            question = qa["question"]
            t0 = _now()
            try:
                hits, status = await _search(client, args.base_url, question, args.agent, args.limit, token)
            except Exception as exc:
                query_results.append({
                    "question": question,
                    "status": 0,
                    "hit_count": 0,
                    "latency_ms": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                error_count += 1
                continue

            latency_ms = (_now() - t0) * 1000.0
            hit_count = len(hits)
            query_times.append(latency_ms)

            query_results.append({
                "question": question,
                "status": status,
                "hit_count": hit_count,
                "latency_ms": round(latency_ms, 2),
            })

            if status != 200:
                error_count += 1

        cpu_after = _read_cpu_snapshot()

    finished_at = datetime.now(timezone.utc).isoformat()

    # Query summary stats
    query_summary = _latency_stats(query_times)
    query_summary["error_count"] = error_count
    query_summary["total_queries"] = len(query_results)

    # Error rate check
    if query_results and error_count / len(query_results) > 0.10:
        exit_code = 3
    else:
        exit_code = 0

    # Ingest summary
    ingest_summary = {
        "turns_sent": turns_sent,
        "total_seconds": round(ingest_seconds, 2),
        "turns_per_sec": round(turns_per_sec, 2),
        "vector_failures": vector_failures_total,
    }

    # CPU fraction
    cpu_fraction = _cpu_busy_fraction(cpu_before, cpu_after)

    # Provenance
    config = {
        "base_url": args.base_url,
        "agent": args.agent,
        "dataset": str(dataset_path),
        "conversation_index": args.conversation_index,
        "queries": args.queries,
        "limit": args.limit,
        "pings": args.pings,
        "label": args.label,
        "shape": args.shape,
    }
    provenance = collect(config, None, str(dataset_path))

    output = {
        "label": args.label,
        "shape": args.shape,
        "base_url": args.base_url,
        "version": version,
        "ping": ping_result,
        "ingest": ingest_summary,
        "queries": {
            "summary": query_summary,
            "per_query": query_results,
        },
        "system_cpu_busy_fraction": cpu_fraction,
        "provenance": provenance,
        "started_at": started_at,
        "finished_at": finished_at,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    if exit_code == 3:
        print(f"WARNING: error rate {error_count}/{len(query_results)} > 10%", file=sys.stderr)

    return exit_code


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Topology client probe for taOSmd Block T. "
                    "Drives a remote taOSmd HTTP service and records latency/ingest/CPU."
    )
    p.add_argument("--base-url", required=True, help="Base URL of taOSmd HTTP server (e.g. http://host:7700)")
    p.add_argument("--token-file", help="Path to file containing bearer token (sent as Authorization header)")
    p.add_argument("--agent", default="e049-topology", help="Agent name for ingest/search (default: e049-topology)")
    p.add_argument("--dataset", default=_DEFAULT_DATASET, help=f"LoCoMo dataset JSON (default: {_DEFAULT_DATASET})")
    p.add_argument("--conversation-index", type=int, default=0, help="Conversation index to use (default: 0)")
    p.add_argument("--queries", type=int, default=50, help="Number of queries to run (default: 50)")
    p.add_argument("--limit", type=int, default=10, help="Search limit per query (default: 10)")
    p.add_argument("--pings", type=int, default=20, help="Number of ping packets (default: 20)")
    p.add_argument("--label", required=True, help="Free-text row name (e.g. T-b)")
    p.add_argument("--shape", required=True, choices=["whole-service", "embedder-only"], help="Probe shape")
    p.add_argument("--out", required=True, help="Output JSON path")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(_parse_args())))