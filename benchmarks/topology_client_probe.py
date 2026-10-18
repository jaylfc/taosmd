#!/usr/bin/env python3
"""Topology client probe for taOSmd (E-049 Block T).

Drives a running taOSmd HTTP service from another machine and records
end-to-end latency, ingest rate, ping, and client CPU. Client side only;
no server changes required.

Mirrors retrieval_latency_probe.py for style, dataset loading, and failure posture.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_BENCH_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _BENCH_DIR)

from benchmarks.bench_provenance import collect

from locomo_runner import (  # noqa: E402
    _DEFAULT_DATASET,
    _session_keys,
)


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


def _parse_ping_output(output: str) -> list[float] | None:
    """Parse ping -c output and return list of per-packet times in ms.
    Returns None if parsing fails."""
    times: list[float] = []
    for line in output.splitlines():
        match = re.search(r"time=([0-9.]+)\s*ms", line)
        if match:
            try:
                times.append(float(match.group(1)))
            except ValueError:
                pass
    return times if times else None


def _read_proc_stat() -> tuple[int, int] | None:
    """Read /proc/stat and return (busy_ticks, total_ticks) across all CPUs.
    Busy = user + nice + system + irq + softirq + steal.
    Returns None if unreadable."""
    try:
        with open("/proc/stat", "r") as f:
            for line in f:
                if line.startswith("cpu "):
                    parts = line.split()
                    if len(parts) < 8:
                        return None
                    ticks = [int(x) for x in parts[1:8]]
                    user, nice, system, idle, iowait, irq, softirq = ticks[:7]
                    steal = ticks[7] if len(ticks) > 7 else 0
                    busy = user + nice + system + irq + softirq + steal
                    total = busy + idle + iowait
                    return busy, total
    except (OSError, ValueError):
        pass
    return None


async def _make_client(base_url: str, token: str | None) -> httpx.AsyncClient:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.AsyncClient(base_url=base_url, headers=headers, timeout=120.0)


async def _run_ping(host: str, pings: int) -> dict | None:
    """Run system ping and return parsed result dict, or error dict, or None on exception."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ping", "-c", str(pings), host,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        if proc.returncode == 0:
            times = _parse_ping_output(stdout.decode())
            if times is not None:
                return {
                    "p50_ms": round(_pctl(times, 0.50), 2),
                    "p95_ms": round(_pctl(times, 0.95), 2),
                    "raw_ms": times,
                }
            return {"error": "failed to parse ping output"}
        return {"error": f"ping exited {proc.returncode}: {stderr.decode().strip()}"}
    except (asyncio.TimeoutError, FileNotFoundError, OSError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def run(args: argparse.Namespace) -> int:
    started_at = datetime.utcnow().isoformat() + "Z"

    # a. GET /version
    parsed = urlparse(args.base_url)
    host = parsed.hostname or "localhost"

    token = None
    if args.token_file:
        try:
            token = Path(args.token_file).expanduser().read_text().strip()
        except OSError as exc:
            print(f"ERROR: failed to read token file: {exc}", file=sys.stderr)
            return 2

    client = await _make_client(args.base_url, token)
    try:
        resp = await client.get("/version")
        if resp.status_code != 200:
            print(f"ERROR: /version returned {resp.status_code}: {resp.text}", file=sys.stderr)
            return 2
        version_json = resp.json()
    except Exception as exc:
        print(f"ERROR: /version request failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        await client.aclose()

    # b. Ping
    ping_result = await _run_ping(host, args.pings)

    # c. Ingest
    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    conversations = json.loads(dataset_path.read_text())
    if args.conversation_index >= len(conversations):
        print(f"ERROR: conversation index {args.conversation_index} out of range (max {len(conversations)-1})", file=sys.stderr)
        return 2
    conv = conversations[args.conversation_index]
    conversation = conv.get("conversation", conv)

    # Read CPU before ingest
    cpu_before = _read_proc_stat()

    client = await _make_client(args.base_url, token)
    try:
        turns_sent = 0
        vector_failures_sum = 0
        ingest_start = time.perf_counter()

        batch: list[dict] = []
        for session_key, _dt in _session_keys(conversation):
            for turn in conversation.get(session_key) or []:
                text = (turn.get("text") or "").strip()
                if not text:
                    continue
                batch.append({"text": text})
                if len(batch) >= 50:
                    resp = await client.post(
                        "/ingest/batch",
                        json={"items": batch, "agent": args.agent},
                    )
                    if resp.status_code != 200:
                        print(f"ERROR: /ingest/batch returned {resp.status_code}: {resp.text}", file=sys.stderr)
                        return 2
                    data = resp.json()
                    vector_failures_sum += data.get("vector_failures", 0)
                    turns_sent += len(batch)
                    batch = []

        if batch:
            resp = await client.post(
                "/ingest/batch",
                json={"items": batch, "agent": args.agent},
            )
            if resp.status_code != 200:
                print(f"ERROR: /ingest/batch returned {resp.status_code}: {resp.text}", file=sys.stderr)
                return 2
            data = resp.json()
            vector_failures_sum += data.get("vector_failures", 0)
            turns_sent += len(batch)

        ingest_s = time.perf_counter() - ingest_start
        ingest_result = {
            "turns_sent": turns_sent,
            "ingest_s": round(ingest_s, 2),
            "turns_per_sec": round(turns_sent / ingest_s, 2) if ingest_s > 0 else 0.0,
            "vector_failures": vector_failures_sum,
        }
    finally:
        await client.aclose()

    # Read CPU after queries (we'll read after step d)
    # Actually, spec says read before step c and after step d
    # So we need to read cpu_after after queries complete
    cpu_after = _read_proc_stat()

    # d. Queries
    qas = conv.get("qa", []) or []
    selected_qas = []
    for qa in qas:
        category = int(qa.get("category", 0))
        if category != 5:
            selected_qas.append(qa)
        if len(selected_qas) >= args.queries:
            break

    client = await _make_client(args.base_url, token)
    try:
        query_times: list[float] = []
        query_results: list[dict] = []
        error_count = 0

        for qa in selected_qas:
            question = qa["question"]
            t0 = time.perf_counter()
            try:
                resp = await client.post(
                    "/search",
                    json={"query": question, "agent": args.agent, "limit": args.limit},
                )
                query_ms = (time.perf_counter() - t0) * 1000.0
                hit_count = len(resp.json().get("results", [])) if resp.status_code == 200 else 0
                query_results.append({
                    "question": question,
                    "latency_ms": round(query_ms, 2),
                    "hit_count": hit_count,
                    "http_status": resp.status_code,
                })
                if resp.status_code == 200:
                    query_times.append(query_ms)
                else:
                    error_count += 1
            except Exception as exc:
                query_ms = (time.perf_counter() - t0) * 1000.0
                query_results.append({
                    "question": question,
                    "latency_ms": round(query_ms, 2),
                    "hit_count": 0,
                    "http_status": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                error_count += 1

        # Read CPU after queries
        cpu_after = _read_proc_stat()

        # Check error rate
        total_queries = len(selected_qas)
        if total_queries > 0 and error_count / total_queries > 0.10:
            # Still write output before exiting
            pass  # We'll write output and then exit with code 3
    finally:
        await client.aclose()

    # e. Client CPU
    client_cpu_busy_fraction = None
    if cpu_before and cpu_after:
        busy_before, total_before = cpu_before
        busy_after, total_after = cpu_after
        busy_delta = busy_after - busy_before
        total_delta = total_after - total_before
        if total_delta > 0:
            client_cpu_busy_fraction = round(busy_delta / total_delta, 4)

    # f. Write output
    finished_at = datetime.utcnow().isoformat() + "Z"

    query_latencies = [r["latency_ms"] for r in query_results if r.get("http_status") == 200]
    query_summary = {
        "total": len(query_results),
        "errors": error_count,
        "error_rate": round(error_count / len(query_results), 4) if query_results else 0.0,
        "latency": _latency_stats(query_latencies),
        "per_query": query_results,
    }

    config_dict = {
        "base_url": args.base_url,
        "agent": args.agent,
        "dataset": args.dataset,
        "conversation_index": args.conversation_index,
        "queries": args.queries,
        "limit": args.limit,
        "pings": args.pings,
        "label": args.label,
        "shape": args.shape,
    }
    provenance = collect(config_dict, None, args.dataset)

    output = {
        "label": args.label,
        "shape": args.shape,
        "base_url": args.base_url,
        "version": version_json,
        "ping": ping_result,
        "ingest": ingest_result,
        "queries": query_summary,
        "client_cpu_busy_fraction": client_cpu_busy_fraction,
        "provenance": provenance,
        "started_at": started_at,
        "finished_at": finished_at,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    # Check error rate for exit code
    total_queries = len(query_results)
    if total_queries > 0 and error_count / total_queries > 0.10:
        print(f"ERROR: {error_count}/{total_queries} queries errored (> 10% threshold)", file=sys.stderr)
        return 3

    print(f"wrote {out_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Topology client probe for taOSmd (E-049 Block T). "
                    "Client-side only: measures /search latency, ingest rate, "
                    "ping, and client CPU against a running HTTP service."
    )
    p.add_argument("--base-url", required=True,
                   help="Base URL of taOSmd HTTP service (e.g. http://host:7700)")
    p.add_argument("--token-file",
                   help="Path to file containing bearer token (sent as Authorization header)")
    p.add_argument("--agent", default="e049-topology",
                   help="Agent name to use for ingest/search (default: e049-topology)")
    p.add_argument("--dataset", default=_DEFAULT_DATASET,
                   help=f"LoCoMo dataset JSON (default: {_DEFAULT_DATASET})")
    p.add_argument("--conversation-index", type=int, default=0,
                   help="Conversation index to use from dataset (default: 0)")
    p.add_argument("--queries", type=int, default=50,
                   help="Number of QAs to query (skipping category 5) (default: 50)")
    p.add_argument("--limit", type=int, default=10,
                   help="Search limit per query (default: 10)")
    p.add_argument("--pings", type=int, default=20,
                   help="Number of ping packets to send (default: 20)")
    p.add_argument("--label", required=True,
                   help="Free-text row name for this run (e.g. T-b)")
    p.add_argument("--shape", required=True, choices=["whole-service", "embedder-only"],
                   help="Topology shape label")
    p.add_argument("--out", required=True,
                   help="Output JSON path")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(_parse_args())))