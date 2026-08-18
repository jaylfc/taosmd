#!/usr/bin/env python3
"""Measure upstream usage-window rollover propagation via the real API.

At an Anthropic usage-window reset boundary, the five_hour window
identity (resets_at) does not update instantly.  This harness polls the
REAL Anthropic usage API at https://api.anthropic.com/api/oauth/usage,
one request per second, and records the wall-clock delta between the
nominal resets_at and the first response serving the new window.

The constant this measures -- MIN_LEAD_SECONDS in the out-of-repo
resume-arm-time helper -- is labelled UNMEASURED until a live
measurement succeeds.  If no credentials are available, the run writes
an UNMEASURED evidence record to benchmarks/results/ so the label is
never silently retired.

Usage:
    TAOSMD_ANTHROPIC_CREDS=/path/creds.json python3 benchmarks/measure_upstream_propagation.py
    python3 benchmarks/measure_upstream_propagation.py --reset-at 2026-08-17T08:00:00+00:00 --creds /path/creds.json
    python3 benchmarks/measure_upstream_propagation.py --dry-run --creds /path/creds.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import httpx

_BENCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCH_DIR.parent
_RESULTS_DIR = _BENCH_DIR / "results"
_EVIDENCE_FILE = _RESULTS_DIR / "tsk-rcnct6_upstream_propagation.json"

USAGE_API_URL = "https://api.anthropic.com/api/oauth/usage"

_HEADERS = {
    "anthropic-beta": "oauth-2025-04-20",
}


def resolve_token(creds_path: str | None = None) -> str:
    """Resolve the Anthropic OAuth token from a credentials file.

    The path is configurable via --creds or TAOSMD_ANTHROPIC_CREDS so no
    hardcoded absolute path is baked into the source.
    """
    path = creds_path or os.environ.get("TAOSMD_ANTHROPIC_CREDS")
    if not path:
        raise SystemExit(
            "No credentials path: set TAOSMD_ANTHROPIC_CREDS or pass --creds"
        )
    with open(path, "r") as f:
        creds = json.load(f)
    return creds["claudeAiOauth"]["accessToken"]


def fetch_usage(token: str, client: httpx.Client | None = None) -> dict:
    """Fetch live usage data from the real Anthropic usage API.

    Returns the parsed JSON body.  Does NOT return canned or mock data.
    """
    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=15.0)
    try:
        resp = client.get(
            USAGE_API_URL,
            headers={**_HEADERS, "Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.json()
    finally:
        if owns_client:
            client.close()


def parse_reset(resets_at: str) -> datetime.datetime:
    """Parse a resets_at ISO timestamp into an aware datetime."""
    cleaned = resets_at.replace("Z", "+00:00")
    dt = datetime.datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def measure_at_boundary(
    reset_at: datetime.datetime,
    token: str,
    poll_interval: float = 1.0,
    max_wait: float = 600.0,
    fetch_fn=None,
    sleep_fn=None,
) -> dict:
    """Poll the real API each second around a reset boundary.

    Records the wall-clock delta between *reset_at* and the first
    response whose five_hour.resets_at differs from the pre-reset
    value, proving the window identity has rolled over.  Utilization
    is recorded as a corroborating signal but never serves as the flip
    trigger: on a shared account utilization moves inside a window as a
    matter of course, and on an idle window it never moves at all, so
    neither direction is a reliable signal.
    """
    if fetch_fn is None:
        fetch_fn = fetch_usage
    if sleep_fn is None:
        sleep_fn = time.sleep

    samples: list[dict] = []
    now = datetime.datetime.now(datetime.timezone.utc)

    pre_util = None
    pre_resets_at = None

    # Capture the pre-boundary baseline BEFORE waiting for the boundary,
    # so a propagation faster than the first post-boundary poll is
    # still measurable by construction.
    try:
        pre_usage = fetch_fn(token)
        pre_window = pre_usage.get("five_hour", {})
        pre_util = pre_window.get("utilization")
        pre_resets_at = pre_window.get("resets_at")
    except Exception as exc:  # noqa: BLE001 - record, proceed without baseline
        samples.append({"time": now.isoformat(), "pre_boundary_error": f"{type(exc).__name__}: {exc}"})

    # Wait for the boundary to arrive.  This wait is NOT counted
    # against the polling budget.
    while now < reset_at:
        wait = (reset_at - now).total_seconds()
        if wait <= 0:
            break
        sleep_fn(min(wait, poll_interval))
        now = datetime.datetime.now(datetime.timezone.utc)

    # Start the polling budget AFTER the boundary arrives, so the
    # pre-boundary wait does not consume max_wait.  (Previously t0
    # was set before the wait loop, exhausting the budget on a
    # far-future boundary and never polling once.)
    t0 = time.monotonic()

    while (time.monotonic() - t0) < max_wait:
        now = datetime.datetime.now(datetime.timezone.utc)
        try:
            usage = fetch_fn(token)
        except Exception as exc:  # noqa: BLE001 - record, keep polling
            samples.append({"time": now.isoformat(), "error": f"{type(exc).__name__}: {exc}"})
            sleep_fn(poll_interval)
            continue

        window = usage.get("five_hour", {})
        cur_resets_at = window.get("resets_at")
        util = window.get("utilization")

        samples.append({
            "time": now.isoformat(),
            "resets_at": cur_resets_at,
            "utilization": util,
        })

        # Key the flip on resets_at changing (window identity), not
        # utilization.  Utilization may move inside a window as a
        # matter of course (shared account), so it cannot trigger a
        # false positive.  A rollover on an idle window is invisible to
        # utilization, so it cannot be the trigger either.
        if (
            pre_resets_at is not None
            and cur_resets_at is not None
            and cur_resets_at != pre_resets_at
        ):
            delta = (now - reset_at).total_seconds()
            return {
                "reset_at": reset_at.isoformat(),
                "flipped_at": now.isoformat(),
                "propagation_seconds": delta,
                "pre_reset_utilization": pre_util,
                "post_reset_utilization": util,
                "pre_resets_at": pre_resets_at,
                "post_resets_at": cur_resets_at,
                "samples": samples,
                "status": "MEASURED",
            }

        sleep_fn(poll_interval)

    return {
        "reset_at": reset_at.isoformat(),
        "flipped_at": None,
        "propagation_seconds": None,
        "pre_reset_utilization": pre_util,
        "pre_resets_at": pre_resets_at,
        "samples": samples,
        "status": "NO_FLIP_DETECTED",
    }


def write_evidence(result: dict, path: Path | None = None) -> Path:
    """Write evidence to the repo's benchmarks/results/ directory."""
    out = path or _EVIDENCE_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return out


def record_unmeasured(creds_path: str | None = None) -> dict:
    """Record that the measurement could not be performed live."""
    reason = "No live Anthropic credentials available in this environment."
    if creds_path and not Path(creds_path).exists():
        reason = f"Credentials file not found: {creds_path}"
    result = {
        "measurement": "upstream usage-window rollover propagation",
        "target_constant": "MIN_LEAD_SECONDS",
        "target_location": "MIN_LEAD_SECONDS constant in the out-of-repo resume-arm-time helper",
        "status": "UNMEASURED",
        "reason": reason,
        "procedure": (
            "At a reset boundary, poll the live Anthropic usage API "
            "https://api.anthropic.com/api/oauth/usage each second and "
            "record the wall-clock delta between the nominal resets_at and "
            "the first response serving the new five_hour window (detected "
            "by resets_at changing). "
            "See measure_at_boundary() in this file."
        ),
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "measured_samples": [],
        "script": "benchmarks/measure_upstream_propagation.py",
    }
    write_evidence(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure upstream usage-window rollover propagation via the real Anthropic API."
    )
    parser.add_argument("--creds", help="Path to credentials JSON (or set TAOSMD_ANTHROPIC_CREDS)")
    parser.add_argument("--reset-at", help="ISO timestamp of the reset boundary to measure")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between polls")
    parser.add_argument("--max-wait", type=float, default=600.0, help="Max seconds to wait for flip")
    parser.add_argument("--dry-run", action="store_true", help="Check credentials and API reachability")
    args = parser.parse_args()

    try:
        token = resolve_token(args.creds)
    except SystemExit:
        result = record_unmeasured(args.creds)
        print(f"No credentials available. Evidence written as UNMEASURED: {_EVIDENCE_FILE}")
        return 1

    if args.dry_run:
        try:
            usage = fetch_usage(token)
            resets_at = usage.get("five_hour", {}).get("resets_at", "N/A")
            print(f"API reachable. five_hour resets_at: {resets_at}")
            return 0
        except Exception as exc:  # noqa: BLE001 - report any failure to reach API
            print(f"API unreachable: {exc}", file=sys.stderr)
            return 2

    if args.reset_at:
        reset_at = parse_reset(args.reset_at)
    else:
        try:
            usage = fetch_usage(token)
        except Exception as exc:  # noqa: BLE001 - report any failure to fetch
            print(f"Failed to fetch usage: {exc}", file=sys.stderr)
            return 2
        resets_at = usage.get("five_hour", {}).get("resets_at")
        if not resets_at:
            print("No five_hour.resets_at in API response", file=sys.stderr)
            return 2
        reset_at = parse_reset(resets_at)
        print(f"Next reset boundary: {reset_at.isoformat()}")

    print(f"Polling the real API at {reset_at.isoformat()} each second...")
    result = measure_at_boundary(
        reset_at, token,
        poll_interval=args.poll_interval,
        max_wait=args.max_wait,
    )
    path = write_evidence(result)
    print(f"Evidence written: {path}")
    print(f"Status: {result['status']}")
    if result.get("propagation_seconds") is not None:
        print(f"Propagation: {result['propagation_seconds']:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
