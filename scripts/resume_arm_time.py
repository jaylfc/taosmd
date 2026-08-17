#!/usr/bin/env python3
"""Resume pair arming helper.

Generates durable crontab lines for the resume pair (primary + retry) and
provides a --fire mode that appends a log line and removes its own
marker-prefixed entry from the crontab.

Usage:
    python3 scripts/resume_arm_time.py <resets_at>
    python3 scripts/resume_arm_time.py --fire --type primary --marker <marker> --armed-at <iso>

The <resets_at> argument is an ISO-8601 timestamp (e.g. 2026-08-17T14:00:00+00:00).
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys


WATCHER_CRON_MINUTES = (6, 16, 26, 36, 46, 56)
MARGIN_MINUTES = 11
RETRY_LEAD_MINUTES = 20


def _next_watcher_tick(after: datetime.datetime) -> datetime.datetime:
    base = after.replace(second=0, microsecond=0)
    for minute in WATCHER_CRON_MINUTES:
        tick = base.replace(minute=minute)
        if tick > after:
            return tick
    next_hour = base + datetime.timedelta(hours=1)
    return next_hour.replace(minute=WATCHER_CRON_MINUTES[0])


def derive(resets_at: datetime.datetime) -> tuple[str, str]:
    primary_tick = _next_watcher_tick(resets_at)
    primary = primary_tick + datetime.timedelta(minutes=MARGIN_MINUTES)
    retry = primary + datetime.timedelta(minutes=RETRY_LEAD_MINUTES)

    def _cron(dt: datetime.datetime) -> str:
        return f"{dt.minute} {dt.hour} * * *"

    return _cron(primary), _cron(retry)


def _marker(fire_type: str, script_path: str) -> str:
    digest = hash(script_path) & 0xFFFFFFFF
    return f"TAOSMD-RESUME-{fire_type.upper()}-{digest:08x}"


def do_fire(fire_type: str, marker: str, armed_at: str, data_dir: str | None = None) -> None:
    body = f"[RESUME DUE] {fire_type} fired for window ending {armed_at}"
    try:
        import asyncio
        from taosmd.service import a2a_send
        asyncio.run(a2a_send(
            sender="resume_arm",
            body=body,
            thread="agent-rules",
            data_dir=data_dir,
        ))
    except Exception:
        pass

    try:
        current = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
        filtered = "\n".join(
            line for line in current.splitlines()
            if marker not in line
        )
        subprocess.run(["crontab", "-"], input=filtered, text=True, check=True)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("resets_at", nargs="?")
    parser.add_argument("--fire", action="store_true")
    parser.add_argument("--type", dest="fire_type")
    parser.add_argument("--marker")
    parser.add_argument("--armed-at")
    args = parser.parse_args()

    if args.fire:
        do_fire(args.fire_type, args.marker, args.armed_at)
        return

    if not args.resets_at:
        parser.error("resets_at is required")

    resets_at = datetime.datetime.fromisoformat(args.resets_at)
    primary, retry = derive(resets_at)
    script_path = os.path.abspath(__file__)
    primary_marker = _marker("primary", script_path)
    retry_marker = _marker("retry", script_path)
    armed_iso = resets_at.isoformat()

    print("USER CRONTAB (durable, survives session death)")
    print(
        f"{primary} python3 {script_path} --fire "
        f"--type primary --marker {primary_marker} "
        f"--armed-at {armed_iso}  # {primary_marker}"
    )
    print(
        f"{retry} python3 {script_path} --fire "
        f"--type retry --marker {retry_marker} "
        f"--armed-at {armed_iso}  # {retry_marker}"
    )


if __name__ == "__main__":
    main()
