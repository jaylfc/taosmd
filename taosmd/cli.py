"""``taosmd`` command-line interface.

Today this is just an agent registry CLI — list, add, remove agents.
Will grow as more operational surface needs to be reachable from the
shell (e.g. compaction, re-indexing, exporting an agent's shelf).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from .agents import (
    AgentExistsError,
    AgentNotFoundError,
    AgentRegistry,
    InvalidAgentNameError,
)


def _fmt_ts(epoch: int) -> str:
    if not epoch:
        return "—"
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _agent_list(registry: AgentRegistry, json_out: bool) -> int:
    agents = registry.list_agents()
    if json_out:
        print(json.dumps(agents, indent=2))
        return 0
    if not agents:
        print("No agents registered.")
        return 0
    # Tiny table — no external deps, fixed column widths.
    print(f"{'NAME':<24} {'DISPLAY':<24} {'CREATED':<18} {'LAST INGEST':<18} {'CHUNKS':>8}")
    for a in agents:
        print(
            f"{a['name']:<24} "
            f"{a.get('display_name', a['name'])[:24]:<24} "
            f"{_fmt_ts(a.get('created_at', 0)):<18} "
            f"{_fmt_ts(a.get('last_ingest_at', 0)):<18} "
            f"{a.get('total_chunks', 0):>8}"
        )
    return 0


def _agent_add(registry: AgentRegistry, name: str, display_name: str, clobber: bool) -> int:
    try:
        record = registry.register_agent(name, display_name=display_name, clobber=clobber)
    except AgentExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("hint: re-run with --clobber to overwrite the existing record.", file=sys.stderr)
        return 2
    except InvalidAgentNameError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Registered agent: {record['name']}")
    return 0


def _agent_rm(registry: AgentRegistry, name: str, drop_data: bool) -> int:
    try:
        registry.delete_agent(name, drop_data=drop_data)
    except AgentNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if drop_data:
        print(f"Removed agent {name!r} and deleted its index files.")
    else:
        print(f"Removed agent {name!r}. Index files preserved (use --drop-data to delete).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="taosmd")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to the taosmd data directory (default: ./data)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    agent = sub.add_parser("agent", help="Manage registered agents")
    agent_sub = agent.add_subparsers(dest="agent_cmd", required=True)

    list_p = agent_sub.add_parser("list", help="List all registered agents")
    list_p.add_argument("--json", action="store_true", help="Output as JSON")

    add_p = agent_sub.add_parser("add", help="Register a new agent")
    add_p.add_argument("name")
    add_p.add_argument("--display-name", default="", help="Human-readable name (defaults to name)")
    add_p.add_argument("--clobber", action="store_true", help="Overwrite an existing record")

    rm_p = agent_sub.add_parser("rm", help="Remove a registered agent")
    rm_p.add_argument("name")
    rm_p.add_argument("--drop-data", action="store_true", help="Also delete the agent's index files")

    args = parser.parse_args(argv)
    registry = AgentRegistry(args.data_dir)

    if args.cmd == "agent":
        if args.agent_cmd == "list":
            return _agent_list(registry, args.json)
        if args.agent_cmd == "add":
            return _agent_add(registry, args.name, args.display_name, args.clobber)
        if args.agent_cmd == "rm":
            return _agent_rm(registry, args.name, args.drop_data)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
