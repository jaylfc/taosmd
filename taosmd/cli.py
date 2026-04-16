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
    LIBRARIAN_TASKS,
    FANOUT_LEVELS,
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


def _librarian_show(registry: AgentRegistry, name: str, json_out: bool) -> int:
    try:
        lib = registry.get_librarian(name)
    except AgentNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if json_out:
        print(json.dumps(lib, indent=2))
        return 0
    enabled = "ON" if lib.get("enabled", True) else "OFF"
    model = lib.get("model") or "(install default)"
    print(f"Agent:      {name}")
    print(f"Librarian:  {enabled}")
    print(f"Model:      {model}")
    print("Tasks:")
    tasks = lib.get("tasks", {})
    for t in LIBRARIAN_TASKS:
        state = "on " if tasks.get(t, True) else "off"
        print(f"  [{state}] {t}")
    return 0


def _librarian_set(
    registry: AgentRegistry,
    name: str,
    enabled: bool | None,
    model: str | None,
    clear_model: bool,
    enable_tasks: list[str],
    disable_tasks: list[str],
    fanout: str | None = None,
    fanout_auto_scale: bool | None = None,
) -> int:
    tasks: dict[str, bool] = {}
    for t in enable_tasks:
        tasks[t] = True
    for t in disable_tasks:
        tasks[t] = False
    try:
        lib = registry.set_librarian(
            name,
            enabled=enabled,
            model=model,
            tasks=tasks or None,
            clear_model=clear_model,
            fanout=fanout,
            fanout_auto_scale=fanout_auto_scale,
        )
    except AgentNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(lib, indent=2))
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

    # ----- librarian subcommands -------------------------------------------
    lib_p = sub.add_parser("librarian", help="Per-agent librarian (LLM enrichment) controls")
    lib_sub = lib_p.add_subparsers(dest="librarian_cmd", required=True)

    show_p = lib_sub.add_parser("show", help="Show an agent's librarian config")
    show_p.add_argument("name")
    show_p.add_argument("--json", action="store_true")

    set_p = lib_sub.add_parser("set", help="Patch an agent's librarian config")
    set_p.add_argument("name")
    set_p.add_argument(
        "--on", dest="enabled_on", action="store_true",
        help="Master switch ON (LLM enrichment runs)",
    )
    set_p.add_argument(
        "--off", dest="enabled_off", action="store_true",
        help="Master switch OFF (only verbatim archive + vector + keyword)",
    )
    set_p.add_argument(
        "--model", default=None,
        help="provider:model override (e.g. ollama:qwen3:4b)",
    )
    set_p.add_argument(
        "--clear-model", action="store_true",
        help="Revert to the install default model",
    )
    set_p.add_argument(
        "--enable", action="append", default=[], metavar="TASK",
        help=f"Enable a specific task. Repeat for multiple. Tasks: {', '.join(LIBRARIAN_TASKS)}",
    )
    set_p.add_argument(
        "--disable", action="append", default=[], metavar="TASK",
        help="Disable a specific task. Repeat for multiple.",
    )
    set_p.add_argument(
        "--fanout",
        choices=list(FANOUT_LEVELS),
        default=None,
        metavar="LEVEL",
        help=(
            "Fan-out retrieval level: off (K=1), low (K=3), med (K=10), high (K=20). "
            "Low is safe on Pi-class hardware; high suits GPU workers with large context."
        ),
    )
    set_p.add_argument(
        "--auto-scale",
        dest="fanout_auto_scale",
        choices=["on", "off"],
        default=None,
        metavar="on|off",
        help=(
            "Auto-scale fanout tier on workers with GPU + TurboQuant + ≥12 GB VRAM. "
            "When on, low→med and med→high automatically on capable workers."
        ),
    )

    args = parser.parse_args(argv)
    registry = AgentRegistry(args.data_dir)

    if args.cmd == "agent":
        if args.agent_cmd == "list":
            return _agent_list(registry, args.json)
        if args.agent_cmd == "add":
            return _agent_add(registry, args.name, args.display_name, args.clobber)
        if args.agent_cmd == "rm":
            return _agent_rm(registry, args.name, args.drop_data)

    if args.cmd == "librarian":
        if args.librarian_cmd == "show":
            return _librarian_show(registry, args.name, args.json)
        if args.librarian_cmd == "set":
            if args.enabled_on and args.enabled_off:
                print("error: --on and --off are mutually exclusive", file=sys.stderr)
                return 2
            enabled: bool | None = None
            if args.enabled_on:
                enabled = True
            elif args.enabled_off:
                enabled = False
            fanout_auto_scale: bool | None = None
            if args.fanout_auto_scale == "on":
                fanout_auto_scale = True
            elif args.fanout_auto_scale == "off":
                fanout_auto_scale = False
            return _librarian_set(
                registry,
                args.name,
                enabled,
                args.model,
                args.clear_model,
                args.enable,
                args.disable,
                fanout=args.fanout,
                fanout_auto_scale=fanout_auto_scale,
            )

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
