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
    print(f"Agent:      {name}")
    print(f"Librarian:  {enabled}")
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
            tasks=tasks or None,
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


def _memory_model_get() -> int:
    from . import config  # noqa: PLC0415

    model = config.get_memory_model()
    print(model if model else "(default)")
    return 0


def _memory_model_set(model: str | None, clear: bool) -> int:
    from . import config  # noqa: PLC0415

    if not clear and not model:
        print("error: provide --model <provider:model> or --clear", file=sys.stderr)
        return 2
    try:
        config.set_memory_model(model or "", clear=clear)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if clear:
        print("Memory model cleared (using default).")
    else:
        print(f"Memory model set: {model}")
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


_RESOLUTION_BY_ACTION = {
    "accept": "accepted",
    "reject": "rejected",
    "modify": "modified",
}


def _review_cmd(args: argparse.Namespace) -> int:
    """Handle ``taosmd review`` subcommand."""
    import asyncio
    from pathlib import Path
    from .pending_decisions import PendingDecisionsStore
    from .knowledge_graph import TemporalKnowledgeGraph

    kg_path = Path(args.data_dir) / "knowledge-graph.db"
    if not kg_path.exists():
        print(f"error: knowledge-graph.db not found at {kg_path}", file=sys.stderr)
        print("  Run `python -m taosmd.auto_setup` first.", file=sys.stderr)
        return 2

    async def _run() -> int:
        store = PendingDecisionsStore(db_path=kg_path)
        await store.init()
        try:
            if args.resolve_id is not None:
                return await _review_resolve_single(store, args, kg_path)
            return await _review_browse(store, args, kg_path)
        finally:
            await store.close()

    return asyncio.run(_run())


async def _review_resolve_single(
    store, args: argparse.Namespace, kg_path,
) -> int:
    if args.action is None:
        print("error: --resolve requires --action {accept|reject|modify}", file=sys.stderr)
        return 2
    resolution = _RESOLUTION_BY_ACTION[args.action]
    decision = await store.get(args.resolve_id)
    if decision is None:
        print(f"error: no pending decision with id {args.resolve_id!r}", file=sys.stderr)
        return 2

    # Apply the KG mutation that the resolution implies. The store records
    # the resolution either way; if the KG mutation fails the user sees the
    # error and can retry.
    if resolution == "accepted" and decision["suggested_action"] == "invalidate_old_add_new":
        from .knowledge_graph import TemporalKnowledgeGraph
        kg = TemporalKnowledgeGraph(db_path=kg_path)
        await kg.init()
        try:
            for old_id in decision["old_triple_ids"]:
                await kg.invalidate(old_id)
            await kg.add_triple(
                subject=decision["subject"],
                predicate=decision["predicate"],
                obj=decision["new_object"],
                confidence=decision["new_triple_confidence"],
                source=decision["source"],
            )
        finally:
            await kg.close()

    ok = await store.resolve(args.resolve_id, resolution=resolution, note=args.note)
    if not ok:
        print(f"error: decision {args.resolve_id} was already resolved", file=sys.stderr)
        return 1
    print(f"Decision {args.resolve_id} -> {resolution}")
    return 0


async def _review_browse(
    store, args: argparse.Namespace, kg_path,
) -> int:
    pending = await store.list_pending(subject=args.subject, limit=args.limit)

    if args.review_json:
        print(json.dumps(pending, indent=2))
        return 0

    if not pending:
        print("No pending decisions. KG is in sync.")
        return 0

    if args.review_list:
        for d in pending:
            print(_format_decision_short(d))
        print(f"\n{len(pending)} pending decision(s). Run `taosmd review` without --list to resolve interactively.")
        return 0

    # Interactive mode
    print(f"{len(pending)} pending decision(s). Press y to accept, n to reject, "
          "s to skip, q to quit, ? for help.\n")
    accepted = rejected = skipped = 0
    for i, decision in enumerate(pending, 1):
        print(f"--- Decision {i}/{len(pending)} (id: {decision['id']}) ---")
        print(_format_decision_full(decision))
        while True:
            resp = input("\n[y]es accept / [n]o reject / [s]kip / [q]uit / [?] help: ").strip().lower()
            if resp in {"y", "n", "s", "q", "?"}:
                break
            print("  Please enter y, n, s, q, or ?")
        if resp == "?":
            print(_review_help())
            continue
        if resp == "q":
            break
        if resp == "s":
            skipped += 1
            continue
        note = input("  Note (optional, ENTER to skip): ").strip()
        action = "accept" if resp == "y" else "reject"
        sub_args = argparse.Namespace(
            data_dir=args.data_dir,
            resolve_id=decision["id"],
            action=action,
            note=note,
        )
        rc = await _review_resolve_single(store, sub_args, kg_path)
        if rc == 0:
            if resp == "y":
                accepted += 1
            else:
                rejected += 1
        print()

    print(f"Done. Accepted: {accepted}, Rejected: {rejected}, Skipped: {skipped}.")
    return 0


def _format_decision_short(d: dict) -> str:
    return (
        f"  {d['id']}  {d['kind']:<25}  "
        f"{d['subject']} {d['predicate']} -> {d['new_object']!r}"
    )


def _format_decision_full(d: dict) -> str:
    created = datetime.fromtimestamp(d["created_at"], tz=timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"  Kind:        {d['kind']}",
        f"  Subject:     {d['subject']}",
        f"  Predicate:   {d['predicate']}",
        f"  New object:  {d['new_object']}",
        f"  Confidence:  {d['new_triple_confidence']:.2f}",
        f"  Detected:    {created}",
        f"  Source:      {d['source'] or '(none)'}",
    ]
    if d.get("evidence"):
        lines.append(f"  Evidence:    {d['evidence']}")
    if d.get("old_triple_ids"):
        lines.append(f"  Conflicts with: {len(d['old_triple_ids'])} existing triple(s)")
    lines.append(f"  Suggested:   {d['suggested_action']}")
    return "\n".join(lines)


def _review_help() -> str:
    return (
        "\n  y (accept)  apply the suggested action: invalidate the old triple(s),\n"
        "              write the new one with the recorded confidence.\n"
        "  n (reject)  leave the KG alone. The new claim is recorded as rejected\n"
        "              so it doesn't re-queue.\n"
        "  s (skip)    decide later; the decision stays in the queue.\n"
        "  q (quit)    stop reviewing. Anything you already accepted/rejected\n"
        "              stays applied.\n"
    )


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

    # ----- memory-model subcommand (system-wide model) ------------------
    mm_p = sub.add_parser(
        "memory-model",
        help="Get/set the system-wide memory (Librarian) model",
    )
    mm_sub = mm_p.add_subparsers(dest="memory_model_cmd", required=True)

    mm_sub.add_parser("get", help="Print the current memory model (or '(default)')")

    mm_set_p = mm_sub.add_parser("set", help="Set or clear the memory model")
    mm_set_p.add_argument(
        "--model", default=None,
        help="provider:model string (e.g. ollama:qwen3:4b)",
    )
    mm_set_p.add_argument(
        "--clear", action="store_true",
        help="Unset the memory model (revert to default)",
    )

    # ----- review subcommand (pending-decisions queue) -------------------
    review_p = sub.add_parser(
        "review",
        help="Review pending KG-update decisions deferred by the nightly librarian",
    )
    review_p.add_argument(
        "--list", dest="review_list", action="store_true",
        help="Print pending decisions and exit (no interactive prompt)",
    )
    review_p.add_argument(
        "--json", dest="review_json", action="store_true",
        help="Output as JSON (implies --list)",
    )
    review_p.add_argument(
        "--limit", type=int, default=20,
        help="Max pending decisions to show / iterate over (default 20)",
    )
    review_p.add_argument(
        "--subject", default=None,
        help="Filter pending decisions to a specific subject entity",
    )
    review_p.add_argument(
        "--resolve", dest="resolve_id", default=None,
        help="Resolve a single decision by ID (non-interactive)",
    )
    review_p.add_argument(
        "--action", choices=["accept", "reject", "modify"], default=None,
        help="Resolution action when using --resolve",
    )
    review_p.add_argument(
        "--note", default="",
        help="Free-form note attached to the resolution",
    )

    # ----- supersede subcommand (retire stale chunks from vector recall) -
    supersede_p = sub.add_parser(
        "supersede",
        help="Soft-hide vector chunk(s) whose text contains --match from active "
             "recall (zero-loss: raw rows are retained, only excluded from search)",
    )
    supersede_p.add_argument(
        "--agent", default=None,
        help="Agent name (accepted for symmetry; the vector store is per data dir)",
    )
    supersede_p.add_argument(
        "--match", required=True,
        help="Substring; every active chunk whose stored text contains it is superseded",
    )

    # ----- serve subcommand (local HTTP/REST API) -----------------------
    serve_p = sub.add_parser(
        "serve",
        help="Run the local HTTP/REST memory API (stdlib server, zero deps)",
    )
    serve_p.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default 127.0.0.1, localhost-only). "
             "Use 0.0.0.0 to expose on the LAN — no auth, so gate it yourself.",
    )
    serve_p.add_argument(
        "--port", type=int, default=7833,
        help="Bind port (default 7833)",
    )
    serve_p.add_argument(
        "--serve-data-dir", dest="serve_data_dir", default=None,
        help="Data dir for served memory (default: $TAOSMD_DATA_DIR or ~/.taosmd)",
    )

    # ----- mcp subcommand (MCP server over stdio) -----------------------
    mcp_p = sub.add_parser(
        "mcp",
        help="Run the MCP memory server over stdio (requires taosmd[mcp])",
    )
    mcp_p.add_argument(
        "--mcp-data-dir", dest="mcp_data_dir", default=None,
        help="Data dir for served memory (default: $TAOSMD_DATA_DIR or ~/.taosmd)",
    )

    args = parser.parse_args(argv)

    if args.cmd == "serve":
        from . import http_server  # noqa: PLC0415
        return http_server.serve(
            host=args.host, port=args.port, data_dir=args.serve_data_dir,
        )

    if args.cmd == "mcp":
        from . import mcp_server  # noqa: PLC0415
        try:
            return mcp_server.serve_stdio(data_dir=args.mcp_data_dir)
        except mcp_server.MissingMCPDependencyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.cmd == "supersede":
        import asyncio  # noqa: PLC0415
        from . import service  # noqa: PLC0415
        result = asyncio.run(
            service.supersede(args.match, agent=args.agent, data_dir=args.data_dir)
        )
        print(f"superseded {result['superseded']} chunk(s) matching {result['match']!r}")
        return 0

    registry = AgentRegistry(args.data_dir)

    if args.cmd == "review":
        return _review_cmd(args)

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
                args.enable,
                args.disable,
                fanout=args.fanout,
                fanout_auto_scale=fanout_auto_scale,
            )

    if args.cmd == "memory-model":
        if args.memory_model_cmd == "get":
            return _memory_model_get()
        if args.memory_model_cmd == "set":
            return _memory_model_set(args.model, args.clear)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
