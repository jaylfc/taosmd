"""``taosmd`` command-line interface.

Today this is just an agent registry CLI: list, add, remove agents.
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
    # Tiny table, no external deps, fixed column widths.
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


def _generator_profile_list(data_dir=None) -> int:
    from . import generator_profiles as gp
    from . import config
    active = config.get_generator_profile(data_dir=data_dir) or gp.default_profile_id()
    for p in gp.list_profiles():
        mark = "*" if p.id == active else " "
        print(f"{mark} {p.id:16} {p.label}")
    return 0


def _generator_profile_show(profile_id: str, data_dir=None) -> int:
    from . import generator_profiles as gp
    p = gp.get_profile(profile_id)
    if p is None:
        print(f"error: unknown profile {profile_id!r}", file=sys.stderr)
        return 1
    print(f"{p.id}: {p.label}")
    print(f"workload: {p.workload}")
    for tier, model in p.models.items():
        print(f"  {tier:9} {model or '(retrieval-only)'}")
    if p.notes:
        print(f"notes: {p.notes}")
    return 0


def _generator_profile_set(profile_id: str, agent=None, data_dir=None) -> int:
    from . import generator_profiles as gp
    from . import config, agents
    if gp.get_profile(profile_id) is None:
        print(f"error: unknown profile {profile_id!r}", file=sys.stderr)
        return 1
    if agent:
        try:
            agents.set_agent_generator_profile(agent, profile_id, data_dir=data_dir)
        except agents.AgentNotFoundError:
            print(f"error: agent {agent!r} is not registered", file=sys.stderr)
            return 1
        print(f"agent {agent}: generator profile = {profile_id}")
    else:
        config.set_generator_profile(profile_id, data_dir=data_dir)
        print(f"global generator profile = {profile_id}")
    return 0


def _config_set_server(url: str | None, clear: bool) -> int:
    from . import config  # noqa: PLC0415

    if not clear and not url:
        print("error: provide <url> or --clear", file=sys.stderr)
        return 2
    try:
        config.set_server_url(url or "", clear=clear)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if clear:
        print("Server URL cleared (local mode).")
    else:
        print(f"Remote server URL set: {url}")
    return 0


def _config_set_token(token: str | None, clear: bool) -> int:
    from . import config  # noqa: PLC0415

    if not clear and not token:
        print("error: provide <token> or --clear", file=sys.stderr)
        return 2
    try:
        config.set_server_token(token or "", clear=clear)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if clear:
        print("Server token cleared.")
    else:
        print("Server token stored.")
    return 0


def _config_show() -> int:
    from . import config  # noqa: PLC0415

    url = config.get_server_url()
    token = config.get_server_token()
    model = config.get_memory_model()
    print(f"server_url   : {url or '(unset, local mode)'}")
    print(f"server_token : {'(set)' if token else '(unset)'}")
    print(f"memory_model : {model or '(default)'}")
    return 0


def _format_a2a_line(msg: dict, show_thread: bool = False) -> str:
    """Render one A2A message as a single human-readable line.

    Shared by ``a2a-poll`` and ``a2a-watch``. Default format:
    ``[<ts UTC>] <sender>[ (reply_to=N)] <body>``. When ``show_thread`` is
    True (all-channels mode) the channel is included:
    ``[<ts UTC>] (<thread>) <sender>[ (reply_to=N)] <body>``.
    """
    from datetime import datetime, timezone  # noqa: PLC0415

    ts = msg.get("ts", 0)
    try:
        ts_str = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    except Exception:  # noqa: BLE001 - never let a bad ts crash the line
        ts_str = str(ts)
    from_ = msg.get("from", "?")
    body = msg.get("body", "")
    reply = f" (reply_to={msg.get('reply_to')})" if msg.get("reply_to") else ""
    thread = f" ({msg.get('thread')})" if show_thread else ""
    return f"[{ts_str}]{thread} <{from_}>{reply} {body}"


def _a2a_stream(server_url: str, channel: str | None, exclude: str | None, since: float | None):
    """Yield new A2A messages from the bus SSE endpoint, one at a time.

    Holds a long-lived connection to ``GET {server}/a2a/stream?thread=&since=``
    and yields each parsed message dict as it arrives. When ``channel`` is
    ``None``, ``""`` or ``"all"``, the ``thread`` filter is omitted so the
    server streams EVERY channel over the one connection (and newly created or
    renamed channels appear automatically). Robustness mirrors ``a2a-poll``:

    * **id-dedup** - the server stream filters by timestamp (``ts > since``),
      which can miss or repeat same-second siblings across a reconnect. We
      track the max message id seen and drop anything ``<= last_id``, so the
      stream is exactly-once regardless of timestamp granularity.
    * **client-side exclude** - messages from ``exclude`` (usually our own
      agent) are skipped but still advance the cursor.
    * **auto-reconnect** - on any read/connection error the connection is
      retried with a short backoff, resuming from ``last_ts`` rewound by one
      second (id-dedup absorbs the overlap). The bus emits ``: keepalive``
      frames every second, so a dropped peer is detected promptly.

    This is a generator: callers (``a2a-watch``, ``a2a-bridge``) decide what to
    do with each message. It runs until the caller stops consuming or the
    process is interrupted.
    """
    import json  # noqa: PLC0415
    import time  # noqa: PLC0415
    import urllib.parse  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    all_mode = channel in (None, "", "all")
    last_id = -1
    last_ts = since if since is not None else time.time()
    backoff = 1.0

    while True:
        # Rewind a second so a same-timestamp sibling is not skipped; id-dedup
        # below drops anything we have already emitted.
        params = {"since": max(0.0, last_ts - 1.0)}
        if not all_mode:
            params["thread"] = channel  # omit -> server streams every thread
        q = urllib.parse.urlencode(params)
        url = f"{server_url.rstrip('/')}/a2a/stream?{q}"
        try:
            req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                backoff = 1.0  # connected: reset backoff
                for raw in resp:
                    line = raw.decode("utf-8", "replace").rstrip("\n")
                    if not line.startswith("data:"):
                        continue  # keepalive (": ...") or blank frame separator
                    payload = line[len("data:"):].strip()
                    if not payload:
                        continue
                    try:
                        msg = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    try:
                        msg_id = int(msg.get("id"))
                    except (TypeError, ValueError):
                        continue
                    ts = msg.get("ts")
                    if isinstance(ts, (int, float)):
                        last_ts = max(last_ts, float(ts))
                    if msg_id <= last_id:
                        continue
                    last_id = msg_id
                    if exclude and msg.get("from") == exclude:
                        continue  # advance cursor (done above) but do not emit
                    yield msg
        except KeyboardInterrupt:
            return
        except Exception:  # noqa: BLE001 - any transport error: reconnect with backoff
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


def _resolve_a2a_server(args: argparse.Namespace) -> str:
    """Resolve the bus base URL for streaming commands.

    Streaming requires the HTTP server (``taosmd serve``); unlike ``a2a-poll``
    there is no in-process fallback. Order: explicit ``--server`` flag,
    then ``$TAOSMD_SERVER_URL``, then the configured server_url, then the
    local default ``http://127.0.0.1:7900``.
    """
    import os  # noqa: PLC0415

    explicit = getattr(args, "server", None)
    if explicit:
        return explicit
    env = os.environ.get("TAOSMD_SERVER_URL")
    if env:
        return env
    try:
        from . import config as _config  # noqa: PLC0415

        configured = _config.get_server_url()
        if configured:
            return configured
    except Exception:  # noqa: BLE001 - config is optional
        pass
    return "http://127.0.0.1:7900"


def _a2a_watch_cmd(args: argparse.Namespace) -> int:
    """Handle ``taosmd a2a-watch``: stream new messages, one line each.

    Holds the bus SSE and prints one line per new message in the same format
    as ``a2a-poll``, flushing immediately so a consumer (or a harness Monitor)
    gets instant pickup. ``--count N`` exits after N messages (0 = forever).
    When ``--channel`` is omitted (or ``all``), every channel is streamed over
    one connection and each line is prefixed with its ``(thread)``.
    """
    server_url = _resolve_a2a_server(args)
    channel = getattr(args, "channel", None)
    exclude = getattr(args, "exclude", None)
    count = getattr(args, "count", 0) or 0
    all_mode = channel in (None, "", "all")

    emitted = 0
    try:
        for msg in _a2a_stream(server_url, channel, exclude, since=None):
            print(_format_a2a_line(msg, show_thread=all_mode), flush=True)
            emitted += 1
            if count and emitted >= count:
                break
    except KeyboardInterrupt:
        return 0
    return 0


def _a2a_bridge_cmd(args: argparse.Namespace) -> int:
    """Handle ``taosmd a2a-bridge``: run a trigger command on each new message.

    On every new message, spawn ``--trigger`` (a shell command) with the
    message JSON on stdin. This is the only way to wake a dormant local
    session: a headless agent can be spawned on message arrival.

    * ``--debounce S`` coalesces a burst: messages arriving within S seconds of
      the last spawn are batched and passed together (as a JSON array) to the
      next spawn.
    * ``--max-concurrency N`` caps simultaneous trigger processes; while N are
      running, new messages are batched for the next free slot rather than
      piling up overlapping spawns.
    """
    import json  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import time  # noqa: PLC0415

    server_url = _resolve_a2a_server(args)
    exclude = getattr(args, "exclude", None)
    trigger = args.trigger
    debounce = max(0.0, getattr(args, "debounce", 0.0) or 0.0)
    max_conc = max(1, getattr(args, "max_concurrency", 1) or 1)
    count = getattr(args, "count", 0) or 0

    running: list[subprocess.Popen] = []
    last_spawn = 0.0
    fired = 0

    def _reap() -> None:
        running[:] = [p for p in running if p.poll() is None]

    def _spawn(batch: list[dict]) -> None:
        nonlocal last_spawn, fired
        payload = json.dumps(batch[0] if len(batch) == 1 else batch)
        try:
            proc = subprocess.Popen(
                trigger, shell=True, stdin=subprocess.PIPE, text=True,
            )
        except Exception as exc:  # noqa: BLE001 - bad trigger command
            print(
                f"taosmd a2a-bridge: failed to spawn trigger ({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )
            return
        try:
            assert proc.stdin is not None
            proc.stdin.write(payload)
            proc.stdin.close()
        except Exception:  # noqa: BLE001 - process may have exited already
            pass
        running.append(proc)
        last_spawn = time.time()
        fired += 1

    try:
        for msg in _a2a_stream(server_url, args.channel, exclude, since=None):
            _reap()
            now = time.time()
            within_debounce = debounce and (now - last_spawn) < debounce
            if len(running) >= max_conc or within_debounce:
                # Wait for a slot / for the debounce window to pass, batching
                # any further messages that the generator would yield next.
                while len(running) >= max_conc or (
                    debounce and (time.time() - last_spawn) < debounce
                ):
                    time.sleep(0.1)
                    _reap()
            _spawn([msg])
            if count and fired >= count:
                break
    except KeyboardInterrupt:
        return 0
    return 0


def _a2a_poll_cmd(args: argparse.Namespace) -> int:
    """Handle ``taosmd a2a-poll``: fetch new messages and update state file."""
    import asyncio  # noqa: PLC0415
    import json  # noqa: PLC0415 (already imported at module level but guard for type-checker)
    import os  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    state_file = Path(args.state_file).expanduser()
    channel = args.channel

    # --- load / initialise state (BEFORE the fetch, so the ts cursor can
    # bound the query) ----------------------------------------------------
    state: dict = {}
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError):
            state = {}
    last_id: int = state.get(channel, -1)
    last_ts: float | None = None
    try:
        last_ts = float(state.get("_ts", {}).get(channel))
    except (TypeError, ValueError, AttributeError):
        last_ts = None

    # Fetch from just before the ts cursor (1s overlap; the id filter below
    # dedups). A plain most-recent-500 fetch silently dropped anything older
    # than the 500th message after a long cron outage; the cursor plus a wide
    # window bounds the query to genuinely-new rows instead.
    since = (last_ts - 1.0) if last_ts is not None else None
    fetch_limit = 2000

    # --- resolve messages from remote or local service -------------------
    # A fetch failure (bus unreachable, server down) must not dump a traceback
    # into a cron inbox, and must not touch the state file (so the cursor is
    # preserved and we do not re-emit the whole channel on the next run).
    server_url = getattr(args, "server", None)
    try:
        if server_url:
            # One-shot override: create a temporary RemoteClient.
            from .remote import RemoteClient  # noqa: PLC0415
            client = RemoteClient(server_url)

            async def _fetch(since):
                return await client.a2a_feed(thread=channel, since=since, limit=fetch_limit)

            messages = asyncio.run(_fetch(since))
        else:
            from . import service  # noqa: PLC0415

            async def _fetch_local(since):
                return await service.a2a_feed(thread=channel, since=since, limit=fetch_limit)

            messages = asyncio.run(_fetch_local(since))
    except Exception as exc:  # noqa: BLE001 - any fetch error: one line, no traceback
        print(
            f"taosmd a2a-poll: could not reach the bus "
            f"({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
        return 2

    if len(messages) >= fetch_limit:
        # The window overflowed; the feed returns the most recent N, so
        # messages between the cursor and the oldest returned row were
        # dropped. Say so instead of pretending the poll was complete.
        print(
            f"taosmd a2a-poll: WARNING fetched {fetch_limit} messages (window "
            f"full) — older unseen messages on {channel!r} may have been skipped",
            file=sys.stderr,
        )

    # --- filter to only new messages ------------------------------------
    exclude = getattr(args, "exclude", None)
    new_messages = []
    for msg in messages:
        msg_id = msg.get("id")
        # IDs from the archive are integers; coerce defensively.
        try:
            msg_id_int = int(msg_id)
        except (TypeError, ValueError):
            continue
        try:
            ts_val = float(msg.get("ts"))
            last_ts = ts_val if last_ts is None else max(last_ts, ts_val)
        except (TypeError, ValueError):
            pass
        if msg_id_int <= last_id:
            continue
        if exclude and msg.get("from") == exclude:
            # Skip messages from the excluded sender (usually "ourselves"),
            # but still advance last_id so we don't re-see them next poll.
            last_id = max(last_id, msg_id_int)
            continue
        new_messages.append((msg_id_int, msg))

    # --- print new messages and update state ----------------------------
    for msg_id_int, msg in sorted(new_messages, key=lambda t: t[0]):
        print(_format_a2a_line(msg))
        last_id = max(last_id, msg_id_int)

    # --- persist state (atomic: tmp write + os.replace) -----------------
    # An in-place write that crashes mid-flush leaves corrupt JSON, which the
    # next run treats as a reset (cursor -1) and re-emits the whole channel.
    # Write to a temp file in the same dir, then atomically rename over it.
    state[channel] = last_id
    if last_ts is not None:
        state.setdefault("_ts", {})[channel] = last_ts
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_file.with_name(state_file.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, state_file)
    return 0


def _install_skill_cmd(args: argparse.Namespace) -> int:
    """Handle ``taosmd install-skill``: copy the packaged skill into ~/.claude/skills/."""
    import shutil  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415
    from importlib.resources import files as _pkg_files  # noqa: PLC0415

    dest_dir = Path("~/.claude/skills/taosmd-a2a").expanduser()
    skill_src_dir = Path(__file__).parent / "skills" / "taosmd-a2a"

    if not skill_src_dir.is_dir():
        # Fallback: try importlib.resources (wheel installs)
        try:
            ref = _pkg_files("taosmd").joinpath("skills/taosmd-a2a")
            # Convert Traversable to a concrete path via __file__ approach.
            skill_src_dir = Path(__file__).parent / "skills" / "taosmd-a2a"
        except Exception:
            pass

    if not skill_src_dir.is_dir():
        print("error: packaged skill not found in taosmd/skills/taosmd-a2a/", file=sys.stderr)
        print("  Re-install the package to include skill assets.", file=sys.stderr)
        return 2

    force = getattr(args, "force", False)
    skill_md = dest_dir / "SKILL.md"
    if skill_md.exists() and not force:
        print(f"Skill already installed at {dest_dir}")
        print("  Re-run with --force to overwrite.")
        return 0

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(skill_src_dir), str(dest_dir), dirs_exist_ok=True)
    print(f"taosmd-a2a skill installed at {dest_dir}")
    return 0


def _setup_prompt_cmd(args: argparse.Namespace) -> int:
    import json  # noqa: PLC0415
    from . import recipes  # noqa: PLC0415
    from . import setup_prompt  # noqa: PLC0415

    if getattr(args, "device_info", None):
        try:
            with open(args.device_info, "r", encoding="utf-8") as fh:
                device_info = json.load(fh)
        except Exception as exc:
            print(f"error: could not read --device-info {args.device_info}: {exc}")
            return 1
    else:
        device_info = recipes.local_probe()

    print(setup_prompt.render_setup_prompt(device_info, getattr(args, "needs", None)))
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


def _fmt_task_line(t: dict) -> str:
    """Format a single task as one line of output."""
    extra = ""
    if t.get("assignee"):
        extra += f" @{t['assignee']}"
    if t.get("priority"):
        extra += f" p{t['priority']}"
    status = t.get("status", "open")
    return f"[{t['id']}] ({status}) {t['title']}{extra}"


def _tasks_cmd(args: argparse.Namespace) -> int:
    """Handle ``taosmd tasks`` subcommand group."""
    import asyncio  # noqa: PLC0415

    from . import service  # noqa: PLC0415

    data_dir = args.data_dir

    if args.tasks_cmd == "add":
        task = asyncio.run(
            service.task_create(
                args.title,
                body=args.body,
                project=args.project,
                assignee=args.assignee,
                priority=args.priority,
                depends_on=args.depends_on,
                created_by=args.created_by,
                data_dir=data_dir,
            )
        )
        print(_fmt_task_line(task))
        return 0

    if args.tasks_cmd == "list":
        tasks = asyncio.run(
            service.task_list(
                status=args.status,
                project=args.project,
                assignee=args.assignee,
                limit=args.limit,
                data_dir=data_dir,
            )
        )
        if not tasks:
            print("No tasks found.")
            return 0
        for t in tasks:
            print(_fmt_task_line(t))
        return 0

    if args.tasks_cmd == "ready":
        tasks = asyncio.run(
            service.task_ready(
                project=args.project,
                assignee=args.assignee,
                limit=args.limit,
                data_dir=data_dir,
            )
        )
        if not tasks:
            print("No ready tasks.")
            return 0
        for t in tasks:
            print(_fmt_task_line(t))
        return 0

    if args.tasks_cmd == "prime":
        result = asyncio.run(
            service.task_prime(
                project=args.project,
                assignee=args.assignee,
                data_dir=data_dir,
            )
        )
        print(result["text"])
        return 0

    if args.tasks_cmd == "start":
        opts: dict = {"status": "in_progress"}
        if getattr(args, "assignee", None):
            opts["assignee"] = args.assignee
        try:
            task = asyncio.run(
                service.task_update(args.task_id, data_dir=data_dir, **opts)
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(_fmt_task_line(task))
        return 0

    if args.tasks_cmd == "close":
        try:
            task = asyncio.run(
                service.task_update(args.task_id, status="closed", data_dir=data_dir)
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(_fmt_task_line(task))
        return 0

    if args.tasks_cmd == "block":
        try:
            edge = asyncio.run(
                service.task_add_edge(
                    args.blocker_id, args.task_id, "blocks", "cli",
                    data_dir=data_dir,
                )
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"edge: {edge['from_id']} --[blocks]--> {edge['to_id']}")
        return 0

    return 1


def _projects_cmd(args: argparse.Namespace) -> int:
    """Handle ``taosmd projects`` and ``taosmd shelves``: project-scoped discovery."""
    import asyncio  # noqa: PLC0415

    from . import service  # noqa: PLC0415

    data_dir = args.data_dir
    if args.cmd == "projects":
        projects = asyncio.run(service.list_projects(data_dir=data_dir))
        if not projects:
            print("No project-tagged memories found.")
            return 0
        for p in projects:
            agents = ", ".join(sorted(p.get("agents", [])))
            print(
                f"{p['project_id']:<14} agents=[{agents}]  "
                f"last_ingest={int(p.get('last_ingest') or 0)}"
            )
        return 0

    # shelves
    shelves = asyncio.run(service.list_shelves(project=args.project, data_dir=data_dir))
    if not shelves:
        print(f"No shelves found for project {args.project!r}.")
        return 0
    for s in shelves:
        print(
            f"{s['agent']:<24} facts={s['facts']}  "
            f"last_ingest={int(s.get('last_ingest') or 0)}"
        )
    return 0


def _reconcile_cmd(args: argparse.Namespace) -> int:
    """Handle ``taosmd reconcile``: compare archive to vector store and repair gaps."""
    import asyncio  # noqa: PLC0415

    from . import service  # noqa: PLC0415
    from .agents import AgentRegistry  # noqa: PLC0415

    repair = not args.check
    data_dir = args.data_dir

    agent_names: list[str]
    if args.agent:
        agent_names = [args.agent]
    else:
        # Reconcile all registered agents.
        registry = AgentRegistry(data_dir)
        agent_names = [a["name"] for a in registry.list_agents()]
        if not agent_names:
            print("No registered agents found.")
            return 0

    any_missing = False
    for agent in agent_names:
        result = asyncio.run(service.reconcile(agent=agent, data_dir=data_dir, repair=repair))
        mode = "check" if not repair else "repair"
        status = "ok" if result["checked_ok"] else "MISSING"
        readded_col = f"  re-added={result['readded']}" if repair else ""
        print(
            f"[{mode}] {result['agent']:<24} "
            f"archive={result['archive_turns']}  "
            f"vector={result['vector_entries']}  "
            f"missing={result['missing']}  "
            f"{status}"
            f"{readded_col}"
        )
        if not result["checked_ok"]:
            any_missing = True

    if any_missing and not repair:
        print(
            "\nhint: run without --check to repair, "
            "or `taosmd reconcile` to re-add missing entries."
        )
    return 0


def _reindex_cmd(args: argparse.Namespace) -> int:
    """Handle ``taosmd reindex``: rebuild an agent's vector store from the archive.

    Clears the agent's vector rows and re-adds every archive turn, which
    re-embeds each under the CURRENTLY configured embedder. Used to cut an agent
    over to a new embedder (e.g. arctic-embed-s); the zero-loss archive is the
    source of truth so reindex is safe to re-run.
    """
    import asyncio  # noqa: PLC0415

    from . import service  # noqa: PLC0415
    from .agents import AgentRegistry  # noqa: PLC0415

    check = args.check
    data_dir = args.data_dir

    print(
        "note: reindex clears the agent's vector rows and rebuilds them from the "
        "zero-loss archive under the CURRENTLY configured embedder. Set "
        "vector_memory.embed_model to the target embedder (e.g. arctic-embed-s) "
        "BEFORE running."
    )

    agent_names: list[str]
    if args.agent:
        agent_names = [args.agent]
    else:
        # Reindex all registered agents.
        registry = AgentRegistry(data_dir)
        agent_names = [a["name"] for a in registry.list_agents()]
        if not agent_names:
            print("No registered agents found.")
            return 0

    any_mismatch = False
    for agent in agent_names:
        result = asyncio.run(service.reindex(agent=agent, data_dir=data_dir, check=check))
        mode = "check" if check else "reindex"
        status = "ok" if result["reindexed_ok"] else "MISMATCH"
        print(
            f"[{mode}] {result['agent']:<24} "
            f"archive={result['archive_turns']}  "
            f"vector={result['vector_before']}  "
            f"cleared={result['cleared']}  "
            f"re-added={result['readded']}  "
            f"{status}"
        )
        if not result["reindexed_ok"]:
            any_mismatch = True

    if check:
        print("\nhint: run without --check to clear and rebuild the vector store.")
    return 1 if any_mismatch and not check else 0


def _claims_cmd(args: argparse.Namespace) -> int:
    """Handle ``taosmd claims`` subcommand group."""
    import asyncio  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from .claims.store import ClaimStore  # noqa: PLC0415

    if args.claims_cmd == "status":
        store = ClaimStore(db_path=Path(args.data_dir) / "claims.db")
        rate = asyncio.run(_claims_rate(store))
        checked = sum(
            rate.get(s, 0) for s in ("supported", "partial", "unsupported", "contradicted")
        )
        hall_rate = rate.get("hallucination_rate", 0.0)
        print(
            f"claims: unverified={rate.get('unverified', 0)}"
            f"  supported={rate.get('supported', 0)}"
            f"  partial={rate.get('partial', 0)}"
            f"  unsupported={rate.get('unsupported', 0)}"
            f"  contradicted={rate.get('contradicted', 0)}"
            f"  checked={checked}"
            f"  hallucination_rate={hall_rate:.3f}"
        )
        return 0

    return 1


async def _claims_rate(store) -> dict:
    await store.init()
    try:
        return await store.rate()
    finally:
        await store.close()


def _verify_cmd(args: argparse.Namespace) -> int:
    """Handle ``taosmd verify``: run a verification pass over unverified claims."""
    import asyncio  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import httpx  # noqa: PLC0415

    from .archive import ArchiveStore  # noqa: PLC0415
    from .claims.store import ClaimStore  # noqa: PLC0415
    from .claims.verifier import LocalEntailmentVerifier  # noqa: PLC0415
    from .claims.verify_pass import verify_pass  # noqa: PLC0415

    data_dir = Path(args.data_dir)

    async def _run() -> int:
        store = ClaimStore(db_path=data_dir / "claims.db")
        await store.init()
        archive = ArchiveStore(
            archive_dir=data_dir / "archive",
            index_path=data_dir / "archive-index.db",
        )
        await archive.init()
        try:
            fetch_spans = await _fetch_spans(archive)
            with httpx.Client() as client:
                verifier = LocalEntailmentVerifier(
                    client=client,
                    ollama_url=args.ollama_url,
                    model=args.model,
                )
                n = await verify_pass(store, verifier, fetch_spans, batch=args.batch)
            rate = await store.rate()
        finally:
            await store.close()
            await archive.close()
        hall_rate = rate.get("hallucination_rate", 0.0)
        print(
            f"verified {n} claims;"
            f" rate: hallucination_rate={hall_rate:.3f}"
            f"  supported={rate.get('supported', 0)}"
            f"  partial={rate.get('partial', 0)}"
            f"  unsupported={rate.get('unsupported', 0)}"
            f"  contradicted={rate.get('contradicted', 0)}"
            f"  unverified={rate.get('unverified', 0)}"
        )
        return 0

    return asyncio.run(_run())


async def _fetch_spans(archive):
    async def fetch(span_ids: list[int]) -> list[str]:
        out = []
        for sid in span_ids:
            ev = await archive.get_event(sid)
            if not ev:
                continue
            data = ev.get("data") or {}
            txt = (data.get("content") if isinstance(data, dict) else None) or ev.get("summary") or ""
            if txt:
                out.append(txt)
        return out
    return fetch


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser.

    Extracted from ``main()`` so tests can parse args without executing handlers.
    ``main()`` calls this then dispatches based on the parsed result.
    """
    parser = argparse.ArgumentParser(prog="taosmd")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to the taosmd data directory (default: ./data)",
    )
    parser.add_argument(
        "--server",
        default=None,
        metavar="URL",
        help="Remote taOSmd server URL (overrides TAOSMD_SERVER_URL and config.json for this invocation)",
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

    # ----- generator-profile subcommand ---------------------------------
    gp_p = sub.add_parser("generator-profile", help="select the answer generator by workload")
    gp_sub = gp_p.add_subparsers(dest="generator_profile_cmd", required=True)
    gp_sub.add_parser("list", help="list profiles and mark the active one")
    gp_show = gp_sub.add_parser("show", help="show a profile's per-tier models")
    gp_show.add_argument("profile_id")
    gp_set = gp_sub.add_parser("set", help="set the active profile")
    gp_set.add_argument("profile_id")
    gp_set.add_argument("--agent", default=None, help="set per-agent instead of global")

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

    # ----- config subcommand (server URL + token + show) ----------------
    cfg_p = sub.add_parser(
        "config",
        help="Get/set connection config (remote server URL, bearer token, memory model)",
    )
    cfg_sub = cfg_p.add_subparsers(dest="config_cmd", required=True)

    # config set-server
    ss_p = cfg_sub.add_parser("set-server", help="Set or clear the remote server URL")
    ss_p.add_argument(
        "url", nargs="?", default=None,
        help="Base URL of the remote taOSmd server, e.g. http://pi.local:7900",
    )
    ss_p.add_argument("--clear", action="store_true", help="Unset the server URL (revert to local mode)")

    # config set-token
    st_p = cfg_sub.add_parser("set-token", help="Set or clear the remote server bearer token")
    st_p.add_argument(
        "token", nargs="?", default=None,
        help="Bearer token for the remote server. Stored in config.json; "
             "use TAOSMD_TOKEN env var to avoid on-disk storage.",
    )
    st_p.add_argument("--clear", action="store_true", help="Unset the token")

    # config show
    cfg_sub.add_parser("show", help="Print the resolved server_url and whether a token is set")

    # ----- install-skill subcommand ------------------------------------
    install_skill_p = sub.add_parser(
        "install-skill",
        help="Copy the taosmd-a2a Claude skill into ~/.claude/skills/taosmd-a2a/",
    )
    install_skill_p.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing installation",
    )

    # ----- setup-prompt subcommand ------------------------------------
    setup_prompt_p = sub.add_parser(
        "setup-prompt",
        help="Print a hardware-tailored agent prompt for installing taOSmd",
    )
    setup_prompt_p.add_argument(
        "--device-info", metavar="FILE", default=None,
        help="Read device_info JSON from FILE instead of probing (for tests / "
             "generating a prompt for a different machine)",
    )
    setup_prompt_p.add_argument(
        "--needs", default=None,
        help="A short note on the user's stated needs (e.g. 'audit trail'); "
             "leans the recommended profile",
    )

    # ----- a2a-poll subcommand ----------------------------------------
    a2a_poll_p = sub.add_parser(
        "a2a-poll",
        help="Fetch new A2A messages since the last poll (cron-friendly, updates state file)",
    )
    a2a_poll_p.add_argument(
        "--channel", required=True,
        help="Channel name to poll (e.g. the project channel name)",
    )
    a2a_poll_p.add_argument(
        "--server", default=None,
        help="Override the remote server URL for this poll (e.g. http://pi.local:7900). "
             "Defaults to TAOSMD_SERVER_URL or the configured server_url.",
    )
    a2a_poll_p.add_argument(
        "--state-file",
        dest="state_file",
        default="~/.taosmd/a2a-poll-state.json",
        help="JSON file that stores the last-seen message ID per channel "
             "(default: ~/.taosmd/a2a-poll-state.json)",
    )
    a2a_poll_p.add_argument(
        "--exclude", default=None,
        help="Skip messages from this sender (e.g. your own agent name)",
    )

    # ----- a2a-watch subcommand (realtime SSE stream) -------------------
    a2a_watch_p = sub.add_parser(
        "a2a-watch",
        help="Stream new A2A messages in realtime over SSE (one line each); "
             "wrap in a Monitor for instant in-session pickup",
    )
    a2a_watch_p.add_argument(
        "--channel", default=None,
        help="Channel name to watch. Omit (or pass 'all') to stream EVERY "
             "channel over one connection, auto-including new/renamed ones; "
             "each line is then prefixed with its (thread).",
    )
    a2a_watch_p.add_argument(
        "--server", default=None,
        help="Bus server URL (default: TAOSMD_SERVER_URL, configured server_url, "
             "else http://127.0.0.1:7900). Requires a running `taosmd serve`.",
    )
    a2a_watch_p.add_argument(
        "--exclude", default=None,
        help="Skip messages from this sender (e.g. your own agent name)",
    )
    a2a_watch_p.add_argument(
        "--count", type=int, default=0,
        help="Exit after printing N new messages (default 0 = run forever)",
    )

    # ----- a2a-bridge subcommand (exec-on-message dormant wake) ---------
    a2a_bridge_p = sub.add_parser(
        "a2a-bridge",
        help="Run a trigger command on each new A2A message (the message JSON "
             "is piped to the command's stdin); wakes a dormant local session",
    )
    a2a_bridge_p.add_argument(
        "--channel", default=None,
        help="Channel name to bridge. Omit (or pass 'all') to fire on messages "
             "from EVERY channel over one connection, auto-including new ones.",
    )
    a2a_bridge_p.add_argument(
        "--trigger", required=True,
        help="Shell command to run per new message; the message JSON is written "
             "to its stdin (a coalesced batch is a JSON array)",
    )
    a2a_bridge_p.add_argument(
        "--server", default=None,
        help="Bus server URL (default: TAOSMD_SERVER_URL, configured server_url, "
             "else http://127.0.0.1:7900). Requires a running `taosmd serve`.",
    )
    a2a_bridge_p.add_argument(
        "--exclude", default=None,
        help="Skip messages from this sender (e.g. your own agent name)",
    )
    a2a_bridge_p.add_argument(
        "--debounce", type=float, default=0.0,
        help="Coalesce a burst: messages within S seconds of the last spawn are "
             "batched into the next trigger run (default 0 = no debounce)",
    )
    a2a_bridge_p.add_argument(
        "--max-concurrency", dest="max_concurrency", type=int, default=1,
        help="Max simultaneous trigger processes (default 1)",
    )
    a2a_bridge_p.add_argument(
        "--count", type=int, default=0,
        help="Exit after firing the trigger N times (default 0 = run forever)",
    )

    # ----- serve subcommand (local HTTP/REST API) -----------------------
    serve_p = sub.add_parser(
        "serve",
        help="Run the local HTTP/REST memory API (stdlib server, zero deps)",
    )
    serve_p.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default 127.0.0.1, localhost-only). "
             "Use 0.0.0.0 to expose on the LAN, no auth, so gate it yourself.",
    )
    serve_p.add_argument(
        "--port", type=int, default=7900,
        help="Bind port (default 7900)",
    )
    serve_p.add_argument(
        "--serve-data-dir", dest="serve_data_dir", default=None,
        help="Data dir for served memory (default: $TAOSMD_DATA_DIR or ~/.taosmd)",
    )
    _svc = serve_p.add_mutually_exclusive_group()
    _svc.add_argument(
        "--install-service", action="store_true",
        help="Install taosmd serve as a persistent background service "
             "(systemd user unit on Linux, LaunchAgent on macOS, "
             "PowerShell instructions on Windows) and exit.",
    )
    _svc.add_argument(
        "--uninstall-service", action="store_true",
        help="Stop and remove the background service installed by --install-service.",
    )
    _svc.add_argument(
        "--service-status", action="store_true",
        help="Print the running status of the background service.",
    )

    # ----- reconcile subcommand ----------------------------------------
    reconcile_p = sub.add_parser(
        "reconcile",
        help="Detect (and repair) archive turns missing from the vector store "
             "due to a crash between the two sequential writes in ingest(). "
             "The archive is the source of truth; this re-adds any absent entries "
             "without resurrecting superseded/corrected content.",
    )
    reconcile_p.add_argument(
        "--agent", default=None,
        help="Agent name to reconcile. When omitted, all registered agents are reconciled.",
    )
    reconcile_p.add_argument(
        "--check", action="store_true",
        help="Dry-run: report missing counts without modifying the vector store.",
    )

    # ----- reindex subcommand ------------------------------------------
    reindex_p = sub.add_parser(
        "reindex",
        help="Re-embed an agent's vector store from the zero-loss archive under "
             "the CURRENTLY configured embedder. Used to cut an agent over to a "
             "new embedder (e.g. arctic-embed-s): clears the agent's vector rows "
             "and re-adds every archive turn. The archive is never touched, so "
             "reindex is per-agent and safe to re-run.",
    )
    reindex_p.add_argument(
        "--agent", default=None,
        help="Agent name to reindex. When omitted, all registered agents are reindexed.",
    )
    reindex_p.add_argument(
        "--check", action="store_true",
        help="Dry-run: report archive/vector counts without modifying the vector store.",
    )

    # ----- project discovery subcommands --------------------------------
    sub.add_parser(
        "projects",
        help="List projects that have stored memories (project id, agents, last activity)",
    )
    shelves_p = sub.add_parser(
        "shelves",
        help="List the agent shelves that have memories within a project",
    )
    shelves_p.add_argument(
        "--project", required=True,
        help="Project id (from `taosmd projects` or taosmd.get_project_id())",
    )

    # ----- tasks subcommand group ---------------------------------------
    tasks_p = sub.add_parser(
        "tasks",
        help="Dependency-aware task graph: add, list, ready queue, prime briefing",
    )
    tasks_sub = tasks_p.add_subparsers(dest="tasks_cmd", required=True)

    # tasks add
    t_add_p = tasks_sub.add_parser("add", help="Create a new task")
    t_add_p.add_argument("--title", required=True, help="Task title")
    t_add_p.add_argument("--body", default=None, help="Task description / body")
    t_add_p.add_argument("--project", default=None, help="Project id")
    t_add_p.add_argument("--assignee", default=None, help="Agent handle to assign")
    t_add_p.add_argument("--priority", type=int, default=0, help="Priority (higher = more urgent, default 0)")
    t_add_p.add_argument(
        "--depends-on", dest="depends_on", action="append", default=None,
        metavar="TASK_ID",
        help="Task IDs that must close before this task is ready. Repeat for multiple.",
    )
    t_add_p.add_argument("--by", dest="created_by", default="cli", help="Creator handle (default: cli)")

    # tasks list
    t_list_p = tasks_sub.add_parser("list", help="List tasks with optional filters")
    t_list_p.add_argument("--status", default=None, help="Filter by status (open|in_progress|blocked|closed|superseded)")
    t_list_p.add_argument("--project", default=None, help="Filter by project id")
    t_list_p.add_argument("--assignee", default=None, help="Filter by assignee")
    t_list_p.add_argument("--limit", type=int, default=50, help="Max results (default 50)")

    # tasks ready
    t_ready_p = tasks_sub.add_parser("ready", help="Show tasks that are ready to run (no active blockers)")
    t_ready_p.add_argument("--project", default=None, help="Filter by project id")
    t_ready_p.add_argument("--assignee", default=None, help="Filter by assignee")
    t_ready_p.add_argument("--limit", type=int, default=20, help="Max results (default 20)")

    # tasks prime
    t_prime_p = tasks_sub.add_parser("prime", help="Print the session-bootstrap briefing")
    t_prime_p.add_argument("--project", default=None, help="Filter by project id")
    t_prime_p.add_argument("--assignee", default=None, help="Filter by assignee")

    # tasks start (update status to in_progress)
    t_start_p = tasks_sub.add_parser("start", help="Mark a task as in progress")
    t_start_p.add_argument("task_id", help="Task ID")
    t_start_p.add_argument("--assignee", default=None, help="Assign to this agent handle")

    # tasks close (update status to closed)
    t_close_p = tasks_sub.add_parser("close", help="Mark a task as closed")
    t_close_p.add_argument("task_id", help="Task ID")

    # tasks block (add a blocks edge: blocker -> blocked)
    t_block_p = tasks_sub.add_parser("block", help="Mark one task as blocked by another")
    t_block_p.add_argument("task_id", help="The task to mark as blocked")
    t_block_p.add_argument("--by", dest="blocker_id", required=True, metavar="BLOCKER_ID",
                            help="The task that is blocking it")

    # ----- mcp subcommand (MCP server over stdio) -----------------------
    mcp_p = sub.add_parser(
        "mcp",
        help="Run the MCP memory server over stdio (requires taosmd[mcp])",
    )
    mcp_p.add_argument(
        "--mcp-data-dir", dest="mcp_data_dir", default=None,
        help="Data dir for served memory (default: $TAOSMD_DATA_DIR or ~/.taosmd)",
    )

    # ----- claims subcommand group -------------------------------------
    claims_p = sub.add_parser(
        "claims",
        help="Inspect and manage verifiable claims (hallucination tracking)",
    )
    claims_sub = claims_p.add_subparsers(dest="claims_cmd", required=True)

    claims_sub.add_parser("status", help="Print the live claim rate (hallucination_rate and counts)")

    # ----- verify subcommand ------------------------------------------
    verify_p = sub.add_parser(
        "verify",
        help="Run a verification pass over unverified claims via a local Ollama model",
    )
    verify_p.add_argument(
        "--model", default="qwen3:4b-instruct-2507",
        help="Ollama model to use as verifier (default: qwen3:4b-instruct-2507)",
    )
    verify_p.add_argument(
        "--ollama-url", dest="ollama_url", default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    verify_p.add_argument(
        "--batch", type=int, default=100,
        help="Claims to pull per batch (default: 100)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "config":
        if args.config_cmd == "set-server":
            return _config_set_server(args.url, args.clear)
        if args.config_cmd == "set-token":
            return _config_set_token(args.token, args.clear)
        if args.config_cmd == "show":
            return _config_show()

    if args.cmd == "install-skill":
        return _install_skill_cmd(args)

    if args.cmd == "setup-prompt":
        return _setup_prompt_cmd(args)

    if args.cmd == "a2a-poll":
        return _a2a_poll_cmd(args)

    if args.cmd == "a2a-watch":
        return _a2a_watch_cmd(args)

    if args.cmd == "a2a-bridge":
        return _a2a_bridge_cmd(args)

    if args.cmd == "serve":
        from . import service_install  # noqa: PLC0415
        if args.install_service:
            return service_install.install_service(
                host=args.host, port=args.port, data_dir=args.serve_data_dir,
            )
        if args.uninstall_service:
            return service_install.uninstall_service()
        if args.service_status:
            return service_install.service_status()
        from . import http_server  # noqa: PLC0415
        return http_server.serve(
            host=args.host, port=args.port, data_dir=args.serve_data_dir,
        )

    if args.cmd == "tasks":
        return _tasks_cmd(args)

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

    if args.cmd == "reconcile":
        return _reconcile_cmd(args)

    if args.cmd == "reindex":
        return _reindex_cmd(args)

    if args.cmd in ("projects", "shelves"):
        return _projects_cmd(args)

    if args.cmd == "claims":
        return _claims_cmd(args)

    if args.cmd == "verify":
        return _verify_cmd(args)

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

    if args.cmd == "generator-profile":
        if args.generator_profile_cmd == "list":
            return _generator_profile_list(data_dir=args.data_dir)
        if args.generator_profile_cmd == "show":
            return _generator_profile_show(args.profile_id, data_dir=args.data_dir)
        if args.generator_profile_cmd == "set":
            return _generator_profile_set(args.profile_id, agent=args.agent, data_dir=args.data_dir)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
