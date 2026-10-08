import asyncio
import logging
from pathlib import Path

from taosmd import project as project_mod
from taosmd import service
from taosmd.hooks import cursors, parser

logger = logging.getLogger(__name__)


async def sync_session(
    data_dir: str,
    session_id: str,
    transcript_path: str,
    cwd: str,
    agent: str = "claude-code",
    timeout: int = 5,
) -> dict:
    store = cursors.CursorStore(str(Path(data_dir) / "capture-cursors.db"))
    await store.init()
    try:
        return await _do_sync(
            data_dir, session_id, transcript_path, cwd, agent, store, timeout
        )
    finally:
        await store.close()


async def _do_sync(
    data_dir: str,
    session_id: str,
    transcript_path: str,
    cwd: str,
    agent: str,
    store: cursors.CursorStore,
    timeout: int,
) -> dict:
    cursor = await store.get(session_id)
    old_offset = cursor["byte_offset"] if cursor else 0

    path = Path(transcript_path)
    if not path.exists():
        return {"ok": True, "session_id": session_id, "ingested": 0, "skipped": 0}

    file_bytes = path.read_bytes()
    file_size = len(file_bytes)

    if file_size < old_offset:
        old_offset = 0

    raw_bytes = file_bytes[old_offset:]
    if not raw_bytes:
        await store.update(session_id, transcript_path, old_offset, None)
        return {"ok": True, "session_id": session_id, "ingested": 0, "skipped": 0}

    chunks = raw_bytes.split(b"\n")
    complete_chunks = chunks[:-1] if chunks else []

    bytes_consumed = sum(len(c) + 1 for c in complete_chunks)
    new_offset = old_offset + bytes_consumed

    all_items: list[dict] = []
    skipped = 0
    last_entry_id: str | None = None
    current_offset = old_offset

    for chunk in complete_chunks:
        raw_line = chunk.decode("utf-8", errors="ignore")
        line_offset = current_offset

        if not raw_line.strip():
            current_offset += len(chunk) + 1
            continue

        items, base_id = parser.parse_entry(line_offset, raw_line, session_id)
        if items is None:
            skipped += 1
            current_offset += len(chunk) + 1
            continue

        for item in items:
            all_items.append(item)
            last_entry_id = item["id"]

        current_offset += len(chunk) + 1

    if not all_items:
        await store.update(session_id, transcript_path, new_offset, last_entry_id)
        return {
            "ok": True,
            "session_id": session_id,
            "ingested": 0,
            "skipped": skipped,
        }

    project_id = project_mod.get_project_id(cwd=cwd)

    for item in all_items:
        item["metadata"]["transcript_path"] = transcript_path

    try:
        result = await asyncio.wait_for(
            service.ingest_batch(
                all_items,
                agent=agent,
                data_dir=data_dir,
                project=project_id,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.error(
            "hooks sync: timeout session_id=%s error_type=TimeoutError",
            session_id,
        )
        await service.reconcile(agent=agent, data_dir=data_dir, repair=True)
        return {
            "ok": False,
            "timed_out": True,
            "session_id": session_id,
            "ingested": 0,
            "skipped": skipped,
        }
    except Exception as exc:
        logger.error(
            "hooks sync: write failed session_id=%s error_type=%s",
            session_id,
            type(exc).__name__,
        )
        return {
            "ok": False,
            "error": type(exc).__name__,
            "session_id": session_id,
            "ingested": 0,
            "skipped": skipped,
        }

    await store.update(session_id, transcript_path, new_offset, last_entry_id)
    return {
        "ok": True,
        "session_id": session_id,
        "ingested": result.get("ingested", 0),
        "skipped": skipped + result.get("skipped", 0),
    }


async def sync_all(data_dir: str, agent: str = "claude-code", timeout: int = 5) -> dict:
    store = cursors.CursorStore(str(Path(data_dir) / "capture-cursors.db"))
    await store.init()
    sessions = await store.all_sessions()
    total_ingested = 0
    total_skipped = 0
    any_failed = False

    for session in sessions:
        result = await _do_sync(
            data_dir,
            session["session_id"],
            session["transcript_path"],
            "",
            agent,
            store,
            timeout,
        )
        if not result.get("ok", False):
            any_failed = True
        total_ingested += result.get("ingested", 0)
        total_skipped += result.get("skipped", 0)

    await store.close()
    return {
        "ok": not any_failed,
        "ingested": total_ingested,
        "skipped": total_skipped,
    }
