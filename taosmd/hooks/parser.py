import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

_MAX_FIELD_BYTES = 8 * 1024

_NOISE_TYPES = frozenset(
    {
        "queue-operation",
        "attachment",
        "system",
        "file-history-snapshot",
        "last-prompt",
        "ai-title",
        "cost-state",
    }
)


def _cap_field(value: str) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) > _MAX_FIELD_BYTES:
        return encoded[:_MAX_FIELD_BYTES].decode("utf-8", "ignore"), True
    return value, False


def _iso_to_epoch(ts_str: str) -> float:
    ts_str = ts_str.strip()
    if not ts_str:
        return time.time()
    ts_str = ts_str.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.timestamp()
    except (ValueError, TypeError):
        return time.time()


def _fallback_id(byte_offset: int, raw_line: str) -> str:
    key = f"{byte_offset}{raw_line}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def parse_entry(
    byte_offset: int, raw_line: str, session_id: str
) -> tuple[list[dict] | None, str | None]:
    stripped = raw_line.strip()
    if not stripped:
        return None, None

    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return None, None

    if not isinstance(obj, dict):
        return None, None

    entry_type = obj.get("type")
    if entry_type is None:
        return None, None

    if entry_type in _NOISE_TYPES:
        return None, None

    session_id_from = obj.get("sessionId") or session_id
    uuid = obj.get("uuid", "") or ""
    timestamp_str = obj.get("timestamp", "") or ""
    ts_epoch = _iso_to_epoch(str(timestamp_str))

    if uuid:
        base_id = f"claude-code:{session_id_from}:{uuid}"
    else:
        base_id = _fallback_id(byte_offset, stripped)

    items: list[dict[str, Any]] = []

    if entry_type == "user":
        message = obj.get("message", {})
        if not isinstance(message, dict):
            return None, None
        content = message.get("content", "")

        if isinstance(content, list):
            text_parts: list[str] = []
            tr_blocks: list[dict] = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", "") or "")
                elif btype == "tool_result":
                    tr_blocks.append(block)

            if tr_blocks:
                for idx, tr_block in enumerate(tr_blocks):
                    tr_content = tr_block.get("content", "") or ""
                    if isinstance(tr_content, list):
                        text_parts_tr: list[str] = []
                        for sub in tr_content:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                text_parts_tr.append(sub.get("text", "") or "")
                        tr_text = "\n".join(text_parts_tr)
                    elif isinstance(tr_content, str):
                        tr_text = tr_content
                    else:
                        tr_text = str(tr_content)
                    tr_text, truncated = _cap_field(tr_text)
                    item_id = f"{base_id}:tool_result:{idx}"
                    items.append(
                        {
                            "id": item_id,
                            "text": tr_text,
                            "metadata": {
                                "source": "hook:claude-code",
                                "session_id": session_id_from,
                                "role": "user",
                                "kind": "tool_result",
                                "timestamp": ts_epoch,
                                "transcript_path": "",
                                "truncated": truncated,
                            },
                        }
                    )
                return items, base_id

            if text_parts:
                text = "\n".join(text_parts)
                if text:
                    text, truncated = _cap_field(text)
                    items.append(
                        {
                            "id": base_id,
                            "text": text,
                            "metadata": {
                                "source": "hook:claude-code",
                                "session_id": session_id_from,
                                "role": "user",
                                "kind": "message",
                                "timestamp": ts_epoch,
                                "transcript_path": "",
                                "truncated": truncated,
                            },
                        }
                    )
                    return items, base_id
            return None, None

        if isinstance(content, str):
            text, truncated = _cap_field(content)
            if text:
                items.append(
                    {
                        "id": base_id,
                        "text": text,
                        "metadata": {
                            "source": "hook:claude-code",
                            "session_id": session_id_from,
                            "role": "user",
                            "kind": "message",
                            "timestamp": ts_epoch,
                            "transcript_path": "",
                            "truncated": truncated,
                        },
                    }
                )
                return items, base_id
            return None, None

        return None, None

    if entry_type == "assistant":
        message = obj.get("message", {})
        if not isinstance(message, dict):
            return None, None
        content = message.get("content", [])

        if not isinstance(content, list):
            return None, None

        text_parts = []
        tu_blocks: list[dict] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", "") or "")
            elif btype == "thinking":
                continue
            elif btype == "tool_use":
                tu_blocks.append(block)

        if text_parts:
            text = "\n".join(text_parts)
            if text:
                text, truncated = _cap_field(text)
                items.append(
                    {
                        "id": base_id,
                        "text": text,
                        "metadata": {
                            "source": "hook:claude-code",
                            "session_id": session_id_from,
                            "role": "assistant",
                            "kind": "message",
                            "timestamp": ts_epoch,
                            "transcript_path": "",
                            "truncated": truncated,
                        },
                    }
                )

        for idx, tub in enumerate(tu_blocks):
            tub_input = tub.get("input", {})
            if isinstance(tub_input, dict):
                tub_text = json.dumps(tub_input, separators=(",", ":"))
            else:
                tub_text = str(tub_input)
            tub_text, truncated = _cap_field(tub_text)
            item_id = f"{base_id}:tool_use:{idx}"
            items.append(
                {
                    "id": item_id,
                    "text": tub_text,
                    "metadata": {
                        "source": "hook:claude-code",
                        "session_id": session_id_from,
                        "role": "assistant",
                        "kind": "tool_use",
                        "timestamp": ts_epoch,
                        "transcript_path": "",
                        "truncated": truncated,
                        "tool_name": tub.get("name", "") or "",
                        "tool_id": tub.get("id", "") or "",
                    },
                }
            )

        return items if items else None, base_id

    return None, None
