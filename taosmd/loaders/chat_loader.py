"""ChatLoader — reads chat-shaped JSON files into a ``ChatBlob``.

Accepted shapes:

  Shape A — list of message dicts (matches OpenAI / Anthropic message
  arrays and most chat-export tools):
    [
      {"role": "user", "content": "hi", "timestamp": 1716000000.0},
      {"role": "assistant", "content": "hello", "alias": "Alex"}
    ]

  Shape B — object with a ``messages`` key (matches several agent
  frameworks that include extra session metadata at the top level):
    {"messages": [...], "agent": "alex", "session_id": "..."}

  Each message can include any of ``role``, ``content``, ``alias``,
  ``timestamp`` — missing fields default to ``""`` / ``0.0``.

Anything else falls through to ``DocLoader`` via the registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from ._safety import DEFAULT_MAX_BYTES, check_size, resolve_within
from .blob import ChatBlob, ChatMessage
from .interface import LoaderInterface


class ChatLoader(LoaderInterface):
    loader_name: ClassVar[str] = "chat"
    supported_extensions: ClassVar[tuple[str, ...]] = ("chat.json", "messages.json")
    supported_mime_types: ClassVar[tuple[str, ...]] = ("application/x-chat+json",)

    @classmethod
    def can_handle(cls, extension: str = "", mime_type: str = "") -> bool:
        """Same as base + match the multi-suffix forms ``*.chat.json`` /
        ``*.messages.json`` even when the path's ``Path.suffix`` only
        captures ``.json``."""
        if super().can_handle(extension=extension, mime_type=mime_type):
            return True
        ext = extension.lower().lstrip(".") if extension else ""
        return ext.endswith("chat.json") or ext.endswith("messages.json")

    async def load(
        self,
        file_path: str | Path,
        *,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
        base_dir: str | Path | None = None,
        **kwargs,
    ) -> ChatBlob:
        path = Path(file_path)
        safe_path = resolve_within(file_path, base_dir)
        check_size(safe_path, max_bytes)
        with open(safe_path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "messages" in data:
            raw_messages = data["messages"]
        elif isinstance(data, list):
            raw_messages = data
        else:
            raise ValueError(
                f"ChatLoader: {path} doesn't match chat shape "
                "(expected list of messages or {messages: [...]})"
            )

        messages: list[ChatMessage] = []
        for m in raw_messages:
            if not isinstance(m, dict):
                continue
            messages.append(ChatMessage(
                role=str(m.get("role", "")),
                content=str(m.get("content", "")),
                alias=str(m.get("alias", "")),
                timestamp=float(m.get("timestamp", 0.0) or 0.0),
            ))

        raw_text = "\n".join(
            f"[{m.alias or m.role}] {m.content}" for m in messages
        )
        return ChatBlob(
            source_path=str(path),
            raw_text=raw_text,
            messages=messages,
        )
