"""EmailLoader — reads RFC 5322 .eml into an ``EmailBlob``.

Uses the stdlib ``email`` module so we don't pull a parser dep. Handles
both ``.eml`` and ``.mbox`` files; for mbox, the first message in the
file is used (multi-message mbox would need a different envelope — a
``MailboxBlob`` containing many ``EmailBlob``s; defer until we have a
real caller).

Threading via ``Message-Id`` + ``In-Reply-To`` headers — both surfaced
on the blob so downstream extractors can rebuild thread trees later.
"""

from __future__ import annotations

import email
from email import policy
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path
from typing import ClassVar

from ._safety import DEFAULT_MAX_BYTES, check_size, resolve_within
from .blob import EmailBlob
from .interface import LoaderInterface


class EmailLoader(LoaderInterface):
    loader_name: ClassVar[str] = "email"
    supported_extensions: ClassVar[tuple[str, ...]] = ("eml", "mbox")
    supported_mime_types: ClassVar[tuple[str, ...]] = (
        "message/rfc822", "application/mbox",
    )

    async def load(
        self,
        file_path: str | Path,
        *,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
        base_dir: str | Path | None = None,
        **kwargs,
    ) -> EmailBlob:
        path = Path(file_path)
        safe_path = resolve_within(file_path, base_dir)
        check_size(safe_path, max_bytes)
        with open(safe_path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)

        # body — prefer text/plain part, fall back to flat string.
        if msg.is_multipart():
            body = ""
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body = part.get_content().strip()
                    except (AttributeError, LookupError):
                        body = ""
                    if body:
                        break
        else:
            try:
                body = msg.get_content().strip()
            except (AttributeError, LookupError):
                body = ""

        recipients = [
            addr for _, addr in getaddresses(msg.get_all("To", []))
            if addr
        ]
        sent_at = 0.0
        date_hdr = msg.get("Date", "")
        if date_hdr:
            try:
                sent_at = parsedate_to_datetime(date_hdr).timestamp()
            except (TypeError, ValueError):
                sent_at = 0.0

        sender_raw = msg.get("From", "")
        sender_addrs = getaddresses([sender_raw]) if sender_raw else []
        sender = sender_addrs[0][1] if sender_addrs else sender_raw

        subject = msg.get("Subject", "")
        message_id = (msg.get("Message-Id", "") or "").strip("<>")
        in_reply_to = (msg.get("In-Reply-To", "") or "").strip("<>")

        raw_text = (
            f"From: {sender}\nTo: {', '.join(recipients)}\n"
            f"Subject: {subject}\n\n{body}"
        )
        return EmailBlob(
            source_path=str(path),
            raw_text=raw_text,
            sender=sender,
            recipients=recipients,
            subject=subject,
            body=body,
            sent_at=sent_at,
            message_id=message_id,
            in_reply_to=in_reply_to,
        )
