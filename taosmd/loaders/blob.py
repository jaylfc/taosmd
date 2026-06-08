"""Typed blob envelopes: what loaders produce, what extractors consume.

The discriminated union (``Blob.kind`` is one of ``BlobType`` enum values)
lets downstream consumers ``match`` on type rather than inspecting field
shapes. Inspired by memobase's blob.py (``ChatBlob``, ``TranscriptBlob``,
etc.) but pared down to what we actually need at the taOSmd tier.

The envelopes are intentionally NOT pydantic; keeping them stdlib-only
keeps the loader package importable without pulling pydantic into
non-LLM code paths (matters for Pi tier where the dep matrix is tight).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BlobType(str, Enum):
    """The 4 first-class data shapes taOSmd's loaders produce.

    More can be added (image, code, calendar-event) when we have a
    concrete consumer; resist letting this grow speculatively.
    """
    CHAT = "chat"
    TRANSCRIPT = "transcript"
    EMAIL = "email"
    DOC = "doc"


@dataclass
class ChatMessage:
    """One turn in a chat-shaped exchange.

    ``alias`` is the speaker-display-name field memobase uses to support
    multi-party threading (when "User" isn't always the same person).
    Empty string when irrelevant.
    """
    role: str                 # "user" / "assistant" / "system" / custom
    content: str
    alias: str = ""
    timestamp: float = 0.0    # POSIX seconds; 0.0 when unknown


@dataclass
class TranscriptStamp:
    """One stamp in a transcript-shaped recording.

    ``start_timestamp_in_seconds`` is offset from start of recording, not
    wall clock, matching what Whisper / VTT / SRT all give back. ``speaker``
    is whichever label the diarisation produced (often "speaker_0" /
    "speaker_1"); empty when undiarised.
    """
    text: str
    start_timestamp_in_seconds: float = 0.0
    speaker: str = ""


@dataclass
class Blob:
    """Base envelope. Every typed blob inherits this + sets ``kind``.

    ``source_path`` is the filesystem path the loader read from (for
    provenance). ``raw_text`` is a flattened-text view of the blob,
    handy fallback for legacy ingest paths that want a single string.
    Empty when expensive to compute; consumers should rely on the typed
    fields when present.
    """
    kind: BlobType
    source_path: str = ""
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatBlob(Blob):
    """Chat-shaped blob: a sequence of role+content turns."""
    kind: BlobType = BlobType.CHAT
    messages: list[ChatMessage] = field(default_factory=list)


@dataclass
class TranscriptBlob(Blob):
    """Transcript-shaped blob: speaker-stamped text segments + offsets."""
    kind: BlobType = BlobType.TRANSCRIPT
    transcripts: list[TranscriptStamp] = field(default_factory=list)


@dataclass
class EmailBlob(Blob):
    """Email-shaped blob: structured headers + body + threading id.

    ``in_reply_to`` is the RFC 5322 Message-Id we're replying to; empty
    when the email starts a new thread. ``message_id`` is this email's
    own id, used by downstream threading.
    """
    kind: BlobType = BlobType.EMAIL
    sender: str = ""
    recipients: list[str] = field(default_factory=list)
    subject: str = ""
    body: str = ""
    sent_at: float = 0.0          # POSIX seconds; 0.0 when unknown
    message_id: str = ""
    in_reply_to: str = ""


@dataclass
class DocBlob(Blob):
    """Plain doc-shaped blob: text or markdown content.

    The catch-all type. Use this when no specialised loader handles a
    file's format; the extraction step then operates on ``raw_text`` /
    ``content`` directly without typed structure to lean on.
    """
    kind: BlobType = BlobType.DOC
    title: str = ""
    content: str = ""
