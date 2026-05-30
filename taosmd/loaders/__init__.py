"""Typed ingestors for taOSmd.

Cognee-style ``LoaderInterface`` (one file in, one typed ``Blob`` out) +
memobase-style discriminated-union envelopes (``ChatBlob``,
``TranscriptBlob``, ``EmailBlob``, ``DocBlob``) so downstream extractors
see structured data rather than a string blob.

The registry picks a loader by file extension + MIME type; concrete
loaders convert their source format into the canonical typed envelope.
Existing string-based ingest paths (``process_conversation_turn`` etc.)
keep working unchanged — this package adds an alternative typed path
for callers that have format-specific data on disk.

Usage::

    from taosmd.loaders import pick_loader

    blob = await pick_loader("meeting.transcript.json").load(
        "meeting.transcript.json"
    )
    # blob is a TranscriptBlob with .transcripts: list[TranscriptStamp]
    for stamp in blob.transcripts:
        print(stamp.speaker, stamp.start_timestamp_in_seconds, stamp.text)

See ``reference_memory_systems_survey.md`` for the design rationale.
"""

from .blob import (
    Blob,
    BlobType,
    ChatBlob,
    ChatMessage,
    DocBlob,
    EmailBlob,
    TranscriptBlob,
    TranscriptStamp,
)
from .interface import LoaderInterface
from .registry import REGISTRY, pick_loader, register_loader
from .chat_loader import ChatLoader
from .doc_loader import DocLoader
from .email_loader import EmailLoader
from .transcript_loader import TranscriptLoader

__all__ = [
    "Blob",
    "BlobType",
    "ChatBlob",
    "ChatMessage",
    "DocBlob",
    "EmailBlob",
    "TranscriptBlob",
    "TranscriptStamp",
    "LoaderInterface",
    "ChatLoader",
    "DocLoader",
    "EmailLoader",
    "TranscriptLoader",
    "REGISTRY",
    "pick_loader",
    "register_loader",
]
