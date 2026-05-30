"""Tests for taosmd.loaders — typed ingestors per data type."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from taosmd.loaders import (
    Blob,
    BlobType,
    ChatBlob,
    ChatLoader,
    DocBlob,
    DocLoader,
    EmailBlob,
    EmailLoader,
    TranscriptBlob,
    TranscriptLoader,
    pick_loader,
    register_loader,
    REGISTRY,
)
from taosmd.loaders.interface import LoaderInterface
from taosmd.loaders.registry import _path_to_extension


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Path-extension helper
# ---------------------------------------------------------------------------


def test_path_to_extension_handles_multi_suffix():
    assert _path_to_extension("meeting.transcript.json") == "transcript.json"
    assert _path_to_extension("chat.json") == "json"
    assert _path_to_extension("notes.md") == "md"
    assert _path_to_extension("README") == ""
    assert _path_to_extension("a.b.c.d") == "c.d"


# ---------------------------------------------------------------------------
# ChatLoader
# ---------------------------------------------------------------------------


def test_chat_loader_handles_list_shape(tmp_path):
    src = tmp_path / "session.chat.json"
    src.write_text(json.dumps([
        {"role": "user", "content": "hi", "timestamp": 1716000000.0},
        {"role": "assistant", "content": "hello", "alias": "Alex"},
    ]))
    blob = _run(ChatLoader().load(src))

    assert isinstance(blob, ChatBlob)
    assert blob.kind == BlobType.CHAT
    assert blob.source_path == str(src)
    assert len(blob.messages) == 2
    assert blob.messages[0].role == "user"
    assert blob.messages[0].timestamp == 1716000000.0
    assert blob.messages[1].alias == "Alex"
    assert "[user] hi" in blob.raw_text
    assert "[Alex] hello" in blob.raw_text


def test_chat_loader_handles_object_shape(tmp_path):
    src = tmp_path / "wrapped.chat.json"
    src.write_text(json.dumps({
        "messages": [{"role": "user", "content": "hi"}],
        "agent": "alex",
    }))
    blob = _run(ChatLoader().load(src))
    assert len(blob.messages) == 1
    assert blob.messages[0].content == "hi"


def test_chat_loader_rejects_unexpected_shape(tmp_path):
    src = tmp_path / "bad.chat.json"
    src.write_text(json.dumps({"not_messages": "wrong"}))
    with pytest.raises(ValueError, match="chat shape"):
        _run(ChatLoader().load(src))


def test_chat_loader_skips_non_dict_rows(tmp_path):
    src = tmp_path / "mixed.chat.json"
    src.write_text(json.dumps([
        {"role": "user", "content": "hi"},
        "this is not a message",
        {"role": "assistant", "content": "ok"},
    ]))
    blob = _run(ChatLoader().load(src))
    assert len(blob.messages) == 2


# ---------------------------------------------------------------------------
# TranscriptLoader
# ---------------------------------------------------------------------------


def test_transcript_loader_whisper_segments(tmp_path):
    src = tmp_path / "meeting.transcript.json"
    src.write_text(json.dumps({
        "segments": [
            {"start": 0.0, "end": 4.2, "text": "Hello.", "speaker": "spk_0"},
            {"start": 4.2, "end": 7.0, "text": "Hi there.", "speaker": "spk_1"},
        ],
    }))
    blob = _run(TranscriptLoader().load(src))

    assert isinstance(blob, TranscriptBlob)
    assert blob.kind == BlobType.TRANSCRIPT
    assert len(blob.transcripts) == 2
    assert blob.transcripts[0].speaker == "spk_0"
    assert blob.transcripts[0].start_timestamp_in_seconds == 0.0
    assert blob.transcripts[1].text == "Hi there."


def test_transcript_loader_canonical_shape(tmp_path):
    src = tmp_path / "canonical.transcript.json"
    src.write_text(json.dumps({
        "transcripts": [
            {"start_timestamp_in_seconds": 0.5,
             "speaker": "alice", "text": "Welcome."}
        ],
    }))
    blob = _run(TranscriptLoader().load(src))
    assert blob.transcripts[0].speaker == "alice"
    assert blob.transcripts[0].start_timestamp_in_seconds == 0.5


def test_transcript_loader_plain_list(tmp_path):
    src = tmp_path / "list.transcript.json"
    src.write_text(json.dumps([
        {"start": 1.0, "speaker": "bob", "text": "Done."}
    ]))
    blob = _run(TranscriptLoader().load(src))
    assert len(blob.transcripts) == 1
    assert blob.transcripts[0].text == "Done."


def test_transcript_loader_drops_empty_text(tmp_path):
    src = tmp_path / "with_empty.transcript.json"
    src.write_text(json.dumps({
        "segments": [
            {"start": 0.0, "text": "", "speaker": "spk_0"},
            {"start": 1.0, "text": "Real.", "speaker": "spk_0"},
            {"start": 2.0, "text": "   ", "speaker": "spk_0"},
        ],
    }))
    blob = _run(TranscriptLoader().load(src))
    assert len(blob.transcripts) == 1
    assert blob.transcripts[0].text == "Real."


# ---------------------------------------------------------------------------
# EmailLoader
# ---------------------------------------------------------------------------


_SAMPLE_EML = b"""From: alice@example.com
To: bob@example.com, carol@example.com
Subject: Re: lunch plans
Date: Thu, 30 May 2026 12:00:00 +0000
Message-Id: <abc123@example.com>
In-Reply-To: <xyz789@example.com>
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8

Yes, 1pm sounds good. See you then.
"""


def test_email_loader_extracts_headers_and_body(tmp_path):
    src = tmp_path / "msg.eml"
    src.write_bytes(_SAMPLE_EML)
    blob = _run(EmailLoader().load(src))

    assert isinstance(blob, EmailBlob)
    assert blob.kind == BlobType.EMAIL
    assert blob.sender == "alice@example.com"
    assert "bob@example.com" in blob.recipients
    assert "carol@example.com" in blob.recipients
    assert blob.subject == "Re: lunch plans"
    assert "Yes, 1pm sounds good" in blob.body
    assert blob.message_id == "abc123@example.com"
    assert blob.in_reply_to == "xyz789@example.com"
    assert blob.sent_at > 0  # Date parsed successfully


def test_email_loader_handles_missing_threading_headers(tmp_path):
    src = tmp_path / "fresh.eml"
    src.write_bytes(
        b"From: alice@example.com\nTo: bob@example.com\n"
        b"Subject: Hi\n\nHello.\n"
    )
    blob = _run(EmailLoader().load(src))
    assert blob.message_id == ""
    assert blob.in_reply_to == ""
    assert blob.sent_at == 0.0


# ---------------------------------------------------------------------------
# DocLoader
# ---------------------------------------------------------------------------


def test_doc_loader_plain_text(tmp_path):
    src = tmp_path / "notes.txt"
    src.write_text("Just some text.\nMore on the next line.")
    blob = _run(DocLoader().load(src))

    assert isinstance(blob, DocBlob)
    assert blob.kind == BlobType.DOC
    assert blob.content == "Just some text.\nMore on the next line."
    assert blob.title == ""


def test_doc_loader_extracts_markdown_title(tmp_path):
    src = tmp_path / "readme.md"
    src.write_text("# Project taOSmd\n\nSome description.")
    blob = _run(DocLoader().load(src))
    assert blob.title == "Project taOSmd"
    assert "Some description." in blob.content


# ---------------------------------------------------------------------------
# Registry / pick_loader
# ---------------------------------------------------------------------------


def test_pick_loader_chat(tmp_path):
    p = tmp_path / "x.chat.json"
    p.write_text("[]")
    loader = pick_loader(p)
    assert isinstance(loader, ChatLoader)


def test_pick_loader_transcript(tmp_path):
    p = tmp_path / "x.transcript.json"
    p.write_text("[]")
    loader = pick_loader(p)
    assert isinstance(loader, TranscriptLoader)


def test_pick_loader_email(tmp_path):
    p = tmp_path / "x.eml"
    p.write_bytes(b"From: a@b\nSubject: t\n\n")
    loader = pick_loader(p)
    assert isinstance(loader, EmailLoader)


def test_pick_loader_doc_fallback_for_md(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("# h")
    loader = pick_loader(p)
    assert isinstance(loader, DocLoader)


def test_pick_loader_doc_fallback_for_unknown(tmp_path):
    p = tmp_path / "weird.frobnicated"
    p.write_text("anything")
    loader = pick_loader(p)
    # No specialised loader claims .frobnicated; catch-all wins.
    assert isinstance(loader, DocLoader)


def test_register_loader_inserts_before_doc():
    class _StubLoader(LoaderInterface):
        loader_name = "stub"
        supported_extensions = ("stub",)
        supported_mime_types = ()
        async def load(self, file_path, **kwargs):
            return DocBlob(source_path=str(file_path), content="")

    before = list(REGISTRY)
    try:
        register_loader(_StubLoader)
        assert _StubLoader in REGISTRY
        assert REGISTRY.index(_StubLoader) < REGISTRY.index(DocLoader)
    finally:
        REGISTRY[:] = before


# ---------------------------------------------------------------------------
# End-to-end: pick + load
# ---------------------------------------------------------------------------


def test_pick_and_load_e2e(tmp_path):
    """The high-level API a caller actually uses."""
    src = tmp_path / "session.chat.json"
    src.write_text(json.dumps([{"role": "user", "content": "Hi."}]))

    loader = pick_loader(src)
    blob = _run(loader.load(src))

    assert isinstance(blob, ChatBlob)
    assert blob.messages[0].content == "Hi."
