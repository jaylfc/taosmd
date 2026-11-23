"""Tests for sealed secret spans in the raw archive (tsk-q6yxdv).

The raw JSONL line keeps the exact secret bytes, AES-GCM encrypted under a
per-record data key wrapped to the install X25519 key. Every readable view
(JSONL, archive_index.data_json, archive_fts) shows the redacted form.
ArchiveStore.reveal() (Python API only) returns the caller's original bytes.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import stat

import pytest
from cryptography.exceptions import InvalidTag

from taosmd import keywrap
from taosmd.archive import ArchiveStore


TOKEN = "sk-ant-api03-" + "A" * 40
SENSITIVE = {
    "content": "\n  ledger entry: " + TOKEN + "  \t\ncaf\xe9\n",
    "nested": [{"messages": [{"body": "in " + TOKEN + " out"}]}],
    "plain": "no secrets here",
}
SUMMARY = "ticket " + TOKEN + " -- " + "B" * 20


def _store(tmp_path, keys_dir=None):
    return ArchiveStore(
        archive_dir=tmp_path / "archive",
        index_path=tmp_path / "archive-index.db",
        keys_dir=keys_dir,
    )


def _raw_line(tmp_path):
    jsonl_files = list((tmp_path / "archive").glob("**/*.jsonl"))
    assert len(jsonl_files) == 1, jsonl_files
    return jsonl_files[0].read_text(encoding="utf-8").strip().splitlines()[-1]


def _flip_b64(entry, key):
    e = copy.deepcopy(entry)
    raw = bytearray(base64.b64decode(e[key]))
    raw[0] ^= 1
    e[key] = base64.b64encode(bytes(raw)).decode()
    return e


@pytest.mark.asyncio
async def test_seal_raw_line_hides_token_and_reveal_round_trip(tmp_path):
    """(a) Token bytes appear nowhere in JSONL, data_json or FTS; reveal is exact."""
    store = _store(tmp_path)
    await store.init()
    try:
        row_id = await store.record(
            "conversation", dict(SENSITIVE), summary=SUMMARY
        )
        assert row_id > 0

        # The caller's input is untouched.
        assert SENSITIVE["content"].count(TOKEN) == 1

        # The raw JSONL line: no secret bytes anywhere, enc_spans present.
        line = _raw_line(tmp_path)
        assert TOKEN not in line
        record = json.loads(line)
        assert record["secret_schema"] == 2
        assert record["enc_spans"], "enc_spans must list the sealed spans"
        paths = {tuple(e["path"]) for e in record["enc_spans"]}
        assert ("summary",) in paths
        assert ("data", "content") in paths
        assert ("data", "nested", 0, "messages", 0, "body") in paths
        for entry in record["enc_spans"]:
            assert TOKEN not in json.dumps(entry)
        assert record["data"]["plain"] == "no secrets here"

        # data_json and FTS show the redacted form, no secret bytes.
        row = store._conn.execute(
            "SELECT data_json FROM archive_index WHERE id = ?", (row_id,)
        ).fetchone()
        assert TOKEN not in row["data_json"]
        assert "[REDACTED:anthropic_key]" in row["data_json"]
        fts = store._conn.execute(
            "SELECT content FROM archive_fts WHERE rowid = ?", (row_id,)
        ).fetchone()
        assert fts is not None
        assert TOKEN not in fts["content"]

        # reveal returns the original input byte for byte.
        revealed = await store.reveal(record)
        assert revealed["data"]["content"] == SENSITIVE["content"]
        assert revealed["data"]["nested"][0]["messages"][0]["body"] == (
            "in " + TOKEN + " out"
        )
        assert revealed["summary"] == SUMMARY
        assert "enc_spans" not in revealed
        assert "secret_schema" not in revealed
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_seal_no_secret_unchanged(tmp_path):
    """Records without spans keep no enc_spans / secret_schema keys."""
    store = _store(tmp_path)
    await store.init()
    try:
        await store.record("conversation", {"content": "just hello"}, summary="hi")
        line = _raw_line(tmp_path)
        record = json.loads(line)
        assert "enc_spans" not in record
        assert "secret_schema" not in record
        assert record["data"]["content"] == "just hello"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_tampering_makes_reveal_raise(tmp_path):
    """(b) One flipped byte in ct, nonce, wrapped DEK or any AAD field fails."""
    store = _store(tmp_path)
    await store.init()
    try:
        await store.record("conversation", {"content": "key " + TOKEN}, summary=TOKEN)
        line = _raw_line(tmp_path)
        record = json.loads(line)
        entry = record["enc_spans"][0]

        wrap_log = (
            store._keys_dir / "wraps" / f"{store._install_key.recipient_kid}.jsonl"
        )
        wrap_entries = [
            json.loads(line_)
            for line_ in wrap_log.read_text(encoding="utf-8").splitlines()
            if line_
        ]
        assert wrap_entries
        original_wrap_text = wrap_log.read_text(encoding="utf-8")

        base_event = {
            "timestamp": record["timestamp"],
            "event_type": record["event_type"],
            "data": {"content": record["data"]["content"]},
            "summary": record["summary"],
            "enc_spans": [copy.deepcopy(entry)],
            "secret_schema": 2,
        }

        # ct flip: leaf ciphertext no longer authenticates.
        tampered = copy.deepcopy(base_event)
        tampered["enc_spans"][0] = _flip_b64(entry, "ct")
        with pytest.raises((InvalidTag, ValueError)):
            await store.reveal(tampered)

        # nonce flip: same leaf path, different nonce.
        tampered = copy.deepcopy(base_event)
        tampered["enc_spans"][0] = _flip_b64(entry, "nonce")
        with pytest.raises((InvalidTag, ValueError)):
            await store.reveal(tampered)

        # Event-level AAD fields bind the span to its place and fail.
        for field, value in (
            ("timestamp", record["timestamp"] + 1.0),
            ("event_type", "other"),
        ):
            tampered = copy.deepcopy(base_event)
            tampered[field] = value
            with pytest.raises((InvalidTag, ValueError)):
                await store.reveal(tampered)

        # Wrapped DEK flip: the wrap log no longer unwraps the DEK.
        tampered_wrap = copy.deepcopy(wrap_entries[0])
        raw = bytearray(base64.b64decode(tampered_wrap["wrapped_dek"]))
        raw[0] ^= 1
        tampered_wrap["wrapped_dek"] = base64.b64encode(bytes(raw)).decode()
        entries = [
            e if e.get("dek_id") != tampered_wrap["dek_id"] else tampered_wrap
            for e in wrap_entries
        ]
        wrap_log.write_text(
            "".join(json.dumps(e, separators=(",", ":")) + "\n" for e in entries),
            encoding="utf-8",
        )
        try:
            with pytest.raises((InvalidTag, ValueError, KeyError)):
                await store.reveal(copy.deepcopy(base_event))
        finally:
            wrap_log.write_text(original_wrap_text, encoding="utf-8")

        # Entry-level AAD fields bind the span to its place and fail.
        for field, value in (
            ("start", entry["start"] + 1),
            ("end", entry["end"] + 1),
            ("kind", "other_kind"),
            ("kid", "0000000000000000"),
        ):
            tampered = copy.deepcopy(base_event)
            tampered["enc_spans"][0][field] = value
            with pytest.raises((InvalidTag, ValueError, KeyError)):
                await store.reveal(tampered)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_key_file_modes(tmp_path):
    """(c) Install key directory 0700, private key file 0600."""
    keys_dir = tmp_path / "keys"
    store = _store(tmp_path, keys_dir=keys_dir)
    await store.init()
    try:
        await store.record("conversation", {"content": "key " + TOKEN}, summary=TOKEN)
        assert stat.S_IMODE(os.stat(keys_dir).st_mode) == 0o700
        private = keys_dir / keywrap.PRIVATE_KEY_FILE
        assert private.exists()
        assert stat.S_IMODE(os.stat(private).st_mode) == 0o600
        wraps_dir = keys_dir / keywrap.WRAPS_DIR
        assert stat.S_IMODE(os.stat(wraps_dir).st_mode) == 0o700
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_init_fails_loud_on_unreadable_key(tmp_path):
    """(c cont.) init() fails loud if a key exists but cannot be read."""
    keys_dir = tmp_path / "keys"
    store = _store(tmp_path, keys_dir=keys_dir)
    await store.init()
    await store.close()
    private = keys_dir / keywrap.PRIVATE_KEY_FILE
    private.chmod(0o000)
    try:
        store2 = _store(tmp_path, keys_dir=keys_dir)
        with pytest.raises(Exception):
            await store2.init()
    finally:
        try:
            private.chmod(0o600)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_rewrap_to_second_key(tmp_path):
    """(d) Re-wrap to a second key pair: reveal works with the second private
    key, the install wrap entry is gone, and the archive JSONL file is
    byte-identical before and after.
    """
    keys_dir_1 = tmp_path / "keys1"
    keys_dir_2 = tmp_path / "keys2"
    store = _store(tmp_path, keys_dir=keys_dir_1)
    await store.init()
    try:
        await store.record(
            "conversation", {"content": "key " + TOKEN}, summary=TOKEN
        )
        jsonl = list(tmp_path.glob("**/*.jsonl"))[0]
        before = jsonl.read_bytes()
        wrap_log = keys_dir_1 / "wraps" / f"{store._install_key.recipient_kid}.jsonl"
        wrap_entries = [
            json.loads(line_)
            for line_ in wrap_log.read_text(encoding="utf-8").splitlines()
            if line_
        ]
        assert len(wrap_entries) == 1
        dek_id = wrap_entries[0]["dek_id"]

        install_2 = keywrap.init_install_key(keys_dir_2)
        re_wrapped = keywrap.rewrap(
            [dek_id],
            install_2.public_key,
            install_2.recipient_kid,
            install_key=store._install_key,
            to_keys_dir=keys_dir_2,
        )
        assert re_wrapped == [dek_id]

        # Install wrap entry is gone, the new recipient log has it.
        install_entries = [
            json.loads(line_)
            for line_ in wrap_log.read_text(encoding="utf-8").splitlines()
            if line_
        ]
        assert all(e["dek_id"] != dek_id for e in install_entries)
        new_log = keys_dir_2 / "wraps" / f"{install_2.recipient_kid}.jsonl"
        new_entries = [
            json.loads(line_)
            for line_ in new_log.read_text(encoding="utf-8").splitlines()
            if line_
        ]
        assert any(e["dek_id"] == dek_id for e in new_entries)

        # Archive line byte-identical before and after.
        assert jsonl.read_bytes() == before

        # Reveal works with the second private key.
        store2 = _store(tmp_path, keys_dir=keys_dir_2)
        await store2.init()
        try:
            stored = json.loads(_raw_line(tmp_path))
            revealed = await store2.reveal(stored)
            assert revealed["data"]["content"] == "key " + TOKEN
            assert revealed["summary"] == TOKEN
        finally:
            await store2.close()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_legacy_lines_untouched(tmp_path):
    """(e) A store written before this change opens, reads and appends with
    no change to existing lines (hash-compare every existing file).
    """
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir(parents=True)

    # A legacy line exactly as origin/master wrote it: no enc_spans, no
    # secret_schema, sha256 over the line without the sha256 key.
    legacy_event = {
        "timestamp": 1000000000.0,
        "event_type": "conversation",
        "agent_name": "bot",
        "app_id": None,
        "project": None,
        "summary": "[REDACTED:anthropic_key]",
        "data": {"content": "key [REDACTED:anthropic_key]"},
    }
    legacy_line = json.dumps(legacy_event, default=str)
    legacy_event["sha256"] = hashlib.sha256(legacy_line.encode()).hexdigest()
    legacy_line = json.dumps(legacy_event, default=str)
    day_dir = archive_dir / "1970" / "01"
    day_dir.mkdir(parents=True)
    legacy_file = day_dir / "01.jsonl"
    legacy_file.write_text(legacy_line + "\n", encoding="utf-8")
    legacy_bytes = legacy_file.read_bytes()

    store = _store(tmp_path)
    await store.init()
    try:
        # Verify passes on the legacy line.
        report = await store.verify_day("1970-01-01")
        assert report["total"] == 1
        assert report["ok"] == 1

        # Appending a new record does not alter existing files.
        await store.record("conversation", {"content": "key " + TOKEN}, summary=TOKEN)
        assert legacy_file.read_bytes() == legacy_bytes
        for path in archive_dir.rglob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    assert ArchiveStore.verify_entry(line), path
    finally:
        await store.close()
