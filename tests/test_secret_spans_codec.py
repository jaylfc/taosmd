import base64
import json
import os
import stat
from pathlib import Path
from typing import NamedTuple

import pytest
from cryptography.exceptions import InvalidTag

from taosmd.secret_spans import FileKeyProvider, encode_leaf, reveal_leaf


class Span(NamedTuple):
    start: int
    end: int
    kind: str
    replacement: str


def _span_for(text: str, token: str, kind: str = "test_secret", replacement: str = "[REDACTED:test_secret]") -> Span:
    start = text.index(token)
    return Span(start, start + len(token), kind, replacement)


@pytest.mark.parametrize(
    "text,token",
    [
        ("  lead ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 tail\n", "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
        ("line1\nsecret TOKEN value\nline3\n", "TOKEN"),
        ("café → TOKEN ü", "TOKEN"),
    ],
)
def test_round_trip(tmp_path, text, token):
    provider = FileKeyProvider(tmp_path / "keys")
    span = _span_for(text, token)
    redacted, enc = encode_leaf(text, [span], ts=123.0, event_type="conversation", path=["msg"], provider=provider)

    assert token not in redacted
    assert token not in json.dumps(enc)
    assert reveal_leaf(redacted, enc, ts=123.0, event_type="conversation", provider=provider) == text


@pytest.mark.parametrize("tamper", ["ct", "nonce", "ts", "event_type", "path", "start", "kind"])
def test_tamper_raises(tmp_path, tamper):
    text = "before SECRET after"
    provider = FileKeyProvider(tmp_path / "keys")
    span = _span_for(text, "SECRET")
    redacted, enc = encode_leaf(text, [span], ts=1.0, event_type="evt", path=["p"], provider=provider)
    assert len(enc) == 1
    e = dict(enc[0])
    args = {"ts": 1.0, "event_type": "evt"}

    if tamper == "ct":
        ct = bytearray(base64.b64decode(e["ct"]))
        ct[0] ^= 1
        e["ct"] = base64.b64encode(bytes(ct)).decode()
    elif tamper == "nonce":
        nonce = bytearray(base64.b64decode(e["nonce"]))
        nonce[0] ^= 1
        e["nonce"] = base64.b64encode(bytes(nonce)).decode()
    elif tamper == "ts":
        args["ts"] = 2.0
    elif tamper == "event_type":
        args["event_type"] = "other"
    elif tamper == "path":
        e["path"] = ["other"]
    elif tamper == "start":
        e["start"] = e["start"] + 1
    elif tamper == "kind":
        e["kind"] = "other_kind"

    with pytest.raises((InvalidTag, ValueError)):
        reveal_leaf(redacted, [e], provider=provider, **args)


def test_key_file_modes_and_stable_current(tmp_path):
    key_dir = tmp_path / "keys"
    provider = FileKeyProvider(key_dir)
    kid1, key1 = provider.current()

    assert (stat.S_IMODE(os.stat(key_dir).st_mode)) == 0o700
    assert (stat.S_IMODE(os.stat(key_dir / f"archive-{kid1}.key").st_mode)) == 0o600
    assert (stat.S_IMODE(os.stat(key_dir / "current").st_mode)) == 0o600

    kid2, key2 = provider.current()
    assert kid2 == kid1
    assert key2 == key1
    assert len(key1) == 32


def test_get_invalid_key_raises(tmp_path):
    key_dir = tmp_path / "keys"
    provider = FileKeyProvider(key_dir)
    kid, _ = provider.current()

    bad_kid = "bad0000000000000"
    bad = key_dir / f"archive-{bad_kid}.key"
    os.write(os.open(bad, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), b"x" * 31)

    with pytest.raises(ValueError):
        provider.get(bad_kid)
    with pytest.raises(KeyError):
        provider.get("missing000000000000")


def test_round_trip_multiple_spans(tmp_path):
    provider = FileKeyProvider(tmp_path / "keys")

    text1 = "a ghp_" + "A" * 36 + " mid sk-" + "B" * 24 + " end"
    span1_1 = Span(text1.index("ghp_"), text1.index("ghp_") + 4 + 36, "github_pat", "[REDACTED:github_pat]")
    span1_2 = Span(text1.index("sk-"), text1.index("sk-") + 3 + 24, "openai_key", "[REDACTED:openai_key]")
    redacted1, enc1 = encode_leaf(text1, [span1_1, span1_2], ts=1.0, event_type="evt", path=["p"], provider=provider)
    assert "ghp_" not in redacted1
    assert "sk-" not in redacted1
    assert "ghp_" not in json.dumps(enc1)
    assert "sk-" not in json.dumps(enc1)
    assert reveal_leaf(redacted1, enc1, ts=1.0, event_type="evt", provider=provider) == text1

    text2 = "α秘β" + "X" * 10 + "γ密δ" + "Y" * 5 + "ε"
    span2_1 = Span(text2.index("X"), text2.index("X") + 10, "kind1", "LONGER_REPLACEMENT")
    span2_2 = Span(text2.index("Y"), text2.index("Y") + 5, "kind2", "SHORT")
    span2_3 = Span(0, 1, "kind3", "Z")
    redacted2, enc2 = encode_leaf(text2, [span2_1, span2_2, span2_3], ts=2.0, event_type="evt", path=["p"], provider=provider)
    assert reveal_leaf(redacted2, enc2, ts=2.0, event_type="evt", provider=provider) == text2
    assert "X" * 10 not in redacted2
    assert "Y" * 5 not in redacted2
    assert text2[0] not in redacted2


def test_current_concurrent_first_use(tmp_path, monkeypatch):
    key_dir = tmp_path / "keys"
    provider = FileKeyProvider(key_dir)

    original_open = os.open
    call_count = {"count": 0}
    existing_kid_holder = {"kid": None, "key": None}

    def wrapped_open(path, flags, mode=0o777, *args, **kwargs):
        if path.name == "current" and (flags & os.O_CREAT) and (flags & os.O_EXCL):
            call_count["count"] += 1
            if call_count["count"] == 1:
                existing_kid = "deadbeefdeadbeef"
                existing_key = os.urandom(32)
                existing_kid_holder["kid"] = existing_kid
                existing_kid_holder["key"] = existing_key
                (key_dir / f"archive-{existing_kid}.key").write_bytes(existing_key)
                os.makedirs(key_dir, exist_ok=True)
                (key_dir / "current").write_text(existing_kid, encoding="utf-8")
                raise FileExistsError
        return original_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", wrapped_open)

    kid, key = provider.current()
    assert kid == existing_kid_holder["kid"]
    assert key == existing_kid_holder["key"]

    key_files = list(key_dir.glob("archive-*.key"))
    assert len(key_files) == 1
    assert key_files[0].name == f"archive-{kid}.key"


def test_get_rejects_bad_kid(tmp_path, monkeypatch):
    key_dir = tmp_path / "keys"
    provider = FileKeyProvider(key_dir)
    provider.current()

    def fail_exists(self):
        raise AssertionError("filesystem should not be touched")

    def fail_read_bytes(self):
        raise AssertionError("filesystem should not be touched")

    monkeypatch.setattr(Path, "exists", property(fail_exists))
    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    for bad_kid in ["../x", "ABCDEF0123456789", "", "z" * 16, "a" * 15, "a" * 17]:
        with pytest.raises(KeyError):
            provider.get(bad_kid)
