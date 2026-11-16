import base64
import json
import os
import stat
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

    bad = key_dir / "archive-bad.key"
    os.write(os.open(bad, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), b"x" * 31)

    with pytest.raises(ValueError):
        provider.get("bad")
    with pytest.raises(KeyError):
        provider.get("missing")
