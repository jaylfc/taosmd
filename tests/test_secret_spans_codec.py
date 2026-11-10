"""Tests for taosmd.secret_spans — AES-GCM span codec and 0600 key provider.

Behaviour-only, fully offline. Spans are built by hand with a small
NamedTuple; find_secret_spans() is not involved here.
"""

from __future__ import annotations

import base64
import json
import os
import stat
from collections import namedtuple

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from taosmd.secret_spans import (
    FileKeyProvider,
    encode_leaf,
    reveal_leaf,
)

Span = namedtuple("Span", "start end kind replacement")

SECRET = "ghp_ABCDEFGHIJKLMNOP1234567890"
REPLACEMENT = "[REDACTED:github_pat]"

ROUND_TRIP_CASES = [
    # (text, secret, replacement)
    (f"  lead {SECRET} tail\n", SECRET, REPLACEMENT),
    (
        "config line one\n" + SECRET + "\nconfig line three\n",
        SECRET,
        REPLACEMENT,
    ),
    ("café → TOKEN ü", "TOKEN", "[REDACTED:token]"),
]


def _span_for(text: str, secret: str, kind: str = "pat") -> Span:
    start = text.find(secret)
    assert start != -1, secret
    return Span(start, start + len(secret), kind, "[REDACTED:" + kind + "]")


def _provider(tmp_path) -> FileKeyProvider:
    return FileKeyProvider(tmp_path / "keys")


@pytest.mark.parametrize("text,secret,replacement", ROUND_TRIP_CASES)
def test_round_trip_preserves_original_and_hides_secret(
    tmp_path, text, secret, replacement
):
    provider = _provider(tmp_path)
    ts = 1730000000.5
    event_type = "ingest"
    path = ["inbox", "turn-3"]
    span = _span_for(text, secret, "github_pat")
    redacted, enc = encode_leaf(
        text, [span], ts=ts, event_type=event_type, path=path, provider=provider
    )
    assert reveal_leaf(redacted, enc, ts=ts, event_type=event_type, provider=provider) == text
    assert secret not in redacted
    assert secret not in json.dumps(enc)


def test_reveal_returns_exact_original_multi_span(tmp_path):
    provider = _provider(tmp_path)
    text = "a " + SECRET + " b " + SECRET + " c"
    ts = 1730000001.0
    event_type = "ingest"
    path = ["inbox"]
    spans = [
        Span(2, 2 + len(SECRET), "github_pat", REPLACEMENT),
        Span(len(text) - len(SECRET), len(text), "github_pat", REPLACEMENT),
    ]
    redacted, enc = encode_leaf(
        text, spans, ts=ts, event_type=event_type, path=path, provider=provider
    )
    assert reveal_leaf(redacted, enc, ts=ts, event_type=event_type, provider=provider) == text
    assert SECRET not in redacted


_TAMPER_TS = 1730000002.0
_TAMPER_EVENT = "ingest"
_TAMPER_PATH = ["inbox"]


def _tamper_base(tmp_path):
    provider = _provider(tmp_path)
    text = "prefix " + SECRET + " suffix"
    span = _span_for(text, SECRET, "github_pat")
    redacted, enc = encode_leaf(
        text,
        [span],
        ts=_TAMPER_TS,
        event_type=_TAMPER_EVENT,
        path=_TAMPER_PATH,
        provider=provider,
    )
    return provider, redacted, enc[0]


def _flip_b64(entry, field, index=0):
    data = bytearray(base64.b64decode(entry[field]))
    data[index] ^= 0xFF
    entry[field] = base64.b64encode(bytes(data)).decode("ascii")


TAMPER_CASES = [
    ("ct byte flip", lambda e: _flip_b64(e, "ct")),
    ("nonce byte flip", lambda e: _flip_b64(e, "nonce")),
]


@pytest.mark.parametrize("label,mutate", TAMPER_CASES)
def test_tamper_ct_or_nonce_raises_invalid_tag(tmp_path, label, mutate):
    provider, redacted, entry = _tamper_base(tmp_path)
    mutate(entry)
    with pytest.raises(InvalidTag):
        reveal_leaf(
            redacted,
            [entry],
            ts=_TAMPER_TS,
            event_type=_TAMPER_EVENT,
            provider=provider,
        )


@pytest.mark.parametrize(
    "label,mutate",
    [
        ("ts changed", lambda e: None),
        ("event_type changed", lambda e: None),
        ("path changed", lambda e: e.update(path=["other"])),
    ],
)
def test_tamper_aad_field_raises(tmp_path, label, mutate):
    provider, redacted, entry = _tamper_base(tmp_path)
    entry = dict(entry)
    mutate(entry)
    kwargs = {"ts": _TAMPER_TS, "event_type": _TAMPER_EVENT, "provider": provider}
    if label == "ts changed":
        kwargs["ts"] = _TAMPER_TS + 1.0
    elif label == "event_type changed":
        kwargs["event_type"] = "other"
    with pytest.raises((InvalidTag, ValueError)):
        reveal_leaf(redacted, [entry], **kwargs)


def test_tamper_start_raises(tmp_path):
    provider, redacted, entry = _tamper_base(tmp_path)
    entry = dict(entry)
    entry["start"] = entry["start"] + 1
    with pytest.raises((InvalidTag, ValueError)):
        reveal_leaf(
            redacted,
            [entry],
            ts=_TAMPER_TS,
            event_type=_TAMPER_EVENT,
            provider=provider,
        )


def test_tamper_kind_raises(tmp_path):
    provider, redacted, entry = _tamper_base(tmp_path)
    entry = dict(entry)
    entry["kind"] = "other"
    with pytest.raises((InvalidTag, ValueError)):
        reveal_leaf(
            redacted,
            [entry],
            ts=_TAMPER_TS,
            event_type=_TAMPER_EVENT,
            provider=provider,
        )


def test_placeholder_mismatch_raises(tmp_path):
    provider, redacted, entry = _tamper_base(tmp_path)
    bad = redacted.replace(entry["replacement"], "NOPE", 1)
    with pytest.raises(ValueError, match="placeholder mismatch"):
        reveal_leaf(
            bad,
            [entry],
            ts=_TAMPER_TS,
            event_type=_TAMPER_EVENT,
            provider=provider,
        )


def test_key_file_modes_and_stable_kid(tmp_path):
    provider = _provider(tmp_path)
    key_dir = tmp_path / "keys"
    kid1, key1 = provider.current()
    key_mode = stat.S_IMODE(
        os.stat(key_dir / f"archive-{kid1}.key").st_mode
    )
    dir_mode = stat.S_IMODE(os.stat(key_dir).st_mode)
    assert key_mode == 0o600
    assert dir_mode == 0o700
    kid2, key2 = provider.current()
    assert kid2 == kid1
    assert key2 == key1


def test_get_rejects_short_key_and_unknown(tmp_path):
    provider = _provider(tmp_path)
    kid, _ = provider.current()
    key_path = provider.key_dir / f"archive-{kid}.key"
    key_path.write_bytes(b"x" * 31)
    with pytest.raises(ValueError):
        provider.get(kid)
    with pytest.raises(KeyError):
        provider.get("does-not-exist")


def test_aad_is_compact_json():
    from taosmd.secret_spans import _aad

    out = _aad(1.5, "e", ["p"], 1, 2, "k", "kid")
    assert out == b'[1.5,"e",["p"],1,2,"k","kid"]'


def test_decrypt_uses_real_aesgcm(tmp_path):
    provider = _provider(tmp_path)
    kid, key = provider.current()
    nonce = os.urandom(12)
    ct = AESGCM(key).encrypt(nonce, b"plaintext", b"aad")
    plaintext = AESGCM(key).decrypt(nonce, ct, b"aad")
    assert plaintext == b"plaintext"
