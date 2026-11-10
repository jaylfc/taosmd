"""Encrypted raw-archive secret spans (taOSmd).

AES-GCM codec for secret spans redacted out of raw archive leaves, plus a
0600 filesystem key provider. Not yet wired into record(); that is a follow
up. `cryptography` is a core dependency (D1) and each archive leaf uses a
standalone local 0600 key file (D2).
"""

from __future__ import annotations

import base64
import json
import os
import secrets
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class FileKeyProvider:
    """Filesystem-backed AES-GCM key provider with standalone 0600 key files.

    Keys live in ``key_dir``. ``current`` names the active kid; each kid's
    32-byte key lives in ``archive-<kid>.key`` mode 0o600, and ``key_dir``
    itself is mode 0o700. Key bytes are never logged.
    """

    def __init__(self, key_dir: str | Path) -> None:
        self.key_dir = Path(key_dir)

    def current(self) -> tuple[str, bytes]:
        current_path = self.key_dir / "current"
        try:
            kid = current_path.read_text().strip()
        except (FileNotFoundError, NotADirectoryError):
            os.makedirs(self.key_dir, exist_ok=True)
            os.chmod(self.key_dir, 0o700)
            kid = secrets.token_hex(8)
            key = os.urandom(32)
            key_path = self.key_dir / f"archive-{kid}.key"
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(key)
            fd = os.open(current_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as handle:
                handle.write(kid)
            return kid, key
        return kid, self.get(kid)

    def get(self, kid: str) -> bytes:
        path = self.key_dir / f"archive-{kid}.key"
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise KeyError(kid) from None
        if len(data) != 32:
            raise ValueError(
                f"key for kid {kid!r} is {len(data)} bytes, expected 32"
            )
        return data


def _aad(ts, event_type, path, start, end, kind, kid) -> bytes:
    return json.dumps(
        [ts, event_type, path, start, end, kind, kid],
        separators=(",", ":"),
    ).encode()


def encode_leaf(
    text: str,
    spans,
    *,
    ts: float,
    event_type: str,
    path: list,
    provider: FileKeyProvider,
) -> tuple[str, list[dict]]:
    """Encrypt each secret span and splice its replacement over it.

    Returns ``(redacted_text, enc)`` where ``enc`` is one dict per span with
    path, start, end, kind, kid, nonce, ct, replacement.
    """
    kid, key = provider.current()
    ordered = sorted(spans, key=lambda span: span.start)
    parts: list[str] = []
    enc: list[dict] = []
    cursor = 0
    for span in ordered:
        parts.append(text[cursor : span.start])
        parts.append(span.replacement)
        nonce = os.urandom(12)
        ct = AESGCM(key).encrypt(
            nonce,
            text[span.start : span.end].encode("utf-8"),
            _aad(ts, event_type, path, span.start, span.end, span.kind, kid),
        )
        enc.append(
            {
                "path": path,
                "start": span.start,
                "end": span.end,
                "kind": span.kind,
                "kid": kid,
                "nonce": base64.b64encode(nonce).decode("ascii"),
                "ct": base64.b64encode(ct).decode("ascii"),
                "replacement": span.replacement,
            }
        )
        cursor = span.end
    parts.append(text[cursor:])
    return "".join(parts), enc


def reveal_leaf(
    redacted: str,
    enc: list[dict],
    *,
    ts,
    event_type: str,
    provider: FileKeyProvider,
) -> str:
    """Decrypt each span back into its placeholder, rebuilding the original."""
    ordered = sorted(enc, key=lambda entry: entry["start"])
    parts: list[str] = []
    cursor = 0
    shift = 0
    for entry in ordered:
        start = entry["start"]
        end = entry["end"]
        replacement = entry["replacement"]
        pos = start + shift
        if redacted[pos : pos + len(replacement)] != replacement:
            raise ValueError("placeholder mismatch")
        parts.append(redacted[cursor:pos])
        key = provider.get(entry["kid"])
        nonce = base64.b64decode(entry["nonce"])
        ct = base64.b64decode(entry["ct"])
        plaintext = AESGCM(key).decrypt(
            nonce,
            ct,
            _aad(
                ts,
                event_type,
                entry["path"],
                start,
                end,
                entry["kind"],
                entry["kid"],
            ),
        ).decode("utf-8")
        if len(plaintext) != end - start:
            raise ValueError(
                f"plaintext length {len(plaintext)} != span {end - start}"
            )
        parts.append(plaintext)
        cursor = pos + len(replacement)
        shift += len(replacement) - (end - start)
    parts.append(redacted[cursor:])
    return "".join(parts)
