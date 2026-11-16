"""AES-GCM encrypted raw-archive secret spans (spec 4/5/6)."""

from __future__ import annotations

import base64
import json
import os
import secrets
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class FileKeyProvider:
    def __init__(self, key_dir: str | Path):
        self.key_dir = Path(key_dir)

    def current(self) -> tuple[str, bytes]:
        current_path = self.key_dir / "current"
        if self.key_dir.exists() and current_path.exists():
            kid = current_path.read_text(encoding="utf-8").strip()
            return kid, self.get(kid)

        os.makedirs(self.key_dir, exist_ok=True)
        os.chmod(self.key_dir, 0o700)

        kid = secrets.token_hex(8)
        key = os.urandom(32)

        key_path = self.key_dir / f"archive-{kid}.key"
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, key)
        finally:
            os.close(fd)

        fd = os.open(current_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, kid.encode("utf-8"))
        finally:
            os.close(fd)

        return kid, key

    def get(self, kid: str) -> bytes:
        key_path = self.key_dir / f"archive-{kid}.key"
        if not key_path.exists():
            raise KeyError(kid)
        key = key_path.read_bytes()
        if len(key) != 32:
            raise ValueError("key must be exactly 32 bytes")
        return key


def _aad(ts, event_type, path, start, end, kind, kid) -> bytes:
    return json.dumps([ts, event_type, path, start, end, kind, kid], separators=(",", ":")).encode()


def encode_leaf(text: str, spans, *, ts: float, event_type: str, path: list, provider) -> tuple[str, list[dict]]:
    spans = sorted(spans, key=lambda s: (s.start, s.end))
    redacted_text = text
    enc: list[dict] = []
    shift = 0
    for span in spans:
        plaintext = text[span.start:span.end]
        kid, key = provider.current()
        nonce = os.urandom(12)
        ct = AESGCM(key).encrypt(
            nonce,
            plaintext.encode("utf-8"),
            _aad(ts, event_type, path, span.start, span.end, span.kind, kid),
        )
        enc.append(
            {
                "path": path,
                "start": span.start,
                "end": span.end,
                "kind": span.kind,
                "kid": kid,
                "nonce": base64.b64encode(nonce).decode(),
                "ct": base64.b64encode(ct).decode(),
                "replacement": span.replacement,
            }
        )
        pos = span.start + shift
        redacted_text = redacted_text[:pos] + span.replacement + redacted_text[pos + (span.end - span.start):]
        shift += len(span.replacement) - (span.end - span.start)
    return redacted_text, enc


def reveal_leaf(redacted: str, enc: list[dict], *, ts, event_type, provider) -> str:
    result = redacted
    shift = 0
    for entry in sorted(enc, key=lambda e: e["start"]):
        start = entry["start"]
        end = entry["end"]
        kind = entry["kind"]
        kid = entry["kid"]
        replacement = entry["replacement"]
        path = entry["path"]
        nonce = base64.b64decode(entry["nonce"])
        ct = base64.b64decode(entry["ct"])
        pos = start + shift
        if result[pos:pos + len(replacement)] != replacement:
            raise ValueError("placeholder mismatch")
        plaintext_bytes = AESGCM(provider.get(kid)).decrypt(
            nonce,
            ct,
            _aad(ts, event_type, path, start, end, kind, kid),
        )
        plaintext = plaintext_bytes.decode("utf-8")
        if len(plaintext) != end - start:
            raise ValueError("plaintext length mismatch")
        result = result[:pos] + plaintext + result[pos + len(replacement):]
        shift += len(plaintext) - len(replacement)
    return result
