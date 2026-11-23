"""Per-record data-key wrapping to an install X25519 key.

Every archive record that carries secret spans gets a fresh random 32-byte
data key (DEK). The DEK encrypts the spans through the existing AES-GCM codec
(taosmd.secret_spans) and is itself wrapped to the install X25519 public key
-- X25519 ECDH with an ephemeral key, HKDF-SHA256, AES-256-GCM -- and appended
to the append-only wrap log ``<keys_dir>/wraps/<recipient_kid>.jsonl``. The
wrap entry is fsynced before the archive line is written, so a crash can leave
an orphaned wrap entry but never a sealed line whose DEK cannot be unwrapped.

The install key pair lives in ``keys_dir``: directory mode 0700, private key
file mode 0600, created on first init if absent, never under the archive
directory. ``init_install_key()`` fails loud if a key exists but cannot be
read.

``rewrap()`` moves DEKs from the install key to a new recipient key without
touching any archive line, so sealed records are re-lockable under a new key.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.hashes import SHA256

PRIVATE_KEY_FILE = "install_private_key"
PUBLIC_KEY_FILE = "install_public_key"
WRAPS_DIR = "wraps"
WRAP_KDF_INFO = b"taosmd/archive/v1/dek-wrap-key"
WRAP_NONCE_INFO = b"taosmd/archive/v1/dek-wrap-nonce"
WRAP_AAD = b"taosmd/archive/v1/dek-wrap-aad"


class InstallKeyError(Exception):
    """The install key exists but cannot be loaded or created."""


class InstallKey:
    """The install X25519 key pair sealed records are wrapped to."""

    def __init__(
        self,
        keys_dir: Path,
        private_key: X25519PrivateKey,
        public_key: X25519PublicKey,
    ) -> None:
        self.keys_dir = keys_dir
        self.private_key = private_key
        self.public_key = public_key
        self.recipient_kid = hashlib.sha256(public_key.public_bytes_raw()).hexdigest()


def init_install_key(keys_dir: str | Path) -> InstallKey:
    """Create or load the install key pair for ``keys_dir``.

    The directory is created mode 0700 on first use and the private key file
    mode 0600. Existing key files are loaded; an unreadable or invalid key
    raises InstallKeyError -- there is never a silent fallback at record time.
    """
    keys_dir = Path(keys_dir)
    if not keys_dir.exists():
        keys_dir.mkdir(parents=True)
        os.chmod(keys_dir, 0o700)

    private_path = keys_dir / PRIVATE_KEY_FILE
    public_path = keys_dir / PUBLIC_KEY_FILE

    if private_path.exists():
        try:
            raw = private_path.read_bytes()
            private_key = X25519PrivateKey.from_private_bytes(raw)
        except (OSError, ValueError) as exc:
            raise InstallKeyError(
                f"install key at {private_path} cannot be read: {exc}"
            ) from exc
    else:
        private_key = X25519PrivateKey.generate()
        fd = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, private_key.private_bytes_raw())
        finally:
            os.close(fd)

    public_key = private_key.public_key()
    if not public_path.exists():
        fd = os.open(public_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, public_key.public_bytes_raw())
        finally:
            os.close(fd)

    return InstallKey(keys_dir, private_key, public_key)


def _wrap_key_and_nonce(shared_secret: bytes) -> tuple[bytes, bytes]:
    key = HKDF(
        algorithm=SHA256(), length=32, salt=None, info=WRAP_KDF_INFO
    ).derive(shared_secret)
    nonce = HKDF(
        algorithm=SHA256(), length=12, salt=None, info=WRAP_NONCE_INFO
    ).derive(shared_secret)
    return key, nonce


def wrap_dek(install_key: InstallKey, dek: bytes) -> dict[str, str]:
    """Wrap DEK bytes to the install public key.

    Returns ``{"wrapped_dek": b64, "eph_pub": b64}``. The GCM nonce is derived
    from the ephemeral agreement (HKDF), so no nonce is stored with the entry.
    """
    ephemeral = X25519PrivateKey.generate()
    shared = ephemeral.exchange(install_key.public_key)
    wrap_key, nonce = _wrap_key_and_nonce(shared)
    ct = AESGCM(wrap_key).encrypt(nonce, dek, WRAP_AAD)
    return {
        "wrapped_dek": base64.b64encode(ct).decode("ascii"),
        "eph_pub": base64.b64encode(
            ephemeral.public_key().public_bytes_raw()
        ).decode("ascii"),
    }


def _parse_recipient_key(value) -> X25519PublicKey:
    if isinstance(value, X25519PublicKey):
        return value
    if isinstance(value, str):
        raw = base64.b64decode(value)
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    else:
        raise TypeError(
            "to_public_key must be an X25519 key, raw bytes, or base64 str"
        )
    return X25519PublicKey.from_public_bytes(raw)


def unwrap_dek(install_key: InstallKey, dek_id: str) -> bytes:
    """Unwrap a DEK by scanning the install wrap log for ``dek_id``."""
    log_path = install_key.keys_dir / WRAPS_DIR / f"{install_key.recipient_kid}.jsonl"
    if not log_path.exists():
        raise KeyError(dek_id)
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("dek_id") == dek_id:
                break
        else:
            raise KeyError(dek_id)
    ephemeral_pub = _parse_recipient_key(entry["eph_pub"])
    shared = install_key.private_key.exchange(ephemeral_pub)
    wrap_key, nonce = _wrap_key_and_nonce(shared)
    return AESGCM(wrap_key).decrypt(
        nonce, base64.b64decode(entry["wrapped_dek"]), WRAP_AAD
    )


class DekProvider:
    """Per-record provider for encode_leaf / reveal_leaf.

    ``current()`` returns the record's own DEK; ``get(dek_id)`` unwraps any DEK
    from the install wrap log (the current DEK is logged there too, so reveal
    in a later process only needs the on-disk log).
    """

    def __init__(
        self,
        install_key: InstallKey,
        dek_id: str | None = None,
        dek: bytes | None = None,
    ) -> None:
        self._install_key = install_key
        self._dek_id = dek_id
        self._dek = dek

    def current(self) -> tuple[str, bytes]:
        if self._dek is None:
            raise KeyError("no current DEK")
        return self._dek_id, self._dek

    def get(self, dek_id: str) -> bytes:
        if self._dek is not None and dek_id == self._dek_id:
            return self._dek
        return unwrap_dek(self._install_key, dek_id)


def _wrap_log_path(install_key: InstallKey) -> Path:
    return (
        install_key.keys_dir / WRAPS_DIR / f"{install_key.recipient_kid}.jsonl"
    )


def append_wrap(
    install_key: InstallKey, dek_id: str, wrapped_dek: str, eph_pub: str
) -> str:
    """Append one wrap entry to the install wrap log and fsync it.

    Returns the ``line_key`` written with the entry. Callers must fsync this
    before writing the archive line that references the DEK.
    """
    log_path = _wrap_log_path(install_key)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(log_path.parent, 0o700)
    line_key = secrets.token_hex(16)
    entry = {
        "line_key": line_key,
        "dek_id": dek_id,
        "recipient_kid": install_key.recipient_kid,
        "wrapped_dek": wrapped_dek,
        "eph_pub": eph_pub,
    }
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(
            fd, (json.dumps(entry, separators=(",", ":")) + "\n").encode("utf-8")
        )
        os.fsync(fd)
    finally:
        os.close(fd)
    return line_key


def _load_wrap_entries(install_key: InstallKey) -> list[dict]:
    log_path = _wrap_log_path(install_key)
    if not log_path.exists():
        return []
    entries: list[dict] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _save_wrap_entries(install_key: InstallKey, entries: list[dict]) -> None:
    """Write entries to a temp file, fsync, then atomically replace the log."""
    log_path = _wrap_log_path(install_key)
    tmp_path = log_path.with_name(log_path.name + ".tmp")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        for entry in entries:
            os.write(
                fd,
                (json.dumps(entry, separators=(",", ":")) + "\n").encode("utf-8"),
            )
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp_path, log_path)
    dir_fd = None
    try:
        dir_fd = os.open(str(log_path.parent), os.O_RDONLY)
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        if dir_fd is not None:
            os.close(dir_fd)


def rewrap(
    dek_ids: list[str],
    to_public_key,
    to_kid: str,
    *,
    install_key: InstallKey | None = None,
    keys_dir: str | Path | None = None,
    to_keys_dir: str | Path | None = None,
) -> list[str]:
    """Re-wrap DEKs from the install key to a new recipient key.

    Each DEK in ``dek_ids`` is unwrapped with the install key, wrapped to
    ``to_public_key``, and the new wrap is decrypted to verify it opens. New
    wrap entries are appended to ``<to_keys_dir>/wraps/<to_kid>.jsonl``
    (default: the install key's ``keys_dir``) and fsynced, and only then are
    the matching install wrap entries removed (remaining entries written to a
    temp file, fsynced, atomically replaced).

    Archive lines are never touched: a sealed record stays byte-identical and
    becomes decryptable under the new key.

    Returns the list of DEK ids that were re-wrapped.
    """
    if install_key is None:
        if keys_dir is None:
            keys_dir = Path("data") / "keys"
        install_key = init_install_key(keys_dir)

    dest_dir = Path(to_keys_dir) if to_keys_dir is not None else install_key.keys_dir
    to_pub = _parse_recipient_key(to_public_key)

    new_entries: list[dict] = []
    now = time.time_ns()
    for dek_id in dek_ids:
        dek = unwrap_dek(install_key, dek_id)
        ephemeral = X25519PrivateKey.generate()
        shared = ephemeral.exchange(to_pub)
        wrap_key, nonce = _wrap_key_and_nonce(shared)
        ct = AESGCM(wrap_key).encrypt(nonce, dek, WRAP_AAD)
        # Verify the new wrap opens before any install log entry is removed.
        if AESGCM(wrap_key).decrypt(nonce, ct, WRAP_AAD) != dek:
            raise ValueError(f"new wrap did not open for dek {dek_id}")
        line_key = hashlib.sha256(
            f"{dek_id}:{to_kid}:{now}".encode("utf-8")
        ).hexdigest()
        new_entries.append(
            {
                "line_key": line_key,
                "dek_id": dek_id,
                "recipient_kid": to_kid,
                "wrapped_dek": base64.b64encode(ct).decode("ascii"),
                "eph_pub": base64.b64encode(
                    ephemeral.public_key().public_bytes_raw()
                ).decode("ascii"),
            }
        )

    new_log_path = dest_dir / WRAPS_DIR / f"{to_kid}.jsonl"
    new_log_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(new_log_path.parent, 0o700)
    fd = os.open(new_log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        for entry in new_entries:
            os.write(
                fd,
                (json.dumps(entry, separators=(",", ":")) + "\n").encode("utf-8"),
            )
        os.fsync(fd)
    finally:
        os.close(fd)

    # Only now: drop the install wrap entries for these DEKs.
    wanted = set(dek_ids)
    remaining = [
        entry for entry in _load_wrap_entries(install_key)
        if entry.get("dek_id") not in wanted
    ]
    _save_wrap_entries(install_key, remaining)
    return list(dek_ids)
