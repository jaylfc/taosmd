"""Collections: named, typed containers for content indexed from a folder.

A collection is a first-class row (not a metadata tag): it names one source
folder, tracks its indexing lifecycle (created -> indexing -> ready | error),
and composes with the existing scoping surfaces through two side tables:

- **links** attach a collection to one or more projects for discovery. A link
  row is typed ``{type: "taos" | "git", id}`` because the taOS project id
  (``prj-xxx``) and the taOSmd git-fingerprint project id (12 hex) are
  different namespaces; a collection for a repo that taOS also manages
  typically carries one of each. Links are metadata only and never grant
  query access (no transitivity).
- **grants** give a named agent query access to a collection. A grant row is
  ``(canonical_id, scope='collection', collection_id)``, unique together,
  and is enforced at search time: collection hits are only returned to
  agents holding a grant.

Safety: a collection's ``source_path`` must resolve inside one of the
directories in the ``collections.allowed_roots`` config list (default EMPTY,
which turns the feature off), checked at create time and again at every
index. Symlink escapes are rejected by ``loaders._safety.resolve_within``.

Zero-loss: DELETE is an alias for archive (``status='archived'``,
reversible); re-indexing supersedes replaced rows via the existing
``valid_to`` machinery and never deletes; only the wipe surface destroys.

Content rows live in the normal vector/archive stores under the agent name
``<collection id>`` (collection ids match the agent-name grammar by
construction), so the existing per-agent search scoping doubles as the
collection scoping mechanism with no new query machinery.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import secrets
import sqlite3
import time
from pathlib import Path

from . import config as _config
from .loaders import check_size, resolve_within
from .loaders.registry import REGISTRY as _LOADER_REGISTRY, _path_to_extension

logger = logging.getLogger(__name__)

KINDS = ("docs", "codebase", "mixed")
STATUSES = ("created", "indexing", "ready", "error", "archived")
LINK_TYPES = ("taos", "git")
GRANT_SCOPE = "collection"

#: Per-file size cap for the folder walker. Deliberately far below the
#: loaders' generous 100 MB default: a docs collection should never contain
#: a file this large, and the cap stops a stray artifact from bloating the
#: index. Overridable per ingest_folder call.
DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024

#: Directory names never descended into, regardless of gitignore rules.
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".ruff_cache", ".pytest_cache",
})

#: Extensions treated as binary and skipped without opening the file.
_BINARY_EXTS = frozenset({
    "png", "jpg", "jpeg", "gif", "webp", "ico", "bmp", "pdf", "zip", "gz",
    "bz2", "xz", "tar", "7z", "whl", "so", "dylib", "dll", "exe", "bin",
    "onnx", "gguf", "pt", "safetensors", "db", "sqlite", "sqlite3", "woff",
    "woff2", "ttf", "eot", "otf", "mp3", "mp4", "mov", "avi", "wav", "flac",
    "pyc", "pyo", "class", "jar", "o", "a", "ds_store",
})


class CollectionNotFoundError(KeyError):
    """Raised when referencing a collection id that does not exist."""


def _new_collection_id() -> str:
    """``col-`` + 12 lowercase hex. Matches the agent-name grammar
    (``^[a-z][a-z0-9_-]{0,62}$``) so the id can double as the agent name
    under which the collection's content rows are stored."""
    return f"col-{secrets.token_hex(6)}"


class CollectionStore:
    """SQLite-backed store for collection rows, links, grants, and the
    per-file hash state that backs incremental re-index.

    Synchronous by design (like :class:`taosmd.agents.AgentRegistry`); all
    callers route through the single service loop so the thread-affine
    connection is used from exactly one thread.
    """

    def __init__(self, data_dir) -> None:
        self._data_dir = os.fspath(data_dir)
        path = Path(self._data_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path / "collections.db"))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS collections (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                source_path TEXT NOT NULL,
                embedder TEXT,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_indexed REAL,
                stats_json TEXT NOT NULL DEFAULT '{}',
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS collection_links (
                collection_id TEXT NOT NULL,
                type TEXT NOT NULL,
                ext_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(collection_id, type, ext_id)
            );
            CREATE TABLE IF NOT EXISTS collection_grants (
                canonical_id TEXT NOT NULL,
                scope TEXT NOT NULL DEFAULT 'collection',
                collection_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(canonical_id, scope, collection_id)
            );
            CREATE TABLE IF NOT EXISTS collection_files (
                collection_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(collection_id, file_path)
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ----- validation ------------------------------------------------------

    def resolve_source_path(self, source_path: str) -> Path:
        """Validate ``source_path`` against the allowed roots; return it resolved.

        Raises ``ValueError`` when no allowed roots are configured (feature
        off), when the path escapes every root (including via symlink), or
        when it is not an existing directory. Called at create time AND again
        at every index, so a root removed from config retroactively disables
        indexing of collections created under it.
        """
        roots = _config.get_collections_allowed_roots(self._data_dir)
        if not roots:
            raise ValueError(
                "collections are disabled: no collections.allowed_roots configured "
                "(set the config key or TAOSMD_COLLECTIONS_ALLOWED_ROOTS)"
            )
        resolved = None
        for root in roots:
            try:
                resolved = resolve_within(source_path, root)
                break
            except ValueError:
                continue
        if resolved is None:
            raise ValueError(
                f"source_path {source_path!r} is outside every configured allowed root"
            )
        if not resolved.is_dir():
            raise ValueError(f"source_path {source_path!r} is not a directory")
        return resolved

    # ----- CRUD ------------------------------------------------------------

    def create(
        self,
        *,
        name: str,
        kind: str,
        source_path: str,
        embedder: str | None = None,
    ) -> dict:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {'|'.join(KINDS)}, got {kind!r}")
        if embedder is not None and (not isinstance(embedder, str) or not embedder.strip()):
            raise ValueError("embedder must be a non-empty string when provided")
        resolved = self.resolve_source_path(source_path)
        cid = _new_collection_id()
        self._conn.execute(
            "INSERT INTO collections (id, name, kind, source_path, embedder, status, "
            "created_at, stats_json) VALUES (?, ?, ?, ?, ?, 'created', ?, '{}')",
            (cid, name.strip(), kind, str(resolved),
             embedder.strip() if embedder else None, time.time()),
        )
        self._conn.commit()
        return self.get(cid)

    def _row(self, collection_id: str) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT * FROM collections WHERE id = ?", (collection_id,)
        ).fetchone()
        if row is None:
            raise CollectionNotFoundError(f"collection {collection_id!r} not found")
        return row

    def get(self, collection_id: str) -> dict:
        row = self._row(collection_id)
        try:
            stats = json.loads(row["stats_json"])
        except (json.JSONDecodeError, TypeError):
            stats = {}
        links = [
            {"type": r["type"], "id": r["ext_id"]}
            for r in self._conn.execute(
                "SELECT type, ext_id FROM collection_links WHERE collection_id = ? "
                "ORDER BY created_at",
                (collection_id,),
            )
        ]
        grants = [
            r["canonical_id"]
            for r in self._conn.execute(
                "SELECT canonical_id FROM collection_grants "
                "WHERE collection_id = ? AND scope = ? ORDER BY created_at",
                (collection_id, GRANT_SCOPE),
            )
        ]
        return {
            "id": row["id"],
            "name": row["name"],
            "kind": row["kind"],
            "source_path": row["source_path"],
            "embedder": row["embedder"],
            "status": row["status"],
            "created_at": row["created_at"],
            "last_indexed": row["last_indexed"],
            "stats": stats if isinstance(stats, dict) else {},
            "error": row["error"],
            "links": links,
            "grants": grants,
        }

    def list(
        self, *, project: str | None = None, include_archived: bool = False
    ) -> list[dict]:
        """All collections, oldest first. ``project`` filters to collections
        holding a link whose ``ext_id`` matches, regardless of link type
        (taOS ``prj-*`` ids and git fingerprints are different namespaces,
        so a raw id is unambiguous in practice)."""
        rows = self._conn.execute(
            "SELECT id, status FROM collections ORDER BY created_at"
        ).fetchall()
        out = []
        for row in rows:
            if not include_archived and row["status"] == "archived":
                continue
            col = self.get(row["id"])
            if project is not None and not any(
                link["id"] == project for link in col["links"]
            ):
                continue
            out.append(col)
        return out

    def archive(self, collection_id: str) -> dict:
        """Archive a collection (the DELETE alias). Reversible: the row and
        every content row stay on disk; search-time grant resolution skips
        archived collections, which hides the content from query."""
        self._row(collection_id)  # 404 check
        self._conn.execute(
            "UPDATE collections SET status = 'archived' WHERE id = ?",
            (collection_id,),
        )
        self._conn.commit()
        return self.get(collection_id)

    def set_status(
        self,
        collection_id: str,
        status: str,
        *,
        error: str | None = None,
        last_indexed: float | None = None,
    ) -> None:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {'|'.join(STATUSES)}, got {status!r}")
        self._row(collection_id)
        self._conn.execute(
            "UPDATE collections SET status = ?, error = ?, "
            "last_indexed = COALESCE(?, last_indexed) WHERE id = ?",
            (status, error if status == "error" else None, last_indexed, collection_id),
        )
        self._conn.commit()

    def set_stats(self, collection_id: str, stats: dict) -> None:
        self._row(collection_id)
        self._conn.execute(
            "UPDATE collections SET stats_json = ? WHERE id = ?",
            (json.dumps(stats), collection_id),
        )
        self._conn.commit()

    # ----- links -----------------------------------------------------------

    def link(self, collection_id: str, link_type: str, ext_id: str) -> dict:
        if link_type not in LINK_TYPES:
            raise ValueError(
                f"link type must be one of {'|'.join(LINK_TYPES)}, got {link_type!r}"
            )
        if not isinstance(ext_id, str) or not ext_id.strip():
            raise ValueError("link id must be a non-empty string")
        self._row(collection_id)
        self._conn.execute(
            "INSERT OR IGNORE INTO collection_links "
            "(collection_id, type, ext_id, created_at) VALUES (?, ?, ?, ?)",
            (collection_id, link_type, ext_id.strip(), time.time()),
        )
        self._conn.commit()
        return self.get(collection_id)

    def unlink(self, collection_id: str, link_type: str, ext_id: str) -> dict:
        self._row(collection_id)
        self._conn.execute(
            "DELETE FROM collection_links "
            "WHERE collection_id = ? AND type = ? AND ext_id = ?",
            (collection_id, link_type, ext_id),
        )
        self._conn.commit()
        return self.get(collection_id)

    # ----- grants ----------------------------------------------------------

    def grant(self, collection_id: str, canonical_id: str) -> dict:
        if not isinstance(canonical_id, str) or not canonical_id.strip():
            raise ValueError("agent (canonical_id) must be a non-empty string")
        self._row(collection_id)
        self._conn.execute(
            "INSERT OR IGNORE INTO collection_grants "
            "(canonical_id, scope, collection_id, created_at) VALUES (?, ?, ?, ?)",
            (canonical_id.strip(), GRANT_SCOPE, collection_id, time.time()),
        )
        self._conn.commit()
        return self.get(collection_id)

    def revoke(self, collection_id: str, canonical_id: str) -> dict:
        self._row(collection_id)
        self._conn.execute(
            "DELETE FROM collection_grants "
            "WHERE canonical_id = ? AND scope = ? AND collection_id = ?",
            (canonical_id, GRANT_SCOPE, collection_id),
        )
        self._conn.commit()
        return self.get(collection_id)

    def has_grant(self, canonical_id: str, collection_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM collection_grants "
            "WHERE canonical_id = ? AND scope = ? AND collection_id = ?",
            (canonical_id, GRANT_SCOPE, collection_id),
        ).fetchone()
        return row is not None

    # ----- per-file hash state (incremental re-index) ----------------------

    def file_states(self, collection_id: str) -> dict[str, str]:
        return {
            r["file_path"]: r["content_hash"]
            for r in self._conn.execute(
                "SELECT file_path, content_hash FROM collection_files "
                "WHERE collection_id = ?",
                (collection_id,),
            )
        }

    def set_file_state(self, collection_id: str, file_path: str, content_hash: str) -> None:
        self._conn.execute(
            "INSERT INTO collection_files (collection_id, file_path, content_hash, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(collection_id, file_path) "
            "DO UPDATE SET content_hash = excluded.content_hash, "
            "updated_at = excluded.updated_at",
            (collection_id, file_path, content_hash, time.time()),
        )
        self._conn.commit()

    def remove_file_state(self, collection_id: str, file_path: str) -> None:
        self._conn.execute(
            "DELETE FROM collection_files WHERE collection_id = ? AND file_path = ?",
            (collection_id, file_path),
        )
        self._conn.commit()
