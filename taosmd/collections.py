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

import asyncio
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


# ---------------------------------------------------------------------------
# Folder walker (gitignore-aware, stdlib only)
# ---------------------------------------------------------------------------

def _parse_gitignore(path: Path) -> list[tuple[str, bool, bool]]:
    """Parse one .gitignore into ``(pattern, negate, dir_only)`` rules.

    Simplified gitwildmatch: comments and blanks dropped, ``!`` negation,
    trailing ``/`` marks dir-only, leading ``/`` anchors to the .gitignore's
    own directory, patterns without ``/`` match the basename at any depth,
    patterns with ``/`` are fnmatch-ed against the path relative to the
    .gitignore's directory.
    # upgrade-path: full gitwildmatch (``**`` semantics, escaped chars) if the
    # simplified rules ever mis-walk a real repo.
    """
    rules: list[tuple[str, bool, bool]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return rules
    for line in lines:
        line = line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        negate = line.startswith("!")
        if negate:
            line = line[1:]
        dir_only = line.endswith("/")
        pattern = line.rstrip("/")
        if pattern:
            rules.append((pattern, negate, dir_only))
    return rules


def _match_one(pattern: str, rel: str, name: str) -> bool:
    if pattern.startswith("/"):
        return fnmatch.fnmatch(rel, pattern[1:])
    if "/" in pattern:
        return fnmatch.fnmatch(rel, pattern)
    return fnmatch.fnmatch(name, pattern)


def _is_ignored(
    rel_posix: str,
    name: str,
    is_dir: bool,
    rule_sets: list[tuple[str, list[tuple[str, bool, bool]]]],
) -> bool:
    """Apply the collected .gitignore rule sets to one path.

    ``rule_sets`` is ``[(prefix, rules)]`` where ``prefix`` is the rule
    file's directory relative to the source root (``""`` for the root).
    Later-matching rules win, mirroring git's last-match-wins semantics.
    """
    ignored = False
    for prefix, rules in rule_sets:
        if prefix:
            if not rel_posix.startswith(prefix + "/"):
                continue
            sub = rel_posix[len(prefix) + 1:]
        else:
            sub = rel_posix
        for pattern, negate, dir_only in rules:
            if dir_only and not is_dir:
                continue
            if _match_one(pattern, sub, name):
                ignored = not negate
    return ignored


def _loader_for(path: Path):
    """Return an instance of the loader that explicitly claims ``path``,
    or ``None`` when no registered loader does.

    This deliberately does NOT use ``pick_loader``'s catch-all fallback:
    the walker only ingests files a loader positively claims (DocLoader
    md/txt/markdown/rst, ChatLoader *.chat.json, ...), so arbitrary source
    files are skipped rather than mangled through the doc path.
    """
    import mimetypes  # noqa: PLC0415

    ext = _path_to_extension(path)
    mime, _ = mimetypes.guess_type(str(path))
    for loader_cls in _LOADER_REGISTRY:
        if loader_cls.can_handle(extension=ext, mime_type=mime or ""):
            return loader_cls()
    return None


def collect_files(
    source_root: Path | str,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> tuple[list[tuple[Path, str]], dict]:
    """Walk ``source_root`` and return ``([(abs_path, rel_posix)], skips)``.

    Gitignore-aware (root and nested .gitignore files), skips VCS/dependency
    directories and hidden directories, binary files (by extension and by a
    null-byte sniff), files over ``max_file_bytes``, symlinks that escape the
    root, and files no registered loader claims. ``skips`` counts each skip
    reason so ingest stats can surface them.
    """
    root = Path(source_root).resolve()
    skips = {
        "skipped_ignored": 0,
        "skipped_binary": 0,
        "skipped_size": 0,
        "skipped_symlink": 0,
        "skipped_unclaimed": 0,
    }
    files: list[tuple[Path, str]] = []
    rule_sets: list[tuple[str, list[tuple[str, bool, bool]]]] = []

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dpath = Path(dirpath)
        rel_dir = "" if dpath == root else dpath.relative_to(root).as_posix()

        gi = dpath / ".gitignore"
        if gi.is_file():
            rules = _parse_gitignore(gi)
            if rules:
                rule_sets.append((rel_dir, rules))

        keep_dirs = []
        for d in sorted(dirnames):
            rel_d = f"{rel_dir}/{d}" if rel_dir else d
            if d in _SKIP_DIRS or d.startswith("."):
                continue
            if _is_ignored(rel_d, d, True, rule_sets):
                skips["skipped_ignored"] += 1
                continue
            keep_dirs.append(d)
        dirnames[:] = keep_dirs

        for fname in sorted(filenames):
            fpath = dpath / fname
            rel_f = f"{rel_dir}/{fname}" if rel_dir else fname
            if _is_ignored(rel_f, fname, False, rule_sets):
                skips["skipped_ignored"] += 1
                continue
            try:
                resolve_within(fpath, root)
            except ValueError:
                skips["skipped_symlink"] += 1
                continue
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
            if ext in _BINARY_EXTS:
                skips["skipped_binary"] += 1
                continue
            try:
                check_size(fpath, max_file_bytes)
            except ValueError:
                skips["skipped_size"] += 1
                continue
            except OSError:
                skips["skipped_symlink"] += 1
                continue
            if _loader_for(fpath) is None:
                skips["skipped_unclaimed"] += 1
                continue
            try:
                with open(fpath, "rb") as fh:
                    head = fh.read(1024)
            except OSError:
                skips["skipped_symlink"] += 1
                continue
            if b"\x00" in head:
                skips["skipped_binary"] += 1
                continue
            files.append((fpath, rel_f))
    return files, skips


# ---------------------------------------------------------------------------
# Chunker (zero-dep)
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars`` characters.

    Greedy paragraph packing: paragraphs (blank-line separated) are packed
    into chunks until the cap; a paragraph longer than the cap is hard-split
    at the nearest space. Zero-loss: concatenating the chunks preserves every
    paragraph's content.
    # upgrade-path: heading/structure-aware chunking (keep md sections whole)
    # and token-based budgets once the Phase-1 eval says boundaries matter.
    """
    import re  # noqa: PLC0415

    text = text.strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for para in paras:
        while len(para) > max_chars:
            cut = para.rfind(" ", 1, max_chars)
            if cut <= 0:
                cut = max_chars
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(para[:cut].strip())
            para = para[cut:].strip()
        if not para:
            continue
        if not cur:
            cur = para
        elif len(cur) + 2 + len(para) <= max_chars:
            cur = f"{cur}\n\n{para}"
        else:
            chunks.append(cur)
            cur = para
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# Ingest pipeline
# ---------------------------------------------------------------------------

def _supersede_collection_rows(vmem, collection_id: str, file_path: str) -> int:
    """Soft-supersede the active vector rows of one collection file.

    Zero-loss: rows are stamped ``valid_to`` (the existing supersede
    machinery) with a ``hidden_by: collection-reindex:<ts>`` marker in their
    metadata; nothing is deleted and the archive rows are untouched. Used on
    re-index for changed and deleted files, mirroring the shelf-archive
    pattern in :mod:`taosmd.admin`.
    """
    ts = time.time()
    marker = f"collection-reindex:{ts}"
    rows = vmem._conn.execute(
        "SELECT id, metadata_json FROM vector_memory WHERE valid_to IS NULL"
    ).fetchall()
    superseded = 0
    for row in rows:
        try:
            meta = json.loads(row["metadata_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        user_md = meta.get("metadata") if isinstance(meta, dict) else None
        if not isinstance(user_md, dict):
            continue
        if user_md.get("collection_id") != collection_id:
            continue
        if user_md.get("file_path") != file_path:
            continue
        meta["hidden_by"] = marker
        vmem._conn.execute(
            "UPDATE vector_memory SET valid_to = ?, metadata_json = ? "
            "WHERE id = ? AND valid_to IS NULL",
            (ts, json.dumps(meta), row["id"]),
        )
        superseded += 1
    if superseded:
        vmem._conn.commit()
        vmem._bm25_dirty = True
    return superseded


def _hash_and_chunk(
    text: str, chunk_chars: int, prior_hash: str | None
) -> tuple[str, list[str] | None]:
    """CPU-bound half of the per-file ingest step, run off the event loop.

    Returns ``(file_hash, chunks)``; ``chunks`` is ``None`` when the hash
    matches ``prior_hash`` (unchanged file, nothing to chunk).
    """
    file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if prior_hash == file_hash:
        return file_hash, None
    return file_hash, chunk_text(text, max_chars=chunk_chars)


async def ingest_folder(
    collection_id: str,
    *,
    data_dir=None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    chunk_chars: int = 2000,
) -> dict:
    """Walk a collection's source folder and index its documents.

    Incremental by content hash: files whose hash matches the stored state
    are skipped; changed files get their old rows superseded (never deleted)
    and their new chunks ingested; files that disappeared from the source
    are superseded too. Chunks route through :func:`taosmd.api.ingest_batch`
    under the collection id as the agent namespace, so the batch dedup and
    metadata preservation come for free and every chunk lands in the
    zero-loss archive.

    Sets the collection status to ``indexing`` for the duration, then
    ``ready`` (stats updated, ``last_indexed`` stamped) or ``error`` (the
    failure recorded on the row). Raises on validation errors so a caller
    driving it synchronously still sees them.
    """
    from . import api as _api  # noqa: PLC0415 - avoid import cycle at module load

    resolved_dir = _api._resolve_data_dir(data_dir)
    store = CollectionStore(resolved_dir)
    col = store.get(collection_id)
    if col["status"] == "archived":
        raise ValueError(f"collection {collection_id!r} is archived; unarchive before indexing")
    if col["embedder"]:
        # Per-collection embedder is stored and returned now (the mechanism);
        # Phase 1 indexes with the global default regardless.
        logger.info(
            "collection %s requests embedder %r; Phase 1 indexes with the "
            "global default embedder", collection_id, col["embedder"],
        )
    store.set_status(collection_id, "indexing")
    try:
        source_root = store.resolve_source_path(col["source_path"])
        prior = store.file_states(collection_id)
        # The walk (os.walk + per-file stat + null-byte sniff over every
        # candidate) is blocking filesystem work; run it on a worker thread so
        # the 202+poll contract holds and the single service loop keeps
        # serving /search and /ingest while a collection indexes.
        files, skips = await asyncio.to_thread(
            collect_files, source_root, max_file_bytes=max_file_bytes,
        )

        stores = await _api._ensure_stores(data_dir)
        vmem = stores["vector"]

        items: list[dict] = []
        indexed: list[tuple[str, str]] = []
        changed: list[str] = []
        unchanged = 0
        errors: list[str] = []
        seen: set[str] = set()

        for abs_path, rel in files:
            seen.add(rel)
            loader = _loader_for(abs_path)
            if loader is None:  # pragma: no cover - collect_files already filtered
                continue
            try:
                blob = await loader.load(
                    abs_path, max_bytes=max_file_bytes, base_dir=source_root
                )
            except Exception as exc:  # noqa: BLE001 - per-file failures are non-fatal
                errors.append(f"{rel}: {type(exc).__name__}: {exc}")
                continue
            text = blob.raw_text or getattr(blob, "content", "") or ""
            if not text.strip():
                continue
            # Hashing + chunking are CPU-bound; off-loop like the walk above.
            file_hash, chunks = await asyncio.to_thread(
                _hash_and_chunk, text, chunk_chars, prior.get(rel)
            )
            if chunks is None:
                unchanged += 1
                continue
            if rel in prior:
                changed.append(rel)
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.sha256(
                    f"{collection_id}:{rel}:{file_hash}:{i}".encode("utf-8")
                ).hexdigest()
                items.append({
                    "text": chunk,
                    "id": chunk_id,
                    "metadata": {
                        "collection_id": collection_id,
                        "file_path": rel,
                        "source": "collection",
                        "chunk_index": i,
                        "file_hash": file_hash,
                    },
                })
            indexed.append((rel, file_hash))

        deleted = sorted(set(prior) - seen)

        chunks_superseded = 0
        for rel in [*changed, *deleted]:
            chunks_superseded += _supersede_collection_rows(vmem, collection_id, rel)

        if items:
            result = await _api.ingest_batch(items, agent=collection_id, data_dir=data_dir)
        else:
            result = {"ingested": 0, "skipped": 0}

        for rel, file_hash in indexed:
            store.set_file_state(collection_id, rel, file_hash)
        for rel in deleted:
            store.remove_file_state(collection_id, rel)

        now = time.time()
        stats = {
            "files_indexed": len(indexed),
            "files_unchanged": unchanged,
            "files_deleted": len(deleted),
            "files_total": len(store.file_states(collection_id)),
            "chunks_ingested": result.get("ingested", 0),
            "chunks_skipped": result.get("skipped", 0),
            "chunks_superseded": chunks_superseded,
            "errors": errors[:20],
            **skips,
        }
        if result.get("vector_failures"):
            stats["vector_failures"] = result["vector_failures"]
            stats["degraded"] = True
        store.set_stats(collection_id, stats)
        store.set_status(collection_id, "ready", last_indexed=now)
        return stats
    except Exception as exc:
        store.set_status(collection_id, "error", error=f"{type(exc).__name__}: {exc}")
        raise
