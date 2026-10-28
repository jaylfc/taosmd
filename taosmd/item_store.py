import json
import hashlib

import numpy as np

from . import _db


class SpaceMismatchError(ValueError):
    """Raised when an embedder's space_id does not match the query space_id."""
    pass


SCHEMA = """
CREATE TABLE IF NOT EXISTS blobs(
    sha256 TEXT PRIMARY KEY,
    mime TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    stored_path TEXT NOT NULL,
    first_seen_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS items(
    item_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    blob_sha256 TEXT,
    span_json TEXT NOT NULL DEFAULT '{}',
    parent_item_id TEXT,
    scope_agent TEXT,
    scope_collection TEXT,
    archive_event_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    valid_from REAL,
    valid_to REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS derivations(
    deriv_id TEXT PRIMARY KEY,
    item_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('caption','ocr','asr','summary','text')),
    producer_id TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS embedders(
    embedder_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    revision TEXT NOT NULL,
    file_sha256 TEXT NOT NULL,
    runtime TEXT NOT NULL,
    quant TEXT NOT NULL,
    pooling TEXT NOT NULL,
    prefix_version TEXT NOT NULL,
    normalised INTEGER NOT NULL,
    dim INTEGER NOT NULL,
    space_id TEXT NOT NULL,
    layout TEXT NOT NULL CHECK (layout IN ('pooled','multi')),
    binary_quant INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS embeddings(
    target_kind TEXT NOT NULL CHECK (target_kind IN ('item','derivation')),
    target_id TEXT NOT NULL,
    embedder_id TEXT NOT NULL,
    modality TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL,
    n_vecs INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    PRIMARY KEY (target_kind, target_id, embedder_id)
);
CREATE TABLE IF NOT EXISTS coverage(
    embedder_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    n_targets INTEGER NOT NULL,
    n_embedded INTEGER NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (embedder_id, scope)
);
CREATE VIRTUAL TABLE IF NOT EXISTS derivations_fts USING fts5(deriv_id UNINDEXED, text);
"""


def _stable_hash(values):
    """Compute sha256 hex of json-encoded values with stable ordering."""
    data = json.dumps(values, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def item_id(archive_event_id: str, span: dict) -> str:
    return _stable_hash([archive_event_id, span])


def deriv_id(item_id: str, kind: str, producer_id: str) -> str:
    return _stable_hash([item_id, kind, producer_id])


def embedder_id(model, revision, file_sha256, runtime, quant, pooling,
                prefix_version, normalised, dim, layout, binary_quant=0) -> str:
    return _stable_hash([
        model,
        revision,
        file_sha256,
        runtime,
        quant,
        pooling,
        prefix_version,
        normalised,
        dim,
        layout,
        binary_quant,
    ])


class ItemStore:
    def __init__(self, path):
        self.path = path
        self.conn = _db.connect(path)
        self.init()

    def init(self):
        _db.run_schema(self.conn, SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def put_item(self, archive_event_id, span, kind, created_at, **optional_columns):
        iid = item_id(archive_event_id, span)
        span_json = json.dumps(span, sort_keys=True)
        optional = {
            'blob_sha256': optional_columns.get('blob_sha256'),
            'parent_item_id': optional_columns.get('parent_item_id'),
            'scope_agent': optional_columns.get('scope_agent'),
            'scope_collection': optional_columns.get('scope_collection'),
            'valid_from': optional_columns.get('valid_from'),
            'valid_to': optional_columns.get('valid_to'),
            'metadata_json': optional_columns.get('metadata_json', '{}'),
        }
        cur = self.conn.cursor()
        cur.execute(
            'INSERT OR IGNORE INTO items (item_id, kind, blob_sha256, span_json, parent_item_id, '
            'scope_agent, scope_collection, archive_event_id, created_at, valid_from, valid_to, metadata_json) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                iid,
                kind,
                optional['blob_sha256'],
                span_json,
                optional['parent_item_id'],
                optional['scope_agent'],
                optional['scope_collection'],
                archive_event_id,
                created_at,
                optional['valid_from'],
                optional['valid_to'],
                optional['metadata_json'],
            ),
        )
        self.conn.commit()
        return iid

    def put_derivation(self, item_id, kind, producer_id, text, created_at):
        did = deriv_id(item_id, kind, producer_id)
        cur = self.conn.cursor()
        cur.execute(
            'INSERT OR IGNORE INTO derivations (deriv_id, item_id, kind, producer_id, text, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (did, item_id, kind, producer_id, text, created_at),
        )
        if cur.rowcount == 1:
            cur.execute(
                'INSERT OR IGNORE INTO derivations_fts (deriv_id, text) VALUES (?, ?)',
                (did, text),
            )
        self.conn.commit()
        return did

    def register_embedder(self, model, revision, file_sha256, runtime, quant, pooling,
                          prefix_version, normalised, dim, space_id, layout, binary_quant=0):
        eid = embedder_id(model, revision, file_sha256, runtime, quant, pooling,
                          prefix_version, normalised, dim, layout, binary_quant)
        cur = self.conn.cursor()
        cur.execute('SELECT space_id FROM embedders WHERE embedder_id = ?', (eid,))
        row = cur.fetchone()
        if row:
            if row[0] != space_id:
                raise SpaceMismatchError
            return eid
        cur.execute(
            'INSERT INTO embedders (embedder_id, model, revision, file_sha256, runtime, quant, '
            'pooling, prefix_version, normalised, dim, space_id, layout, binary_quant) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (eid, model, revision, file_sha256, runtime, quant, pooling,
             prefix_version, normalised, dim, space_id, layout, binary_quant),
        )
        self.conn.commit()
        return eid

    def put_embedding(self, target_kind, target_id, embedder_id, modality, vec: np.ndarray, created_at):
        cur = self.conn.cursor()
        cur.execute('SELECT dim, layout FROM embedders WHERE embedder_id = ?', (embedder_id,))
        row = cur.fetchone()
        if not row:
            raise KeyError(embedder_id)
        embedder_dim, layout = row[0], row[1]
        vec = np.asarray(vec, dtype=np.float32)
        if layout == 'pooled':
            if vec.ndim != 1 or vec.shape[0] != embedder_dim:
                raise ValueError
            n_vecs = 1
        elif layout == 'multi':
            if vec.ndim != 2 or vec.shape[1] != embedder_dim:
                raise ValueError
            n_vecs = int(vec.shape[0])
        else:
            raise ValueError
        cur.execute(
            'INSERT OR REPLACE INTO embeddings (target_kind, target_id, embedder_id, modality, '
            'dim, vec, n_vecs, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (target_kind, target_id, embedder_id, modality, embedder_dim,
             vec.tobytes(), n_vecs, created_at),
        )
        self.conn.commit()

    def score(self, query_vec, query_space_id, embedder_id):
        cur = self.conn.cursor()
        cur.execute('SELECT dim, layout, space_id FROM embedders WHERE embedder_id = ?', (embedder_id,))
        row = cur.fetchone()
        if not row:
            raise KeyError(embedder_id)
        embedder_dim, layout, space_id = row[0], row[1], row[2]
        if space_id != query_space_id:
            raise SpaceMismatchError
        if layout != 'pooled':
            raise ValueError('score only supports pooled layout')
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if q.shape[0] != embedder_dim:
            raise ValueError
        cur.execute(
            'SELECT target_kind, target_id, dim, vec FROM embeddings WHERE embedder_id = ?',
            (embedder_id,),
        )
        results = []
        qn = np.linalg.norm(q)
        if qn == 0.0:
            qn = 1.0
        for target_kind, target_id, dim, vec_blob in cur.fetchall():
            v = np.frombuffer(vec_blob, dtype=np.float32)
            if v.shape[0] != dim:
                continue
            vn = np.linalg.norm(v)
            if vn == 0.0:
                vn = 1.0
            cos = float(np.dot(q, v) / (qn * vn))
            results.append((target_kind, target_id, cos))
        results.sort(key=lambda x: x[2], reverse=True)
        return results
