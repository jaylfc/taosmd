import numpy as np
import pytest

from taosmd.item_store import (
    ItemStore,
    SpaceMismatchError,
    item_id,
    deriv_id,
    embedder_id,
)

def test_deterministic_ids_stable_and_different(tmp_path):
    # Stable across calls
    assert item_id('ev1', {'x': 1}) == item_id('ev1', {'x': 1})
    # span key order must not change item_id
    assert item_id('ev1', {'x': 1, 'y': 2}) == item_id('ev1', {'y': 2, 'x': 1})
    # Different when inputs change
    assert item_id('ev1', {'x': 1}) != item_id('ev2', {'x': 1})
    assert item_id('ev1', {'x': 1}) != item_id('ev1', {'x': 2})
    assert deriv_id('i1', 'caption', 'p1') == deriv_id('i1', 'caption', 'p1')
    assert deriv_id('i1', 'caption', 'p1') != deriv_id('i2', 'caption', 'p1')
    assert deriv_id('i1', 'caption', 'p1') != deriv_id('i1', 'ocr', 'p1')
    assert deriv_id('i1', 'caption', 'p1') != deriv_id('i1', 'caption', 'p2')
    eid1 = embedder_id('m', 'r', 'f', 'cpu', 'f32', 'mean', 'v1', 1, 384, 'pooled', 0)
    eid2 = embedder_id('m', 'r', 'f', 'cpu', 'f32', 'mean', 'v1', 1, 384, 'pooled', 0)
    assert eid1 == eid2
    assert eid1 != embedder_id('m2', 'r', 'f', 'cpu', 'f32', 'mean', 'v1', 1, 384, 'pooled', 0)
    # space_id is NOT part of embedder_id
    assert eid1 == embedder_id('m', 'r', 'f', 'cpu', 'f32', 'mean', 'v1', 1, 384, 'pooled', 0)

def test_put_item_twice_same_event_span(tmp_path):
    store = ItemStore(tmp_path / 'test.db')
    iid1 = store.put_item('ev1', {'x': 1}, 'text', 123.0)
    iid2 = store.put_item('ev1', {'x': 1}, 'text', 123.0)
    assert iid1 == iid2
    cur = store.conn.cursor()
    cur.execute('SELECT count(*) FROM items')
    assert cur.fetchone()[0] == 1
    store.close()


def test_put_derivation_two_producers(tmp_path):
    store = ItemStore(tmp_path / 'test.db')
    iid = store.put_item('ev1', {'x': 1}, 'text', 123.0)
    d1 = store.put_derivation(iid, 'caption', 'p1', 'hello world', 124.0)
    d2 = store.put_derivation(iid, 'caption', 'p2', 'hi there', 125.0)
    assert d1 != d2
    cur = store.conn.cursor()
    cur.execute("SELECT deriv_id FROM derivations_fts WHERE derivations_fts MATCH 'hello'")
    res = [r[0] for r in cur.fetchall()]
    assert d1 in res
    cur.execute("SELECT deriv_id FROM derivations_fts WHERE derivations_fts MATCH 'there'")
    res = [r[0] for r in cur.fetchall()]
    assert d2 in res
    store.close()


def test_register_embedder_idempotent_and_space_mismatch(tmp_path):
    store = ItemStore(tmp_path / 'test.db')
    eid1 = store.register_embedder('m', 'r', 'f', 'cpu', 'f32', 'mean', 'v1', 1, 384, 'space1', 'pooled', 0)
    eid2 = store.register_embedder('m', 'r', 'f', 'cpu', 'f32', 'mean', 'v1', 1, 384, 'space1', 'pooled', 0)
    assert eid1 == eid2
    with pytest.raises(SpaceMismatchError):
        store.register_embedder('m', 'r', 'f', 'cpu', 'f32', 'mean', 'v1', 1, 384, 'space2', 'pooled', 0)
    store.close()


def test_score_matching_space(tmp_path):
    store = ItemStore(tmp_path / 'test.db')
    eid = store.register_embedder('m', 'r', 'f', 'cpu', 'f32', 'mean', 'v1', 1, 3, 'space1', 'pooled', 0)
    store.put_embedding('item', 't1', eid, 'text', np.array([1.0, 0.0, 0.0], dtype=np.float32), 123.0)
    store.put_embedding('item', 't2', eid, 'text', np.array([0.0, 1.0, 0.0], dtype=np.float32), 124.0)
    store.put_embedding('item', 't3', eid, 'text', np.array([0.6, 0.8, 0.0], dtype=np.float32), 125.0)
    res = store.score(np.array([1.0, 0.0, 0.0], dtype=np.float32), 'space1', eid)
    assert res[0][1] == 't1'
    assert res[0][0] == 'item'
    store.close()


def test_score_space_mismatch_raises(tmp_path):
    store = ItemStore(tmp_path / 'test.db')
    eid = store.register_embedder('m', 'r', 'f', 'cpu', 'f32', 'mean', 'v1', 1, 3, 'space1', 'pooled', 0)
    store.put_embedding('item', 't1', eid, 'text', np.array([1.0, 0.0, 0.0], dtype=np.float32), 123.0)
    with pytest.raises(SpaceMismatchError):
        store.score(np.array([1.0, 0.0, 0.0], dtype=np.float32), 'space2', eid)
    store.close()


def test_put_embedding_validation(tmp_path):
    store = ItemStore(tmp_path / 'test.db')
    eid = store.register_embedder('m', 'r', 'f', 'cpu', 'f32', 'mean', 'v1', 1, 3, 'space1', 'pooled', 0)
    with pytest.raises(ValueError):
        store.put_embedding('item', 't1', eid, 'text', np.array([1.0, 0.0], dtype=np.float32), 123.0)
    with pytest.raises(KeyError):
        store.put_embedding('item', 't1', 'nope', 'text', np.array([1.0, 0.0, 0.0], dtype=np.float32), 123.0)
    store.close()


def test_reopen_idempotent(tmp_path):
    store = ItemStore(tmp_path / 'test.db')
    eid = store.register_embedder('m', 'r', 'f', 'cpu', 'f32', 'mean', 'v1', 1, 3, 'space1', 'pooled', 0)
    iid = store.put_item('ev1', {'x': 1}, 'text', 123.0)
    store.put_derivation(iid, 'caption', 'p1', 'hello', 124.0)
    store.close()
    store2 = ItemStore(tmp_path / 'test.db')
    eid2 = store2.register_embedder('m', 'r', 'f', 'cpu', 'f32', 'mean', 'v1', 1, 3, 'space1', 'pooled', 0)
    iid2 = store2.put_item('ev1', {'x': 1}, 'text', 123.0)
    assert eid2 == eid
    assert iid2 == iid
    cur = store2.conn.cursor()
    cur.execute('SELECT count(*) FROM embedders')
    assert cur.fetchone()[0] == 1
    cur.execute('SELECT count(*) FROM items')
    assert cur.fetchone()[0] == 1
    cur.execute('SELECT count(*) FROM derivations')
    assert cur.fetchone()[0] == 1
    store2.close()


def test_put_derivation_idempotent_fts_no_duplicates(tmp_path):
    store = ItemStore(tmp_path / 'test.db')
    iid = store.put_item('ev1', {'x': 1}, 'text', 123.0)
    text = 'bicycle repair manual'
    store.put_derivation(iid, 'caption', 'p1', text, 124.0)
    store.put_derivation(iid, 'caption', 'p1', text, 125.0)
    store.put_derivation(iid, 'caption', 'p1', text, 126.0)
    cur = store.conn.cursor()
    cur.execute("SELECT count(*) FROM derivations_fts WHERE derivations_fts MATCH 'bicycle'")
    assert cur.fetchone()[0] == 1
    cur.execute('SELECT count(*) FROM derivations')
    assert cur.fetchone()[0] == 1
    store.close()
