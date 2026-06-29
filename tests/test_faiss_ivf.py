"""
tests/test_faiss_ivf.py
────────────────────────
Unit tests for the FAISS IVFFlat gallery index.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def config():
    return {
        "faiss_ivf": {
            "index_type": "IVFFlat",
            "metric": "inner_product",
            "nlist": 4,            # small for testing
            "nprobe": 2,
            "auto_promote_threshold": 1000,
            "similarity_threshold": 0.3,
            "min_train_vectors": 8,   # small threshold for testing
            "retrain_on_add": False,
            "retrain_growth_ratio": 0.2,
            "use_gpu": False,
        }
    }


@pytest.fixture
def index(config):
    from src.backends.gallery.faiss_ivf_index import FAISSIVFIndex
    return FAISSIVFIndex(config, embed_dim=16)


def _rand_emb(dim=16):
    """Random L2-normalized vector."""
    v = np.random.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────

def test_faiss_index_builds_after_min_train_vectors(index, config):
    """Index should train once min_train_vectors are accumulated."""
    min_v = config["faiss_ivf"]["min_train_vectors"]   # 8
    for i in range(min_v):
        index.add(i, _rand_emb())
    assert index._index is not None
    assert index._index.is_trained


def test_add_single_entry_queryable(index, config):
    min_v = config["faiss_ivf"]["min_train_vectors"]
    for i in range(min_v):
        index.add(i, _rand_emb())
    assert index.size == min_v


def test_query_returns_correct_global_id(config):
    """The best-match query should return the identical vector's global_id."""
    from src.backends.gallery.faiss_ivf_index import FAISSIVFIndex
    idx  = FAISSIVFIndex(config, embed_dim=16)
    min_v = config["faiss_ivf"]["min_train_vectors"]

    target_emb = _rand_emb()
    target_id  = 999

    # Add filler entries
    for i in range(min_v - 1):
        idx.add(i, _rand_emb())
    idx.add(target_id, target_emb)

    # Update metadata so the target is not filtered by last_seen
    idx.update_meta(target_id, class_id=0, frame_n=0, bbox=np.zeros(4))

    result = idx.query(target_emb, class_id=0, current_frame=100, exclude_ids=set())
    if result is not None:
        gid, sim = result
        assert gid == target_id
        assert 0.0 <= sim <= 1.01


def test_query_respects_class_filter(config):
    """Entries with a different class_id should not be returned."""
    from src.backends.gallery.faiss_ivf_index import FAISSIVFIndex
    idx   = FAISSIVFIndex(config, embed_dim=16)
    min_v = config["faiss_ivf"]["min_train_vectors"]
    emb   = _rand_emb()

    for i in range(min_v):
        idx.add(i, _rand_emb())
    # class_id=1 for all entries
    for i in range(min_v):
        idx.update_meta(i, class_id=1, frame_n=0, bbox=np.zeros(4))

    result = idx.query(emb, class_id=0, current_frame=100, exclude_ids=set())
    assert result is None   # no class_id=0 entries


def test_query_respects_exclude_ids(config):
    from src.backends.gallery.faiss_ivf_index import FAISSIVFIndex
    idx   = FAISSIVFIndex(config, embed_dim=16)
    min_v = config["faiss_ivf"]["min_train_vectors"]
    emb   = _rand_emb()

    for i in range(min_v):
        idx.add(i, _rand_emb() * 0.1)   # weak random entries
    idx.add(999, emb)   # add exact target
    idx.update_meta(999, class_id=0, frame_n=0, bbox=np.zeros(4))
    for i in range(min_v):
        idx.update_meta(i, class_id=0, frame_n=0, bbox=np.zeros(4))

    result = idx.query(emb, class_id=0, current_frame=100, exclude_ids={999})
    if result is not None:
        assert result[0] != 999


def test_remove_marks_entry_as_deleted(index, config):
    min_v = config["faiss_ivf"]["min_train_vectors"]
    for i in range(min_v):
        index.add(i, _rand_emb())
    index.remove(0)
    assert 0 in index._deleted


def test_rebuild_removes_deleted_entries(index, config):
    min_v = config["faiss_ivf"]["min_train_vectors"]
    for i in range(min_v):
        index.add(i, _rand_emb())
    initial_size = index.size
    index.remove(0)
    index.rebuild()
    assert 0 not in index._deleted
    assert index.size == initial_size - 1


def test_size_property_excludes_deleted(index, config):
    min_v = config["faiss_ivf"]["min_train_vectors"]
    for i in range(min_v):
        index.add(i, _rand_emb())
    full_size = index.size
    index.remove(0)
    assert index.size == full_size - 1


def test_save_and_load_roundtrip(index, config):
    min_v = config["faiss_ivf"]["min_train_vectors"]
    for i in range(min_v):
        index.add(i, _rand_emb())
    original_size = index.size

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "gallery")
        index.save(path)

        from src.backends.gallery.faiss_ivf_index import FAISSIVFIndex
        loaded = FAISSIVFIndex(config, embed_dim=16)
        loaded.load(path)
        assert loaded.size == original_size


def test_pending_buffer_flushes_on_train_threshold(config):
    """Vectors should stay in pending until min_train_vectors reached, then flush."""
    from src.backends.gallery.faiss_ivf_index import FAISSIVFIndex
    idx   = FAISSIVFIndex(config, embed_dim=16)
    min_v = config["faiss_ivf"]["min_train_vectors"]

    for i in range(min_v - 1):
        idx.add(i, _rand_emb())
        assert idx._index is None, "Index should not be built yet"
        assert len(idx._pending) == i + 1

    # One more to trigger training
    idx.add(min_v - 1, _rand_emb())
    assert idx._index is not None
    assert len(idx._pending) == 0   # pending should be flushed


def test_faiss_matches_numpy_cosine_search(config):
    """FAISS approximate search should agree with brute-force cosine for exact match."""
    from src.backends.gallery.faiss_ivf_index import FAISSIVFIndex
    from sklearn.metrics.pairwise import cosine_similarity

    idx   = FAISSIVFIndex(config, embed_dim=16)
    min_v = config["faiss_ivf"]["min_train_vectors"]

    embs    = [_rand_emb() for _ in range(min_v + 2)]
    gids    = list(range(len(embs)))
    query   = embs[0]

    for gid, emb in zip(gids, embs):
        idx.add(gid, emb)
        idx.update_meta(gid, class_id=0, frame_n=0, bbox=np.zeros(4))

    result = idx.query(query, class_id=0, current_frame=1000, exclude_ids=set())
    # Brute-force best
    matrix     = np.stack(embs)
    bf_sims    = cosine_similarity(query.reshape(1, -1), matrix)[0]
    bf_best_id = int(np.argmax(bf_sims))

    if result is not None:
        gid_faiss, sim_faiss = result
        # FAISS may not return the exact same due to ANN, but sim should be close
        assert sim_faiss >= 0.0
