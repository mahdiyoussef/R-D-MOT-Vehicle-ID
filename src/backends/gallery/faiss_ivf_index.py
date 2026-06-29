"""
src/backends/gallery/faiss_ivf_index.py
────────────────────────────────────────
FAISS IVFFlat gallery index for fast approximate cosine similarity search
over large Re-ID galleries (>1000 identities).

Key design decisions:
  - FAISS stores raw embeddings; metadata (class_id, last_seen, bbox) lives
    in a parallel Python dict keyed by global_id.
  - FAISS assigns sequential integer IDs (faiss_id); a bidirectional mapping
    between faiss_id and global_id handles translation.
  - Deletions: FAISS IVFFlat doesn't support efficient single-vector removal.
    We use tombstone sets and periodic full rebuilds.
  - GPU support: if faiss-gpu is installed and use_gpu=true, index moves to GPU.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.backends.gallery.base import BaseGalleryIndex

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GalleryEntryMeta:
    global_id:        int
    mean_embedding:   np.ndarray          # running mean, L2-normalized
    class_id:         int
    last_seen_frame:  int
    bbox_last:        np.ndarray
    hit_count:        int = 0
    re_id_count:      int = 0


# ─────────────────────────────────────────────────────────────────────────────

class FAISSIVFIndex(BaseGalleryIndex):
    """
    FAISS IVFFlat (or IVFPQ / IVFScalarQuantizer) gallery index.

    Starts buffering vectors until min_train_vectors are collected,
    then trains the IVF quantizer and flushes the buffer.
    Subsequent add() calls insert directly into the trained index.

    Cosine similarity is implemented as inner product on L2-normalized vectors.
    """

    def __init__(self, config: dict, embed_dim: int) -> None:
        self.cfg       = config.get("faiss_ivf", {})
        self.embed_dim = embed_dim

        self._index     = None                         # faiss.Index
        self._meta:     dict[int, GalleryEntryMeta] = {}   # global_id → meta
        self._id_map:   dict[int, int] = {}            # faiss_id  → global_id
        self._rev_map:  dict[int, int] = {}            # global_id → faiss_id
        self._pending:  list[tuple[int, np.ndarray]] = []  # pre-train buffer
        self._deleted:  set[int] = set()               # global_ids tombstoned
        self._faiss_id_counter  = 0
        self._last_size_at_retrain = 0

        self._sim_threshold = float(self.cfg.get("similarity_threshold", 0.45))

    # ──────────────────────────────────────────────────────────────────────────
    def _build_index(self):
        """Build the FAISS index according to config."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS is not installed. Install with:\n"
                "  pip install faiss-cpu  (or faiss-gpu for GPU support)"
            )

        index_type = self.cfg.get("index_type", "IVFFlat")
        nlist      = int(self.cfg.get("nlist", 128))
        metric     = faiss.METRIC_INNER_PRODUCT

        quantizer = faiss.IndexFlatIP(self.embed_dim)

        if index_type == "IVFFlat":
            index = faiss.IndexIVFFlat(quantizer, self.embed_dim, nlist, metric)
        elif index_type == "IVFPQ":
            m     = int(self.cfg.get("ivfpq_m",     64))
            nbits = int(self.cfg.get("ivfpq_nbits",  8))
            index = faiss.IndexIVFPQ(quantizer, self.embed_dim, nlist, m, nbits)
        elif index_type == "IVFScalarQuantizer":
            from faiss import ScalarQuantizer
            index = faiss.IndexIVFScalarQuantizer(
                quantizer, self.embed_dim, nlist,
                faiss.ScalarQuantizer.QT_8bit, metric,
            )
        else:
            raise ValueError(f"Unknown FAISS index_type: {index_type}")

        index.nprobe = int(self.cfg.get("nprobe", 16))

        # Optional GPU support
        if self.cfg.get("use_gpu", False):
            try:
                res = faiss.StandardGpuResources()
                gpu_id = int(self.cfg.get("gpu_id", 0))
                index  = faiss.index_cpu_to_gpu(res, gpu_id, index)
                logger.info("FAISS index moved to GPU %d.", gpu_id)
            except Exception as e:
                logger.warning("FAISS GPU failed (%s). Falling back to CPU.", e)

        return index

    # ──────────────────────────────────────────────────────────────────────────
    def _train_index(self, embeddings: np.ndarray) -> None:
        """Train the IVF quantizer on the provided embeddings."""
        min_vecs = int(self.cfg.get("min_train_vectors", 256))
        if len(embeddings) < min_vecs:
            raise RuntimeError(
                f"Need >= {min_vecs} vectors to train IVF index, got {len(embeddings)}"
            )
        if self._index is None:
            self._index = self._build_index()
        self._index.train(embeddings.astype(np.float32))
        logger.info(
            "FAISS IVF index trained on %d vectors (index_type=%s, nlist=%d).",
            len(embeddings),
            self.cfg.get("index_type", "IVFFlat"),
            self.cfg.get("nlist", 128),
        )

    # ──────────────────────────────────────────────────────────────────────────
    def _flush_pending(self) -> None:
        """Train the index and add all buffered vectors."""
        if not self._pending:
            return
        gids  = [gid for gid, _ in self._pending]
        embs  = np.array([e for _, e in self._pending], dtype=np.float32)
        self._train_index(embs)

        faiss_ids = np.arange(
            self._faiss_id_counter,
            self._faiss_id_counter + len(gids),
            dtype=np.int64,
        )
        self._index.add_with_ids(embs, faiss_ids)

        for gid, fid in zip(gids, faiss_ids):
            self._id_map[int(fid)] = gid
            self._rev_map[gid]     = int(fid)
        self._faiss_id_counter += len(gids)
        self._pending.clear()

    # ──────────────────────────────────────────────────────────────────────────
    def add(self, global_id: int, embedding: np.ndarray) -> None:
        """Insert or update an embedding for global_id."""
        emb = embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        if global_id in self._meta:
            # Update running mean
            meta = self._meta[global_id]
            n     = meta.hit_count + 1
            new_mean = (meta.mean_embedding * meta.hit_count + emb) / n
            new_mean /= (np.linalg.norm(new_mean) + 1e-8)
            meta.mean_embedding  = new_mean
            meta.hit_count       = n
            meta.last_seen_frame = meta.last_seen_frame  # will be updated by pipeline

            # Update FAISS vector: mark old as deleted, add new
            if global_id in self._rev_map and self._index is not None and self._index.is_trained:
                old_fid = self._rev_map[global_id]
                # Remove old mapping (FAISS doesn't support true removal, use tombstone)
                if old_fid in self._id_map:
                    del self._id_map[old_fid]
                # Add updated embedding with new faiss_id
                new_fid = self._faiss_id_counter
                self._faiss_id_counter += 1
                fid_arr = np.array([new_fid], dtype=np.int64)
                self._index.add_with_ids(new_mean.reshape(1, -1), fid_arr)
                self._id_map[new_fid] = global_id
                self._rev_map[global_id] = new_fid
        else:
            # New identity
            meta = GalleryEntryMeta(
                global_id       = global_id,
                mean_embedding  = emb,
                class_id        = 0,       # will be updated by pipeline
                last_seen_frame = 0,
                bbox_last       = np.zeros(4, dtype=np.float32),
                hit_count       = 1,
            )
            self._meta[global_id] = meta

            min_vecs = int(self.cfg.get("min_train_vectors", 256))
            if self._index is None or not self._index.is_trained:
                self._pending.append((global_id, emb))
                if len(self._pending) >= min_vecs:
                    self._flush_pending()
            else:
                fid = self._faiss_id_counter
                self._faiss_id_counter += 1
                self._index.add_with_ids(emb.reshape(1, -1),
                                         np.array([fid], dtype=np.int64))
                self._id_map[fid]       = global_id
                self._rev_map[global_id] = fid

        self._maybe_retrain()

    # ──────────────────────────────────────────────────────────────────────────
    def update_meta(
        self,
        global_id:   int,
        class_id:    int,
        frame_n:     int,
        bbox:        np.ndarray,
    ) -> None:
        """Update metadata fields (class, last_seen, bbox). Called by pipeline."""
        if global_id in self._meta:
            self._meta[global_id].class_id        = class_id
            self._meta[global_id].last_seen_frame  = frame_n
            self._meta[global_id].bbox_last        = bbox

    # ──────────────────────────────────────────────────────────────────────────
    def query(
        self,
        embedding:     np.ndarray,
        class_id:      int,
        current_frame: int,
        exclude_ids:   set[int],
        top_k:         int = 5,
    ) -> tuple[int, float] | None:
        """
        Approximate cosine search via FAISS, with post-filtering.
        """
        if self._index is None or not self._index.is_trained or len(self._meta) == 0:
            return None

        emb = embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        # Over-fetch to allow for post-filtering
        k = min(top_k * 5, max(1, self._index.ntotal))
        sims, faiss_ids = self._index.search(emb.reshape(1, -1), k)
        sims    = sims[0]
        fids    = faiss_ids[0]

        best_gid  = None
        best_sim  = -1.0

        for fid, sim in zip(fids, sims):
            if fid < 0:
                continue
            gid = self._id_map.get(int(fid))
            if gid is None:
                continue
            if gid in self._deleted:
                continue
            meta = self._meta.get(gid)
            if meta is None:
                continue
            if meta.class_id != class_id:
                continue
            if current_frame - meta.last_seen_frame < 10:
                continue
            if gid in exclude_ids:
                continue
            if float(sim) > best_sim:
                best_sim = float(sim)
                best_gid = gid

        if best_gid is not None and best_sim >= self._sim_threshold:
            return (best_gid, best_sim)
        return None

    # ──────────────────────────────────────────────────────────────────────────
    def remove(self, global_id: int) -> None:
        """Tombstone a global_id. Actual FAISS removal happens at rebuild()."""
        self._deleted.add(global_id)
        logger.debug("FAISSIVFIndex: tombstoned global_id=%d.", global_id)

    # ──────────────────────────────────────────────────────────────────────────
    def rebuild(self) -> None:
        """Full index rebuild — removes all tombstoned entries."""
        active = {
            gid: meta for gid, meta in self._meta.items()
            if gid not in self._deleted
        }
        if not active:
            logger.info("FAISSIVFIndex rebuild: no active entries, skipping.")
            return

        n_removed = len(self._deleted)
        n_kept    = len(active)

        gids = list(active.keys())
        embs = np.array([m.mean_embedding for m in active.values()], dtype=np.float32)

        # Reset all FAISS state
        self._index          = None
        self._id_map.clear()
        self._rev_map.clear()
        self._faiss_id_counter = 0
        self._meta           = dict(active)
        self._deleted.clear()
        self._pending.clear()

        # Rebuild
        min_vecs = int(self.cfg.get("min_train_vectors", 256))
        if len(embs) >= min_vecs:
            self._index = self._build_index()
            self._index.train(embs)
            faiss_ids = np.arange(len(gids), dtype=np.int64)
            self._index.add_with_ids(embs, faiss_ids)
            for gid, fid in zip(gids, faiss_ids):
                self._id_map[int(fid)]  = gid
                self._rev_map[gid]       = int(fid)
            self._faiss_id_counter = len(gids)
        else:
            # Not enough data — go back to pending buffer
            for gid, emb in zip(gids, embs):
                self._pending.append((gid, emb))

        self._last_size_at_retrain = n_kept
        logger.info(
            "FAISSIVFIndex rebuild complete: removed=%d, kept=%d, index_type=%s.",
            n_removed, n_kept, self.cfg.get("index_type", "IVFFlat"),
        )

    # ──────────────────────────────────────────────────────────────────────────
    def _maybe_retrain(self) -> None:
        """Trigger rebuild if gallery grew by > retrain_growth_ratio since last retrain."""
        if not self.cfg.get("retrain_on_add", False):
            return
        current = self.size
        ratio   = float(self.cfg.get("retrain_growth_ratio", 0.2))
        if (
            self._last_size_at_retrain > 0
            and current - self._last_size_at_retrain
                > self._last_size_at_retrain * ratio
        ):
            logger.info(
                "FAISSIVFIndex: growth ratio exceeded (%.0f%%). Rebuilding…",
                ratio * 100,
            )
            self.rebuild()

    # ──────────────────────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Save FAISS index + metadata to disk."""
        try:
            import faiss
        except ImportError:
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._index is not None and self._index.is_trained:
            faiss.write_index(self._index, str(path.with_suffix(".faiss")))

        meta_path = path.with_suffix(".meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump({
                "meta":              self._meta,
                "id_map":            self._id_map,
                "rev_map":           self._rev_map,
                "pending":           self._pending,
                "deleted":           self._deleted,
                "faiss_id_counter":  self._faiss_id_counter,
                "embed_dim":         self.embed_dim,
            }, f)
        logger.info("FAISSIVFIndex saved → %s", path)

    # ──────────────────────────────────────────────────────────────────────────
    def load(self, path: str) -> None:
        """Load FAISS index + metadata from disk."""
        try:
            import faiss
        except ImportError:
            return

        path = Path(path)
        faiss_path = path.with_suffix(".faiss")
        meta_path  = path.with_suffix(".meta.pkl")

        if faiss_path.exists():
            self._index = faiss.read_index(str(faiss_path))
            if int(self._index.d) != self.embed_dim:
                raise ValueError(
                    f"Saved index has embed_dim={self._index.d}, "
                    f"expected {self.embed_dim}."
                )

        if meta_path.exists():
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
            self._meta              = data["meta"]
            self._id_map            = data["id_map"]
            self._rev_map           = data["rev_map"]
            self._pending           = data.get("pending", [])
            self._deleted           = data.get("deleted", set())
            self._faiss_id_counter  = data["faiss_id_counter"]

        logger.info(
            "FAISSIVFIndex loaded ← %s  (%d identities, %d tombstoned).",
            path, len(self._meta), len(self._deleted),
        )

    # ──────────────────────────────────────────────────────────────────────────
    @property
    def size(self) -> int:
        return len(self._meta) - len(self._deleted)
