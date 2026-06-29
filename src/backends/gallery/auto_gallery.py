"""
src/backends/gallery/auto_gallery.py
─────────────────────────────────────
Auto-promoting gallery manager: starts with NumpyGalleryIndex and seamlessly
upgrades to FAISSIVFIndex when the gallery exceeds a configurable threshold.

Promotion is transparent to callers — all public methods delegate to whichever
backend is currently active.
"""

from __future__ import annotations

import logging

import numpy as np

from src.backends.gallery.base import BaseGalleryIndex
from src.backends.gallery.numpy_index import NumpyGalleryIndex

logger = logging.getLogger(__name__)


class AutoGalleryManager(BaseGalleryIndex):
    """
    Transparent wrapper that starts with NumpyGalleryIndex and promotes
    to FAISSIVFIndex when size > cfg['faiss_ivf']['auto_promote_threshold'].

    Promotion is seamless: existing embeddings are migrated to FAISS,
    metadata is preserved, and all subsequent calls use the FAISS backend.
    """

    def __init__(self, config: dict, embed_dim: int, gallery_ref) -> None:
        self.cfg        = config
        self.embed_dim  = embed_dim
        self.threshold  = int(
            config.get("faiss_ivf", {}).get("auto_promote_threshold", 1000)
        )
        self._backend: BaseGalleryIndex = NumpyGalleryIndex(config, gallery_ref)
        self._promoted  = False
        self._gallery_ref = gallery_ref

    # ──────────────────────────────────────────────────────────────────────────
    def _promote_to_faiss(self) -> None:
        """
        Migrate all existing gallery entries from NumpyGalleryIndex to FAISS.
        """
        from src.backends.gallery.faiss_ivf_index import FAISSIVFIndex

        logger.info(
            "AutoGalleryManager: gallery reached %d identities — "
            "promoting to FAISS IVFFlat…",
            self.size,
        )

        faiss_idx = FAISSIVFIndex(self.cfg, self.embed_dim)

        # Migrate all embeddings from the underlying PersistentGallery
        for pid, data in self._gallery_ref.gallery.items():
            if not data["embeddings"]:
                continue
            mean_emb = np.mean(data["embeddings"], axis=0).astype(np.float32)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm
            faiss_idx.add(pid, mean_emb)
            faiss_idx.update_meta(
                global_id   = pid,
                class_id    = data.get("class", 0),
                frame_n     = data.get("last_seen", 0),
                bbox        = np.zeros(4, dtype=np.float32),
            )

        self._backend  = faiss_idx
        self._promoted = True
        logger.info(
            "AutoGalleryManager: promotion complete — FAISS IVFFlat "
            "now serving %d identities.",
            faiss_idx.size,
        )

    # ──────────────────────────────────────────────────────────────────────────
    def add(self, global_id: int, embedding: np.ndarray) -> None:
        self._backend.add(global_id, embedding)
        if not self._promoted and self.size >= self.threshold:
            self._promote_to_faiss()

    def query(
        self,
        embedding:     np.ndarray,
        class_id:      int,
        current_frame: int,
        exclude_ids:   set[int],
        top_k:         int = 5,
    ) -> tuple[int, float] | None:
        return self._backend.query(embedding, class_id, current_frame, exclude_ids, top_k)

    def remove(self, global_id: int) -> None:
        self._backend.remove(global_id)

    def rebuild(self) -> None:
        self._backend.rebuild()

    # ──────────────────────────────────────────────────────────────────────────
    @property
    def size(self) -> int:
        return self._backend.size

    @property
    def backend_name(self) -> str:
        return "faiss_ivf" if self._promoted else "numpy"

    @property
    def name(self) -> str:
        return f"AutoGalleryManager({self.backend_name})"
