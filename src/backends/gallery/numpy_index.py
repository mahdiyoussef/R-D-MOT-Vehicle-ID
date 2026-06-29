"""
src/backends/gallery/numpy_index.py
────────────────────────────────────
Thin adapter wrapping the existing PersistentGallery as a BaseGalleryIndex.
Preserves all v1.0 gallery behavior with no logic duplication.
"""

from __future__ import annotations

import logging

import numpy as np

from src.backends.gallery.base import BaseGalleryIndex

logger = logging.getLogger(__name__)


class NumpyGalleryIndex(BaseGalleryIndex):
    """
    Adapter: wraps the existing PersistentGallery as a BaseGalleryIndex.
    All similarity search, filtering, and metadata management is delegated
    to the existing gallery instance.
    """

    def __init__(self, config: dict, gallery_ref) -> None:
        """
        Parameters
        ----------
        config      : Full pipeline config dict.
        gallery_ref : Existing PersistentGallery instance.
        """
        self._gallery = gallery_ref
        self._similarity_threshold = config.get("gallery", {}).get(
            "similarity_threshold", 0.45
        )

    # ──────────────────────────────────────────────────────────────────────────
    def add(self, global_id: int, embedding: np.ndarray) -> None:
        """
        Keep the FAISS/numpy index in sync with the PersistentGallery.
        For the numpy backend the gallery itself is the authoritative store,
        so add() is a no-op (gallery.update_known / register_or_recover
        already handle this in the pipeline).
        """
        # The PersistentGallery already stores embeddings internally;
        # this index is just an interface adapter — no extra work needed.
        pass

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
        Delegate to the existing PersistentGallery._query() method.
        """
        best_id, best_score = self._gallery._query(
            query_emb      = embedding,
            current_frame  = current_frame,
            cls_id         = class_id,
        )
        if (
            best_id is not None
            and best_score >= self._similarity_threshold
            and best_id not in exclude_ids
        ):
            return (best_id, best_score)
        return None

    # ──────────────────────────────────────────────────────────────────────────
    def remove(self, global_id: int) -> None:
        """Remove from the underlying gallery."""
        if global_id in self._gallery.gallery:
            del self._gallery.gallery[global_id]

    # ──────────────────────────────────────────────────────────────────────────
    def rebuild(self) -> None:
        """No index rebuild needed for flat numpy search."""
        pass

    # ──────────────────────────────────────────────────────────────────────────
    @property
    def size(self) -> int:
        return len(self._gallery.gallery)
