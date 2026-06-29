"""
src/backends/gallery/base.py
────────────────────────────
Abstract base class for all gallery index backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseGalleryIndex(ABC):
    """Abstract interface for Re-ID gallery similarity search."""

    @abstractmethod
    def add(self, global_id: int, embedding: np.ndarray) -> None:
        """Insert or update embedding for global_id."""

    @abstractmethod
    def query(
        self,
        embedding: np.ndarray,
        class_id: int,
        current_frame: int,
        exclude_ids: set[int],
        top_k: int = 5,
    ) -> tuple[int, float] | None:
        """
        Return (global_id, cosine_similarity) of best match, or None.
        Must filter by class_id, recency, and exclude_ids.
        """

    @abstractmethod
    def remove(self, global_id: int) -> None:
        """Remove a stale identity from the index."""

    @abstractmethod
    def rebuild(self) -> None:
        """Rebuild the internal index after bulk add/remove."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of identities currently indexed."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
