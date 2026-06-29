"""
src/backends/tracking/base.py
─────────────────────────────
Abstract base class for all multi-object tracker backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackerOutput:
    """Standardised output from any tracker backend."""
    tracker_id:        int
    bbox:              np.ndarray    # [x1, y1, x2, y2]
    confidence:        float
    class_id:          int
    is_new:            bool          # True = first confirmed appearance or re-appeared
    frames_since_seen: int = 0


class BaseTracker(ABC):
    """Abstract interface for all multi-object trackers."""

    @abstractmethod
    def update(
        self,
        frame: np.ndarray,
        detections: list[dict],
        embeddings: dict[int, np.ndarray] | None = None,
    ) -> list[TrackerOutput]:
        """
        Associate detections to tracks for one frame.

        Parameters
        ----------
        frame      : Full BGR frame (needed by BoT-SORT for CMC)
        detections : [{'bbox': [x1,y1,x2,y2], 'conf': float, 'cls': int}]
        embeddings : Pre-computed Re-ID embeddings keyed by detection index
                     (used by BoT-SORT-ReID / StrongSORT for appearance fusion)

        Returns
        -------
        list of confirmed TrackerOutput objects
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all track state."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def needs_embeddings(self) -> bool:
        """Return True if this tracker uses pre-computed Re-ID embeddings."""
        return False
