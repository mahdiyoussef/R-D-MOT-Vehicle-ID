"""
src/tracker.py
──────────────
Stage 2 — Short-Term Multi-Object Tracking
Wraps StrongSORT (via boxmot) to assign frame-local track IDs using
a Kalman Filter (motion) and lightweight ReID (appearance).
IDs are short-lived; persistent identity is handled by the Gallery.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VehicleTracker:
    """
    StrongSORT tracker wrapper.

    Parameters
    ----------
    reid_weights : str | Path
        Path to the lightweight ReID weights used internally by StrongSORT
        (e.g. osnet_x0_25_msmt17.pt — auto-downloaded on first run).
    device       : str   'cuda:0' or 'cpu'.
    half         : bool  FP16 mode.
    max_age      : int   Frames to keep a lost track alive.
    min_hits     : int   Confirmed detections before reporting a track.
    iou_threshold: float IoU for matching detections to existing tracks.
    """

    def __init__(
        self,
        reid_weights: str | Path = "models/trackers/osnet_x0_25_msmt17.pt",
        device: str = "cuda:0",
        half: bool = True,
        max_age: int = 60,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        from boxmot.trackers.tracker_zoo import create_tracker

        self.tracker = create_tracker(
            tracker_type="strongsort",
            reid_weights=Path(reid_weights),
            device=torch.device(device),
            half=half,
            evolve_param_dict={
                "max_age": max_age,
                "min_hits": min_hits,
                "iou_threshold": iou_threshold,
            }
        )
        logger.info("VehicleTracker (StrongSORT) initialised — device=%s", device)

    # ──────────────────────────────────────────────────────────────────────────
    def update(
        self,
        detections: np.ndarray,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Update tracker with current-frame detections.

        Parameters
        ----------
        detections : np.ndarray, shape (N, 6)
            [x1, y1, x2, y2, confidence, class_id]
        frame      : np.ndarray
            BGR frame (H, W, 3).

        Returns
        -------
        np.ndarray, shape (M, 8)
            [x1, y1, x2, y2, track_id, confidence, class_id, det_idx]
            Empty array with shape (0, 8) when nothing is tracked.
        """
        if detections is None or len(detections) == 0:
            # Feed empty array so Kalman states can age out
            detections = np.empty((0, 6), dtype=np.float32)

        tracks = self.tracker.update(detections, frame)

        if tracks is None or len(tracks) == 0:
            return np.empty((0, 8), dtype=np.float32)

        return np.array(tracks, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Reset internal tracker state (e.g. between video clips)."""
        self.tracker.reset()
        logger.debug("VehicleTracker state reset.")
