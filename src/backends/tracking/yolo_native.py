"""
src/backends/tracking/yolo_native.py
─────────────────────────────────────
YOLO Native Tracker Backend (v2.0)

Uses Ultralytics YOLO's built-in `.track()` method, which bundles
BoT-SORT or ByteTrack directly into the detection forward pass.
This eliminates the need for a separate tracker library (boxmot),
reduces latency (single forward pass for detection + tracking),
and provides a lightweight, zero-config tracking option.

Supported tracker configs (via Ultralytics):
  - botsort.yaml   (default, uses BoT-SORT with Kalman + IoU + optional ReID)
  - bytetrack.yaml (ByteTrack, no appearance features)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.backends.tracking.base import BaseTracker, TrackerOutput

logger = logging.getLogger(__name__)


class YOLONativeTracker(BaseTracker):
    """
    Tracker backend using Ultralytics YOLO's integrated `.track()` API.

    This replaces the entire Detection + Tracking two-step with a single
    YOLO call that returns both bounding boxes AND persistent track IDs.

    Parameters (from config['yolo_native'])
    ----------
    tracker_type   : str   'botsort' or 'bytetrack' (Ultralytics built-in configs)
    track_high_thresh : float  High confidence threshold for first-pass association
    track_low_thresh  : float  Low confidence threshold for second-pass (ByteTrack)
    new_track_thresh  : float  Threshold to initialize a new track
    track_buffer      : int    Frames to keep lost tracks alive (max_age equivalent)
    match_thresh      : float  IoU matching threshold
    """

    def __init__(self, config: dict) -> None:
        self._cfg = config.get("yolo_native", {})
        self._tracker_type = self._cfg.get("tracker_type", "botsort")
        self._track_high_thresh = self._cfg.get("track_high_thresh", 0.5)
        self._track_low_thresh = self._cfg.get("track_low_thresh", 0.1)
        self._new_track_thresh = self._cfg.get("new_track_thresh", 0.6)
        self._track_buffer = self._cfg.get("track_buffer", 60)
        self._match_thresh = self._cfg.get("match_thresh", 0.8)

        self._known_ids: set[int] = set()  # track IDs seen in previous frames
        self._frame_n: int = 0

        logger.info(
            "YOLONativeTracker initialized — tracker_type=%s, buffer=%d",
            self._tracker_type, self._track_buffer,
        )

    # ──────────────────────────────────────────────────────────────────────────
    def update(
        self,
        frame: np.ndarray,
        detections: list[dict],
        embeddings: dict[int, np.ndarray] | None = None,
    ) -> list[TrackerOutput]:
        """
        Not used directly in the standard pipeline flow.
        YOLO native tracking is integrated at the pipeline level
        via `run_yolo_track()`.
        """
        return []

    def run_yolo_track(
        self,
        model,
        frame: np.ndarray,
        conf: float = 0.4,
        iou: float = 0.45,
        imgsz: int = 640,
        classes: list[int] | None = None,
        device: str = "cuda:0",
        half: bool = True,
    ) -> np.ndarray:
        """
        Run YOLO detection + tracking in a single forward pass.

        Parameters
        ----------
        model   : Ultralytics YOLO model instance
        frame   : BGR frame (H, W, 3)
        conf    : Detection confidence threshold
        iou     : NMS IoU threshold
        imgsz   : Input resolution
        classes : List of class IDs to track
        device  : Device string
        half    : FP16 mode

        Returns
        -------
        np.ndarray, shape (M, 8)
            [x1, y1, x2, y2, track_id, confidence, class_id, det_idx]
            Compatible with the standard pipeline track format.
        """
        results = model.track(
            frame,
            persist=True,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=classes,
            device=device,
            half=half,
            verbose=False,
            tracker=f"{self._tracker_type}.yaml",
        )[0]

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            self._frame_n += 1
            return np.empty((0, 8), dtype=np.float32)

        xyxy = boxes.xyxy.cpu().numpy()       # (N, 4)
        confs = boxes.conf.cpu().numpy()       # (N,)
        clss = boxes.cls.cpu().numpy()         # (N,)

        # Track IDs — may be None if tracking fails on some detections
        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            # Fallback: assign sequential IDs
            track_ids = np.arange(len(xyxy), dtype=int)

        n = len(xyxy)
        det_idx = np.arange(n, dtype=np.float32)

        tracks = np.column_stack([
            xyxy,                          # x1, y1, x2, y2
            track_ids.reshape(-1, 1),      # track_id
            confs.reshape(-1, 1),          # confidence
            clss.reshape(-1, 1),           # class_id
            det_idx.reshape(-1, 1),        # det_idx
        ]).astype(np.float32)

        # Update known IDs for is_new detection
        new_ids = set(track_ids.tolist()) - self._known_ids
        self._known_ids.update(track_ids.tolist())
        self._frame_n += 1

        return tracks

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Reset internal state."""
        self._known_ids.clear()
        self._frame_n = 0
        logger.debug("YOLONativeTracker state reset.")

    @property
    def name(self) -> str:
        return f"YOLONativeTracker ({self._tracker_type})"
