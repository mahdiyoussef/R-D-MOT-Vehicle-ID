"""
src/backends/tracking/legacy_tracker.py
────────────────────────────────────────
Thin adapter wrapping the existing VehicleTracker (StrongSORT via boxmot)
as a BaseTracker subclass. Preserves full v1.0 behavior.
"""

from __future__ import annotations

import logging

import numpy as np

from src.backends.tracking.base import BaseTracker, TrackerOutput

logger = logging.getLogger(__name__)


class LegacyTrackerBackend(BaseTracker):
    """
    Adapter: wraps the existing VehicleTracker (StrongSORT via boxmot)
    as a BaseTracker without duplicating any logic.

    The `embeddings` parameter from the BaseTracker interface is ignored
    since the legacy tracker manages Re-ID internally via boxmot.
    """

    def __init__(self, config: dict) -> None:
        trk_cfg = config.get("tracking", {})
        self._reid_weights  = trk_cfg.get("reid_weights",  "models/trackers/osnet_x0_25_msmt17.pt")
        self._device        = config.get("pipeline", {}).get("device", "cpu")
        self._half          = trk_cfg.get("half",          True)
        self._max_age       = trk_cfg.get("max_age",       60)
        self._min_hits      = trk_cfg.get("min_hits",      3)
        self._iou_threshold = trk_cfg.get("iou_threshold", 0.3)
        self._tracker       = None
        self._frame_count   = 0
        self._known_track_ids: set[int] = set()

    def _get_tracker(self):
        if self._tracker is None:
            from src.tracker import VehicleTracker
            self._tracker = VehicleTracker(
                reid_weights  = self._reid_weights,
                device        = self._device,
                half          = self._half,
                max_age       = self._max_age,
                min_hits      = self._min_hits,
                iou_threshold = self._iou_threshold,
            )
        return self._tracker

    # ──────────────────────────────────────────────────────────────────────────
    def update(
        self,
        frame: np.ndarray,
        detections: list[dict],
        embeddings: dict[int, np.ndarray] | None = None,
    ) -> list[TrackerOutput]:
        """
        Convert dict detections → numpy array, delegate to VehicleTracker,
        and convert the output back to TrackerOutput objects.
        """
        tracker = self._get_tracker()

        if not detections:
            det_array = np.empty((0, 6), dtype=np.float32)
        else:
            rows = []
            for d in detections:
                bbox = d["bbox"]
                rows.append([bbox[0], bbox[1], bbox[2], bbox[3],
                              d.get("conf", 1.0), d.get("cls", 0)])
            det_array = np.array(rows, dtype=np.float32)

        raw_tracks = tracker.update(det_array, frame)   # [M, 8]
        self._frame_count += 1

        outputs: list[TrackerOutput] = []
        for row in raw_tracks:
            x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
            tid             = int(row[4])
            conf            = float(row[5])
            cls             = int(row[6])
            is_new          = tid not in self._known_track_ids
            self._known_track_ids.add(tid)
            outputs.append(TrackerOutput(
                tracker_id        = tid,
                bbox              = np.array([x1, y1, x2, y2], dtype=np.float32),
                confidence        = conf,
                class_id          = cls,
                is_new            = is_new,
                frames_since_seen = 0,
            ))

        return outputs

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        if self._tracker is not None:
            self._tracker.reset()
        self._frame_count = 0
        self._known_track_ids.clear()
        logger.debug("LegacyTrackerBackend state reset.")
