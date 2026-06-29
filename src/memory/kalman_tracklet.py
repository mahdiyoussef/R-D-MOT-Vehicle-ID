"""
src/memory/kalman_tracklet.py
──────────────────────────────
Kalman-filter-based occlusion buffer for lost tracks.

Handles vehicle occlusion (tunnels, blind spots) by:
  1. On track loss → freeze the Kalman state (position + velocity)
  2. Each frame → advance Kalman prediction for all lost tracks
  3. On new detection → compute spatial proximity to expected re-entry zone
     and use proximity as a soft boost to the cosine similarity score
  4. After `lost_buffer_frames` → signal the gallery to archive the track

The spatial zone is a SOFT gate, not a hard filter — a vehicle exiting
a tunnel at a different lane is still considered. Proximity boosts the
candidate score but does not exclude non-proximate candidates.

Kalman model: constant-velocity, 4 states: [cx, cy, vx, vy]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constant-velocity Kalman filter
# ─────────────────────────────────────────────────────────────────────────────

class ConstantVelocityKalman:
    """
    Constant-velocity 2D Kalman filter for bounding box center tracking.

    State vector : [cx, cy, vx, vy]
    Observation  : [cx, cy]
    """

    def __init__(self, initial_cx: float, initial_cy: float) -> None:
        self.state = np.array([initial_cx, initial_cy, 0.0, 0.0], dtype=np.float64)

        # State transition matrix (constant velocity model)
        self.transition_matrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Observation matrix (we only observe position, not velocity)
        self.observation_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Uncertainty matrices
        self.covariance     = np.eye(4, dtype=np.float64) * 100
        self.process_noise  = np.eye(4, dtype=np.float64) * 1.0
        self.measurement_noise = np.eye(2, dtype=np.float64) * 10.0

    def predict_next_position(self) -> np.ndarray:
        """Advance state by one timestep. Returns predicted (cx, cy)."""
        self.state      = self.transition_matrix @ self.state
        self.covariance = self.transition_matrix @ self.covariance @ self.transition_matrix.T + self.process_noise
        return self.state[:2].copy()

    def update_with_observation(self, observed_center: np.ndarray) -> None:
        """Update filter state with an observed (cx, cy) measurement."""
        z         = observed_center[:2]
        residual  = z - self.observation_matrix @ self.state
        S         = self.observation_matrix @ self.covariance @ self.observation_matrix.T + self.measurement_noise
        K         = self.covariance @ self.observation_matrix.T @ np.linalg.inv(S)
        self.state      = self.state + K @ residual
        self.covariance = (np.eye(4) - K @ self.observation_matrix) @ self.covariance


# ─────────────────────────────────────────────────────────────────────────────
# Lost track data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LostTrackRecord:
    """State for a single vehicle that has disappeared from the scene."""
    global_id:          int
    kalman_filter:      ConstantVelocityKalman
    predicted_bbox:     np.ndarray   # [x1, y1, x2, y2] — updated each frame
    frame_lost:         int
    class_id:           int
    last_view_label:    str = "ambiguous"


# ─────────────────────────────────────────────────────────────────────────────
# Tracklet Memory Manager
# ─────────────────────────────────────────────────────────────────────────────

class KalmanTrackletMemory:
    """
    Lost-track buffer with Kalman-extrapolated position predictions.

    Parameters
    ----------
    config : pipeline config dict (reads from 'tracklet_memory' section)
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("tracklet_memory", {})
        self.enabled              = cfg.get("enabled", True)
        self.max_lost_frames      = int(cfg.get("lost_buffer_frames", 300))
        self.spatial_search_radius = float(cfg.get("spatial_zone_radius_px", 150))
        self.use_kalman           = cfg.get("kalman_enabled", True)

        self._lost_tracks: dict[int, LostTrackRecord] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_track_loss(
        self,
        global_id:   int,
        last_bbox:   np.ndarray,  # [x1, y1, x2, y2]
        frame_index: int,
        class_id:    int = 0,
        view_label:  str = "ambiguous",
    ) -> None:
        """
        Called when a tracked vehicle disappears.
        Initialises a frozen Kalman filter at the last known bounding box center.
        """
        if not self.enabled:
            return

        cx = (last_bbox[0] + last_bbox[2]) / 2.0
        cy = (last_bbox[1] + last_bbox[3]) / 2.0

        self._lost_tracks[global_id] = LostTrackRecord(
            global_id       = global_id,
            kalman_filter   = ConstantVelocityKalman(cx, cy),
            predicted_bbox  = last_bbox.copy(),
            frame_lost      = frame_index,
            class_id        = class_id,
            last_view_label = view_label,
        )
        logger.debug("KalmanTracklet: registered loss for gid=%d at frame=%d", global_id, frame_index)

    def advance_all_predictions(self, frame_index: int) -> list[int]:
        """
        Advance Kalman predictions for all lost tracks by one frame.
        Returns the global_ids of tracks whose buffer has expired
        (ready to be archived in gallery cold storage).
        """
        if not self.enabled:
            return []

        expired_ids: list[int] = []
        for gid, record in list(self._lost_tracks.items()):
            frames_since_loss = frame_index - record.frame_lost

            if frames_since_loss > self.max_lost_frames:
                expired_ids.append(gid)
                del self._lost_tracks[gid]
                logger.debug("KalmanTracklet: gid=%d expired after %d frames", gid, frames_since_loss)
                continue

            if self.use_kalman:
                pred_cx, pred_cy = record.kalman_filter.predict_next_position()
                # Update predicted_bbox to center at new prediction, keep original dimensions
                bbox_width  = record.predicted_bbox[2] - record.predicted_bbox[0]
                bbox_height = record.predicted_bbox[3] - record.predicted_bbox[1]
                record.predicted_bbox = np.array([
                    pred_cx - bbox_width / 2,  pred_cy - bbox_height / 2,
                    pred_cx + bbox_width / 2,  pred_cy + bbox_height / 2,
                ], dtype=np.float32)

        return expired_ids

    def get_nearby_lost_tracks(
        self,
        query_bbox: np.ndarray,
        class_id:   int = 0,
    ) -> list[tuple[int, float]]:
        """
        Find lost tracks whose extrapolated position is within the spatial search
        radius of the query bounding box center.

        Returns
        -------
        List of (global_id, proximity_boost) sorted by proximity descending.
        proximity_boost in [0.0, 1.0]: 1.0 = exact center overlap, 0.0 = at edge.
        """
        if not self.enabled:
            return []

        query_cx = (query_bbox[0] + query_bbox[2]) / 2.0
        query_cy = (query_bbox[1] + query_bbox[3]) / 2.0

        nearby: list[tuple[int, float]] = []
        for gid, record in self._lost_tracks.items():
            if record.class_id != class_id:
                continue

            track_cx = (record.predicted_bbox[0] + record.predicted_bbox[2]) / 2.0
            track_cy = (record.predicted_bbox[1] + record.predicted_bbox[3]) / 2.0
            distance = float(np.sqrt((query_cx - track_cx) ** 2 + (query_cy - track_cy) ** 2))

            if distance <= self.spatial_search_radius:
                proximity_boost = max(0.0, 1.0 - distance / self.spatial_search_radius)
                nearby.append((gid, proximity_boost))

        return sorted(nearby, key=lambda pair: -pair[1])

    def mark_track_recovered(self, global_id: int) -> None:
        """Remove a track from the lost buffer after successful re-identification."""
        if global_id in self._lost_tracks:
            del self._lost_tracks[global_id]
            logger.debug("KalmanTracklet: gid=%d recovered and removed from buffer", global_id)

    def is_currently_lost(self, global_id: int) -> bool:
        return global_id in self._lost_tracks

    @property
    def num_lost_tracks(self) -> int:
        return len(self._lost_tracks)

    def get_all_lost_ids(self) -> list[int]:
        return list(self._lost_tracks.keys())
