"""
src/geometry/camera_topology.py
───────────────────────────────
Spatio-Temporal Transition Constraints for multi-camera Re-ID.

Defines the physical topology of a camera network. By knowing the physical distance
between cameras and the maximum possible speed of a vehicle, we can compute the
minimum travel time required to move from Camera A to Camera B.

Matches that violate this temporal constraint (e.g., appearing in Camera B 2 seconds
after leaving Camera A, when the physical drive takes 30 seconds) are mathematically
impossible and are strictly rejected.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class CameraTopology:
    """
    Manages spatio-temporal constraints across a multi-camera network.
    
    Parameters
    ----------
    config : dict
        The global pipeline config containing the `camera_topology` section.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        top_cfg = config.get("camera_topology", {})
        self.enabled = top_cfg.get("enabled", False)
        
        # fps used to convert frame diffs to seconds
        self.fps = top_cfg.get("fps", 30)
        
        # maximum physically possible vehicle speed in meters/second
        self.max_speed_mps = top_cfg.get("max_speed_kph", 120.0) / 3.6
        
        # Dictionary of camera positions: { cam_id: (x, y) }
        self.cameras: Dict[int, tuple[float, float]] = {}
        for cam in top_cfg.get("cameras", []):
            self.cameras[int(cam["id"])] = (float(cam["x"]), float(cam["y"]))

        if self.enabled and not self.cameras:
            logger.warning("Camera topology is enabled but no cameras are defined.")

    def _euclidean_distance(self, cam_a: int, cam_b: int) -> float:
        """Compute 2D distance between two cameras in meters."""
        if cam_a not in self.cameras or cam_b not in self.cameras:
            return 0.0
        xa, ya = self.cameras[cam_a]
        xb, yb = self.cameras[cam_b]
        return ((xa - xb)**2 + (ya - yb)**2)**0.5

    def is_transition_possible(
        self, 
        cam_a: int, 
        frame_a: int, 
        cam_b: int, 
        frame_b: int
    ) -> bool:
        """
        Check if it's physically possible for a vehicle to travel from cam_a to cam_b
        within the given frame difference.

        Parameters
        ----------
        cam_a   : Source camera ID
        frame_a : Frame index at source camera
        cam_b   : Destination camera ID
        frame_b : Frame index at destination camera

        Returns
        -------
        bool
            True if the transition is physically possible, False otherwise.
        """
        if not self.enabled:
            return True
            
        if cam_a == cam_b:
            # Same camera: always possible, temporal direction must be valid
            return frame_b >= frame_a

        # Time elapsed in seconds
        delta_frames = abs(frame_b - frame_a)
        delta_seconds = delta_frames / self.fps

        # Physical distance in meters
        distance_m = self._euclidean_distance(cam_a, cam_b)
        
        if distance_m == 0.0:
            return True

        # Minimum time required to traverse the distance at max speed
        min_required_seconds = distance_m / self.max_speed_mps

        # If actual time elapsed is strictly less than the absolute minimum required time,
        # the transition is physically impossible.
        if delta_seconds < min_required_seconds:
            logger.debug(
                "Spatio-temporal rejection: Cam %d -> Cam %d. "
                "Dist: %.1fm, Time: %.1fs, Min Required: %.1fs",
                cam_a, cam_b, distance_m, delta_seconds, min_required_seconds
            )
            return False

        return True
