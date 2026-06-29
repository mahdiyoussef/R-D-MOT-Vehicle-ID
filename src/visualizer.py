"""
src/visualizer.py
─────────────────
Stage 6 (output) — Drawing & Annotation
Renders bounding boxes, persistent ID labels, status badges, and
centroid trajectory trails onto BGR frames.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Colour scheme by track status  (BGR tuples)
_COLOURS: dict[str, tuple[int, int, int]] = {
    "new":       (0,   255,   0),    # Green
    "recovered": (255, 128,   0),    # Orange
    "tracked":   (0,   255, 255),    # Yellow
}

# View label badge colours (BGR)
_VIEW_COLOURS: dict[str, tuple[int, int, int]] = {
    "front":      (255,  80,  80),   # Blue
    "rear":       (80,   80, 255),   # Red
    "side_left":  (80,  200,  80),   # Green
    "side_right": (200, 180,  50),   # Teal
    "top_down":   (180,  80, 200),   # Purple
    "ambiguous":  (140, 140, 140),   # Gray
}

_FONT            = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE      = 0.55
_FONT_THICKNESS  = 1
_BOX_THICKNESS   = 2
_TRAJ_THICKNESS  = 2


class Visualizer:
    """
    Annotates frames with bounding boxes, persistent ID labels,
    and centroid trajectory trails.

    Parameters
    ----------
    trajectory_length : int  Max number of past centroids to draw.
    show_confidence   : bool Append detection confidence to label.
    show_trajectory   : bool Draw centroid path.
    """

    def __init__(
        self,
        trajectory_length: int = 30,
        show_confidence:   bool = True,
        show_trajectory:   bool = True,
    ) -> None:
        self.trajectory_length = trajectory_length
        self.show_confidence   = show_confidence
        self.show_trajectory   = show_trajectory

        # persistent_id → deque of (cx, cy) centroid positions
        self._trajectories: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=trajectory_length)
        )

    # ──────────────────────────────────────────────────────────────────────────
    def draw(
        self,
        frame:              np.ndarray,
        tracks:             np.ndarray,
        persistent_id_map:  Dict[int, int],
        status_map:         Optional[Dict[int, str]] = None,
        view_label_map:     Optional[Dict[int, str]] = None,
    ) -> np.ndarray:
        """
        Annotate ``frame`` with all active tracks.

        Parameters
        ----------
        frame             : BGR numpy array (H, W, 3).
        tracks            : np.ndarray (M, 8)
                            [x1,y1,x2,y2, track_id, conf, class_id, det_idx]
        persistent_id_map : {short_term_track_id → persistent_id}
        status_map        : {persistent_id → 'new'|'recovered'|'tracked'}
                            If None, every box is drawn as 'tracked'.
        view_label_map    : {persistent_id → 'front'|'rear'|'side_left'|...}
                            If provided, draws a view badge on each box.

        Returns
        -------
        Annotated frame (copy).
        """
        out = frame.copy()
        if tracks is None or len(tracks) == 0:
            return out

        if status_map is None:
            status_map = {}
        if view_label_map is None:
            view_label_map = {}

        for t in tracks:
            x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
            tid   = int(t[4])
            conf  = float(t[5])
            cls   = int(t[6])

            pid    = persistent_id_map.get(tid, -1)
            status = status_map.get(pid, "tracked")
            colour = _COLOURS.get(status, _COLOURS["tracked"])

            # ── Trajectory ───────────────────────────────────────────────────
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if pid != -1:
                self._trajectories[pid].append((cx, cy))
                if self.show_trajectory:
                    self._draw_trajectory(out, pid, colour)

            # ── Bounding box ─────────────────────────────────────────────────
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, _BOX_THICKNESS)

            # ── View badge (v4.0) ─────────────────────────────────────────────
            view_label = view_label_map.get(pid, "")
            if view_label:
                view_colour = _VIEW_COLOURS.get(view_label, _VIEW_COLOURS["ambiguous"])
                # Short display label
                view_abbr = {
                    "front":      "FRONT",
                    "rear":       "REAR",
                    "side_left":  "SIDE-L",
                    "side_right": "SIDE-R",
                    "top_down":   "TOP",
                    "ambiguous":  "?",
                }.get(view_label, view_label.upper())
                self._put_label(out, view_abbr, x1, y1, view_colour)

            # ── Main label ────────────────────────────────────────────────────
            from src.detector import VEHICLE_CLASSES
            cls_name = VEHICLE_CLASSES.get(cls, "vehicle")
            pid_str  = f"VID-{pid}" if pid != -1 else "VID-?"
            label    = f"{pid_str} | {cls_name} | {status}"
            if self.show_confidence:
                label += f" {conf:.2f}"

            # Draw main label below the view badge (offset down by ~20px if view shown)
            label_y = y1 - 22 if view_label and y1 > 44 else y1
            self._put_label(out, label, x1, label_y, colour)

        return out

    # ──────────────────────────────────────────────────────────────────────────
    def _draw_trajectory(
        self,
        frame:  np.ndarray,
        pid:    int,
        colour: Tuple[int, int, int],
    ) -> None:
        pts = list(self._trajectories[pid])
        for i in range(1, len(pts)):
            # Fade older segments (alpha proportional to recency)
            alpha = i / len(pts)
            c = tuple(int(v * alpha) for v in colour)
            cv2.line(frame, pts[i - 1], pts[i], c, _TRAJ_THICKNESS)

    @staticmethod
    def _put_label(
        frame:  np.ndarray,
        text:   str,
        x:      int,
        y:      int,
        colour: Tuple[int, int, int],
    ) -> None:
        """Draw a filled background rectangle under the label text."""
        (tw, th), baseline = cv2.getTextSize(
            text, _FONT, _FONT_SCALE, _FONT_THICKNESS
        )
        pad = 4
        y0  = max(y - th - baseline - pad, 0)
        # Background
        cv2.rectangle(frame, (x, y0), (x + tw + pad * 2, y), colour, -1)
        # Text in contrasting colour
        text_colour = (0, 0, 0) if sum(colour) > 400 else (255, 255, 255)
        cv2.putText(
            frame, text,
            (x + pad, y - baseline),
            _FONT, _FONT_SCALE, text_colour, _FONT_THICKNESS,
            cv2.LINE_AA,
        )

    # ──────────────────────────────────────────────────────────────────────────
    def reset_trajectories(self) -> None:
        """Clear all stored centroid trails (e.g. between clips)."""
        self._trajectories.clear()
