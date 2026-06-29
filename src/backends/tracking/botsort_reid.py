"""
src/backends/tracking/botsort_reid.py
──────────────────────────────────────
BoT-SORT with ReID fusion — implemented from scratch to accept external
embeddings from our Re-ID backends (no dependency on ultralytics internals).

Algorithm (per-frame):
  1. CMC:  Estimate camera motion → warp predicted track positions
  2. Kalman predict: advance all track state estimates
  3. High-conf match: Fused IoU+ReID cost → Hungarian assignment
  4. Low-conf match:  IoU-only cost on remaining → 2nd pass
  5. Init new tracks for unmatched high-conf detections
  6. Age and prune stale tracks

Reference: Aharon et al., "BoT-SORT: Robust Associations Multi-Pedestrian
           Tracking", arXiv 2206.14651, 2022.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.backends.tracking.base import BaseTracker, TrackerOutput

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Kalman Filter (simple constant-velocity model for bboxes)
# ─────────────────────────────────────────────────────────────────────────────

class KalmanBoxTracker:
    """
    Simple Kalman filter for a single bounding box.
    State: [cx, cy, s, r, vx, vy, vs]
      cx, cy = centre; s = area; r = aspect ratio (fixed); vx,vy,vs = velocities
    """
    count = 0

    def __init__(self, bbox: np.ndarray) -> None:
        from filterpy.kalman import KalmanFilter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=float)
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=float)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P         *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4]      = self._to_z(bbox)
        KalmanBoxTracker.count += 1

    @staticmethod
    def _to_z(bbox: np.ndarray) -> np.ndarray:
        """Convert [x1,y1,x2,y2] → [cx,cy,s,r]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2
        cy = bbox[1] + h / 2
        s  = w * h
        r  = w / float(h + 1e-6)
        return np.array([[cx], [cy], [s], [r]], dtype=float)

    @staticmethod
    def _to_bbox(x: np.ndarray) -> np.ndarray:
        """Convert state [cx,cy,s,r] → [x1,y1,x2,y2]."""
        cx, cy, s, r = float(x[0].item()), float(x[1].item()), float(x[2].item()), float(x[3].item())
        w = np.sqrt(abs(s * r))
        h = abs(s) / (w + 1e-6)
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dtype=np.float32)

    def predict(self) -> np.ndarray:
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        return self._to_bbox(self.kf.x)

    def update(self, bbox: np.ndarray) -> None:
        self.kf.update(self._to_z(bbox))

    @property
    def bbox(self) -> np.ndarray:
        return self._to_bbox(self.kf.x)


# ─────────────────────────────────────────────────────────────────────────────
# Camera Motion Compensator
# ─────────────────────────────────────────────────────────────────────────────

class CameraMotionCompensator:
    """
    Estimates frame-to-frame camera motion as an affine [2×3] matrix.

    Methods:
      'ecc' — Enhanced Correlation Coefficient minimization
      'orb' — ORB keypoints + RANSAC
      'sof' — Sparse Optical Flow + affine RANSAC
      None  — returns None (no compensation)
    """

    def __init__(self, method: str | None = "ecc", downscale: int = 2) -> None:
        self.method    = method
        self.downscale = max(1, downscale)
        self._prev_gray: Optional[np.ndarray] = None

    def compute(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns a [2,3] affine warp matrix (or None on the first frame
        or if estimation fails).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            gray = cv2.resize(gray, (
                gray.shape[1] // self.downscale,
                gray.shape[0] // self.downscale,
            ))

        M = None
        if self._prev_gray is not None:
            try:
                M = self._estimate(self._prev_gray, gray)
            except Exception as e:
                logger.debug("CMC estimation failed: %s", e)

        self._prev_gray = gray.copy()
        return M

    def _estimate(
        self, prev: np.ndarray, curr: np.ndarray
    ) -> Optional[np.ndarray]:
        if self.method is None:
            return None

        if self.method == "ecc":
            warp = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
            _, warp = cv2.findTransformECC(prev, curr, warp, cv2.MOTION_EUCLIDEAN, criteria)
            return warp

        if self.method == "orb":
            orb = cv2.ORB_create(500)
            kp1, des1 = orb.detectAndCompute(prev, None)
            kp2, des2 = orb.detectAndCompute(curr, None)
            if des1 is None or des2 is None or len(kp1) < 4:
                return None
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            if len(good) < 4:
                return None
            src = np.float32([kp1[m.queryIdx].pt for m in good])
            dst = np.float32([kp2[m.trainIdx].pt for m in good])
            M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
            return M

        if self.method == "sof":
            p0 = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=7)
            if p0 is None or len(p0) < 4:
                return None
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None)
            good_src = p0[st == 1]
            good_dst = p1[st == 1]
            if len(good_src) < 4:
                return None
            M, _ = cv2.estimateAffinePartial2D(good_src, good_dst)
            return M

        return None


# ─────────────────────────────────────────────────────────────────────────────
# IoU helper
# ─────────────────────────────────────────────────────────────────────────────

def _iou_matrix(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of [x1,y1,x2,y2] boxes."""
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)), dtype=np.float32)
    ax1 = bboxes_a[:, 0:1];  ay1 = bboxes_a[:, 1:2]
    ax2 = bboxes_a[:, 2:3];  ay2 = bboxes_a[:, 3:4]
    bx1 = bboxes_b[:, 0];    by1 = bboxes_b[:, 1]
    bx2 = bboxes_b[:, 2];    by2 = bboxes_b[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter_w  = np.maximum(0, inter_x2 - inter_x1)
    inter_h  = np.maximum(0, inter_y2 - inter_y1)
    inter    = inter_w * inter_h
    area_a   = (ax2 - ax1) * (ay2 - ay1)
    area_b   = (bx2 - bx1) * (by2 - by1)
    union    = area_a + area_b - inter
    return (inter / (union + 1e-6)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Track object
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BotSortTrack:
    tracker_id:        int
    detection:         dict
    embedding:         Optional[np.ndarray] = None
    hits:              int = 1
    frames_since_seen: int = 0
    is_new:            bool = True
    _kalman:           KalmanBoxTracker = field(default_factory=lambda: None)
    _embedding_buffer: deque = field(default_factory=lambda: deque(maxlen=30))

    def __post_init__(self):
        bbox = np.array(self.detection["bbox"], dtype=np.float32)
        self._kalman = KalmanBoxTracker(bbox)
        if self.embedding is not None:
            self._embedding_buffer.append(self.embedding)

    @property
    def bbox(self) -> np.ndarray:
        return self._kalman.bbox

    @property
    def confidence(self) -> float:
        return float(self.detection.get("conf", 1.0))

    @property
    def class_id(self) -> int:
        return int(self.detection.get("cls", 0))

    @property
    def mean_embedding(self) -> Optional[np.ndarray]:
        if not self._embedding_buffer:
            return None
        embs = np.stack(self._embedding_buffer)
        mean = embs.mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / (norm + 1e-8)

    def predict(self) -> np.ndarray:
        return self._kalman.predict()

    def update(self, detection: dict, embedding: Optional[np.ndarray]) -> None:
        self.detection = detection
        self.hits += 1
        self.is_new = False
        self.frames_since_seen = 0
        self._kalman.update(np.array(detection["bbox"], dtype=np.float32))
        if embedding is not None:
            self._embedding_buffer.append(embedding)

    def apply_warp(self, M: Optional[np.ndarray]) -> None:
        """Apply affine warp to the predicted bbox center."""
        if M is None:
            return
        bbox = self._kalman.bbox
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        pt = np.array([[cx, cy, 1.0]])
        warped = (M @ pt.T).T.flatten()[:2]
        dx, dy = warped[0] - cx, warped[1] - cy
        # Shift the Kalman state (cx, cy components)
        self._kalman.kf.x[0] += dx
        self._kalman.kf.x[1] += dy


# ─────────────────────────────────────────────────────────────────────────────
# BoT-SORT-ReID Tracker
# ─────────────────────────────────────────────────────────────────────────────

class BotSortReIDTracker(BaseTracker):
    """
    BoT-SORT tracker with Re-ID fusion.

    Implements the full two-pass association pipeline with optional
    camera motion compensation and ReID-fused cost matrix.
    """

    def __init__(self, config: dict) -> None:
        self.cfg          = config.get("botsort_reid", {})
        self._tracks:     dict[int, BotSortTrack] = {}
        self._next_id     = 1
        self._frame_count = 0
        self._cmc         = CameraMotionCompensator(
            method    = self.cfg.get("cmc_method",    "ecc"),
            downscale = self.cfg.get("cmc_downscale", 2),
        )

    # ──────────────────────────────────────────────────────────────────────────
    @property
    def needs_embeddings(self) -> bool:
        return True

    # ──────────────────────────────────────────────────────────────────────────
    def update(
        self,
        frame: np.ndarray,
        detections: list[dict],
        embeddings: dict[int, np.ndarray] | None = None,
    ) -> list[TrackerOutput]:
        self._frame_count += 1
        if embeddings is None:
            embeddings = {}

        # 1. CMC: compute affine warp from previous frame
        warp_matrix = self._cmc.compute(frame)

        # 2. Kalman predict + apply CMC warp
        for trk in self._tracks.values():
            trk.predict()
            trk.apply_warp(warp_matrix)

        # 3. Split high/low confidence detections
        thresh_high = self.cfg.get("match_thresh_high", 0.8)
        high_dets   = [d for d in detections if d.get("conf", 1.0) >= thresh_high]
        low_dets    = [d for d in detections if d.get("conf", 1.0) <  thresh_high]
        # Map detection original index → high/low list index
        high_idx_map = {i: di for di, i in enumerate(
            j for j, d in enumerate(detections) if d.get("conf", 1.0) >= thresh_high)}
        low_idx_map  = {i: di for di, i in enumerate(
            j for j, d in enumerate(detections) if d.get("conf", 1.0) <  thresh_high)}

        high_embs = {high_idx_map[k]: v for k, v in embeddings.items()
                     if k in high_idx_map}
        low_embs  = {low_idx_map[k]:  v for k, v in embeddings.items()
                     if k in low_idx_map}

        # 4. First association: active tracks vs. high-conf detections
        matched_h, unmatched_trk_ids, unmatched_det_h = self._associate(
            list(self._tracks.keys()), high_dets, high_embs, use_reid=True,
        )

        # 5. Second association: unmatched tracks vs. low-conf detections
        matched_l, still_lost, _ = self._associate(
            unmatched_trk_ids, low_dets, low_embs, use_reid=False,
        )

        # 6. Update matched tracks
        for trk_id, det_i in matched_h:
            emb = high_embs.get(det_i)
            self._tracks[trk_id].update(high_dets[det_i], emb)

        for trk_id, det_i in matched_l:
            emb = low_embs.get(det_i)
            self._tracks[trk_id].update(low_dets[det_i], emb)

        # 7. Mark lost tracks
        for trk_id in still_lost:
            self._tracks[trk_id].frames_since_seen += 1

        # 8. Initialize new tracks from unmatched high-conf detections
        for det_i in unmatched_det_h:
            emb = high_embs.get(det_i)
            new_trk = BotSortTrack(
                tracker_id = self._next_id,
                detection  = high_dets[det_i],
                embedding  = emb,
            )
            self._tracks[self._next_id] = new_trk
            self._next_id += 1

        # 9. Prune dead tracks
        max_age = self.cfg.get("max_age", 60)
        dead    = [tid for tid, t in self._tracks.items()
                   if t.frames_since_seen > max_age]
        for tid in dead:
            del self._tracks[tid]

        # 10. Return confirmed tracks (hits >= min_hits)
        min_hits = self.cfg.get("min_hits", 3)
        return [
            TrackerOutput(
                tracker_id        = t.tracker_id,
                bbox              = t.bbox.copy(),
                confidence        = t.confidence,
                class_id          = t.class_id,
                is_new            = t.is_new,
                frames_since_seen = t.frames_since_seen,
            )
            for t in self._tracks.values()
            if t.hits >= min_hits
        ]

    # ──────────────────────────────────────────────────────────────────────────
    def _associate(
        self,
        track_ids: list[int],
        detections: list[dict],
        embeddings: dict[int, np.ndarray],
        use_reid:   bool = True,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Build cost matrix and solve Hungarian assignment.

        Returns
        -------
        matched  : list of (track_id, det_index) pairs
        unmatched_tracks : list of track_ids with no assignment
        unmatched_dets   : list of det indices with no assignment
        """
        if not track_ids or not detections:
            return [], list(track_ids), list(range(len(detections)))

        track_bboxes = np.array([self._tracks[tid].bbox for tid in track_ids])
        det_bboxes   = np.array([np.array(d["bbox"]) for d in detections])

        # IoU cost (1 - IoU)
        iou_mat  = _iou_matrix(track_bboxes, det_bboxes)
        cost_mat = 1.0 - iou_mat

        # ReID cost fusion
        reid_weight = float(self.cfg.get("reid_weight", 0.4)) if use_reid else 0.0
        if use_reid and reid_weight > 0 and embeddings:
            for di, emb in embeddings.items():
                for ti, tid in enumerate(track_ids):
                    trk_emb = self._tracks[tid].mean_embedding
                    if trk_emb is not None:
                        sim = float(np.dot(emb, trk_emb))
                        reid_cost = 1.0 - sim
                        cost_mat[ti, di] = (
                            (1 - reid_weight) * cost_mat[ti, di]
                            + reid_weight * reid_cost
                        )

        # Gate: set infeasible pairs to 1e9
        iou_gate = self.cfg.get("iou_threshold", 0.3)
        cost_mat[iou_mat < iou_gate] = 1e9

        row_ind, col_ind = linear_sum_assignment(cost_mat)

        matched, unmatched_trk, unmatched_det = [], [], []
        matched_det_set, matched_trk_set = set(), set()

        for r, c in zip(row_ind, col_ind):
            if cost_mat[r, c] < 1e8:
                matched.append((track_ids[r], c))
                matched_trk_set.add(track_ids[r])
                matched_det_set.add(c)

        unmatched_trk = [tid for tid in track_ids if tid not in matched_trk_set]
        unmatched_det = [i for i in range(len(detections)) if i not in matched_det_set]

        return matched, unmatched_trk, unmatched_det

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        self._tracks.clear()
        self._next_id     = 1
        self._frame_count = 0
        self._cmc._prev_gray = None
        logger.debug("BotSortReIDTracker state reset.")
