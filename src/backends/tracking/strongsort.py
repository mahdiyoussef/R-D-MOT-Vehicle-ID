"""
src/backends/tracking/strongsort_v2.py
───────────────────────────────────────
StrongSORT v2 — enhanced tracking with:
  a. EMA-based appearance update built on top of the existing boxmot tracker
  b. AFLink: lightweight MLP post-processor for offline tracklet linking
  c. GSI: Gaussian-smoothed cubic spline interpolation for gap filling

Reference: Du et al., "StrongSORT: Make DeepSORT Great Again"
           IEEE TCSVT 2023. https://arxiv.org/abs/2202.13514
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.backends.tracking.base import BaseTracker, TrackerOutput

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# AFLink: Appearance-Free Link model
# ─────────────────────────────────────────────────────────────────────────────

class AFLinkModel(nn.Module):
    """
    Lightweight MLP that predicts the probability that two tracklets belong
    to the same vehicle identity, using only temporal + positional features
    (no appearance features).

    Feature vector (9D) per tracklet pair:
      [delta_t, delta_cx, delta_cy, delta_w, delta_h,
       velocity_cx, velocity_cy, track_len_1, track_len_2]

    Architecture: 9 → 64 → 64 → 1 (sigmoid output)
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),  nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_pair(self, feat: np.ndarray) -> float:
        """Predict linking probability for a single feature vector."""
        t = torch.FloatTensor(feat).unsqueeze(0)
        with torch.no_grad():
            return float(self.net(t).item())

    @classmethod
    def load_from_path(cls, path: str) -> "AFLinkModel":
        model = cls()
        path  = Path(path)
        if path.exists():
            ckpt = torch.load(path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            model.load_state_dict(ckpt, strict=False)
            logger.info("AFLink weights loaded from '%s'.", path)
        else:
            logger.warning(
                "AFLink weights not found at '%s'. "
                "Using randomly initialised model — linking quality will be poor. "
                "Run: python scripts/download_models.py --model aflink",
                path,
            )
        model.eval()
        return model

    @staticmethod
    def _build_feature(
        track_end:   dict,   # {'cx', 'cy', 'w', 'h', 'vx', 'vy', 'len', 'frame'}
        track_start: dict,
    ) -> np.ndarray:
        """Build the 9D feature vector for an AFLink pair."""
        delta_t  = track_start["frame"] - track_end["frame"]
        delta_cx = track_start["cx"]    - track_end["cx"]
        delta_cy = track_start["cy"]    - track_end["cy"]
        delta_w  = track_start["w"]     - track_end["w"]
        delta_h  = track_start["h"]     - track_end["h"]
        vel_cx   = track_end.get("vx", 0.0)
        vel_cy   = track_end.get("vy", 0.0)
        len1     = float(track_end["len"])
        len2     = float(track_start["len"])
        return np.array([delta_t, delta_cx, delta_cy, delta_w, delta_h,
                         vel_cx, vel_cy, len1, len2], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian-Smoothed Interpolator (GSI)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianSmoothInterpolator:
    """
    For track gaps of length 1..gsi_interval:
      1. Fit a cubic spline through confirmed bbox positions on both sides of gap
      2. Generate interpolated positions at gap frames
      3. Apply Gaussian smoothing with kernel bandwidth tau
    """

    def __init__(self, gsi_interval: int = 20, tau: float = 10.0) -> None:
        self.gsi_interval = gsi_interval
        self.tau          = tau

    def interpolate(
        self,
        history: list[tuple[int, np.ndarray]],
    ) -> list[tuple[int, np.ndarray]]:
        """
        Fill gaps in a track's bbox history.

        Parameters
        ----------
        history : list of (frame_n, bbox) sorted by frame_n

        Returns
        -------
        Augmented history with interpolated entries inserted.
        """
        if len(history) < 2:
            return history

        from scipy.interpolate import CubicSpline
        from scipy.ndimage import gaussian_filter1d

        history = sorted(history, key=lambda x: x[0])
        frames  = np.array([h[0] for h in history], dtype=float)
        bboxes  = np.array([h[1] for h in history], dtype=np.float32)

        filled = list(history)

        for i in range(len(history) - 1):
            gap = history[i + 1][0] - history[i][0]
            if 1 < gap <= self.gsi_interval:
                # Interpolate this gap
                f0, f1 = history[i][0], history[i + 1][0]
                gap_frames = np.arange(f0 + 1, f1)

                # Fit cubic splines for each bbox coordinate
                for coord in range(4):
                    y = bboxes[:, coord]
                    cs = CubicSpline(frames, y)
                    interp_vals = cs(gap_frames.astype(float))
                    # Gaussian smoothing
                    sigma = self.tau / (gap + 1e-6)
                    smoothed = gaussian_filter1d(interp_vals, sigma=sigma)
                    # Insert back
                    for j, gf in enumerate(gap_frames):
                        # Find or create entry for this frame
                        entry_idx = next(
                            (k for k, (f, _) in enumerate(filled) if f == int(gf)),
                            None,
                        )
                        if entry_idx is None:
                            new_bbox = bboxes[i].copy()
                            new_bbox[coord] = smoothed[j]
                            filled.append((int(gf), new_bbox))
                        else:
                            arr = list(filled[entry_idx][1])
                            arr[coord] = smoothed[j]
                            filled[entry_idx] = (int(gf), np.array(arr, dtype=np.float32))

        return sorted(filled, key=lambda x: x[0])


# ─────────────────────────────────────────────────────────────────────────────
# StrongSORT v2 Tracker
# ─────────────────────────────────────────────────────────────────────────────

class StrongSORTv2Tracker(BaseTracker):
    """
    StrongSORT v2 — wraps the existing VehicleTracker (boxmot) as its base
    association engine, then adds EMA appearance update, appearance-guided
    re-association for lost tracks, and offline AFLink + GSI post-processing.
    """

    def __init__(self, config: dict) -> None:
        self.cfg            = config.get("strongsort_v2", {})
        self.full_config    = config
        self.ema_alpha      = float(self.cfg.get("ema_alpha", 0.9))

        # Track state
        self._track_embeddings: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )
        self._track_history: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        self._frame_count   = 0
        self._known_ids: set[int] = set()

        # Base tracker (legacy boxmot wrapper)
        self._base_tracker = None

        # AFLink and GSI (optional)
        self._aflink: Optional[AFLinkModel] = None
        self._gsi:    Optional[GaussianSmoothInterpolator] = None

        if self.cfg.get("aflink_enabled", True):
            model_path = self.cfg.get("aflink_model_path", "models/aflink.pth")
            self._aflink = AFLinkModel.load_from_path(model_path)

        if self.cfg.get("gsi_enabled", True):
            self._gsi = GaussianSmoothInterpolator(
                gsi_interval = self.cfg.get("gsi_interval", 20),
                tau          = float(self.cfg.get("gsi_tau", 10.0)),
            )

    def _get_base_tracker(self):
        if self._base_tracker is None:
            from src.backends.tracking.legacy_tracker import LegacyTrackerBackend
            self._base_tracker = LegacyTrackerBackend(self.full_config)
        return self._base_tracker

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

        # 1. Run base association
        base_outputs = self._get_base_tracker().update(frame, detections, None)

        # 2. EMA embedding update for each confirmed track
        for out in base_outputs:
            tid = out.tracker_id
            # Attempt to map detection index to embedding
            # (we use the track's bbox to find the closest detection)
            best_emb = self._find_best_embedding(out.bbox, detections, embeddings)
            if best_emb is not None:
                buf = self._track_embeddings[tid]
                if buf:
                    ema = self.ema_alpha * buf[-1] + (1 - self.ema_alpha) * best_emb
                    norm = np.linalg.norm(ema)
                    buf.append(ema / (norm + 1e-8))
                else:
                    buf.append(best_emb)

        # 3. Record bbox history for GSI
        for out in base_outputs:
            self._track_history[out.tracker_id].append(
                (self._frame_count, out.bbox.copy())
            )

        return base_outputs

    # ──────────────────────────────────────────────────────────────────────────
    def _find_best_embedding(
        self,
        bbox:       np.ndarray,
        detections: list[dict],
        embeddings: dict[int, np.ndarray],
    ) -> Optional[np.ndarray]:
        """Find the embedding whose detection bbox most overlaps with bbox."""
        if not embeddings:
            return None

        best_iou  = 0.2   # minimum IoU to claim
        best_emb  = None

        for det_i, emb in embeddings.items():
            if det_i >= len(detections):
                continue
            d_bbox = np.array(detections[det_i]["bbox"])
            # Compute IoU
            ix1 = max(bbox[0], d_bbox[0])
            iy1 = max(bbox[1], d_bbox[1])
            ix2 = min(bbox[2], d_bbox[2])
            iy2 = min(bbox[3], d_bbox[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_a = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            area_b = (d_bbox[2]-d_bbox[0])*(d_bbox[3]-d_bbox[1])
            iou = inter / (area_a + area_b - inter + 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_emb = emb

        return best_emb

    # ──────────────────────────────────────────────────────────────────────────
    def finalize(
        self,
        all_outputs: list[list[TrackerOutput]],
    ) -> list[list[TrackerOutput]]:
        """
        Offline post-processing (called after all frames are processed):
          1. AFLink: link short tracklets separated by ≤ aflink_window frames
          2. GSI: fill gaps with Gaussian-smoothed interpolation

        Parameters
        ----------
        all_outputs : per-frame list of TrackerOutput lists

        Returns
        -------
        Modified all_outputs with merged tracklets and interpolated gaps.
        """
        if not all_outputs:
            return all_outputs

        # Build tracklet summary for AFLink
        if self._aflink is not None:
            all_outputs = self._apply_aflink(all_outputs)

        # Apply GSI gap filling
        if self._gsi is not None:
            all_outputs = self._apply_gsi(all_outputs)

        return all_outputs

    # ──────────────────────────────────────────────────────────────────────────
    def _apply_aflink(
        self, all_outputs: list[list[TrackerOutput]]
    ) -> list[list[TrackerOutput]]:
        """Post-process tracklet linkage with AFLink MLP."""
        aflink_window = self.cfg.get("aflink_window", 30)
        min_length    = self.cfg.get("aflink_min_length", 2)
        prob_thresh   = self.cfg.get("aflink_prob_thresh", 0.8)

        # Build tracklet summaries
        tracklets: dict[int, dict] = {}
        for frame_n, frame_outs in enumerate(all_outputs):
            for out in frame_outs:
                tid = out.tracker_id
                if tid not in tracklets:
                    cx = (out.bbox[0] + out.bbox[2]) / 2
                    cy = (out.bbox[1] + out.bbox[3]) / 2
                    tracklets[tid] = {
                        "start_frame": frame_n,
                        "end_frame":   frame_n,
                        "start_cx":    cx, "start_cy": cy,
                        "start_w":     out.bbox[2] - out.bbox[0],
                        "start_h":     out.bbox[3] - out.bbox[1],
                        "end_cx":      cx, "end_cy": cy,
                        "end_w":       out.bbox[2] - out.bbox[0],
                        "end_h":       out.bbox[3] - out.bbox[1],
                        "len":         1,
                        "vx": 0.0, "vy": 0.0,
                    }
                else:
                    prev_cx = tracklets[tid]["end_cx"]
                    prev_cy = tracklets[tid]["end_cy"]
                    cx = (out.bbox[0] + out.bbox[2]) / 2
                    cy = (out.bbox[1] + out.bbox[3]) / 2
                    tracklets[tid]["vx"] = cx - prev_cx
                    tracklets[tid]["vy"] = cy - prev_cy
                    tracklets[tid]["end_frame"] = frame_n
                    tracklets[tid]["end_cx"] = cx
                    tracklets[tid]["end_cy"] = cy
                    tracklets[tid]["end_w"]  = out.bbox[2] - out.bbox[0]
                    tracklets[tid]["end_h"]  = out.bbox[3] - out.bbox[1]
                    tracklets[tid]["len"]   += 1

        # Filter short tracklets
        tracklets = {tid: t for tid, t in tracklets.items()
                     if t["len"] >= min_length}

        # Build id-remapping via AFLink
        id_remap: dict[int, int] = {}  # old_id → merged_id

        tids = sorted(tracklets.keys())
        for i, tid1 in enumerate(tids):
            for tid2 in tids[i+1:]:
                t1 = tracklets[tid1]
                t2 = tracklets[tid2]
                gap = t2["start_frame"] - t1["end_frame"]
                if 0 < gap <= aflink_window:
                    end_info   = {"frame": t1["end_frame"],   "cx": t1["end_cx"],
                                  "cy": t1["end_cy"], "w": t1["end_w"],
                                  "h": t1["end_h"], "vx": t1["vx"],
                                  "vy": t1["vy"], "len": t1["len"]}
                    start_info = {"frame": t2["start_frame"], "cx": t2["start_cx"],
                                  "cy": t2["start_cy"], "w": t2["start_w"],
                                  "h": t2["start_h"], "len": t2["len"]}
                    feat = AFLinkModel._build_feature(end_info, start_info)
                    prob = self._aflink.predict_pair(feat)
                    if prob >= prob_thresh:
                        # Merge tid2 into tid1
                        keep = min(tid1, tid2)
                        drop = max(tid1, tid2)
                        id_remap[drop] = keep

        # Apply remap to all_outputs
        if id_remap:
            merged = []
            for frame_outs in all_outputs:
                new_frame = []
                for out in frame_outs:
                    new_id = id_remap.get(out.tracker_id, out.tracker_id)
                    # Resolve chains
                    visited = set()
                    while new_id in id_remap and new_id not in visited:
                        visited.add(new_id)
                        new_id = id_remap[new_id]
                    from dataclasses import replace
                    new_frame.append(TrackerOutput(
                        tracker_id        = new_id,
                        bbox              = out.bbox,
                        confidence        = out.confidence,
                        class_id          = out.class_id,
                        is_new            = out.is_new,
                        frames_since_seen = out.frames_since_seen,
                    ))
                merged.append(new_frame)
            logger.info("AFLink merged %d tracklet pairs.", len(id_remap))
            return merged

        return all_outputs

    # ──────────────────────────────────────────────────────────────────────────
    def _apply_gsi(
        self, all_outputs: list[list[TrackerOutput]]
    ) -> list[list[TrackerOutput]]:
        """Fill track gaps with Gaussian-smoothed interpolated bboxes."""
        # Collect per-track history
        track_histories: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        for frame_n, frame_outs in enumerate(all_outputs):
            for out in frame_outs:
                track_histories[out.tracker_id].append((frame_n, out.bbox.copy()))

        # Interpolate each track
        interpolated: dict[int, dict[int, np.ndarray]] = {}
        for tid, hist in track_histories.items():
            filled = self._gsi.interpolate(hist)
            interpolated[tid] = {f: bbox for f, bbox in filled}

        # Rebuild all_outputs with interpolated frames inserted
        max_frame = len(all_outputs)
        rebuilt = [list(frame_outs) for frame_outs in all_outputs]

        for tid, frame_bbox_map in interpolated.items():
            # Find what was in the original output for this track
            orig_frames = {f for f, _ in track_histories[tid]}
            for frame_n, bbox in frame_bbox_map.items():
                if frame_n not in orig_frames and 0 <= frame_n < max_frame:
                    rebuilt[frame_n].append(TrackerOutput(
                        tracker_id        = tid,
                        bbox              = bbox,
                        confidence        = 1.0,
                        class_id          = 0,
                        is_new            = False,
                        frames_since_seen = 0,
                    ))

        return rebuilt

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        if self._base_tracker is not None:
            self._base_tracker.reset()
        self._track_embeddings.clear()
        self._track_history.clear()
        self._frame_count = 0
        self._known_ids.clear()
        logger.debug("StrongSORTv2Tracker state reset.")
