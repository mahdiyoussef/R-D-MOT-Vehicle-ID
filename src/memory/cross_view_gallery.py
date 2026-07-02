"""
src/memory/cross_view_gallery.py
─────────────────────────────────
Per-view vehicle identity store with cold storage and lifecycle management.

Each identity stores separate embedding buffers per camera viewpoint:

  gallery[global_id] = {
    "front":        deque(maxlen=8),   # L2-normalized embeddings
    "rear":         deque(maxlen=8),
    "side_left":    deque(maxlen=8),
    "side_right":   deque(maxlen=8),
    "top_down":     deque(maxlen=8),
    "ambiguous":    deque(maxlen=8),
    "mean_embeds":  {view: np.ndarray | None},   # cached mean, recomputed on insert
    "part_embeds":  {view: np.ndarray | None},   # (6, D) mean part embedding per view
    "attributes":   AttributeVec,                # EMA-updated semantic attributes
    "last_seen":    int,                         # frame number of last observation
    "class_id":     int,
    "track_status": "active" | "lost" | "expired",
    "lost_since":   int | None,
  }

Identity lifecycle:
  active → (absent > lost_timeout)    → lost
  lost   → (absent > cold_timeout)    → expired → cold storage
  expired → (redetected, sim ≥ 0.85) → active (restored from cold)

IDs are NEVER deleted within a session.
"""

from __future__ import annotations

import logging
import pickle
from collections import deque
from pathlib import Path

import numpy as np
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

from src.feature_extraction.attribute_extractor import AttributeVec

logger = logging.getLogger(__name__)

VIEW_SLOTS = ["front", "rear", "side_left", "side_right", "top_down", "ambiguous"]


def _create_empty_entry(global_id: int, class_id: int, frame_index: int) -> dict:
    return {
        "global_id":    global_id,
        "class_id":     class_id,
        "last_seen":    frame_index,
        "track_status": "active",
        "lost_since":   None,
        "attributes":   AttributeVec(),
        "front":        deque(maxlen=8),
        "rear":         deque(maxlen=8),
        "side_left":    deque(maxlen=8),
        "side_right":   deque(maxlen=8),
        "top_down":     deque(maxlen=8),
        "ambiguous":    deque(maxlen=8),
        "mean_embeds":  {v: None for v in VIEW_SLOTS},
        "part_embeds":  {v: None for v in VIEW_SLOTS},
        "anomaly_embeds": {v: None for v in VIEW_SLOTS},
    }


class CrossViewGallery:
    """
    Per-view vehicle identity store with cold storage and lifecycle FSM.

    Parameters
    ----------
    config    : pipeline config dict
    embed_dim : global embedding dimension
    """

    def __init__(self, config: dict, embed_dim: int = 512) -> None:
        cfg = config.get("gallery_v4", {})
        self.embed_dim = embed_dim

        self._lost_timeout_frames  = int(cfg.get("lost_timeout_frames", 150))
        self._cold_timeout_frames  = int(cfg.get("cold_storage_timeout_frames", 18000))
        self._cold_reentry_thresh  = float(cfg.get("cold_reentry_threshold", 0.85))
        self._similarity_threshold = float(
            config.get("gallery", {}).get("similarity_threshold", 0.85)
        )

        # ── v6.0: Dual-Memory Bank parameters ─────────────────────────────────
        # Time-adaptive re-entry: threshold increases logarithmically with
        # absence duration to require more evidence for longer disappearances.
        self._reentry_base_thresh = float(cfg.get("reentry_base_threshold", 0.70))
        self._reentry_beta        = float(cfg.get("reentry_time_beta", 0.04))
        self._reentry_tau         = float(cfg.get("reentry_time_tau", 150.0))  # frames
        self._assumed_fps         = float(cfg.get("assumed_fps", 30.0))

        self._gallery:      dict[int, dict] = {}
        self._cold_storage: dict[int, dict] = {}
        self._archive_embeds: dict[int, tuple[np.ndarray, int]] = {}  # gid -> (frozen_embed, frame_lost)
        self._next_global_id: int = 1

    # ── Core API ───────────────────────────────────────────────────────────────

    def insert_embedding(
        self,
        global_id:   int,
        embedding:   np.ndarray,
        view_label:  str,
        class_id:    int,
        frame_index: int,
        part_embeds: np.ndarray | None = None,
        anomaly_embeds: np.ndarray | None = None,
        attributes:  AttributeVec | None = None,
        camera_id:   int = 0,
    ) -> None:
        """
        Insert or update a gallery entry for a vehicle.

        Parameters
        ----------
        global_id   : The persistent ID assigned to the vehicle.
        embedding   : Full (global) feature vector.
        view_label  : Perspective classification (front, rear, side_left, side_right, ambiguous).
        class_id    : Detected vehicle class index.
        frame_index : Current timestamp for lifecycle management.
        part_embeds : Local/stripe features if available.
        attributes  : Extracted physical attributes (color, type) if available.
        camera_id   : ID of the camera observing this detection.
        """
        normalized_emb = _l2_normalize(embedding)

        if global_id not in self._gallery:
            self._gallery[global_id] = _create_empty_entry(global_id, class_id, frame_index)
            if global_id >= self._next_global_id:
                self._next_global_id = global_id + 1

        entry = self._gallery[global_id]
        slot  = view_label if view_label in VIEW_SLOTS else "ambiguous"

        # Append to rolling view buffer
        entry[slot].append(normalized_emb)

        # Recompute cached mean for this view slot
        stacked       = np.stack(entry[slot])
        entry["mean_embeds"][slot] = _l2_normalize(stacked.mean(axis=0))

        # Update part embedding cache (running mean)
        if part_embeds is not None:
            existing = entry["part_embeds"][slot]
            entry["part_embeds"][slot] = (
                part_embeds.astype(np.float32)
                if existing is None
                else (existing + part_embeds.astype(np.float32)) / 2.0
            )

        # Update anomaly embedding cache (running mean of attention patches)
        if anomaly_embeds is not None:
            existing_anom = entry["anomaly_embeds"][slot]
            entry["anomaly_embeds"][slot] = (
                anomaly_embeds.astype(np.float32)
                if existing_anom is None
                else (existing_anom + anomaly_embeds.astype(np.float32)) / 2.0
            )

        # Update metadata
        entry["last_seen"] = frame_index
        entry["last_camera_id"] = camera_id
        entry["class_id"]  = class_id
        if entry["track_status"] != "active":
            entry["track_status"] = "active"
            entry["lost_since"]   = None

        # EMA update semantic attributes
        if attributes is not None:
            entry["attributes"].ema_update(attributes)

    def find_best_match(
        self,
        embedding:      np.ndarray,
        view_label:     str,
        class_id:       int,
        frame_index:    int,
        excluded_ids:   set[int],
        part_embeds:    np.ndarray | None = None,
        spatial_boosts: dict[int, float] | None = None,
    ) -> tuple[int | None, float, str]:
        """
        Query gallery for the best matching identity.

        Returns
        -------
        (global_id, score, match_source)
            match_source: 'same_view' | 'cross_view' | 'cold_reentry' | 'no_match'
        """
        normalized_query = _l2_normalize(embedding)
        best_id     = None
        best_score  = -1.0
        best_source = "same_view"

        for gid, entry in self._gallery.items():
            if gid in excluded_ids:
                continue
            if entry["class_id"] != class_id:
                continue
            if entry["track_status"] == "expired":
                continue

            score, source = self._score_entry(entry, normalized_query, view_label, part_embeds)

            if spatial_boosts and gid in spatial_boosts:
                score += spatial_boosts[gid] * 0.05  # soft Kalman proximity boost

            if score > best_score:
                best_score  = score
                best_id     = gid
                best_source = source

        if best_id is not None and best_score >= self._similarity_threshold:
            return best_id, best_score, best_source

        # Check cold storage with time-adaptive threshold (v6.0)
        cold_id, cold_score = self._query_cold_storage(
            normalized_query, view_label, class_id, excluded_ids, part_embeds,
            frame_index=frame_index,
        )
        if cold_id is not None:
            adaptive_thresh = self._compute_adaptive_reentry_threshold(cold_id, frame_index)
            if cold_score >= adaptive_thresh:
                restored_entry = self._cold_storage.pop(cold_id)
                restored_entry["track_status"] = "active"
                restored_entry["lost_since"]   = None
                self._gallery[cold_id] = restored_entry
                self._archive_embeds.pop(cold_id, None)  # Clear archive on re-entry
                logger.info(
                    "Gallery: gid=%d restored from cold storage (sim=%.3f, adaptive_thresh=%.3f)",
                    cold_id, cold_score, adaptive_thresh,
                )
                return cold_id, cold_score, "cold_reentry"

        return None, best_score, "no_match"

    def mint_new_id(self, class_id: int, frame_index: int) -> int:
        """Allocate a new global ID and initialise its gallery entry."""
        new_id = self._next_global_id
        self._next_global_id += 1
        self._gallery[new_id] = _create_empty_entry(new_id, class_id, frame_index)
        return new_id

    def advance_lifecycle(self, frame_index: int) -> tuple[list[int], list[int]]:
        """
        Advance the track_status lifecycle FSM for all entries.

        Returns
        -------
        (newly_lost_ids, newly_expired_ids)
        """
        newly_lost:    list[int] = []
        newly_expired: list[int] = []

        for gid, entry in list(self._gallery.items()):
            frames_absent = frame_index - entry["last_seen"]

            if entry["track_status"] == "active" and frames_absent >= self._lost_timeout_frames:
                entry["track_status"] = "lost"
                entry["lost_since"]   = frame_index
                newly_lost.append(gid)
                # v6.0: Archive the best embedding at the moment of loss
                best_embed = self._best_mean_embed(entry)
                if best_embed is not None:
                    self._archive_embeds[gid] = (best_embed.copy(), frame_index)
                logger.debug("Gallery: gid=%d → LOST at frame=%d (absent %d frames, archive=%s)",
                             gid, frame_index, frames_absent, gid in self._archive_embeds)

            elif entry["track_status"] == "lost":
                lost_duration = frame_index - (entry["lost_since"] or frame_index)
                if lost_duration >= self._cold_timeout_frames:
                    entry["track_status"]    = "expired"
                    self._cold_storage[gid] = entry
                    del self._gallery[gid]
                    newly_expired.append(gid)
                    logger.debug("Gallery: gid=%d → EXPIRED → cold storage at frame=%d", gid, frame_index)

        return newly_lost, newly_expired

    def get_mean_embedding(self, global_id: int, view: str = "ambiguous") -> np.ndarray | None:
        entry = self._gallery.get(global_id)
        return entry["mean_embeds"].get(view) if entry else None

    def get_part_embedding(self, global_id: int, view: str = "ambiguous") -> np.ndarray | None:
        entry = self._gallery.get(global_id)
        return entry["part_embeds"].get(view) if entry else None

    def get_attributes(self, global_id: int) -> AttributeVec | None:
        entry = self._gallery.get(global_id)
        return entry.get("attributes") if entry else None

    def get_all_active_mean_embeddings(self) -> tuple[np.ndarray, list[int]] | None:
        """
        Return stacked mean embeddings + ID list for all active identities.
        Uses the best available view slot per identity.
        """
        embeddings, ids = [], []
        for gid, entry in self._gallery.items():
            if entry["track_status"] == "expired":
                continue
            representative = self._best_mean_embed(entry)
            if representative is not None:
                embeddings.append(representative)
                ids.append(gid)
        if not embeddings:
            return None
        return np.stack(embeddings, axis=0).astype(np.float32), ids

    @property
    def num_active(self) -> int:
        return len(self._gallery)

    @property
    def num_cold(self) -> int:
        return len(self._cold_storage)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_snapshot(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "gallery":      self._gallery,
                "cold_storage": self._cold_storage,
                "next_id":      self._next_global_id,
            }, f)
        logger.info("CrossViewGallery saved → %s (%d active, %d cold)", path, self.num_active, self.num_cold)

    def load_snapshot(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            logger.warning("Gallery snapshot not found at '%s'. Starting fresh.", path)
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._gallery         = data.get("gallery", {})
        self._cold_storage    = data.get("cold_storage", {})
        self._next_global_id  = data.get("next_id", 1)
        logger.info("CrossViewGallery loaded ← %s (%d active, %d cold)", path, self.num_active, self.num_cold)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _score_entry(
        self,
        entry:       dict,
        query_vec:   np.ndarray,
        view_label:  str,
        part_embeds: np.ndarray | None,
    ) -> tuple[float, str]:
        """Score a gallery entry: tries same-view first, then cross-view."""
        from src.geometry.view_geometry import compute_cross_view_cosine

        same_view_mean = entry["mean_embeds"].get(view_label)
        if same_view_mean is not None:
            score = _cosine_sim(query_vec.flatten(), same_view_mean.flatten())
            return score, "same_view"

        best_cross_score = -1.0
        if part_embeds is not None:
            for other_view in VIEW_SLOTS:
                gallery_parts = entry["part_embeds"].get(other_view)
                if gallery_parts is None or other_view == view_label:
                    continue
                cross_score, is_valid = compute_cross_view_cosine(
                    part_embeds, gallery_parts, view_label, other_view
                )
                if is_valid and cross_score > best_cross_score:
                    best_cross_score = cross_score

        if best_cross_score > -1.0:
            return best_cross_score, "cross_view"

        representative = self._best_mean_embed(entry)
        if representative is not None:
            score = _cosine_sim(query_vec.flatten(), representative.flatten())
            return score, "cross_view"

        return -1.0, "no_match"

    def _query_cold_storage(
        self,
        query_vec:    np.ndarray,
        view_label:   str,
        class_id:     int,
        excluded_ids: set[int],
        part_embeds:  np.ndarray | None,
        frame_index:  int = 0,
    ) -> tuple[int | None, float]:
        best_id    = None
        best_score = -1.0
        for gid, entry in self._cold_storage.items():
            if gid in excluded_ids or entry["class_id"] != class_id:
                continue
            score, _ = self._score_entry(entry, query_vec, view_label, part_embeds)

            # v6.0: Also check archive embedding (frozen at time of loss)
            archive = self._archive_embeds.get(gid)
            if archive is not None:
                archived_embed, frame_lost = archive
                archive_score = _cosine_sim(query_vec.flatten(), archived_embed.flatten())
                # Use the better of the two scores
                score = max(score, archive_score)

            if score > best_score:
                best_score = score
                best_id    = gid
        return best_id, best_score

    def _compute_adaptive_reentry_threshold(self, gid: int, current_frame: int) -> float:
        """
        v6.0: Time-adaptive re-entry threshold.

        Short absence  (< 5s):  lower threshold ~0.70 (same vehicle, slight change)
        Medium absence (5-60s): standard        ~0.80
        Long absence   (> 60s): higher          ~0.88+ (more evidence needed)

        Formula: threshold = base + beta * log(1 + absence_frames / tau)
        """
        import math
        entry = self._cold_storage.get(gid) or self._gallery.get(gid)
        if entry is None:
            return self._cold_reentry_thresh
        last_seen = entry.get("last_seen", current_frame)
        absence_frames = max(0, current_frame - last_seen)
        adaptive = self._reentry_base_thresh + self._reentry_beta * math.log(
            1.0 + absence_frames / self._reentry_tau
        )
        # Clamp to [base, 0.95] to avoid impossible thresholds
        return min(adaptive, 0.95)

    @staticmethod
    def _best_mean_embed(entry: dict) -> np.ndarray | None:
        """Return the first non-None mean embedding from any view slot."""
        for view in VIEW_SLOTS:
            mean = entry["mean_embeds"].get(view)
            if mean is not None:
                return mean
        return None


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)
