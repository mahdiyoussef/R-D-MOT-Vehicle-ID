"""
src/matching/cascade_matcher.py
────────────────────────────────
5-step cascaded identity assignment engine.

Replaces the flat Hungarian matcher with a decision tree:

  Step 1 — IoU continuation       (cheapest — same bbox, skip Re-ID)
  Step 2 — Same-view gallery match (cosine on matching view slot)
  Step 3 — Cross-view match        (part-filtered cosine on shared stripes)
  Step 4 — Attribute fallback      (color + class + plate weighted score)
  Step 5 — New ID assignment       (only if all above fail)

Each step only fires if all earlier steps failed.
The engine is stateless — all gallery and tracklet state is owned by
CrossViewGallery and KalmanTrackletMemory respectively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

from src.geometry.view_geometry import compute_cross_view_cosine, are_views_compatible

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

MatchType = Literal[
    "iou_continuation",
    "same_view",
    "cross_view",
    "attribute",
    "new_id",
    "reentry",
]


@dataclass
class DetectionInput:
    """All information for a single detection entering the matcher."""
    bbox:         np.ndarray          # [x1, y1, x2, y2]
    embedding:    np.ndarray          # (D,) L2-normalized global embedding
    view_label:   str                 # from ViewClassifier
    class_id:     int
    class_name:   str = "car"
    part_embeds:  np.ndarray | None = None   # (6, stripe_D) or None
    anomaly_embeds: np.ndarray | None = None # (num_anomalies, anomaly_D) or None
    attributes:   object | None = None        # AttributeVec or None
    confidence:   float = 1.0
    camera_id:    int = 0


@dataclass
class ActiveTrack:
    """Minimal representation of an active short-term track."""
    track_id:   int
    global_id:  int
    bbox:       np.ndarray      # current [x1, y1, x2, y2]
    view_label: str = "ambiguous"


@dataclass
class MatchResult:
    """Output of the cascade engine for a single detection."""
    global_id:        int
    match_type:       MatchType
    match_confidence: float
    is_new:           bool = False
    needs_human_review: bool = False  # flagged because top rejected sim > threshold


# ─────────────────────────────────────────────────────────────────────────────
# IoU utility
# ─────────────────────────────────────────────────────────────────────────────

def compute_bbox_iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """Compute Intersection-over-Union between two [x1,y1,x2,y2] bounding boxes."""
    inter_x1 = max(bbox_a[0], bbox_b[0])
    inter_y1 = max(bbox_a[1], bbox_b[1])
    inter_x2 = min(bbox_a[2], bbox_b[2])
    inter_y2 = min(bbox_a[3], bbox_b[3])
    inter_w  = max(0.0, inter_x2 - inter_x1)
    inter_h  = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a  = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b  = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union   = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Cascade Match Engine
# ─────────────────────────────────────────────────────────────────────────────

class CascadeMatcher:
    """
    5-step cascaded identity assignment engine.

    Parameters
    ----------
    config : pipeline config dict (reads from 'matching_v4' and 'attribute_weights')
    """

    def __init__(self, config: dict) -> None:
        matching_cfg = config.get("matching_v4", {})
        self._attribute_weights = config.get("attribute_weights", {
            "color_match":         0.35,
            "vehicle_class_match": 0.25,
            "plate_chars_match":   0.30,
            "roof_rack_match":     0.05,
            "tow_hitch_match":     0.05,
        })
        
        from src.geometry.camera_topology import CameraTopology
        self.topology = CameraTopology(config)

        self._iou_continuation_thresh   = float(matching_cfg.get("iou_continuation_threshold", 0.50))
        self._same_view_accept_thresh   = float(matching_cfg.get("same_view_match_threshold", 0.75))
        self._same_view_uncertain_thresh = float(matching_cfg.get("same_view_uncertain_low", 0.60))
        self._cross_view_thresh         = float(matching_cfg.get("cross_view_match_threshold", 0.55))
        self._attribute_accept_thresh   = float(matching_cfg.get("attribute_score_threshold", 0.60))
        self._human_review_sim_thresh   = float(matching_cfg.get("human_review_flag_threshold", 0.45))

    def assign_identity(
        self,
        detection:      DetectionInput,
        active_tracks:  list[ActiveTrack],
        gallery,                              # CrossViewGallery instance
        already_assigned_ids: set[int],
        frame_index:    int,
        spatial_boosts: dict[int, float] | None = None,  # from KalmanTrackletMemory
    ) -> MatchResult:
        """
        Run the 5-step decision cascade for a single detection.

        Parameters
        ----------
        detection            : DetectionInput
        active_tracks        : currently active tracks in this frame
        gallery              : CrossViewGallery
        already_assigned_ids : global_ids already assigned in this frame
        frame_index          : current frame number
        spatial_boosts       : gid → Kalman proximity bonus from KalmanTrackletMemory

        Returns
        -------
        MatchResult
        """

        excluded_ids = set(already_assigned_ids)
        
        # ── Pre-filter (C2): Spatio-Temporal Constraints ──────────────────────
        if getattr(self, "topology", None) is not None and self.topology.enabled:
            for gid, entry in gallery._gallery.items():
                if gid in excluded_ids:
                    continue
                last_cam = entry.get("last_camera_id", 0)
                last_frame = entry.get("last_seen", 0)
                if not self.topology.is_transition_possible(last_cam, last_frame, detection.camera_id, frame_index):
                    excluded_ids.add(gid)

        # ── Step 1 — IoU Continuation ─────────────────────────────────────────
        for track in active_tracks:
            if track.global_id in excluded_ids:
                continue
            iou = compute_bbox_iou(detection.bbox, track.bbox)
            if iou >= self._iou_continuation_thresh:
                logger.debug(
                    "Match [1 IoU] gid=%d  iou=%.3f  frame=%d",
                    track.global_id, iou, frame_index,
                )
                return MatchResult(
                    global_id        = track.global_id,
                    match_type       = "iou_continuation",
                    match_confidence = iou,
                )

        # ── Step 2 — Same-View Gallery Match ──────────────────────────────────
        same_view_match = self._match_same_view(detection, gallery, excluded_ids, spatial_boosts)
        if same_view_match is not None:
            best_gid, best_score = same_view_match
            if best_score >= self._same_view_accept_thresh:
                logger.debug("Match [2 same-view] gid=%d sim=%.3f", best_gid, best_score)
                return MatchResult(
                    global_id        = best_gid,
                    match_type       = "same_view",
                    match_confidence = best_score,
                )
            elif best_score >= self._same_view_uncertain_thresh:
                # Uncertain score → try attributes before escalating
                logger.debug("Step 2 UNCERTAIN gid=%d sim=%.3f → trying attribute check", best_gid, best_score)
                attribute_match = self._match_by_attributes(
                    detection, gallery, already_assigned_ids, preferred_gid=best_gid
                )
                if attribute_match is not None:
                    return attribute_match

        # ── Step 3 — Cross-View Match ─────────────────────────────────────────
        cross_view_match = self._match_cross_view(detection, gallery, excluded_ids, spatial_boosts)
        if cross_view_match is not None:
            best_gid, best_score = cross_view_match
            if best_score >= self._cross_view_thresh:
                logger.debug("Match [3 cross-view] gid=%d sim=%.3f", best_gid, best_score)
                return MatchResult(
                    global_id        = best_gid,
                    match_type       = "cross_view",
                    match_confidence = best_score,
                )

        # ── Step 4 — Attribute Fallback ───────────────────────────────────────
        attribute_match = self._match_by_attributes(detection, gallery, excluded_ids)
        if attribute_match is not None:
            return attribute_match

        # ── Step 5 — Assign New Identity ─────────────────────────────────────
        new_global_id = gallery.mint_new_id(detection.class_id, frame_index)

        top_rejected_similarity = self._compute_best_rejected_similarity(
            detection, gallery, already_assigned_ids
        )
        needs_review = top_rejected_similarity > self._human_review_sim_thresh
        if needs_review:
            logger.warning(
                "New ID gid=%d — possible missed match (best_rejected_sim=%.3f > %.3f). "
                "Manual review recommended.",
                new_global_id, top_rejected_similarity, self._human_review_sim_thresh,
            )

        logger.debug("Match [5 new_id] gid=%d frame=%d", new_global_id, frame_index)
        return MatchResult(
            global_id           = new_global_id,
            match_type          = "new_id",
            match_confidence    = 0.0,
            is_new              = True,
            needs_human_review  = needs_review,
        )

    # ── Private step implementations ──────────────────────────────────────────

    def _match_same_view(
        self,
        detection:    DetectionInput,
        gallery,
        excluded_ids: set[int],
        spatial_boosts: dict[int, float] | None,
    ) -> tuple[int, float] | None:
        """Query gallery using same-view mean embeddings. Returns (gid, score) or None."""
        query_vec = detection.embedding.reshape(1, -1)
        best_gid   = None
        best_score = -1.0

        for gid, entry in gallery._gallery.items():
            if gid in excluded_ids:
                continue
            if entry["class_id"] != detection.class_id:
                continue
            if entry["track_status"] == "expired":
                continue

            view_mean = entry["mean_embeds"].get(detection.view_label)
            if view_mean is None:
                continue

            score = _cosine_sim(query_vec.flatten(), view_mean.flatten())

            if spatial_boosts and gid in spatial_boosts:
                score += spatial_boosts[gid] * 0.05  # soft Kalman proximity bonus

            if score > best_score:
                best_score = score
                best_gid   = gid

        return (best_gid, best_score) if best_gid is not None else None

    def _match_cross_view(
        self,
        detection:    DetectionInput,
        gallery,
        excluded_ids: set[int],
        spatial_boosts: dict[int, float] | None,
    ) -> tuple[int, float] | None:
        """Part-stripe + anomaly filtered cross-view cosine. Returns (gid, score) or None."""
        if detection.part_embeds is None and detection.anomaly_embeds is None:
            return None

        best_gid   = None
        best_score = -1.0

        for gid, entry in gallery._gallery.items():
            if gid in excluded_ids:
                continue
            if entry["class_id"] != detection.class_id:
                continue
            if entry["track_status"] == "expired":
                continue

            max_part_score = -1.0
            
            # Check part embeddings (stripes)
            if detection.part_embeds is not None:
                for gallery_view, gallery_parts in entry["part_embeds"].items():
                    if gallery_parts is None:
                        continue
                    if gallery_view == detection.view_label:
                        continue  # already handled by same-view step
                    if not are_views_compatible(detection.view_label, gallery_view):
                        continue

                    score, is_valid = compute_cross_view_cosine(
                        detection.part_embeds,
                        gallery_parts,
                        detection.view_label,
                        gallery_view,
                    )
                    if is_valid and score > max_part_score:
                        max_part_score = score

            # Check anomaly embeddings (C4)
            anomaly_score = -1.0
            if detection.anomaly_embeds is not None:
                for gallery_view, gallery_anomalies in entry.get("anomaly_embeds", {}).items():
                    if gallery_anomalies is None:
                        continue
                    # Anomalies (like dents or custom stickers) don't depend strictly on view compatibility,
                    # but if they are visible in both views, they should match highly.
                    sim = _cosine_sim(detection.anomaly_embeds.flatten(), gallery_anomalies.flatten())
                    if sim > anomaly_score:
                        anomaly_score = sim

            # Fuse scores
            final_score = -1.0
            if max_part_score > -1.0 and anomaly_score > -1.0:
                final_score = 0.7 * max_part_score + 0.3 * anomaly_score
            elif max_part_score > -1.0:
                final_score = max_part_score
            elif anomaly_score > -1.0:
                final_score = anomaly_score
                
            if final_score > -1.0:
                if spatial_boosts and gid in spatial_boosts:
                    final_score += spatial_boosts[gid] * 0.05

                if final_score > best_score:
                    best_score = final_score
                    best_gid   = gid

        return (best_gid, best_score) if best_gid is not None else None

    def _match_by_attributes(
        self,
        detection:     DetectionInput,
        gallery,
        excluded_ids:  set[int],
        preferred_gid: int | None = None,
    ) -> MatchResult | None:
        """Semantic attribute fallback. Returns MatchResult or None."""
        from src.feature_extraction.attribute_extractor import attribute_similarity

        if detection.attributes is None:
            return None

        # Check preferred candidate first (from uncertain Step 2 result)
        candidates = list(gallery._gallery.items())
        if preferred_gid is not None:
            candidates = [(preferred_gid, gallery._gallery[preferred_gid])] + [
                c for c in candidates if c[0] != preferred_gid
            ]

        for gid, entry in candidates:
            if gid in excluded_ids:
                continue
            if entry["class_id"] != detection.class_id:
                continue
            if entry["track_status"] == "expired":
                continue

            gallery_attrs = entry.get("attributes")
            if gallery_attrs is None:
                continue

            score, is_valid = attribute_similarity(
                detection.attributes,
                gallery_attrs,
                self._attribute_weights,
            )

            if is_valid and score >= self._attribute_accept_thresh:
                logger.debug(
                    "Match [4 attribute] gid=%d score=%.3f "
                    "(color=%s class=%s plate=%s)",
                    gid, score,
                    detection.attributes.color,
                    detection.attributes.vehicle_class,
                    detection.attributes.plate_chars,
                )
                return MatchResult(
                    global_id        = gid,
                    match_type       = "attribute",
                    match_confidence = score,
                )

        return None

    def _compute_best_rejected_similarity(
        self,
        detection:    DetectionInput,
        gallery,
        excluded_ids: set[int],
    ) -> float:
        """Compute the highest cosine similarity that was still rejected (for review flagging)."""
        query_vec = detection.embedding.reshape(1, -1)
        best_sim = 0.0
        for gid, entry in gallery._gallery.items():
            if gid in excluded_ids:
                continue
            best_mean = gallery._best_mean_embed(entry)
            if best_mean is None:
                continue
            sim = _cosine_sim(query_vec.flatten(), best_mean.flatten())
            if sim > best_sim:
                best_sim = sim
        return best_sim
