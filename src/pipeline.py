"""
src/pipeline.py
───────────────
Main Pipeline Orchestrator  (v2.0 — strategy-pattern backends)
Connects all 6 stages into a single inference loop:
  Detection → Tracking → Embedding → Gallery → Hungarian Match → Visualise

v2.0 changes (non-breaking):
  - Re-ID, Tracker, and Gallery backends are now selected via config['strategy'].
  - Existing behaviour is preserved when strategy is not present in the config
    (defaults to v1.0 OSNet + legacy tracker + numpy gallery).
  - The frame-processing loop is UNCHANGED.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

from src.detector   import VehicleDetector
from src.visualizer import Visualizer
from src.tracker    import VehicleTracker
from src.feature_extraction.embedder_dispatcher import AppearanceEmbedder
from src.gallery    import PersistentGallery
from src.matching.hungarian_matcher import HungarianMatcher

logger = logging.getLogger(__name__)


def _load_config(config_path: str | Path = "configs/pipeline.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class VehicleReIDPipeline:
    """
    End-to-end Vehicle Persistent ReID pipeline.

    Parameters
    ----------
    config_path : str | Path
        Path to configs/pipeline.yaml.
    gallery_snapshot : str | Path | None
        Optional saved gallery to resume from.
    """

    def __init__(
        self,
        config_path:      str | Path = "configs/pipeline.yaml",
        gallery_snapshot: Optional[str | Path] = None,
    ) -> None:
        cfg = _load_config(config_path)

        det_cfg   = cfg["detection"]
        trk_cfg   = cfg["tracking"]
        emb_cfg   = cfg["embedding"]
        gal_cfg   = cfg["gallery"]
        match_cfg = cfg["matching"]
        vis_cfg   = cfg["visualization"]
        out_cfg   = cfg["output"]

        device    = cfg["pipeline"]["device"]
        half      = cfg["pipeline"]["half_precision"]

        # ── Stage 1: Detection (UNCHANGED) ────────────────────────────────────
        self.detector = VehicleDetector(
            weights          = det_cfg["weights"],
            fallback_weights = det_cfg["fallback_weights"],
            conf             = det_cfg["confidence_threshold"],
            iou              = det_cfg["iou_threshold"],
            imgsz            = det_cfg["imgsz"],
            device           = device,
            half             = half,
        )
        self.reid_class_ids = det_cfg.get("reid_class_ids", [0])

        # ── Stage 3: Re-ID backend (v2.0 — strategy selected) ─────────────────
        # Falls back to OSNet (v1.0) when 'strategy' key is absent from config.
        if "strategy" in cfg:
            from src.backends.factory import (
                build_reid_backend,
                build_tracker_backend,
                build_gallery_index,
                build_matcher,
            )
            self._reid_backend    = build_reid_backend(cfg)
            self._reid_backend.load()
            self._tracker_backend = build_tracker_backend(cfg)
            self._use_v2_backends = True
            # Check if YOLO native tracker is selected (single-pass mode)
            self._use_yolo_native = (
                cfg.get("strategy", {}).get("tracker_backend") == "yolo_native"
            )
        else:
            self._reid_backend    = None
            self._tracker_backend = None
            self._use_v2_backends = False
            self._use_yolo_native = False

        # ── Stage 2 & 3 (legacy fallback): Tracker and Embedder ─────────────────────
        if not self._use_v2_backends:
            self.tracker = VehicleTracker(
                reid_weights  = trk_cfg["reid_weights"],
                device        = device,
                half          = trk_cfg["half"],
                max_age       = trk_cfg["max_age"],
                min_hits      = trk_cfg["min_hits"],
                iou_threshold = trk_cfg["iou_threshold"],
            )
            self.embedder = AppearanceEmbedder(
                weights    = emb_cfg["weights"],
                backbone   = emb_cfg["backbone"],
                input_size = tuple(emb_cfg["input_size"]),
                device     = device,
                batch_size = emb_cfg["batch_size"],
            )
        else:
            self.tracker = None
            self.embedder = None

        # ── Stage 4: Gallery (always the same PersistentGallery) ──────────────
        self.gallery = PersistentGallery(
            threshold     = gal_cfg["similarity_threshold"],
            max_embeddings= gal_cfg["max_embeddings_per_id"],
            timeout       = gal_cfg["gallery_timeout_frames"],
        )

        # ── Stage 4: Gallery Index backend (v2.0/v3.0 strategy) ─────────────────
        if self._use_v2_backends:
            embed_dim = self._reid_backend.embed_dim
            self._gallery_index = build_gallery_index(cfg, embed_dim, self.gallery)
        else:
            self._gallery_index = None

        self.matcher = HungarianMatcher(similarity_threshold=match_cfg["threshold"])

        if match_cfg.get("reranking_enabled", False):
            from src.matching.reranker import KReciprocalReRanker
            self.matcher.reranker = KReciprocalReRanker(
                k1=match_cfg.get("reranking_k1", 20),
                k2=match_cfg.get("reranking_k2", 6),
                lambda_value=match_cfg.get("reranking_lambda", 0.3)
            )
            logger.info("k-Reciprocal Re-Ranking is ACTIVE.")

        # ── Stage 5 (v3.0): GNN Context-Aware Matcher ─────────────────────────
        self._gnn_matcher = None
        if self._use_v2_backends:
            gnn = build_matcher(cfg, self._reid_backend.embed_dim)
            if gnn is not None:
                self._gnn_matcher = gnn
                logger.info("GNN context-aware matcher is ACTIVE.")

        # ── v3.0: Temporal Tracklet Aggregator ────────────────────────────────
        self._temporal_aggregator = None
        if self._use_v2_backends and cfg.get("temporal_aggregator", {}).get("enabled", False):
            from src.feature_extraction.temporal_aggregator import TemporalTrackletAggregator
            self._temporal_aggregator = TemporalTrackletAggregator(
                cfg, self._reid_backend.embed_dim
            )
            logger.info("Temporal tracklet aggregator is ACTIVE.")

        self.visualizer = Visualizer(
            trajectory_length = vis_cfg["trajectory_length"],
            show_confidence   = vis_cfg["show_confidence"],
            show_trajectory   = vis_cfg["show_trajectory"],
        )

        self._prune_interval = out_cfg["gallery_prune_interval"]
        self._log_format     = out_cfg.get("log_format", "jsonl")
        self._v2_cfg         = cfg  # store full config for get_stats()

        # Improvement #2: Feature update interval + quality gate
        self._feature_update_interval = 5  # only re-embed known tracks every N frames
        self._min_crop_area = 1024         # skip crops smaller than 32x32
        self._min_embed_conf = 0.5         # skip low-confidence crops
        if self._use_v2_backends:
            strategy = cfg.get("strategy", {})
            reid_key = strategy.get("reid_backend", "osnet")
            backend_cfg = cfg.get(reid_key, {})
            self._feature_update_interval = backend_cfg.get("feature_update_interval", 5)

        # Improvement #4: Set min_absence to 0 so gallery can match
        # vehicles that were lost even briefly. Duplicate prevention is
        # handled entirely by the excluded_pids mechanism.
        self.gallery.min_absence_frames = 0

        if gallery_snapshot:
            self.gallery.load(gallery_snapshot)

        # Runtime state
        self._persistent_id_map: Dict[int, int]  = {}  # track_id → pid
        self._status_map:        Dict[int, str]   = {}  # pid      → status
        self._frame_n:           int              = 0
        # Grace period: keep stale mappings for N frames before deleting
        # so brief detection flickers don't lose the track→pid link.
        self._stale_track_grace: Dict[int, int]   = {}  # tid → frame_last_seen
        self._stale_grace_frames: int = trk_cfg["max_age"]  # match tracker max_age
        self._view_map_cache: Dict[int, str]      = {}  # pid → view_label
        self._last_bboxes: Dict[int, np.ndarray]  = {}  # tid → bbox [x1, y1, x2, y2]

        strategy = cfg.get("strategy", {})
        logger.info(
            "VehicleReIDPipeline ready (v2.0). "
            "reid=%s  tracker=%s  gallery=%s",
            strategy.get("reid_backend",    "[legacy]"),
            strategy.get("tracker_backend", "[legacy]"),
            strategy.get("gallery_backend", "[legacy]"),
        )

        # ── v4.0 — Cross-View Pipeline Modules ────────────────────────────────
        self._use_v4 = "gallery_v4" in cfg and cfg.get("gallery_v4", {}) != {}

        self._cross_view_gallery  = None
        self._multi_stage_matcher = None
        self._tracklet_memory     = None
        self._view_classifier     = None
        self._attribute_extractor = None

        if self._use_v4:
            from src.memory.cross_view_gallery         import CrossViewGallery
            from src.matching.cascade_matcher           import CascadeMatcher
            from src.memory.kalman_tracklet             import KalmanTrackletMemory
            from src.feature_extraction.view_classifier    import ViewClassifier
            from src.feature_extraction.attribute_extractor import AttributeExtractor

            embed_dim = self._reid_backend.embed_dim if self._reid_backend else 512
            backbone_dim = self._reid_backend.backbone_dim if self._reid_backend else embed_dim // 2

            self._cross_view_gallery  = CrossViewGallery(cfg, embed_dim=embed_dim)
            self._multi_stage_matcher = CascadeMatcher(cfg)
            self._tracklet_memory     = KalmanTrackletMemory(cfg)
            self._view_classifier     = ViewClassifier(
                cfg,
                backbone_dim=backbone_dim,
                device=torch.device(device),
            )
            self._attribute_extractor = AttributeExtractor(
                cfg,
                backbone_dim=backbone_dim,
            )
            from src.feature_extraction.view_synthesizer import GenerativeViewSynthesizer
            self._view_synthesizer = GenerativeViewSynthesizer(cfg, self._reid_backend)
            
            logger.info("Cross-view pipeline modules activated.")

    # ──────────────────────────────────────────────────────────────────────────
    def process_video(
        self,
        input_path:       str | Path,
        output_video_path: str | Path,
        log_path:         Optional[str | Path] = None,
        snapshot_path:    Optional[str | Path] = None,
    ) -> None:
        """
        Run the full pipeline on a video file.

        Parameters
        ----------
        input_path        : Path to source video.
        output_video_path : Where to write the annotated output video.
        log_path          : Optional CSV/JSONL track log output path.
        snapshot_path     : Optional path to save the gallery at the end.
        """
        input_path        = Path(input_path)
        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        fps   = cap.get(cv2.CAP_PROP_FPS) or 30
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (W, H))

        log_handle, log_writer = self._init_logger(log_path)

        logger.info(
            "Processing '%s'  %dx%d @ %.1f fps  (%d frames)",
            input_path.name, W, H, fps, total,
        )
        t0 = time.perf_counter()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, track_records = self.process_frame(frame)
                writer.write(annotated)

                if log_writer and track_records:
                    for rec in track_records:
                        self._write_log_record(log_handle, log_writer, rec)

                if self._frame_n % 100 == 0:
                    elapsed = time.perf_counter() - t0
                    fps_actual = self._frame_n / max(elapsed, 1e-6)
                    logger.info(
                        "frame %5d / %d  |  gallery: %d IDs  |  %.1f fps",
                        self._frame_n, total, len(self.gallery), fps_actual,
                    )
        finally:
            cap.release()
            writer.release()
            if log_handle:
                log_handle.close()

        if snapshot_path:
            self.gallery.save(snapshot_path)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Done. %d frames in %.1fs (%.1f fps). Output: %s",
            self._frame_n, elapsed, self._frame_n / elapsed,
            output_video_path,
        )

    # ──────────────────────────────────────────────────────────────────────────
    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, list[dict]]:
        """
        Process a single BGR frame.

        Returns
        -------
        annotated_frame : np.ndarray
        track_records   : list of dicts (one per active track)
        """
        fn = self._frame_n

        # ── Stage 1 + 2: Detect & Track ────────────────────────────────────────
        if self._use_yolo_native:
            # Single-pass: YOLO .track() does detection + tracking together
            from src.backends.tracking.yolo_native import YOLONativeTracker
            tracks = self._tracker_backend.run_yolo_track(
                model=self.detector.model,
                frame=frame,
                conf=self.detector.conf,
                iou=self.detector.iou,
                imgsz=self.detector.imgsz,
                classes=self.detector._class_ids,
                device=self.detector.device,
                half=self.detector.half,
            )
        elif self._use_v2_backends and self._tracker_backend is not None:
            # v2 tracker backend: detect first, then run v2 tracker
            detections = self.detector.detect(frame)
            det_list = []
            for row in detections:
                det_list.append({
                    "bbox": row[:4].tolist(),
                    "conf": float(row[4]),
                    "cls":  int(row[5]),
                })
            tracker_outputs = self._tracker_backend.update(frame, det_list)
            # Convert TrackerOutput objects to numpy [x1,y1,x2,y2, tid, conf, cls, 0]
            if tracker_outputs:
                rows = []
                for to in tracker_outputs:
                    rows.append([
                        to.bbox[0], to.bbox[1], to.bbox[2], to.bbox[3],
                        to.tracker_id, to.confidence, to.class_id, 0.0,
                    ])
                tracks = np.array(rows, dtype=np.float32)
            else:
                tracks = np.empty((0, 8), dtype=np.float32)
        else:
            # Legacy two-step: detect then track via VehicleTracker
            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections, frame)

        # ── Split into new vs known tracks ────────────────────────────────────
        new_tracks:   list = []
        known_tracks: list = []

        for t in tracks:
            tid = int(t[4])
            if tid not in self._persistent_id_map:
                new_tracks.append(t)
            else:
                known_tracks.append(t)

        # ── Build active PIDs set (PIDs already assigned to live tracks) ──────
        # This prevents the gallery from assigning the same PID to two
        # different vehicles visible in the same frame.
        active_pids: set[int] = set()
        for t in known_tracks:
            tid = int(t[4])
            pid = self._persistent_id_map.get(tid)
            if pid is not None:
                active_pids.add(pid)

        # ── Clean up stale track mappings with grace period ───────────────────
        # Don't delete mappings immediately when a track disappears — the
        # tracker may re-acquire it within a few frames with the SAME track ID.
        active_tids = {int(t[4]) for t in tracks}
        newly_stale = [
            tid for tid in self._persistent_id_map
            if tid not in active_tids and tid not in self._stale_track_grace
        ]
        for tid in newly_stale:
            self._stale_track_grace[tid] = fn
            if self._use_v4 and self._tracklet_memory is not None:
                pid = self._persistent_id_map[tid]
                last_bbox = self._last_bboxes.get(tid)
                if last_bbox is not None:
                    vlabel = self._view_map_cache.get(pid, "ambiguous")
                    self._tracklet_memory.record_track_loss(pid, last_bbox, fn, view_label=vlabel)

        # Revive tracks that reappeared
        revived = [tid for tid in self._stale_track_grace if tid in active_tids]
        for tid in revived:
            del self._stale_track_grace[tid]
            if self._use_v4 and self._tracklet_memory is not None:
                pid = self._persistent_id_map[tid]
                self._tracklet_memory.mark_track_recovered(pid)

        # Delete mappings that have exceeded the grace period
        expired = [
            tid for tid, last_frame in self._stale_track_grace.items()
            if fn - last_frame > self._stale_grace_frames
        ]
        for tid in expired:
            self._persistent_id_map.pop(tid, None)
            self._last_bboxes.pop(tid, None)
            del self._stale_track_grace[tid]

        # Update last bboxes
        for t in tracks:
            self._last_bboxes[int(t[4])] = t[:4].copy()

        if self._use_v4:
            if self._tracklet_memory:
                self._tracklet_memory.advance_all_predictions(fn)
            if self._cross_view_gallery:
                self._cross_view_gallery.advance_lifecycle(fn)

        # ── Stage 3 + 4 + 5: Embed -> Gallery -> Match  (new tracks only) ────
        if new_tracks:
            # ONLY EMBED PERSISTENT RE-ID CLASSES
            reid_target_tracks = [t for t in new_tracks if int(t[6]) in self.reid_class_ids]
            other_tracks       = [t for t in new_tracks if int(t[6]) not in self.reid_class_ids]

            # Bypass gallery for non-reid tracks (like persons in forklift mode)
            for t in other_tracks:
                tid = int(t[4])
                # Assign a generic PID so they render correctly, but DO NOT store embeddings
                pid = self.gallery._next_id
                self.gallery._next_id += 1
                self._persistent_id_map[tid] = pid
                self._status_map[pid] = "tracked"

            if reid_target_tracks:
                if self._use_v4:
                    from src.matching.cascade_matcher import DetectionInput, ActiveTrack
                    bboxes = [t[:4] for t in reid_target_tracks]
                    embeddings = self._reid_backend.extract_batch(frame, bboxes)
                    
                    active_tracks_v4 = []
                    for t in known_tracks:
                        k_tid = int(t[4])
                        k_pid = self._persistent_id_map[k_tid]
                        vlabel = self._view_map_cache.get(k_pid, "ambiguous")
                        active_tracks_v4.append(ActiveTrack(k_tid, k_pid, t[:4], vlabel))

                    batch_assigned_pids: set[int] = set()
                    for i, t in enumerate(reid_target_tracks):
                        tid = int(t[4])
                        cls = int(t[6])
                        emb = embeddings[i]
                        x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                        h_f, w_f = frame.shape[:2]
                        crop_bgr = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
                        
                        view_label = "ambiguous"
                        if self._view_classifier and crop_bgr.size > 0:
                            view_label, _ = self._view_classifier.classify_from_crop(crop_bgr)
                            
                        attributes = None
                        if self._attribute_extractor and crop_bgr.size > 0:
                            cls_name = self.detector.class_name(cls)
                            attributes = self._attribute_extractor.extract(crop_bgr, cls_name)

                        det_input = DetectionInput(
                            bbox=t[:4], embedding=emb, view_label=view_label,
                            class_id=cls, attributes=attributes, confidence=float(t[5])
                        )
                        
                        all_excluded = active_pids | batch_assigned_pids
                        spatial_boosts = None
                        if self._tracklet_memory:
                            nearby = self._tracklet_memory.get_nearby_lost_tracks(t[:4], cls)
                            spatial_boosts = {g: b for g, b in nearby}

                        match_res = self._multi_stage_matcher.assign_identity(
                            det_input, active_tracks_v4, self._cross_view_gallery,
                            all_excluded, fn, spatial_boosts
                        )
                        
                        pid = match_res.global_id
                        self._persistent_id_map[tid] = pid
                        self._status_map[pid] = "new" if match_res.is_new else "recovered"
                        self._view_map_cache[pid] = view_label
                            
                        self._cross_view_gallery.insert_embedding(
                            global_id=pid, embedding=emb, view_label=view_label,
                            class_id=cls, frame_index=fn, attributes=attributes
                        )
                        
                        # C3: Generative View Synthesis (hallucinate missing views for new IDs)
                        if match_res.is_new and getattr(self, "_view_synthesizer", None) is not None:
                            if self._view_synthesizer.enabled and view_label != "ambiguous":
                                synthetic_views = self._view_synthesizer.synthesize_missing_views(crop_bgr, view_label)
                                for syn_view, syn_emb in synthetic_views.items():
                                    if syn_emb is not None:
                                        self._cross_view_gallery.insert_embedding(
                                            global_id=pid, embedding=syn_emb, view_label=syn_view,
                                            class_id=cls, frame_index=fn, attributes=attributes
                                        )

                        if self._tracklet_memory:
                            self._tracklet_memory.mark_track_recovered(pid)
                            
                        active_pids.add(pid)
                        batch_assigned_pids.add(pid)
                else:
                    crops = [
                        AppearanceEmbedder.crop_from_frame(frame, t[:4])
                        for t in reid_target_tracks
                    ]
                    # Improvement #5 (P0): Use v2 backend when available
                    if self._use_v2_backends and self._reid_backend is not None:
                        bboxes = [t[:4] for t in reid_target_tracks]
                        embeddings = self._reid_backend.extract_batch(frame, bboxes)
                    else:
                        embeddings = self.embedder.extract(crops)

                    # Get gallery candidates for matching
                    gallery_result = self.gallery.get_representative_embeddings()

                    if gallery_result is not None:
                        gallery_embs, gallery_ids = gallery_result
                        # v3.0: Use GNN context-aware matcher if available
                        if getattr(self, "_gnn_matcher", None) is not None:
                            # Build spatial info for GNN edge encoding
                            q_bboxes = np.array([t[:4] for t in reid_target_tracks], dtype=np.float32)
                            matches = self._gnn_matcher.match(
                                embeddings, gallery_embs, gallery_ids,
                                query_bboxes=q_bboxes,
                            )
                        else:
                            # v2.0/v1.0: Standard Hungarian
                            matches = self.matcher.match(embeddings, gallery_embs, gallery_ids)
                    else:
                        matches = []

                    # Track PIDs assigned in THIS batch to prevent intra-batch duplicates
                    batch_assigned_pids: set[int] = set()

                    for i, t in enumerate(reid_target_tracks):
                        tid   = int(t[4])
                        cls   = int(t[6])
                        emb   = embeddings[i]

                        # Compute box dimensions for size verification
                        x1, y1 = int(t[0]), int(t[1])
                        x2, y2 = int(t[2]), int(t[3])
                        box_w, box_h = x2 - x1, y2 - y1
                        box_size = (box_w, box_h)

                        # Extract BGR crop for color verification
                        h_f, w_f = frame.shape[:2]
                        crop_bgr = frame[
                            max(0, y1):min(h_f, y2),
                            max(0, x1):min(w_f, x2),
                        ]

                        # Check if Hungarian gave this query a match
                        hit = [(qi, pid) for qi, pid in matches if qi == i]

                        if hit:
                            candidate_pid = hit[0][1]
                            # Only reject if already assigned to another NEW track
                            # in this batch. Active PIDs from known_tracks are fine
                            # to reclaim — it means the tracker gave us a new tid
                            # for a vehicle we already know.
                            if candidate_pid in batch_assigned_pids:
                                hit = []  # fall through to register_or_recover
                            else:
                                # Improvement #3: Color histogram secondary verification
                                color_ok = self.gallery.verify_color(candidate_pid, crop_bgr)
                                # Box-size secondary verification
                                size_ok = self.gallery.verify_box_size(candidate_pid, box_size)
                                if color_ok and size_ok:
                                    pid = candidate_pid
                                    self._persistent_id_map[tid] = pid
                                    self._status_map[pid]         = "recovered"
                                    self.gallery.update_known(pid, emb, fn, box_size=box_size)
                                    active_pids.add(pid)
                                    batch_assigned_pids.add(pid)
                                    continue  # skip the fallback block
                                else:
                                    hit = []  # fall through

                        # No valid match — register or recover with exclusion guard
                        all_excluded = active_pids | batch_assigned_pids
                        pid, status = self.gallery.register_or_recover(
                            emb, fn, cls, excluded_pids=all_excluded,
                            box_size=box_size,
                        )
                        self._persistent_id_map[tid] = pid
                        self._status_map[pid]         = status
                        active_pids.add(pid)
                        batch_assigned_pids.add(pid)
                        # Store color signature for new or recovered gallery entries
                        self.gallery.set_color_signature(pid, crop_bgr)

        # ── Update gallery for known tracks ──────────────────────────────────────
        for t in known_tracks:
            tid = int(t[4])
            cls = int(t[6])
            pid = self._persistent_id_map[tid]
            if self._status_map.get(pid) in ("new", "recovered"):
                self._status_map[pid] = "tracked"

            # Improvement #2: Only re-embed every N frames with quality gate
            if cls in self.reid_class_ids and fn % self._feature_update_interval == 0:
                # Quality gate: skip tiny or low-confidence crops
                crop_w = int(t[2]) - int(t[0])
                crop_h = int(t[3]) - int(t[1])
                conf   = float(t[5])
                known_box_size = (crop_w, crop_h)
                if crop_w * crop_h > self._min_crop_area and conf > self._min_embed_conf:
                    if self._use_v4:
                        emb = self._reid_backend.extract(frame, t[:4])
                        if emb is not None and emb.size > 0:
                            if self._temporal_aggregator is not None:
                                emb = self._temporal_aggregator.update_and_aggregate(tid, emb)
                            vlabel = self._view_map_cache.get(pid, "ambiguous")
                            x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                            h_f, w_f = frame.shape[:2]
                            crop_bgr = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
                            if self._view_classifier and crop_bgr.size > 0:
                                vlabel, _ = self._view_classifier.classify_from_crop(crop_bgr)
                                self._view_map_cache[pid] = vlabel
                            attributes = None
                            if self._attribute_extractor and crop_bgr.size > 0:
                                cls_name = self.detector.class_name(cls)
                                attributes = self._attribute_extractor.extract(crop_bgr, cls_name)
                            
                            self._cross_view_gallery.insert_embedding(
                                global_id=pid, embedding=emb, view_label=vlabel,
                                class_id=cls, frame_index=fn, attributes=attributes
                            )
                    else:
                        # Improvement #5: Use v2 backend for known track updates too
                        if self._use_v2_backends and self._reid_backend is not None:
                            emb = self._reid_backend.extract(frame, t[:4])
                            if emb is not None and emb.size > 0:
                                # v3.0: Temporal aggregation before gallery update
                                if self._temporal_aggregator is not None:
                                    emb = self._temporal_aggregator.update_and_aggregate(tid, emb)
                                self.gallery.update_known(pid, emb, fn, box_size=known_box_size)
                        else:
                            crop = AppearanceEmbedder.crop_from_frame(frame, t[:4])
                            emb  = self.embedder.extract([crop])
                            if emb.size > 0:
                                self.gallery.update_known(pid, emb[0], fn, box_size=known_box_size)

        # ── v4.0: View Classification Cache ───────────────────────────────────
        view_map: dict[int, str] = {}
        if tracks is not None and len(tracks) > 0:
            for t in tracks:
                tid = int(t[4])
                pid = self._persistent_id_map.get(tid, -1)
                if pid == -1:
                    continue
                vlabel = self._view_map_cache.get(pid)
                if vlabel is None:
                    if self._use_v4 and self._view_classifier is not None:
                        x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                        h_f, w_f = frame.shape[:2]
                        crop_bgr = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
                        if crop_bgr.size > 0:
                            vlabel, _ = self._view_classifier.classify_from_crop(crop_bgr)
                            self._view_map_cache[pid] = vlabel
                        else:
                            vlabel = "ambiguous"
                    else:
                        vlabel = "ambiguous"
                view_map[pid] = vlabel

        # ── Stage 6: Visualise ────────────────────────────────────────────────
        annotated = self.visualizer.draw(
            frame, tracks, self._persistent_id_map, self._status_map,
            view_label_map=view_map,
        )

        # ── Build log records ─────────────────────────────────────────────────
        track_records: list[dict] = []
        for t in tracks:
            tid  = int(t[4])
            pid  = self._persistent_id_map.get(tid, -1)
            rec  = {
                "frame_n":       fn,
                "persistent_id": pid,
                "track_id":      tid,
                "x1": int(t[0]), "y1": int(t[1]),
                "x2": int(t[2]), "y2": int(t[3]),
                "class":      int(t[6]),
                "confidence": round(float(t[5]), 4),
                "status":     self._status_map.get(pid, "tracked"),
                "view_label": view_map.get(pid, "ambiguous"),
            }
            track_records.append(rec)

        # ── Periodic gallery prune ────────────────────────────────────────────
        if fn > 0 and fn % self._prune_interval == 0:
            if not self._use_v4:
                self.gallery.prune(fn)

        self._frame_n += 1
        return annotated, track_records

    # ──────────────────────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        """Return a dict of current pipeline statistics and backend names."""
        gallery_index_name = "numpy (legacy)"
        gallery_promoted   = False
        if self._gallery_index is not None:
            gallery_index_name = self._gallery_index.name
            gallery_promoted   = getattr(self._gallery_index, "_promoted", False)

        reid_name    = self._reid_backend.name    if self._reid_backend    else "AppearanceEmbedder (legacy)"
        tracker_name = self._tracker_backend.name if self._tracker_backend else "VehicleTracker (legacy)"
        matcher_name = "GNNContextMatcher" if getattr(self, "_gnn_matcher", None) is not None else "HungarianMatcher (legacy)"

        return {
            "frame_n":         self._frame_n,
            "gallery_size":    len(self.gallery),
            "gallery_backend": gallery_index_name,
            "gallery_promoted": gallery_promoted,
            "reid_backend":    reid_name,
            "tracker_backend": tracker_name,
            "matcher_backend": matcher_name,
            "active_tracks":   len(self._persistent_id_map),
        }

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Reset pipeline state between unrelated video clips."""
        self._persistent_id_map.clear()
        self._status_map.clear()
        self._stale_track_grace.clear()
        self._view_map_cache.clear()
        self._last_bboxes.clear()
        self._frame_n = 0
        if self.tracker is not None:
            self.tracker.reset()
        if getattr(self, "_temporal_aggregator", None) is not None:
            self._temporal_aggregator.reset()
        self.gallery.reset()
        if self._tracker_backend is not None:
            self._tracker_backend.reset()
        self.visualizer.reset_trajectories()

    # ──────────────────────────────────────────────────────────────────────────
    # Logging helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _init_logger(self, log_path):
        if log_path is None:
            return None, None

        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if self._log_format == "csv":
            fh = open(log_path, "w", newline="")
            fieldnames = [
                "frame_n", "persistent_id", "track_id",
                "x1", "y1", "x2", "y2", "class", "confidence", "status",
            ]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            return fh, writer
        else:  # jsonl
            fh = open(log_path, "w")
            return fh, "jsonl"

    @staticmethod
    def _write_log_record(fh, writer, record: dict) -> None:
        if writer == "jsonl":
            fh.write(json.dumps(record) + "\n")
        else:
            writer.writerow(record)
