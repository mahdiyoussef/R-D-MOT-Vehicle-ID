#!/usr/bin/env python3
"""
main.py
-------
Entry point for the Long-Term Forklift Tracking and Re-Identification pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from database import MilvusReIDDatabase
from detector import ForkliftDetectorTracker
from reid import ReIDEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

CLASS_LABELS = {
    0: "forklift",
    1: "person",
}


def _load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _cfg_get(cfg: dict, keys: list[str], default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _parse_ids(value, default: list[int]) -> list[int]:
    if value is None:
        return default
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return [int(p) for p in parts]
    return default


class ForkliftTrackingPipeline:
    """End-to-end pipeline: Detect + Track + ReID + Milvus + Visualize."""

    def __init__(
        self,
        yolo_weights: str,
        reid_weights: str,
        reid_backbone: str,
        tracker_config: str,
        device: str | None,
        conf: float,
        iou: float,
        imgsz: int,
        class_ids: list[int],
        reid_class_ids: list[int],
        similarity_threshold: float,
        ema_alpha: float,
        metric: str,
        update_interval: int,
        min_box_area: int,
        frame_interval: int,
        top_k: int,
        retention_seconds: int,
        milvus_host: str,
        milvus_port: int,
        milvus_collection: str,
        milvus_index_type: str,
    ) -> None:
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.detector = ForkliftDetectorTracker(
            weights=yolo_weights,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            tracker_config=tracker_config,
            class_ids=class_ids,
        )
        self.embedder = ReIDEmbedder(weights=reid_weights, backbone=reid_backbone, device=self.device)
        self.db: MilvusReIDDatabase | None = None
        self.db_dim: int | None = None
        self.similarity_threshold = similarity_threshold
        self.ema_alpha = ema_alpha
        self.metric = metric.upper()
        self.update_interval = max(1, update_interval)
        self.min_box_area = min_box_area
        self.frame_interval = max(1, frame_interval)
        self.top_k = max(1, top_k)
        self.retention_seconds = max(0, retention_seconds)
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.milvus_collection = milvus_collection
        self.milvus_index_type = milvus_index_type
        self.class_ids = set(class_ids)
        self.reid_class_ids = set(reid_class_ids)

        self.track_to_pid: dict[int, int] = {}
        self.pid_status: dict[int, str] = {}
        self.last_extract_frame: dict[int, int] = {}
        self.frame_idx = 0

    def process_video(self, input_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

        logger.info(
            "Processing '%s'  %dx%d @ %.1f fps  (%d frames)",
            input_path.name, w, h, fps, total,
        )
        t0 = time.perf_counter()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, _ = self.process_frame(frame)
                writer.write(annotated)

                if self.frame_idx % 100 == 0:
                    elapsed = time.perf_counter() - t0
                    fps_actual = self.frame_idx / max(elapsed, 1e-6)
                    db_size = self.db.size() if self.db else 0
                    logger.info(
                        "frame %5d / %d  |  gallery: %d IDs  |  %.1f fps",
                        self.frame_idx, total, db_size, fps_actual,
                    )
        finally:
            cap.release()
            writer.release()

        elapsed = time.perf_counter() - t0
        logger.info(
            "Done. %d frames in %.1fs (%.1f fps). Output: %s",
            self.frame_idx, elapsed, self.frame_idx / max(elapsed, 1e-6),
            output_path,
        )

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        detections = self.detector.track(frame)
        current_ts = int(time.time())

        # Resolve new tracks
        new_tracks: list[dict] = []
        for det in detections:
            tid = det["track_id"]
            if tid not in self.track_to_pid:
                new_tracks.append(det)

        if new_tracks:
            crops = []
            crop_tids = []
            for det in new_tracks:
                tid = det["track_id"]
                if not self._should_extract(det, tid):
                    continue
                crop = ReIDEmbedder.crop_from_frame(frame, det["bbox"])
                if crop is None:
                    continue
                crops.append(crop)
                crop_tids.append(tid)

            embeddings = self.embedder.extract(crops) if crops else np.empty((0, 0), dtype=np.float32)
            if embeddings.shape[0] != len(crop_tids):
                logger.warning("Embedding count mismatch; skipping new tracks this frame.")
            else:
                for idx, tid in enumerate(crop_tids):
                    emb = embeddings[idx]
                    self._ensure_db(emb.shape[-1])

                    min_last_seen = None
                    if self.retention_seconds > 0:
                        min_last_seen = current_ts - self.retention_seconds

                    matches = self.db.search(emb, top_k=self.top_k, min_last_seen=min_last_seen)
                    pid, matched = self._best_match(matches)
                    if pid is None:
                        pid = self.db.add_new(emb, last_seen=current_ts)
                        self.pid_status[pid] = "new"
                    else:
                        self.pid_status[pid] = "reid"
                        self.db.update(pid, emb, last_seen=current_ts)

                    self.track_to_pid[tid] = pid
                    self.last_extract_frame[tid] = self.frame_idx

        # Update known tracks periodically
        if self.frame_idx % self.update_interval == 0:
            for det in detections:
                tid = det["track_id"]
                pid = self.track_to_pid.get(tid)
                if pid is None:
                    continue

                if not self._should_extract(det, tid):
                    continue
                crop = ReIDEmbedder.crop_from_frame(frame, det["bbox"])
                if crop is None:
                    continue

                emb = self.embedder.extract([crop])
                if emb.size == 0:
                    continue

                self._ensure_db(emb.shape[-1])
                self.db.update(pid, emb[0], last_seen=current_ts)
                if self.pid_status.get(pid) == "new":
                    self.pid_status[pid] = "tracked"
                self.last_extract_frame[tid] = self.frame_idx

        annotated = self._draw(frame, detections)

        track_records: list[dict] = []
        for det in detections:
            tid = det["track_id"]
            pid = self.track_to_pid.get(tid, -1)
            track_records.append(
                {
                    "frame_n": self.frame_idx,
                    "persistent_id": pid,
                    "track_id": tid,
                    "x1": int(det["bbox"][0]),
                    "y1": int(det["bbox"][1]),
                    "x2": int(det["bbox"][2]),
                    "y2": int(det["bbox"][3]),
                    "confidence": float(det["conf"]),
                    "class_id": int(det["cls"]),
                    "status": self.pid_status.get(pid, "tracked"),
                }
            )

        self.frame_idx += 1
        return annotated, track_records

    def _ensure_db(self, dimension: int) -> None:
        if self.db is None:
            self.db = MilvusReIDDatabase(
                dimension=dimension,
                host=self.milvus_host,
                port=self.milvus_port,
                collection_name=self.milvus_collection,
                metric_type=self.metric,
                index_type=self.milvus_index_type,
                threshold=self.similarity_threshold,
                ema_alpha=self.ema_alpha,
            )
            self.db_dim = dimension
            return

        if self.db_dim is not None and self.db_dim != dimension:
            raise ValueError(
                f"ReID embedding dimension mismatch: db={self.db_dim}, got={dimension}"
            )

    def _draw(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            tid = det["track_id"]
            pid = self.track_to_pid.get(tid, -1)
            status = self.pid_status.get(pid, "tracked")
            cls_id = int(det["cls"])
            cls_name = CLASS_LABELS.get(cls_id, f"cls{cls_id}")

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
            label = f"{cls_name} LTID:{pid} STID:{tid} {status}"
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _should_extract(self, det: dict, track_id: int) -> bool:
        cls_id = int(det["cls"])
        if cls_id not in self.reid_class_ids:
            return False
        area = self._bbox_area(det["bbox"])
        if area < self.min_box_area:
            return False
        last = self.last_extract_frame.get(track_id)
        if last is None:
            return True
        return (self.frame_idx - last) >= self.frame_interval

    @staticmethod
    def _bbox_area(bbox: np.ndarray | list) -> int:
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _best_match(self, matches: list[tuple[int, float]]) -> tuple[int | None, bool]:
        if not matches:
            return None, False

        if self.metric == "COSINE":
            best_pid, best_score = max(matches, key=lambda x: x[1])
            return (best_pid, True) if best_score >= self.similarity_threshold else (None, False)

        best_pid, best_score = min(matches, key=lambda x: x[1])
        return (best_pid, True) if best_score <= self.similarity_threshold else (None, False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Long-Term Forklift Tracking and Re-Identification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/pipeline_milvus.yaml")
    p.add_argument("--input", "-i", required=True, help="Path to input video.")
    p.add_argument(
        "--output",
        "-o",
        default="outputs/videos/forklift_reid.mp4",
        help="Path to output annotated video.",
    )
    p.add_argument("--yolo-weights", default=None)
    p.add_argument("--reid-weights", default=None)
    p.add_argument("--tracker-config", default=None)
    p.add_argument("--device", default=None, help="cuda:0 or cpu")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--class-ids", default=None, help="Comma-separated class IDs")
    p.add_argument("--reid-class-ids", default=None, help="Comma-separated ReID class IDs")
    p.add_argument("--reid-backbone", default=None)
    p.add_argument("--similarity-threshold", type=float, default=None)
    p.add_argument("--ema-alpha", type=float, default=None)
    p.add_argument("--metric", choices=["cosine", "l2"], default=None)
    p.add_argument("--update-interval", type=int, default=None)
    p.add_argument("--min-box-area", type=int, default=None)
    p.add_argument("--frame-interval", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--retention-seconds", type=int, default=None)
    p.add_argument("--milvus-host", default=None)
    p.add_argument("--milvus-port", type=int, default=None)
    p.add_argument("--milvus-collection", default=None)
    p.add_argument("--milvus-index-type", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    cfg = _load_config(args.config)

    yolo_weights = args.yolo_weights or _cfg_get(cfg, ["detector", "weights_path"], "outputs/results/weights/best.pt")
    reid_weights = args.reid_weights or _cfg_get(cfg, ["reid_extractor", "weights_path"], "models/reid/osnet_x0_25_msmt17.pt")
    reid_backbone = args.reid_backbone or _cfg_get(cfg, ["reid_extractor", "backbone"], "osnet_x1_0")
    tracker_config = args.tracker_config or _cfg_get(cfg, ["detector", "tracker_config"], "botsort.yaml")

    device = args.device or _cfg_get(cfg, ["pipeline", "device"], None)
    if device == "auto":
        device = None

    conf = args.conf if args.conf is not None else _cfg_get(cfg, ["detector", "confidence_threshold"], 0.4)
    iou = args.iou if args.iou is not None else _cfg_get(cfg, ["detector", "iou_threshold"], 0.45)
    input_res = _cfg_get(cfg, ["detector", "input_resolution"], [640, 640])
    imgsz = args.imgsz if args.imgsz is not None else int(input_res[0])

    class_ids = _parse_ids(args.class_ids, _cfg_get(cfg, ["detector", "classes"], [0, 1]))
    reid_class_ids = _parse_ids(args.reid_class_ids, _cfg_get(cfg, ["reid_extractor", "reid_classes"], class_ids))

    similarity_threshold = (
        args.similarity_threshold
        if args.similarity_threshold is not None
        else _cfg_get(cfg, ["association_logic", "cosine_similarity_threshold"], 0.85)
    )
    top_k = args.top_k if args.top_k is not None else _cfg_get(cfg, ["association_logic", "top_k_search"], 5)
    retention_seconds = (
        args.retention_seconds
        if args.retention_seconds is not None
        else _cfg_get(cfg, ["association_logic", "max_memory_retention_seconds"], 0)
    )

    min_box_area = (
        args.min_box_area
        if args.min_box_area is not None
        else _cfg_get(cfg, ["reid_extractor", "extraction_trigger", "min_box_area"], 0)
    )
    frame_interval = (
        args.frame_interval
        if args.frame_interval is not None
        else _cfg_get(cfg, ["reid_extractor", "extraction_trigger", "frame_interval"], 5)
    )
    update_interval = args.update_interval if args.update_interval is not None else frame_interval

    ema_alpha = args.ema_alpha if args.ema_alpha is not None else 0.9
    metric = args.metric or _cfg_get(cfg, ["vector_database", "metric_type"], "COSINE")

    milvus_host = args.milvus_host or _cfg_get(cfg, ["vector_database", "host"], "localhost")
    milvus_port = args.milvus_port or _cfg_get(cfg, ["vector_database", "port"], 19530)
    milvus_collection = args.milvus_collection or _cfg_get(cfg, ["vector_database", "collection_name"], "vehicle_embeddings")
    milvus_index_type = args.milvus_index_type or _cfg_get(cfg, ["vector_database", "index_type"], "HNSW")

    pipeline = ForkliftTrackingPipeline(
        yolo_weights=yolo_weights,
        reid_weights=reid_weights,
        reid_backbone=reid_backbone,
        tracker_config=tracker_config,
        device=device,
        conf=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        class_ids=class_ids,
        reid_class_ids=reid_class_ids,
        similarity_threshold=float(similarity_threshold),
        ema_alpha=float(ema_alpha),
        metric=str(metric),
        update_interval=int(update_interval),
        min_box_area=int(min_box_area),
        frame_interval=int(frame_interval),
        top_k=int(top_k),
        retention_seconds=int(retention_seconds),
        milvus_host=str(milvus_host),
        milvus_port=int(milvus_port),
        milvus_collection=str(milvus_collection),
        milvus_index_type=str(milvus_index_type),
    )

    pipeline.process_video(input_path, Path(args.output))


if __name__ == "__main__":
    main()
