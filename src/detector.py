"""
src/detector.py
───────────────
Stage 1 — Vehicle Detection
Wraps YOLOv8/v11 via the ultralytics API. Returns raw bounding boxes,
confidence scores, and class IDs for every vehicle in a frame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

VEHICLE_CLASSES: dict[int, str] = {
    0: "forklift",
    1: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class VehicleDetector:
    """
    YOLO-based vehicle detector.

    Parameters
    ----------
    weights : str | Path
        Path to the YOLO model weights (.pt).  If the fine-tuned checkpoint
        does not exist the ultralytics auto-download fallback is used.
    conf    : float   Confidence threshold (0–1).
    iou     : float   NMS IoU threshold (0–1).
    imgsz   : int     Input image size (square).
    device  : str     'cuda:0' or 'cpu'.
    half    : bool    FP16 inference (GPU only).
    """

    def __init__(
        self,
        weights: str | Path = "models/yolo/yolo11m_vehicle.pt",
        fallback_weights: str = "yolo11m.pt",
        conf: float = 0.4,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "cuda:0",
        half: bool = True,
        use_onnx: bool = True,
    ) -> None:
        from ultralytics import YOLO  # lazy import -- avoids load-time overhead

        weights = Path(weights)
        if not weights.exists():
            logger.warning(
                "Fine-tuned weights not found at '%s'. "
                "Falling back to '%s'.",
                weights,
                fallback_weights,
            )
            weights = fallback_weights  # type: ignore[assignment]

        # Try to load or export ONNX for faster inference
        onnx_loaded = False
        if use_onnx and str(weights).endswith(".pt"):
            onnx_path = Path(str(weights).replace(".pt", ".onnx"))
            if onnx_path.exists():
                logger.info("Loading pre-exported ONNX model: %s", onnx_path)
                self.model = YOLO(str(onnx_path), task="detect")
                onnx_loaded = True
            else:
                try:
                    logger.info("Exporting YOLO model to ONNX for faster inference...")
                    pt_model = YOLO(str(weights), task="detect")
                    exported_path = pt_model.export(
                        format="onnx",
                        imgsz=imgsz,
                        half=False,
                        simplify=True,
                    )
                    if exported_path and Path(exported_path).exists():
                        logger.info("ONNX export successful: %s", exported_path)
                        self.model = YOLO(str(exported_path), task="detect")
                        onnx_loaded = True
                    else:
                        logger.warning("ONNX export did not produce a valid file. Using .pt model.")
                except Exception as e:
                    logger.warning("ONNX export failed (%s). Using .pt model.", e)

        if not onnx_loaded:
            self.model = YOLO(str(weights), task="detect")

        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.half = half if not onnx_loaded else False  # ONNX handles precision internally
        self._class_ids = list(VEHICLE_CLASSES.keys())
        self._is_onnx = onnx_loaded

        logger.info(
            "VehicleDetector initialised -- weights=%s device=%s onnx=%s",
            weights, device, onnx_loaded,
        )

    # ──────────────────────────────────────────────────────────────────────────
    def detect(
        self, frame: np.ndarray
    ) -> np.ndarray:
        """
        Run detection on a single BGR frame.

        Returns
        -------
        np.ndarray, shape (N, 6)
            Each row: [x1, y1, x2, y2, confidence, class_id]
            Empty array if no vehicles found.
        """
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=self._class_ids,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        xyxy   = boxes.xyxy.cpu().numpy()          # (N,4)
        confs  = boxes.conf.cpu().numpy()[:, None]  # (N,1)
        clss   = boxes.cls.cpu().numpy()[:, None]   # (N,1)

        return np.concatenate([xyxy, confs, clss], axis=1).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def class_name(class_id: int) -> str:
        return VEHICLE_CLASSES.get(int(class_id), "unknown")
