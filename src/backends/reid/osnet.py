"""
src/backends/reid/osnet_backend.py
───────────────────────────────────
Thin adapter wrapping the existing AppearanceEmbedder as a BaseReIDBackend.
This backend preserves full backward-compatibility with v1.0.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from src.backends.reid.base import BaseReIDBackend

logger = logging.getLogger(__name__)


class OSNetBackend(BaseReIDBackend):
    """
    Adapter: wraps the existing AppearanceEmbedder (OSNet) as a
    BaseReIDBackend subclass. No logic is duplicated — all work
    is delegated to the original embedder implementation.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        emb_cfg = config.get("embedding", {})
        self._weights    = emb_cfg.get("weights",    "models/reid/osnet_veri776.pth")
        self._backbone   = emb_cfg.get("backbone",   "osnet_x1_0")
        self._input_size = tuple(emb_cfg.get("input_size", [128, 256]))  # (W, H)
        self._batch_size = emb_cfg.get("batch_size", 32)
        self._device     = config.get("pipeline", {}).get("device", "cpu")
        self._embedder   = None

    # ──────────────────────────────────────────────────────────────────────────
    def load(self) -> None:
        from src.feature_extraction.embedder_dispatcher import AppearanceEmbedder  # lazy import
        self._embedder = AppearanceEmbedder(
            weights    = self._weights,
            backbone   = self._backbone,
            input_size = self._input_size,
            device     = self._device,
            batch_size = self._batch_size,
        )
        logger.info("OSNetBackend loaded — backbone=%s", self._backbone)

        # TensorRT Acceleration
        if self._config.get("strategy", {}).get("use_tensorrt", False) and "cuda" in self._device:
            input_shape = (self._batch_size, 3, self._input_size[1], self._input_size[0])  # (B, C, H, W)
            self._embedder.model = self.compile_tensorrt(
                self._embedder.model,
                input_shape=input_shape,
                model_name=f"osnet_{self._backbone}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    def extract(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        cam_label: int = 0,
    ) -> np.ndarray:
        """Extract single embedding from a frame crop."""
        if self._embedder is None:
            self.load()
        crop = AppearanceEmbedder.crop_from_frame(frame, bbox)
        result = self._embedder.extract([crop])
        if result.size == 0:
            return np.zeros(self.embed_dim, dtype=np.float32)
        return result[0]

    # ──────────────────────────────────────────────────────────────────────────
    def extract_batch(
        self,
        frame: np.ndarray,
        bboxes: list[np.ndarray],
        cam_labels: list[int] | None = None,
    ) -> np.ndarray:
        """Batch extraction — delegate to embedder.extract with all crops."""
        if self._embedder is None:
            self.load()
        if not bboxes:
            return np.empty((0, self.embed_dim), dtype=np.float32)

        from src.feature_extraction.embedder_dispatcher import AppearanceEmbedder
        crops = [AppearanceEmbedder.crop_from_frame(frame, bb) for bb in bboxes]
        result = self._embedder.extract(crops)
        if result.size == 0:
            return np.zeros((len(bboxes), self.embed_dim), dtype=np.float32)
        return result

    # ──────────────────────────────────────────────────────────────────────────
    @property
    def embed_dim(self) -> int:
        return 512
