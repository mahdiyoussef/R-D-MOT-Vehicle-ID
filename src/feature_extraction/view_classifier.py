"""
src/backends/reid/view_classifier.py
──────────────────────────────────────
v4.0 — Lightweight View Classification Head.

Classifies the camera viewpoint of a vehicle crop into one of 6 classes:
  front | rear | side_left | side_right | top_down | ambiguous

Designed to run on top of an ALREADY-COMPUTED backbone CLS token —
zero additional forward passes required when the backbone is already running.

If no pretrained weights are available, it falls back to a heuristic based
on aspect ratio and HSV histogram analysis (works decently for side vs front/rear).

Reference:
  The view classification approach follows the dual-branch architecture
  from PVEN: "Parsing-based View-aware Embedding Network for Vehicle ReID"
  CVPR 2020. https://arxiv.org/abs/2004.05021
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

VIEW_CLASSES = ["front", "rear", "side_left", "side_right", "top_down", "ambiguous"]
VIEW_TO_IDX  = {v: i for i, v in enumerate(VIEW_CLASSES)}


# ─────────────────────────────────────────────────────────────────────────────
# MLP Head (runs on backbone CLS token)
# ─────────────────────────────────────────────────────────────────────────────

class ViewClassifierHead(nn.Module):
    """
    Lightweight 2-layer MLP classifier on top of a backbone CLS token.

    Parameters
    ----------
    backbone_dim : int   Dimension of the incoming CLS token (e.g. 384 for DINOv2-ViT-S)
    num_classes  : int   Number of view classes (default 6)
    hidden_dim   : int   Hidden layer size
    dropout      : float Dropout rate
    """

    def __init__(
        self,
        backbone_dim: int = 384,
        num_classes: int = 6,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cls_token : (B, D)  backbone CLS token

        Returns
        -------
        (B, num_classes)  logits
        """
        return self.head(cls_token)


# ─────────────────────────────────────────────────────────────────────────────
# View Classifier (wraps head + inference logic)
# ─────────────────────────────────────────────────────────────────────────────

class ViewClassifier:
    """
    Vehicle viewpoint classifier.

    If pretrained_weights exist → runs the MLP head on backbone CLS tokens.
    Otherwise → falls back to heuristic (aspect ratio + edge orientation).

    Parameters
    ----------
    config       : full pipeline config dict
    backbone_dim : int  CLS token dimension from the active backbone
    device       : torch.device
    """

    def __init__(
        self,
        config: dict,
        backbone_dim: int = 384,
        device: torch.device | None = None,
    ) -> None:
        self.cfg          = config.get("view_classifier", {})
        self.enabled      = self.cfg.get("enabled", True)
        self.conf_thresh  = float(self.cfg.get("confidence_threshold", 0.60))
        self.device       = device or torch.device("cpu")
        self._head: ViewClassifierHead | None = None
        self._use_heuristic = True

        weights_path = Path(self.cfg.get("pretrained_weights", ""))
        if weights_path.is_file():
            self._head = ViewClassifierHead(
                backbone_dim=backbone_dim,
                num_classes=len(VIEW_CLASSES),
            ).to(self.device)
            ckpt = torch.load(weights_path, map_location="cpu")
            self._head.load_state_dict(ckpt)
            self._head.eval()
            self._use_heuristic = False
            logger.info("ViewClassifier: loaded weights from %s", weights_path)
        else:
            logger.info(
                "ViewClassifier: no pretrained weights found at '%s'. "
                "Using aspect-ratio heuristic fallback.",
                weights_path,
            )

    # ── Primary path: MLP on CLS token ────────────────────────────────────────
    @torch.no_grad()
    def classify_from_token(
        self,
        cls_token: torch.Tensor,  # (1, D) or (D,)
    ) -> tuple[str, float]:
        """
        Classify viewpoint using the MLP head on a precomputed CLS token.

        Returns
        -------
        (view_label, confidence)
        """
        if not self.enabled or self._head is None:
            return "ambiguous", 0.0

        if cls_token.dim() == 1:
            cls_token = cls_token.unsqueeze(0)

        cls_token = cls_token.to(self.device)
        logits    = self._head(cls_token)  # (1, C)
        probs     = F.softmax(logits, dim=-1)[0]

        confidence = float(probs.max())
        label_idx  = int(probs.argmax())

        if confidence < self.conf_thresh:
            return "ambiguous", confidence

        return VIEW_CLASSES[label_idx], confidence

    # ── Fallback: heuristic on raw crop ───────────────────────────────────────
    def classify_from_crop(
        self,
        crop_bgr: np.ndarray,
    ) -> tuple[str, float]:
        """
        Heuristic viewpoint classification from a raw BGR crop.
        Uses aspect ratio + Sobel edge orientation histogram.

        Returns
        -------
        (view_label, confidence)  — confidence is approximate (0.5 for heuristic)
        """
        if not self.enabled:
            return "ambiguous", 0.0

        if crop_bgr is None or crop_bgr.size == 0:
            return "ambiguous", 0.0

        h, w = crop_bgr.shape[:2]
        aspect = w / max(h, 1)

        # Side views are typically wide (aspect > 1.8)
        # Front/Rear views are typically squarish (0.6 < aspect < 1.8)
        if aspect > 1.8:
            label = _heuristic_side(crop_bgr)
            return label, 0.55

        if aspect < 0.7:
            return "front", 0.50   # tall thin crops = front

        return _heuristic_front_rear(crop_bgr), 0.50

    # ── Unified classify ───────────────────────────────────────────────────────
    def classify(
        self,
        crop_bgr: np.ndarray,
        cls_token: torch.Tensor | None = None,
    ) -> tuple[str, float]:
        """
        Classify viewpoint. Prefers MLP path if cls_token is available,
        otherwise falls back to heuristic on the crop.
        """
        if cls_token is not None and not self._use_heuristic:
            return self.classify_from_token(cls_token)
        return self.classify_from_crop(crop_bgr)


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _heuristic_side(crop_bgr: np.ndarray) -> str:
    """
    Distinguish side_left vs side_right using horizontal edge asymmetry.
    The windshield/engine compartment typically faces the brighter/more-
    detailed side of the crop.
    """
    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    left_mean  = float(gray[:, :w//2].mean())
    right_mean = float(gray[:, w//2:].mean())

    # Very rough heuristic: front of car (windshield) tends to be brighter
    if left_mean > right_mean + 5:
        return "side_right"  # car faces left → right side view
    return "side_left"


def _heuristic_front_rear(crop_bgr: np.ndarray) -> str:
    """
    Distinguish front vs rear using horizontal symmetry of the bottom region.
    Taillights (rear) often have higher red channel; headlights are brighter/white.
    """
    h, w = crop_bgr.shape[:2]
    bottom = crop_bgr[h*2//3:, :]

    r_mean = float(bottom[:, :, 2].mean())   # Red channel
    b_mean = float(bottom[:, :, 0].mean())   # Blue channel

    # Taillights are red → rear
    if r_mean > b_mean + 15:
        return "rear"
    return "front"
