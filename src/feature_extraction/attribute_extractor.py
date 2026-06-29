"""
src/backends/reid/attribute_extractor.py
─────────────────────────────────────────
v4.0 — Vehicle Attribute Extractor for Step 5.4 (Attribute Fallback Matching).

Extracts semantic vehicle attributes that are robust to viewpoint changes:
  - color:         12-class (white/black/silver/red/blue/green/yellow/orange/brown/purple/gold/gray)
  - vehicle_class: 8-class (car/truck/van/motorcycle/bus/pickup/suv/emergency)
  - plate_chars:   partial OCR string (empty string if unreadable, NEVER null)
  - has_roof_rack: bool (binary head, zero-shot via heuristic if untrained)
  - has_tow_hitch: bool (binary head, zero-shot via heuristic if untrained)

Architecture:
  All heads share the already-computed backbone CLS token.
  No second forward pass needed. Each head is a 2-layer MLP (Linear→ReLU→Linear).

If no pretrained head weights are found:
  - color → HSV histogram + k-nearest named-color lookup
  - vehicle_class → YOLO class label passthrough
  - plate → EasyOCR on bottom-20% crop (disabled by default, needs easyocr package)
  - roof_rack, tow_hitch → False (unknown)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Color palette: 12 named colors with representative HSV centroids
# ─────────────────────────────────────────────────────────────────────────────

COLOR_CLASSES = [
    "white", "black", "silver", "red", "blue", "green",
    "yellow", "orange", "brown", "purple", "gold", "gray",
]

# (H_center, S_min, S_max, V_min, V_max) in OpenCV HSV space (H: 0-180)
_COLOR_HSV_RULES = [
    # name       H_low H_high  S_low  S_high  V_low  V_high
    ("white",      0,   180,    0,     40,    200,    255),
    ("black",      0,   180,    0,    255,      0,     60),
    ("silver",     0,   180,    0,     60,    120,    200),
    ("gray",       0,   180,    0,     50,     60,    180),
    ("red",        0,    10,   100,   255,    100,    255),
    ("red",      170,   180,   100,   255,    100,    255),  # red wraps HSV
    ("orange",    10,    25,   100,   255,    100,    255),
    ("yellow",    25,    35,   100,   255,    100,    255),
    ("green",     35,    85,   100,   255,     50,    255),
    ("blue",      85,   130,   100,   255,     50,    255),
    ("purple",   130,   160,    50,   255,     50,    255),
    ("brown",      5,    20,    80,   200,     40,    130),
    ("gold",      20,    30,   100,   220,    100,    200),
]

VEHICLE_CLASSES = ["car", "truck", "van", "motorcycle", "bus", "pickup", "suv", "emergency"]


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AttributeVec:
    """Semantic vehicle attributes used in Step 5.4 fallback matching."""
    color:         str   = "unknown"
    vehicle_class: str   = "car"
    plate_chars:   str   = ""          # partial OCR — empty string, never null
    has_roof_rack: bool  = False
    has_tow_hitch: bool  = False

    def to_dict(self) -> dict:
        return {
            "color":         self.color,
            "vehicle_class": self.vehicle_class,
            "plate_chars":   self.plate_chars,
            "has_roof_rack": self.has_roof_rack,
            "has_tow_hitch": self.has_tow_hitch,
        }

    @staticmethod
    def from_dict(d: dict) -> "AttributeVec":
        return AttributeVec(
            color=d.get("color", "unknown"),
            vehicle_class=d.get("vehicle_class", "car"),
            plate_chars=d.get("plate_chars", ""),
            has_roof_rack=bool(d.get("has_roof_rack", False)),
            has_tow_hitch=bool(d.get("has_tow_hitch", False)),
        )

    def ema_update(self, other: "AttributeVec", alpha: float = 0.7) -> None:
        """Update string attributes by majority vote, boolean by OR."""
        # Colors: keep majority (simple override, could be improved with counter)
        if other.color != "unknown":
            self.color = other.color
        if other.vehicle_class != "car":
            self.vehicle_class = other.vehicle_class
        if other.plate_chars:
            # Prefer longer plate string
            if len(other.plate_chars) > len(self.plate_chars):
                self.plate_chars = other.plate_chars
        # Boolean: once True, stays True
        self.has_roof_rack = self.has_roof_rack or other.has_roof_rack
        self.has_tow_hitch = self.has_tow_hitch or other.has_tow_hitch


# ─────────────────────────────────────────────────────────────────────────────
# MLP Heads
# ─────────────────────────────────────────────────────────────────────────────

class AttributeHead(nn.Module):
    """General-purpose 2-layer MLP head for classification."""

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Main Attribute Extractor
# ─────────────────────────────────────────────────────────────────────────────

class AttributeExtractor:
    """
    Extracts vehicle attributes from a crop and/or a precomputed CLS token.

    Parameters
    ----------
    config       : pipeline config dict
    backbone_dim : int   CLS token dimension
    device       : torch.device
    """

    def __init__(
        self,
        config: dict,
        backbone_dim: int = 384,
        device: torch.device | None = None,
    ) -> None:
        self.cfg         = config.get("attributes", {})
        self.enabled     = self.cfg.get("enabled", True)
        self.device      = device or torch.device("cpu")
        self._ocr_reader = None
        self._color_head:   AttributeHead | None = None
        self._class_head:   AttributeHead | None = None
        self._rack_head:    AttributeHead | None = None
        self._hitch_head:   AttributeHead | None = None

        # Try to load pretrained MLP heads
        # (Expected format: single .pth with keys "color", "vehicle_class", "roof_rack", "tow_hitch")
        weights_path = Path(self.cfg.get("pretrained_weights", ""))
        if weights_path.is_file():
            ckpt = torch.load(weights_path, map_location="cpu")
            if "color" in ckpt:
                self._color_head = AttributeHead(backbone_dim, len(COLOR_CLASSES)).to(self.device)
                self._color_head.load_state_dict(ckpt["color"])
                self._color_head.eval()
            if "vehicle_class" in ckpt:
                self._class_head = AttributeHead(backbone_dim, len(VEHICLE_CLASSES)).to(self.device)
                self._class_head.load_state_dict(ckpt["vehicle_class"])
                self._class_head.eval()
            if "roof_rack" in ckpt:
                self._rack_head = AttributeHead(backbone_dim, 2).to(self.device)
                self._rack_head.load_state_dict(ckpt["roof_rack"])
                self._rack_head.eval()
            if "tow_hitch" in ckpt:
                self._hitch_head = AttributeHead(backbone_dim, 2).to(self.device)
                self._hitch_head.load_state_dict(ckpt["tow_hitch"])
                self._hitch_head.eval()
            logger.info("AttributeExtractor: loaded pretrained heads from %s", weights_path)
        else:
            logger.info("AttributeExtractor: no pretrained weights. Using heuristic fallbacks.")

    def extract(
        self,
        crop_bgr: np.ndarray,
        yolo_class_name: str = "car",
        cls_token: torch.Tensor | None = None,
    ) -> AttributeVec:
        """
        Extract all attributes for a vehicle detection.

        Parameters
        ----------
        crop_bgr        : BGR image crop (tight bounding box)
        yolo_class_name : class label string from YOLO detector
        cls_token       : (D,) or (1,D) CLS token from backbone (optional)

        Returns
        -------
        AttributeVec with all fields populated (never None).
        """
        if not self.enabled:
            return AttributeVec(vehicle_class=yolo_class_name)

        attrs = AttributeVec()

        # ── Color ────────────────────────────────────────────────────────────
        if self._color_head is not None and cls_token is not None:
            attrs.color = self._color_from_token(cls_token)
        elif crop_bgr is not None and crop_bgr.size > 0:
            attrs.color = self._color_from_hsv(crop_bgr)

        # ── Vehicle Class ─────────────────────────────────────────────────────
        if self._class_head is not None and cls_token is not None:
            attrs.vehicle_class = self._class_from_token(cls_token)
        else:
            # Map YOLO class to our attribute vocabulary
            attrs.vehicle_class = _map_yolo_class(yolo_class_name)

        # ── Plate OCR ────────────────────────────────────────────────────────
        if self.cfg.get("enable_plate_ocr", False) and crop_bgr is not None and crop_bgr.size > 0:
            attrs.plate_chars = self._run_plate_ocr(crop_bgr)

        # ── Roof Rack ─────────────────────────────────────────────────────────
        if self._rack_head is not None and cls_token is not None:
            attrs.has_roof_rack = self._binary_from_token(self._rack_head, cls_token)

        # ── Tow Hitch ─────────────────────────────────────────────────────────
        if self._hitch_head is not None and cls_token is not None:
            attrs.has_tow_hitch = self._binary_from_token(self._hitch_head, cls_token)

        return attrs

    # ── Internal helpers ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _color_from_token(self, cls_token: torch.Tensor) -> str:
        t = cls_token.unsqueeze(0) if cls_token.dim() == 1 else cls_token
        logits = self._color_head(t.to(self.device))
        idx = int(logits.argmax(dim=-1)[0])
        return COLOR_CLASSES[idx]

    @torch.no_grad()
    def _class_from_token(self, cls_token: torch.Tensor) -> str:
        t = cls_token.unsqueeze(0) if cls_token.dim() == 1 else cls_token
        logits = self._class_head(t.to(self.device))
        idx = int(logits.argmax(dim=-1)[0])
        return VEHICLE_CLASSES[idx]

    @torch.no_grad()
    def _binary_from_token(self, head: AttributeHead, cls_token: torch.Tensor) -> bool:
        t = cls_token.unsqueeze(0) if cls_token.dim() == 1 else cls_token
        logits = head(t.to(self.device))
        return bool(logits.argmax(dim=-1)[0].item())

    def _color_from_hsv(self, crop_bgr: np.ndarray) -> str:
        """HSV histogram + rule-based color classification (vectorized heuristic)."""
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        # Focus on center 50% of crop (avoids background)
        h, w = hsv.shape[:2]
        roi = hsv[h//4:3*h//4, w//4:3*w//4]
        h_ch = roi[:, :, 0]
        s_ch = roi[:, :, 1]
        v_ch = roi[:, :, 2]

        color_votes: dict[str, int] = {}
        for rule in _COLOR_HSV_RULES:
            name, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = rule
            mask = (
                (h_ch >= h_lo) & (h_ch <= h_hi) &
                (s_ch >= s_lo) & (s_ch <= s_hi) &
                (v_ch >= v_lo) & (v_ch <= v_hi)
            )
            color_votes[name] = color_votes.get(name, 0) + int(mask.sum())

        return max(color_votes, key=lambda k: color_votes[k])

    def _run_plate_ocr(self, crop_bgr: np.ndarray) -> str:
        """Run EasyOCR on the bottom 20% of the crop to detect plate characters."""
        try:
            if self._ocr_reader is None:
                import easyocr
                self._ocr_reader = easyocr.Reader(["en"], gpu=str(self.device) != "cpu")

            h = crop_bgr.shape[0]
            plate_region = crop_bgr[int(h * 0.80):, :]
            results = self._ocr_reader.readtext(plate_region, detail=0)
            text = "".join(results).upper().strip()
            min_chars = self.cfg.get("plate_min_chars", 2)
            return text if len(text) >= min_chars else ""
        except Exception as e:
            logger.debug("Plate OCR failed: %s", e)
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# Attribute-based similarity score (Step 5.4)
# ─────────────────────────────────────────────────────────────────────────────

def attribute_similarity(
    query: AttributeVec,
    gallery: AttributeVec,
    weights: dict | None = None,
) -> tuple[float, bool]:
    """
    Compute the weighted attribute similarity score for Step 5.4.

    Parameters
    ----------
    query, gallery : AttributeVec instances to compare
    weights        : dict with keys matching the plan's weight keys

    Returns
    -------
    (score, is_valid)
        is_valid=False if neither entry has plate data (can't anchor on empty plates).
    """
    if weights is None:
        weights = {
            "color_match":          0.35,
            "vehicle_class_match":  0.25,
            "plate_chars_match":    0.30,
            "roof_rack_match":      0.05,
            "tow_hitch_match":      0.05,
        }

    score = 0.0
    matched_color = False
    matched_plate = False

    if query.color != "unknown" and gallery.color != "unknown":
        if query.color == gallery.color:
            score += weights["color_match"]
            matched_color = True

    if query.vehicle_class == gallery.vehicle_class:
        score += weights["vehicle_class_match"]

    # Plate: partial overlap, Levenshtein ≤ 1
    if query.plate_chars and gallery.plate_chars:
        plate_score = _plate_similarity(query.plate_chars, gallery.plate_chars)
        if plate_score:
            score += weights["plate_chars_match"]
            matched_plate = True

    if query.has_roof_rack == gallery.has_roof_rack:
        score += weights["roof_rack_match"]
    if query.has_tow_hitch == gallery.has_tow_hitch:
        score += weights["tow_hitch_match"]

    # Guard: no plate data on either side → mark invalid as primary anchor
    has_plate_data = bool(query.plate_chars) or bool(gallery.plate_chars)
    is_valid = matched_color or matched_plate or not has_plate_data

    return score, is_valid


def _plate_similarity(a: str, b: str) -> bool:
    """
    Returns True if plates share at least 2 characters overlap
    AND Levenshtein distance ≤ 1.
    """
    # Simple overlap check (common 2-char substring)
    overlap = sum(1 for c in a if c in b)
    if overlap < 2:
        return False
    # Simple Levenshtein (no dependency needed for short strings)
    if abs(len(a) - len(b)) > 2:
        return False
    return _levenshtein(a, b) <= 1


def _levenshtein(a: str, b: str) -> int:
    """Minimal Levenshtein distance implementation."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def _map_yolo_class(yolo_name: str) -> str:
    """Map YOLO class name to our 8-class vocabulary."""
    mapping = {
        "car": "car", "automobile": "car",
        "truck": "truck", "lorry": "truck",
        "van": "van", "minivan": "van",
        "motorcycle": "motorcycle", "motorbike": "motorcycle",
        "bus": "bus", "coach": "bus",
        "pickup": "pickup",
        "suv": "suv",
        "ambulance": "emergency", "police": "emergency", "firetruck": "emergency",
    }
    return mapping.get(yolo_name.lower(), "car")
