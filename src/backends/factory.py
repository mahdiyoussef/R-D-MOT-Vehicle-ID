"""
src/backends/factory.py
────────────────────────
Backend factory — reads config['strategy'] and returns the correct backend.
The pipeline orchestrator calls these factories at startup and never imports
concrete backends directly, enabling hot-swapping via a single config key.
"""

from __future__ import annotations

import logging

from src.backends.reid.base     import BaseReIDBackend
from src.backends.tracking.base import BaseTracker
from src.backends.gallery.base  import BaseGalleryIndex

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

def build_reid_backend(config: dict) -> BaseReIDBackend:
    """
    Instantiate and return the configured Re-ID embedding backend.

    strategy.reid_backend choices:
      osnet       — OSNet multi-scale CNN
      transreid   — TransReID ViT-S/16 with JPM + SIE + BNNeck
      clipreid    — CLIP ViT-B/16 image encoder (fine-tuned)
      multibranch — Multi-branch global + local feature head
      dinov2      — DINOv2 Foundation Model + LoRA adapters
    """
    backend_name = config.get("strategy", {}).get("reid_backend", "osnet")
    logger.info("Building Re-ID backend: '%s'", backend_name)

    if backend_name == "osnet":
        from src.backends.reid.osnet import OSNetBackend
        return OSNetBackend(config)

    if backend_name == "transreid":
        from src.backends.reid.transreid import TransReIDBackend
        return TransReIDBackend(config)

    if backend_name == "clipreid":
        from src.backends.reid.clip_reid import CLIPReIDBackend
        return CLIPReIDBackend(config)

    if backend_name == "multibranch":
        from src.backends.reid.multibranch import MultiBranchBackend
        return MultiBranchBackend(config)

    if backend_name == "dinov2":
        from src.backends.reid.dinov2_lora import DINOv2ReIDBackend
        return DINOv2ReIDBackend(config)

    raise ValueError(
        f"Unknown reid_backend: '{backend_name}'. "
        f"Valid options: osnet, transreid, clipreid, multibranch, dinov2"
    )


# ─────────────────────────────────────────────────────────────────────────────

def build_tracker_backend(config: dict) -> BaseTracker:
    """
    Instantiate and return the configured short-term tracker backend.

    strategy.tracker_backend choices:
      legacy        — Original StrongSORT-based tracker (via boxmot)
      botsort_reid  — BoT-SORT with ReID fusion + Camera Motion Compensation
      strongsort    — StrongSORT with AFLink + GSI interpolation
      yolo_native   — Ultralytics YOLO built-in tracking (BoT-SORT / ByteTrack)
    """
    backend_name = config.get("strategy", {}).get("tracker_backend", "legacy")
    logger.info("Building tracker backend: '%s'", backend_name)

    if backend_name == "legacy":
        from src.backends.tracking.legacy_tracker import LegacyTrackerBackend
        return LegacyTrackerBackend(config)

    if backend_name == "botsort_reid":
        from src.backends.tracking.botsort_reid import BotSortReIDTracker
        return BotSortReIDTracker(config)

    if backend_name in ("strongsort", "strongsort_v2"):
        from src.backends.tracking.strongsort import StrongSORTv2Tracker
        return StrongSORTv2Tracker(config)

    if backend_name == "yolo_native":
        from src.backends.tracking.yolo_native import YOLONativeTracker
        return YOLONativeTracker(config)

    raise ValueError(
        f"Unknown tracker_backend: '{backend_name}'. "
        f"Valid options: legacy, botsort_reid, strongsort, yolo_native"
    )


# ─────────────────────────────────────────────────────────────────────────────

def build_gallery_index(
    config:      dict,
    embed_dim:   int,
    gallery_ref,
) -> BaseGalleryIndex:
    """
    Instantiate and return the configured gallery index backend.

    strategy.gallery_backend choices:
      numpy         — Flat cosine search (best for < 500 identities)
      faiss_ivf     — FAISS IVFFlat approximate search (scalable)
      auto          — Start with numpy, auto-promote to FAISS at threshold
      probabilistic — Gaussian identity gallery with MLS matching
    """
    backend_name = config.get("strategy", {}).get("gallery_backend", "numpy")
    logger.info("Building gallery index backend: '%s'", backend_name)

    if backend_name == "numpy":
        from src.backends.gallery.numpy_index import NumpyGalleryIndex
        return NumpyGalleryIndex(config, gallery_ref)

    if backend_name == "faiss_ivf":
        from src.backends.gallery.faiss_ivf_index import FAISSIVFIndex
        return FAISSIVFIndex(config, embed_dim)

    if backend_name == "auto":
        from src.backends.gallery.auto_gallery import AutoGalleryManager
        return AutoGalleryManager(config, embed_dim, gallery_ref)

    if backend_name == "probabilistic":
        from src.backends.gallery.probabilistic_gallery import ProbabilisticGalleryIndex
        return ProbabilisticGalleryIndex(config, embed_dim)

    raise ValueError(
        f"Unknown gallery_backend: '{backend_name}'. "
        f"Valid options: numpy, faiss_ivf, auto, probabilistic"
    )


# ─────────────────────────────────────────────────────────────────────────────

def build_matcher(config: dict, embed_dim: int = 768):
    """
    Instantiate the configured matcher backend.

    strategy.matcher_backend choices:
      hungarian — Standard Hungarian algorithm (default)
      gnn       — GNN context-aware association + Hungarian
    """
    matcher_name = config.get("strategy", {}).get("matcher_backend", "hungarian")

    if matcher_name == "hungarian":
        logger.info("Using standard HungarianMatcher.")
        return None  # pipeline.py uses HungarianMatcher directly

    if matcher_name == "gnn":
        from src.matching.gnn_context_matcher import GNNContextMatcher
        matcher = GNNContextMatcher(config, embed_dim)
        logger.info("Built GNNContextMatcher.")
        return matcher

    raise ValueError(
        f"Unknown matcher_backend: '{matcher_name}'. "
        f"Valid options: hungarian, gnn"
    )
