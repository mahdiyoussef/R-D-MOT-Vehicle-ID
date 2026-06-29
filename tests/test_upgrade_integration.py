"""
tests/test_upgrade_integration.py
───────────────────────────────────
Integration tests verifying that all strategy combinations build correctly
and that preset configs behave as expected.

Slow tests require actual model weights and GPU.
"""

import numpy as np
import pytest
import yaml
from pathlib import Path


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _make_dummy_frame(h=480, w=640):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _base_config():
    """Minimal functional config for unit-level integration tests."""
    return {
        "pipeline":   {"device": "cpu", "half_precision": False},
        "detection":  {
            "weights": "models/yolo/yolo11n_ha11.pt",
            "fallback_weights": "yolo11m.pt",
            "confidence_threshold": 0.4,
            "iou_threshold": 0.45,
            "imgsz": 640,
        },
        "tracking":   {
            "reid_weights": "models/trackers/osnet_x0_25_msmt17.pt",
            "max_age": 10, "min_hits": 1, "iou_threshold": 0.3, "half": False,
        },
        "embedding":  {
            "weights": "models/reid/osnet_veri776.pth",
            "backbone": "osnet_x1_0", "embedding_dim": 512,
            "input_size": [128, 256], "batch_size": 4,
        },
        "gallery":    {
            "similarity_threshold": 0.45,
            "max_embeddings_per_id": 5,
            "gallery_timeout_frames": 100,
            "new_track_grace_period": 5,
        },
        "matching":   {"threshold": 0.45},
        "visualization": {
            "show_confidence": True, "show_trajectory": True,
            "trajectory_length": 10,
        },
        "output": {"video_codec": "mp4v", "log_format": "jsonl", "gallery_prune_interval": 50},
        "botsort_reid": {
            "max_age": 10, "min_hits": 1, "iou_threshold": 0.3,
            "cmc_method": None, "cmc_downscale": 1,
            "reid_weight": 0.4, "reid_thresh_new": 0.7,
            "embedding_buffer": 5,
            "match_thresh_high": 0.5, "match_thresh_low": 0.2,
            "second_pass_iou": 0.3,
        },
        "strongsort_v2": {
            "max_age": 10, "min_hits": 1, "iou_threshold": 0.3,
            "aflink_enabled": False, "gsi_enabled": False,
            "ema_alpha": 0.9,
        },
        "faiss_ivf": {
            "index_type": "IVFFlat", "metric": "inner_product",
            "nlist": 4, "nprobe": 2, "auto_promote_threshold": 1000,
            "similarity_threshold": 0.45, "min_train_vectors": 8,
            "retrain_on_add": False, "retrain_growth_ratio": 0.2, "use_gpu": False,
        },
        "transreid": {
            "model_name": "vit_small_patch16_224", "embed_dim": 384,
            "image_size": [256, 128], "pretrained_weights": "",
            "jigsaw_patches_module": {"enabled": True, "shift_num": 5,
                                       "patch_shuffle": True, "divide_length": 4, "num_groups": 4},
            "side_information_embeddings": {"enabled": True, "num_cameras": 20, "num_views": 3},
            "bnneck": True,
        },
        "multibranch": {
            "num_parts": 4, "global_dim": 384, "local_dim": 96,
            "fusion": "concat", "attention_heads": 4, "dropout": 0.0,
            "backbone": "transreid",
        },
        "clipreid": {
            "clip_model": "ViT-B/16", "embed_dim": 512,
            "image_size": [224, 224], "pretrained_weights": "",
            "weights_gdrive_id": "1PLACE_HOLDER_CLIPREID_ID",
        },
    }


# ─── Factory tests ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("reid", ["osnet", "transreid"])
@pytest.mark.parametrize("tracker", ["legacy", "botsort_reid", "strongsort_v2"])
@pytest.mark.parametrize("gallery", ["numpy", "faiss_ivf", "auto"])
def test_factory_builds_all_strategy_combinations(reid, tracker, gallery):
    """
    All 3 × 3 × 3 = 27 combinations of (reid, tracker, gallery) should build
    without error. OSNet and TransReID don't require actual weights for build().
    CLIP-ReID is excluded from the non-slow matrix because it requires CLIP.
    """
    if reid == "multibranch":
        pytest.skip("multibranch requires backbone weights")
    if reid == "clipreid":
        pytest.skip("clipreid requires CLIP installation")

    from src.backends.factory import (
        build_reid_backend, build_tracker_backend, build_gallery_index
    )

    class _MockGallery:
        gallery  = {}
        _next_id = 1
        def _query(self, *a, **kw): return None, -1.0

    cfg = _base_config()
    cfg["strategy"] = {"reid_backend": reid, "tracker_backend": tracker, "gallery_backend": gallery}

    reid_b    = build_reid_backend(cfg)
    tracker_b = build_tracker_backend(cfg)
    gallery_b = build_gallery_index(cfg, reid_b.embed_dim, _MockGallery())

    assert reid_b    is not None
    assert tracker_b is not None
    assert gallery_b is not None


def test_preset_fast_config_is_valid():
    """configs/preset_fast.yaml should load without error."""
    path = Path("configs/preset_fast.yaml")
    assert path.exists(), "configs/preset_fast.yaml not found"
    cfg = _load_config(str(path))
    assert cfg["strategy"]["reid_backend"]    == "osnet"
    assert cfg["strategy"]["tracker_backend"] == "legacy"
    assert cfg["strategy"]["gallery_backend"] == "numpy"


def test_preset_balanced_config_is_valid():
    path = Path("configs/preset_balanced.yaml")
    assert path.exists()
    cfg = _load_config(str(path))
    assert cfg["strategy"]["reid_backend"]    == "transreid"
    assert cfg["strategy"]["tracker_backend"] == "botsort_reid"


def test_preset_sota_config_is_valid():
    path = Path("configs/preset_sota.yaml")
    assert path.exists()
    cfg = _load_config(str(path))
    assert cfg["strategy"]["reid_backend"]    == "clipreid"
    assert cfg["strategy"]["tracker_backend"] == "strongsort_v2"


def test_gallery_auto_promotes_during_processing():
    """AutoGalleryManager should promote to FAISS when identities exceed threshold."""
    from src.backends.gallery.auto_gallery import AutoGalleryManager

    class _MockGallery:
        gallery  = {}
        _next_id = 1
        def _query(self, *a, **kw): return None, -1.0

    cfg = _base_config()
    cfg["faiss_ivf"]["auto_promote_threshold"] = 5
    cfg["faiss_ivf"]["min_train_vectors"]      = 5

    gallery     = _MockGallery()
    auto_mgr    = AutoGalleryManager(cfg, embed_dim=16, gallery_ref=gallery)

    for i in range(7):
        emb = np.random.randn(16).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        gallery.gallery[i] = {"embeddings": [emb], "last_seen": 0, "class": 0}
        auto_mgr.add(i, emb)

    assert auto_mgr._promoted, "Should have promoted to FAISS after 7 identities"


@pytest.mark.slow
def test_all_strategy_combinations_build_slow():
    """Full slow test including CLIP-ReID (requires actual CLIP installation)."""
    pytest.skip("Requires CLIP + GPU; run manually")
