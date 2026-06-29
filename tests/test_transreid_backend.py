"""
tests/test_transreid_backend.py
────────────────────────────────
Unit tests for the TransReID ViT-S/16 backend.
Tests marked @pytest.mark.slow require actual model weights to be present.
"""

import numpy as np
import pytest
import torch

# Skip all tests in this module when timm is not installed
timm = pytest.importorskip(
    "timm",
    reason="timm not installed. Install with: pip install timm",
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_config():
    """Minimal config with CPU device and no pretrained weights."""
    return {
        "pipeline": {"device": "cpu", "half_precision": False},
        "strategy": {"reid_backend": "transreid"},
        "transreid": {
            "model_name": "vit_small_patch16_224",
            "embed_dim": 384,
            "image_size": [256, 128],
            "pretrained_weights": "models/reid/transreid_vit_small_veri776.pth",
            "jigsaw_patches_module": {
                "enabled": True, "shift_num": 5, "patch_shuffle": True,
                "divide_length": 4, "num_groups": 4,
            },
            "side_information_embeddings": {
                "enabled": True, "num_cameras": 20, "num_views": 3,
            },
            "bnneck": True,
        },
    }


@pytest.fixture
def dummy_frame():
    """256×128 BGR dummy frame."""
    return np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox():
    return np.array([10, 10, 118, 246], dtype=np.float32)


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_embed_dim_property_is_384(minimal_config):
    from src.backends.reid.transreid_backend import TransReIDBackend
    backend = TransReIDBackend(minimal_config)
    assert backend.embed_dim == 384


def test_extract_returns_shape_384(minimal_config, dummy_frame, sample_bbox):
    from src.backends.reid.transreid_backend import TransReIDBackend
    backend = TransReIDBackend(minimal_config)
    backend.load()
    emb = backend.extract(dummy_frame, sample_bbox, cam_label=0)
    assert emb.shape == (384,), f"Expected (384,) got {emb.shape}"


def test_extract_output_is_l2_normalized(minimal_config, dummy_frame, sample_bbox):
    from src.backends.reid.transreid_backend import TransReIDBackend
    backend = TransReIDBackend(minimal_config)
    backend.load()
    emb = backend.extract(dummy_frame, sample_bbox, cam_label=0)
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 1e-4, f"Not L2-normalized: norm={norm}"


def test_extract_batch_shape_n_by_384(minimal_config, dummy_frame):
    from src.backends.reid.transreid_backend import TransReIDBackend
    backend = TransReIDBackend(minimal_config)
    backend.load()
    bboxes = [
        np.array([10, 10, 80, 200], dtype=np.float32),
        np.array([20, 20, 100, 220], dtype=np.float32),
        np.array([5,  5,  70, 180], dtype=np.float32),
    ]
    embs = backend.extract_batch(dummy_frame, bboxes)
    assert embs.shape == (3, 384), f"Expected (3, 384) got {embs.shape}"


def test_extract_tiny_crop_returns_zeros_not_raises(minimal_config, dummy_frame):
    """A bbox smaller than 20px on either side should return zeros, not crash."""
    from src.backends.reid.transreid_backend import TransReIDBackend
    backend = TransReIDBackend(minimal_config)
    backend.load()
    tiny_bbox = np.array([10, 10, 25, 15], dtype=np.float32)   # 15w × 5h
    emb = backend.extract(dummy_frame, tiny_bbox, cam_label=0)
    assert emb.shape == (384,)
    assert np.allclose(emb, 0.0), "Tiny crop should return zeros"


def test_jpm_module_output_length_equals_num_groups():
    from src.backends.reid.transreid_backend import JigsawPatchesModule
    jpm = JigsawPatchesModule(embed_dim=384, num_groups=4)
    patch_seq = torch.randn(2, 128, 384)   # B=2, N=128, D=384
    outputs   = jpm(patch_seq)
    assert len(outputs) == 4, f"Expected 4 groups, got {len(outputs)}"
    for g, out in enumerate(outputs):
        assert out.shape == (2, 384), f"Group {g}: expected (2,384) got {out.shape}"


def test_sie_adds_cam_embedding_to_cls_token():
    from src.backends.reid.transreid_backend import SideInfoEmbeddings
    sie = SideInfoEmbeddings(num_cameras=20, num_views=3, embed_dim=384)
    cam_labels = torch.tensor([0, 1, 2])
    out = sie(cam_labels)
    assert out.shape == (3, 384), f"Expected (3,384) got {out.shape}"
    # Non-zero for valid cam ids
    assert not torch.allclose(out, torch.zeros_like(out))


def test_warmup_succeeds(minimal_config):
    from src.backends.reid.transreid_backend import TransReIDBackend
    backend = TransReIDBackend(minimal_config)
    backend.load()
    backend.warmup(n=2)   # should not raise


@pytest.mark.slow
def test_transreid_backend_loads_without_error(minimal_config):
    """Requires actual weights at models/reid/transreid_vit_small_veri776.pth."""
    from src.backends.reid.transreid_backend import TransReIDBackend
    backend = TransReIDBackend(minimal_config)
    backend.load()
    assert backend._model is not None
