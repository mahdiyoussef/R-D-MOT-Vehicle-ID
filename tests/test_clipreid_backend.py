"""
tests/test_clipreid_backend.py
──────────────────────────────
Unit tests for the CLIP-ReID backend.
Slow tests require actual CLIP model to be installed.
"""

import numpy as np
import pytest


@pytest.fixture
def minimal_config():
    return {
        "pipeline": {"device": "cpu"},
        "strategy": {"reid_backend": "clipreid"},
        "clipreid": {
            "clip_model": "ViT-B/16",
            "embed_dim": 512,
            "image_size": [224, 224],
            "pretrained_weights": "models/reid/clipreid_veri776.pth",
            "weights_gdrive_id": "1PLACE_HOLDER_CLIPREID_ID",
        },
    }


@pytest.fixture
def dummy_frame():
    return np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox():
    return np.array([10, 10, 280, 280], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────

def test_embed_dim_property_is_512(minimal_config):
    from src.backends.reid.clipreid_backend import CLIPReIDBackend
    backend = CLIPReIDBackend(minimal_config)
    assert backend.embed_dim == 512


@pytest.mark.slow
def test_clipreid_loads_without_error(minimal_config):
    """Requires: pip install git+https://github.com/openai/CLIP.git"""
    from src.backends.reid.clipreid_backend import CLIPReIDBackend
    backend = CLIPReIDBackend(minimal_config)
    backend.load()
    assert backend._visual_encoder is not None


@pytest.mark.slow
def test_extract_returns_shape_512(minimal_config, dummy_frame, sample_bbox):
    from src.backends.reid.clipreid_backend import CLIPReIDBackend
    backend = CLIPReIDBackend(minimal_config)
    backend.load()
    emb = backend.extract(dummy_frame, sample_bbox)
    assert emb.shape == (512,)


@pytest.mark.slow
def test_extract_output_is_l2_normalized(minimal_config, dummy_frame, sample_bbox):
    from src.backends.reid.clipreid_backend import CLIPReIDBackend
    backend = CLIPReIDBackend(minimal_config)
    backend.load()
    emb = backend.extract(dummy_frame, sample_bbox)
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 1e-4


@pytest.mark.slow
def test_extract_batch_consistent_with_single(minimal_config, dummy_frame):
    from src.backends.reid.clipreid_backend import CLIPReIDBackend
    backend = CLIPReIDBackend(minimal_config)
    backend.load()
    bboxes = [
        np.array([10, 10, 200, 200], dtype=np.float32),
        np.array([20, 20, 250, 250], dtype=np.float32),
    ]
    batch_embs  = backend.extract_batch(dummy_frame, bboxes)
    single_emb0 = backend.extract(dummy_frame, bboxes[0])
    # Batch and single should agree closely
    assert np.allclose(batch_embs[0], single_emb0, atol=1e-4)


def test_clip_normalization_constants_correct():
    """Verify CLIP uses its own mean/std, not ImageNet's."""
    from src.backends.reid.clipreid_backend import _CLIP_MEAN, _CLIP_STD
    assert abs(_CLIP_MEAN[0] - 0.48145466) < 1e-6
    assert abs(_CLIP_STD[0]  - 0.26862954) < 1e-6
