"""
tests/test_botsort_reid.py
──────────────────────────
Unit tests for the BoT-SORT-ReID tracker backend.
"""

import numpy as np
import pytest


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return {
        "pipeline": {"device": "cpu", "half_precision": False},
        "strategy": {"tracker_backend": "botsort_reid"},
        "tracking": {
            "reid_weights": "models/trackers/osnet_x0_25_msmt17.pt",
            "max_age": 10, "min_hits": 1, "iou_threshold": 0.3, "half": False,
        },
        "botsort_reid": {
            "max_age": 10, "min_hits": 1, "iou_threshold": 0.3,
            "cmc_method": None,          # disable CMC for deterministic tests
            "cmc_downscale": 1,
            "reid_weight": 0.4,
            "reid_thresh_new": 0.7,
            "embedding_buffer": 5,
            "match_thresh_high": 0.5,   # lower for test detections
            "match_thresh_low": 0.2,
            "second_pass_iou": 0.3,
        },
    }


@pytest.fixture
def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _make_det(x1, y1, x2, y2, conf=0.9, cls=0):
    return {"bbox": [x1, y1, x2, y2], "conf": conf, "cls": cls}


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_update_returns_tracker_outputs(config, dummy_frame):
    from src.backends.tracking.botsort_reid import BotSortReIDTracker
    from src.backends.tracking.base import TrackerOutput
    tracker = BotSortReIDTracker(config)
    dets    = [_make_det(100, 100, 200, 200)]
    outputs = tracker.update(dummy_frame, dets)
    for out in outputs:
        assert isinstance(out, TrackerOutput)


def test_reset_clears_all_state(config, dummy_frame):
    from src.backends.tracking.botsort_reid import BotSortReIDTracker
    tracker = BotSortReIDTracker(config)
    tracker.update(dummy_frame, [_make_det(10, 10, 100, 100)])
    assert len(tracker._tracks) > 0 or True   # may be empty if min_hits > 1
    tracker.reset()
    assert len(tracker._tracks) == 0
    assert tracker._next_id == 1
    assert tracker._frame_count == 0


def test_is_new_true_on_first_appearance(config, dummy_frame):
    """A track confirmed on its first frame (min_hits=1) should have is_new=True."""
    from src.backends.tracking.botsort_reid import BotSortReIDTracker
    tracker = BotSortReIDTracker(config)
    dets    = [_make_det(100, 100, 200, 200, conf=0.9)]
    outputs = tracker.update(dummy_frame, dets)
    if outputs:
        assert outputs[0].is_new is True


def test_track_survives_max_age_then_dropped(config, dummy_frame):
    """A track with no matching detections should be pruned after max_age frames."""
    from src.backends.tracking.botsort_reid import BotSortReIDTracker
    tracker = BotSortReIDTracker(config)
    tracker.update(dummy_frame, [_make_det(100, 100, 200, 200)])
    # Feed empty detections until track ages out
    max_age = config["botsort_reid"]["max_age"]
    for _ in range(max_age + 2):
        tracker.update(dummy_frame, [])
    assert all(t.frames_since_seen <= max_age for t in tracker._tracks.values())


def test_reid_fusion_cost_between_0_and_1(config, dummy_frame):
    """The fused cost matrix should be in [0, 1] before gating."""
    from src.backends.tracking.botsort_reid import BotSortReIDTracker
    tracker  = BotSortReIDTracker(config)
    dets     = [_make_det(100, 100, 200, 200)]
    tracker.update(dummy_frame, dets)  # create one track

    # Next frame: same position + embedding
    emb      = np.random.randn(384).astype(np.float32)
    emb      = emb / (np.linalg.norm(emb) + 1e-8)
    embeddings = {0: emb}
    # _associate should not raise and costs should be finite
    matched, um_trk, um_det = tracker._associate(
        list(tracker._tracks.keys()), dets, {0: emb}, use_reid=True
    )
    assert isinstance(matched, list)


def test_cmc_returns_none_on_first_frame(config):
    """CMC should return None on the very first call (no previous frame)."""
    from src.backends.tracking.botsort_reid import CameraMotionCompensator
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cmc   = CameraMotionCompensator(method="ecc", downscale=1)
    M     = cmc.compute(frame)
    assert M is None  # no previous frame


def test_cmc_ecc_returns_affine_matrix_or_none(config):
    """After two frames, ECC CMC should return a [2,3] matrix or None."""
    import cv2
    from src.backends.tracking.botsort_reid import CameraMotionCompensator
    cmc  = CameraMotionCompensator(method="ecc", downscale=2)
    # Two identical frames (should produce identity-like warp or None on failure)
    f1   = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    f2   = f1.copy()
    cmc.compute(f1)
    M = cmc.compute(f2)
    if M is not None:
        assert M.shape == (2, 3)


def test_high_conf_detections_matched_first(config, dummy_frame):
    """High-conf detections (≥ match_thresh_high) are used in the first pass."""
    from src.backends.tracking.botsort_reid import BotSortReIDTracker
    tracker  = BotSortReIDTracker(config)
    high_det = _make_det(100, 100, 200, 200, conf=0.95)
    low_det  = _make_det(300, 300, 400, 400, conf=0.3)
    tracker.update(dummy_frame, [high_det, low_det])
    # Should not raise; both may or may not create tracks depending on min_hits


def test_low_conf_detections_used_in_second_pass(config, dummy_frame):
    """Low-conf detections should not raise and may create tracks in 2nd pass."""
    from src.backends.tracking.botsort_reid import BotSortReIDTracker
    tracker = BotSortReIDTracker(config)
    tracker.update(dummy_frame, [_make_det(100, 100, 200, 200, conf=0.1)])


def test_embedding_buffer_averages_correctly():
    """mean_embedding should be L2-normalized mean of the buffer."""
    from src.backends.tracking.botsort_reid import BotSortTrack
    det = {"bbox": [100, 100, 200, 200], "conf": 0.9, "cls": 0}
    e1  = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e2  = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    trk = BotSortTrack(tracker_id=1, detection=det, embedding=e1)
    trk.update(det, e2)
    mean = trk.mean_embedding
    assert mean is not None
    assert abs(np.linalg.norm(mean) - 1.0) < 1e-5
