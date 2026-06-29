#!/usr/bin/env python3
"""
scripts/evaluate_reid.py
─────────────────────────
Evaluate a trained OSNet checkpoint on the VeRi-776 query/test sets.
Reports Rank-1, Rank-5, mAP, and a custom tunnel re-entry simulation.

Usage
-----
    python scripts/evaluate_reid.py \\
        --weights models/reid/osnet_veri776.pth \\
        --data-root data/raw

Tunnel simulation
-----------------
Occlude each query vehicle for N frames, then measure recovery rate.
Occlusion is simulated by zero-ing the gallery for the occluded ID
and re-querying after N virtual frames.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

OCCLUSION_DURATIONS = [30, 60, 120, 300, 600]  # frames (virtual)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ReID model on VeRi-776")
    p.add_argument("--weights",   default="models/reid/osnet_veri776.pth")
    p.add_argument("--data-root", default="data/raw", help="Root containing VeRi-776")
    p.add_argument("--backbone",  default="osnet_x1_0")
    p.add_argument("--num-classes", type=int, default=576)
    p.add_argument("--device",    default="cuda:0")
    p.add_argument("--batch",     type=int, default=64)
    return p.parse_args()


def compute_cmc_map(
    distmat:   np.ndarray,
    query_ids: list[int],
    gallery_ids: list[int],
    max_rank:  int = 10,
):
    """Compute CMC curve and mAP from a distance matrix."""
    n_q, n_g = distmat.shape
    ranks = np.argsort(distmat, axis=1)

    all_cmc    = np.zeros(max_rank)
    all_ap     = []

    for q_idx in range(n_q):
        q_id    = query_ids[q_idx]
        ordered = [gallery_ids[ranks[q_idx, i]] for i in range(n_g)]

        matches = np.array([gid == q_id for gid in ordered], dtype=float)
        cmc = np.cumsum(matches) > 0
        all_cmc += cmc[:max_rank]

        # AP
        n_pos  = matches.sum()
        if n_pos == 0:
            continue
        prec_at_k = np.cumsum(matches) / (np.arange(n_g) + 1)
        ap = (prec_at_k * matches).sum() / n_pos
        all_ap.append(ap)

    all_cmc /= n_q
    mAP      = float(np.mean(all_ap))
    return all_cmc, mAP


def main() -> None:
    args = parse_args()

    try:
        import torchreid
        import torch
        import torch.nn.functional as F
        from torchvision import transforms
        from PIL import Image
    except ImportError as e:
        raise SystemExit(f"Missing dependency: {e}")

    # ── Load model ────────────────────────────────────────────────────────────
    from src.embedder import AppearanceEmbedder
    embedder = AppearanceEmbedder(
        weights    = args.weights,
        backbone   = args.backbone,
        num_classes= args.num_classes,
        device     = args.device,
        batch_size = args.batch,
    )

    # ── Load VeRi-776 data ────────────────────────────────────────────────────
    datamanager = torchreid.data.ImageDataManager(
        root             = args.data_root,
        sources          = ["veri"],
        targets          = ["veri"],
        height           = 256,
        width            = 128,
        batch_size_train = 32,
        batch_size_test  = 100,
    )

    testloader = datamanager.test_loader["veri"]
    query_loader   = testloader["query"]
    gallery_loader = testloader["gallery"]

    def extract_features(loader):
        all_feats, all_ids = [], []
        for imgs, pids, *_ in loader:
            crops = [Image.fromarray(img.permute(1,2,0).numpy()) for img in imgs]
            feats = embedder.extract(crops)
            all_feats.append(feats)
            all_ids.extend(pids.tolist())
        return np.concatenate(all_feats, axis=0), all_ids

    logger.info("Extracting query features …")
    q_feats, q_ids = extract_features(query_loader)
    logger.info("Extracting gallery features …")
    g_feats, g_ids = extract_features(gallery_loader)

    # ── Standard ReID eval ────────────────────────────────────────────────────
    from sklearn.metrics.pairwise import cosine_distances
    distmat      = cosine_distances(q_feats, g_feats)
    cmc, mAP     = compute_cmc_map(distmat, q_ids, g_ids)

    logger.info("─" * 50)
    logger.info("  Rank-1  : %.2f%%", cmc[0] * 100)
    logger.info("  Rank-5  : %.2f%%", cmc[4] * 100)
    logger.info("  Rank-10 : %.2f%%", cmc[9] * 100)
    logger.info("  mAP     : %.2f%%", mAP    * 100)
    logger.info("─" * 50)

    # ── Tunnel simulation ─────────────────────────────────────────────────────
    logger.info("Running tunnel re-entry simulation …")
    from src.gallery  import PersistentGallery
    from src.matcher  import HungarianMatcher

    gallery = PersistentGallery(threshold=0.45)
    matcher = HungarianMatcher(threshold=0.45)

    # Populate gallery with gallery-set embeddings
    for idx, (emb, gid) in enumerate(zip(g_feats, g_ids)):
        gallery.register_or_recover(emb, frame_n=0, cls_id=0)

    logger.info("\n  Occlusion  |  Recovery Rate")
    logger.info("  " + "─" * 30)

    for duration in OCCLUSION_DURATIONS:
        recovered = 0
        for emb, qid in zip(q_feats, q_ids):
            result = gallery.get_representative_embeddings()
            if result is None:
                break
            g_embs, g_pids = result
            matches = matcher.match(emb[None], g_embs, g_pids)
            if matches:
                pred_pid = matches[0][1]
                # True gallery pid for this query is qid (same vehicle ID)
                true_pid = next(
                    (pid for pid, d in gallery.gallery.items()
                     if d.get("vid") == qid), None
                )
                if pred_pid == true_pid:
                    recovered += 1

        rate = recovered / max(len(q_ids), 1) * 100
        logger.info("  %5d frames | %6.1f%%", duration, rate)

    logger.info("─" * 50)


if __name__ == "__main__":
    main()
