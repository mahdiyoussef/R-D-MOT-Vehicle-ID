"""
src/embedder.py
───────────────
Stage 3 — Appearance Embedding Extraction
Wraps OSNet (torchreid) or any timm ViT (TransReID) to extract
L2-normalised appearance descriptors from vehicle crops.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# ImageNet statistics used during VeRi-776 / MSMT17 training
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


class AppearanceEmbedder:
    """
    Vehicle appearance feature extractor.

    Supports two backends:
    - ``osnet``    — OSNet-x1.0 via torchreid (default, faster)
    - ``transreid``— ViT-Base/16 via timm (higher accuracy, more VRAM)

    Parameters
    ----------
    weights      : str | Path   Path to fine-tuned checkpoint (.pth).
    backbone     : str          'osnet_x1_0' or 'vit_base_patch16_224'.
    num_classes  : int          Number of vehicle IDs in training set (VeRi-776 → 576).
    input_size   : tuple        (width, height) for the transform.
    device       : str          'cuda:0' or 'cpu'.
    batch_size   : int          Max crops to process per forward pass.
    """

    def __init__(
        self,
        weights: str | Path = "models/reid/osnet_veri776.pth",
        backbone: str = "osnet_x1_0",
        num_classes: int = 576,
        input_size: tuple[int, int] = (128, 256),  # (W, H)
        device: str = "cuda:0",
        batch_size: int = 32,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.backbone_name = backbone

        self.model = self._build_model(backbone, num_classes, weights)
        self.model.eval().to(self.device)

        w, h = input_size
        self.transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])

        logger.info(
            "AppearanceEmbedder initialised — backbone=%s device=%s weights=%s",
            backbone, self.device, weights,
        )

    # ──────────────────────────────────────────────────────────────────────────
    def _build_model(
        self,
        backbone: str,
        num_classes: int,
        weights: str | Path,
    ) -> torch.nn.Module:
        weights = Path(weights)

        if backbone.startswith("osnet"):
            import torchreid
            model = torchreid.models.build_model(
                name=backbone,
                num_classes=num_classes,
                pretrained=not weights.exists(),  # skip download if we have weights
            )
            if weights.exists():
                state = torch.load(weights, map_location="cpu")
                # Handle various checkpoint formats
                if "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state, strict=False)
                logger.info("Loaded OSNet weights from '%s'.", weights)
            else:
                logger.warning(
                    "Fine-tuned weights not found at '%s'. "
                    "Using ImageNet pretrained OSNet — accuracy will be suboptimal. "
                    "Train with scripts/train_reid.py first.",
                    weights,
                )
        else:
            # TransReID / generic timm ViT backbone
            import timm
            model = timm.create_model(
                backbone, pretrained=not weights.exists(), num_classes=num_classes
            )
            if weights.exists():
                state = torch.load(weights, map_location="cpu")
                if "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state, strict=False)
                logger.info("Loaded ViT weights from '%s'.", weights)

        return model

    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def extract(self, crops: List[Image.Image]) -> np.ndarray:
        """
        Extract L2-normalised embeddings for a list of PIL crops.

        Parameters
        ----------
        crops : list of PIL.Image
            Vehicle crops (RGB).

        Returns
        -------
        np.ndarray, shape (N, D)
            Float32 L2-normalised embedding vectors.
        """
        if not crops:
            return np.empty((0,), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(crops), self.batch_size):
            batch_crops = crops[i : i + self.batch_size]
            tensors = torch.stack([self.transform(c) for c in batch_crops])
            tensors = tensors.to(self.device)

            features = self.model(tensors)  # (B, D)
            features = F.normalize(features, p=2, dim=1)
            all_embeddings.append(features.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def crop_from_frame(
        frame: np.ndarray,
        bbox: np.ndarray | list,
        padding: int = 10,
    ) -> Image.Image:
        """
        Crop a vehicle from a BGR frame given [x1, y1, x2, y2].
        Converts to RGB PIL image for the transform pipeline.
        """
        import cv2

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        crop_bgr = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(crop_rgb)
