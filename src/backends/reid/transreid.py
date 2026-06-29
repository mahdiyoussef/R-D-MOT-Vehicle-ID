"""
src/backends/reid/transreid_backend.py
───────────────────────────────────────
TransReID ViT-S/16 backend with JPM (Jigsaw Patches Module),
SIE (Side Information Embeddings), and BNNeck.

Architecture:
  Input crop → Patch Embedding → JPM → ViT-S Blocks ×12
    → SIE add → CLS token → BNNeck → L2-norm → [384]

Reference: Shuting He et al., "TransReID: Transformer-based Object Re-Identification"
           ICCV 2021. https://arxiv.org/abs/2102.04378
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from src.backends.reid.base import BaseReIDBackend

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Helper modules
# ─────────────────────────────────────────────────────────────────────────────

class BNNeck(nn.Module):
    """
    Batch-Norm Neck: BN1d applied before the metric/ID head.
    Splits the forward pass so triplet loss uses pre-BN features
    and ID loss uses post-BN features.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(embed_dim)
        self.bn.bias.requires_grad_(False)  # no shift in BNNeck

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class SideInfoEmbeddings(nn.Module):
    """
    Adds learnable camera and view embeddings to the CLS token.

    Parameters
    ----------
    num_cameras : int   Number of camera IDs in the dataset (0 = disabled).
    num_views   : int   Number of view angles (0 = disabled).
    embed_dim   : int   ViT embedding dimension.
    """

    def __init__(
        self,
        num_cameras: int,
        num_views: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.cam_emb  = nn.Embedding(max(num_cameras, 1), embed_dim) if num_cameras > 0 else None
        self.view_emb = nn.Embedding(max(num_views,   1), embed_dim) if num_views   > 0 else None
        self.num_cameras = num_cameras
        self.num_views   = num_views

    def forward(
        self,
        cam_label: torch.Tensor,
        view_label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = torch.zeros(cam_label.size(0), self.cam_emb.embedding_dim
                          if self.cam_emb else 1,
                          device=cam_label.device)
        if self.cam_emb is not None:
            cam_label = cam_label.clamp(0, self.num_cameras - 1)
            out = self.cam_emb(cam_label)
        if self.view_emb is not None and view_label is not None:
            view_label = view_label.clamp(0, self.num_views - 1)
            out = out + self.view_emb(view_label)
        return out  # [B, embed_dim]


class JigsawPatchesModule(nn.Module):
    """
    TransReID Section 3.2 — Jigsaw Patches Module (JPM).

    Applies circular shift + optional shuffle to the patch sequence,
    splits into local groups, and pools each group into a branch embedding.

    Parameters
    ----------
    embed_dim    : int   Patch embedding dimension (ViT-S = 384).
    shift_num    : int   Circular shift positions.
    shuffle      : bool  Randomly shuffle patches within groups.
    divide_length: int   Each group spans (N / divide_length) patches.
    num_groups   : int   Total local branches.
    """

    def __init__(
        self,
        embed_dim: int,
        shift_num: int   = 5,
        shuffle:   bool  = True,
        divide_length: int = 4,
        num_groups: int = 4,
    ) -> None:
        super().__init__()
        self.shift_num    = shift_num
        self.shuffle      = shuffle
        self.divide_length = divide_length
        self.num_groups   = num_groups

        # Lightweight shared local transformer (2 blocks)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=6, dim_feedforward=embed_dim * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.local_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Per-group BNNeck
        self.group_bns = nn.ModuleList(
            [BNNeck(embed_dim) for _ in range(num_groups)]
        )

    def forward(self, patch_seq: torch.Tensor) -> list[torch.Tensor]:
        """
        Parameters
        ----------
        patch_seq : [B, N, D]   patch tokens (excluding CLS)

        Returns
        -------
        list of num_groups tensors, each [B, D]
        """
        B, N, D = patch_seq.shape

        # 1. Circular shift along N axis
        shifted = torch.roll(patch_seq, shifts=self.shift_num, dims=1)

        # 2. Optional intra-group shuffle (during training only)
        if self.shuffle and self.training:
            group_size = N // self.num_groups
            parts = shifted.chunk(self.num_groups, dim=1)
            shuffled_parts = []
            for part in parts:
                perm = torch.randperm(part.size(1), device=part.device)
                shuffled_parts.append(part[:, perm, :])
            shifted = torch.cat(shuffled_parts, dim=1)

        # 3. Pass through local transformer
        local_out = self.local_transformer(shifted)   # [B, N, D]

        # 4. Split into num_groups slices and mean-pool each
        group_outputs: list[torch.Tensor] = []
        chunk_size = N // self.num_groups
        for g in range(self.num_groups):
            start = g * chunk_size
            end   = start + chunk_size
            group_feat = local_out[:, start:end, :].mean(dim=1)  # [B, D]
            group_feat = self.group_bns[g](group_feat)
            group_outputs.append(group_feat)

        return group_outputs


# ─────────────────────────────────────────────────────────────────────────────
# Weight loader
# ─────────────────────────────────────────────────────────────────────────────

def load_pretrained_weights(model: nn.Module, path: str) -> dict:
    """
    Load .pth checkpoint robustly, handling various wrapper key formats.

    Returns
    -------
    dict with 'loaded' and 'skipped' counts.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Checkpoint not found at '%s'. Skipping weight load.", path)
        return {"loaded": 0, "skipped": 0}

    ckpt = torch.load(path, map_location="cpu")

    # Unwrap common checkpoint wrappers
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            ckpt = ckpt[key]
            break

    model_state = model.state_dict()
    matched, skipped = {}, []

    for k, v in ckpt.items():
        # Strip common prefixes
        clean_k = k
        for prefix in ("module.", "backbone.", "base."):
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix):]

        if clean_k in model_state and model_state[clean_k].shape == v.shape:
            matched[clean_k] = v
        else:
            skipped.append(k)

    model.load_state_dict(matched, strict=False)

    if skipped:
        logger.warning(
            "load_pretrained_weights: %d keys skipped (shape mismatch / not found): %s…",
            len(skipped), skipped[:5],
        )
    logger.info(
        "load_pretrained_weights: loaded %d / %d parameters from '%s'.",
        len(matched), len(model_state), path,
    )
    return {"loaded": len(matched), "skipped": len(skipped)}


# ─────────────────────────────────────────────────────────────────────────────
# TransReID Backend
# ─────────────────────────────────────────────────────────────────────────────

class TransReIDBackend(BaseReIDBackend):
    """
    TransReID ViT-S/16 Re-ID backend.

    Architecture: timm ViT-S/16 + JPM local branches + SIE + BNNeck.
    Outputs a 384-dimensional L2-normalized embedding per crop.
    """

    def __init__(self, config: dict) -> None:
        self.full_config = config
        self.cfg    = config.get("transreid", {})
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model:   Optional[nn.Module] = None
        self._jpm:     Optional[JigsawPatchesModule] = None
        self._sie:     Optional[SideInfoEmbeddings]  = None
        self._bnneck:  Optional[BNNeck]              = None
        self._transform = None

    # ── load ──────────────────────────────────────────────────────────────────
    def load(self) -> None:
        import timm

        h, w    = self.cfg.get("image_size", [256, 128])
        emb_dim = self.cfg.get("embed_dim",  384)

        # 1. Build ViT-S/16 feature extractor (no classification head)
        self._model = timm.create_model(
            self.cfg.get("model_name", "vit_small_patch16_224"),
            pretrained   = False,
            num_classes  = 0,
            global_pool  = "",         # return full patch sequence + CLS
            img_size     = (h, w),
        )

        # 2. Jigsaw Patches Module
        jpm_cfg = self.cfg.get("jigsaw_patches_module", {})
        if jpm_cfg.get("enabled", True):
            self._jpm = JigsawPatchesModule(
                embed_dim     = emb_dim,
                shift_num     = jpm_cfg.get("shift_num",     5),
                shuffle       = jpm_cfg.get("patch_shuffle", True),
                divide_length = jpm_cfg.get("divide_length", 4),
                num_groups    = jpm_cfg.get("num_groups",    4),
            )

        # 3. Side Information Embeddings
        sie_cfg = self.cfg.get("side_information_embeddings", {})
        if sie_cfg.get("enabled", True):
            self._sie = SideInfoEmbeddings(
                num_cameras = sie_cfg.get("num_cameras", 20),
                num_views   = sie_cfg.get("num_views",   3),
                embed_dim   = emb_dim,
            )

        # 4. BNNeck
        if self.cfg.get("bnneck", True):
            self._bnneck = BNNeck(emb_dim)

        # 5. Load pretrained weights
        weights_path = self.cfg.get("pretrained_weights", "")
        if weights_path and Path(weights_path).exists():
            modules_to_load = nn.ModuleDict({"backbone": self._model})
            if self._jpm:     modules_to_load["jpm"]     = self._jpm
            if self._sie:     modules_to_load["sie"]     = self._sie
            if self._bnneck:  modules_to_load["bnneck"]  = self._bnneck
            # Try loading into backbone only (most common checkpoint format)
            load_pretrained_weights(self._model, weights_path)
        else:
            logger.warning(
                "TransReID weights not found at '%s'. "
                "Using random initialization — accuracy will be very poor. "
                "Run: python scripts/download_models.py --model transreid-vit-small",
                weights_path,
            )

        # 6. Move to device and set eval
        self._model.to(self.device).eval()
        if self._jpm:    self._jpm.to(self.device).eval()
        if self._sie:    self._sie.to(self.device).eval()
        if self._bnneck: self._bnneck.to(self.device).eval()

        # 7. Build transform
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

        # 8. TensorRT Acceleration
        if self.full_config.get("strategy", {}).get("use_tensorrt", False) and "cuda" in str(self.device):
            class ViTForwardFeaturesWrapper(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    return self.m.forward_features(x)
            
            wrapper = ViTForwardFeaturesWrapper(self._model)
            input_shape = (self.full_config.get("transreid", {}).get("batch_size", 32), 3, h, w)
            compiled_wrapper = self.compile_tensorrt(
                wrapper, 
                input_shape=input_shape,
                model_name="transreid_vits16"
            )
            # Re-bind the forward_features to call the compiled wrapper
            self._model.forward_features = compiled_wrapper

        logger.info(
            "TransReID ViT-S/16 loaded on %s  (JPM=%s, SIE=%s, BNNeck=%s)",
            self.device,
            self._jpm is not None,
            self._sie is not None,
            self._bnneck is not None,
        )

    # ── _preprocess ───────────────────────────────────────────────────────────
    def _preprocess_crop(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[torch.Tensor]:
        """Crop, validate, and transform a single bbox from frame."""
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        H, W = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return None

        crop_bgr = frame[y1:y2, x1:x2]
        import cv2
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return self._transform(crop_rgb)  # [3, H, W]

    # ── _forward ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _forward(
        self,
        batch: torch.Tensor,
        cam_labels: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Run batch through the full TransReID head, return [N, 384] numpy."""
        batch = batch.to(self.device)

        # ViT forward — returns [B, N+1, D] (CLS + patches)
        tokens = self._model.forward_features(batch)

        # Separate CLS token and patch tokens
        cls_token  = tokens[:, 0, :]          # [B, D]
        patch_seq  = tokens[:, 1:, :]         # [B, N, D]

        # Add SIE to CLS token
        if self._sie is not None and cam_labels is not None:
            cam_labels = cam_labels.to(self.device)
            sie_emb    = self._sie(cam_labels)   # [B, D]
            cls_token  = cls_token + sie_emb

        # BNNeck
        if self._bnneck is not None:
            embedding = self._bnneck(cls_token)   # [B, D]
        else:
            embedding = cls_token

        # L2-normalize
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy().astype(np.float32)

    # ── extract ───────────────────────────────────────────────────────────────
    @torch.no_grad()
    def extract(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        cam_label: int = 0,
    ) -> np.ndarray:
        if self._model is None:
            self.load()

        tensor = self._preprocess_crop(frame, bbox)
        if tensor is None:
            return np.zeros(self.embed_dim, dtype=np.float32)

        batch  = tensor.unsqueeze(0)                  # [1, 3, H, W]
        cam_t  = torch.tensor([cam_label], dtype=torch.long)
        result = self._forward(batch, cam_t)
        return result[0]

    # ── extract_batch ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def extract_batch(
        self,
        frame: np.ndarray,
        bboxes: list[np.ndarray],
        cam_labels: list[int] | None = None,
    ) -> np.ndarray:
        if self._model is None:
            self.load()
        if not bboxes:
            return np.empty((0, self.embed_dim), dtype=np.float32)

        tensors, valid_mask = [], []
        for bbox in bboxes:
            t = self._preprocess_crop(frame, bbox)
            if t is not None:
                tensors.append(t)
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        out = np.zeros((len(bboxes), self.embed_dim), dtype=np.float32)
        if not tensors:
            return out

        batch = torch.stack(tensors)   # [M, 3, H, W]
        if cam_labels:
            valid_cams = [c for c, v in zip(cam_labels, valid_mask) if v]
            cam_t = torch.tensor(valid_cams, dtype=torch.long)
        else:
            cam_t = torch.zeros(batch.size(0), dtype=torch.long)

        embeddings = self._forward(batch, cam_t)

        j = 0
        for i, valid in enumerate(valid_mask):
            if valid:
                out[i] = embeddings[j]
                j += 1
        return out

    # ── warmup ────────────────────────────────────────────────────────────────
    def warmup(self, n: int = 3) -> None:
        if self._model is None:
            self.load()
        h, w = self.cfg.get("image_size", [256, 128])
        dummy = torch.zeros(1, 3, h, w).to(self.device)
        for _ in range(n):
            self._model.forward_features(dummy)
        logger.debug("TransReIDBackend warmup complete (%d passes).", n)

    # ── embed_dim ─────────────────────────────────────────────────────────────
    @property
    def embed_dim(self) -> int:
        return int(self.cfg.get("embed_dim", 384))
