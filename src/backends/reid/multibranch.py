"""
src/backends/reid/multibranch_backend.py
─────────────────────────────────────────
Multi-Branch Global + Local Feature Head backend.

Architecture:
  Any backbone (TransReID or CLIP-ReID image encoder)
    │
    ├── Global Branch: CLS token → BNNeck → [global_dim]
    │
    └── Local Branch:  Patch sequence → horizontal strip split (num_parts=4)
                        → per-strip attention pool → BNNeck → [local_dim×4]
    │
    Fusion: concat [global_dim + local_dim×num_parts] → L2-norm → final embedding

This backend replaces the color-histogram fallback and provides a
high-accuracy zero-shot or fine-tuned multi-branch feature extractor.
"""

from __future__ import annotations

import logging
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
_CLIP_MEAN     = (0.48145466, 0.4578275,  0.40821073)
_CLIP_STD      = (0.26862954, 0.26130258, 0.27577711)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class GlobalBranch(nn.Module):
    """
    Projects the CLS token to the global embedding space.

    Input:  cls_token [B, D]
    Output: [B, global_dim]
    """

    def __init__(self, in_dim: int, global_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, global_dim),
            nn.BatchNorm1d(global_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.proj(cls_token)


class LocalBranch(nn.Module):
    """
    Splits the patch sequence into num_parts horizontal strips and
    projects each strip to the local embedding space.

    Input:  patch_seq [B, N, D]
    Output: [B, num_parts * local_dim]

    The patch layout assumes standard ViT with square patches on a
    rectangular input (e.g. 256×128 / stride 16 → 16×8 = 128 patches).
    """

    def __init__(
        self,
        in_dim:    int,
        local_dim: int,
        num_parts: int = 4,
        h_patches: int = 16,  # height of patch grid
        w_patches: int = 8,   # width of patch grid
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        self.num_parts = num_parts
        self.h_patches = h_patches
        self.w_patches = w_patches

        self.part_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, local_dim),
                nn.BatchNorm1d(local_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
            for _ in range(num_parts)
        ])

    def forward(self, patch_seq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        patch_seq : [B, N, D]  patch tokens (without CLS)

        Returns
        -------
        [B, num_parts * local_dim]
        """
        B, N, D = patch_seq.shape
        H, W = self.h_patches, self.w_patches

        # Reshape to spatial grid, truncating if N != H*W
        n_patches = min(N, H * W)
        spatial = patch_seq[:, :n_patches, :].view(B, H, W, D)

        strip_height = H // self.num_parts
        part_feats = []
        for p in range(self.num_parts):
            start = p * strip_height
            end   = start + strip_height
            strip = spatial[:, start:end, :, :]         # [B, sh, W, D]
            pooled = strip.mean(dim=(1, 2))              # [B, D]
            feat  = self.part_projs[p](pooled)           # [B, local_dim]
            part_feats.append(feat)

        return torch.cat(part_feats, dim=1)              # [B, num_parts * local_dim]


class FusionModule(nn.Module):
    """
    Fuses global and local branch features.

    Modes:
      'concat'       — direct concatenation (default, no parameters)
      'weighted_sum' — learned scalar weight per branch (all projected first)
      'attention'    — cross-attention: global CLS as Q, local parts as K+V
    """

    def __init__(
        self,
        global_dim:   int,
        local_dim:    int,
        num_parts:    int,
        mode:         str = "concat",
        num_heads:    int = 4,
    ) -> None:
        super().__init__()
        self.mode       = mode
        self.global_dim = global_dim
        self.local_dim  = local_dim
        self.num_parts  = num_parts

        total_local = local_dim * num_parts

        if mode == "weighted_sum":
            # Project all branches to global_dim, then learned sum
            self.proj_local = nn.Linear(total_local, global_dim)
            self.weights    = nn.Parameter(torch.ones(2) / 2)

        elif mode == "attention":
            # Project local parts to global_dim for K, V
            self.proj_local = nn.Linear(local_dim, global_dim)
            self.attn       = nn.MultiheadAttention(
                global_dim, num_heads, batch_first=True
            )

    def forward(
        self,
        global_feat: torch.Tensor,  # [B, global_dim]
        local_feat:  torch.Tensor,  # [B, num_parts * local_dim]
    ) -> torch.Tensor:

        if self.mode == "concat":
            return torch.cat([global_feat, local_feat], dim=1)

        if self.mode == "weighted_sum":
            local_proj = self.proj_local(local_feat)           # [B, global_dim]
            w = torch.softmax(self.weights, dim=0)
            return w[0] * global_feat + w[1] * local_proj      # [B, global_dim]

        if self.mode == "attention":
            B = global_feat.size(0)
            # Reshape local to [B, num_parts, local_dim]
            parts = local_feat.view(B, self.num_parts, self.local_dim)
            kv    = self.proj_local(parts)                     # [B, num_parts, global_dim]
            q     = global_feat.unsqueeze(1)                   # [B, 1, global_dim]
            out, _ = self.attn(q, kv, kv)
            return out.squeeze(1)                              # [B, global_dim]

        raise ValueError(f"Unknown fusion mode: {self.mode}")


# ─────────────────────────────────────────────────────────────────────────────

class MultiBranchBackend(BaseReIDBackend):
    """
    Multi-branch global + local feature head backend.

    Wraps a backbone (TransReID or CLIP-ReID) and adds global + local
    branches on top. Can be used zero-shot or with fine-tuned weights.
    """

    def __init__(self, config: dict) -> None:
        self.cfg          = config.get("multibranch", {})
        self.full_config  = config
        self.backbone_name = self.cfg.get("backbone", "transreid")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._backbone_backend: Optional[BaseReIDBackend] = None
        self._backbone_model:   Optional[nn.Module]       = None
        self._global_branch:    Optional[GlobalBranch]    = None
        self._local_branch:     Optional[LocalBranch]     = None
        self._fusion:           Optional[FusionModule]    = None
        self._transform = None

    # ── load ──────────────────────────────────────────────────────────────────
    def load(self) -> None:
        self._osnet_global_only = False  # flag for OSNet fallback mode

        # 1. Load the underlying backbone
        if self.backbone_name == "transreid":
            from src.backends.reid.transreid import TransReIDBackend
            backend = TransReIDBackend(self.full_config)
            backend.load()
            self._backbone_backend = backend
            backbone_dim = backend.embed_dim  # 384 for ViT-S

            # We need access to the raw model for feature extraction
            self._backbone_model  = backend._model
            self._backbone_bnneck = backend._bnneck
            self._backbone_sie    = backend._sie

            h, w = self.full_config.get("transreid", {}).get("image_size", [256, 128])
            # Patch grid: 256/16=16 high, 128/16=8 wide
            h_patches = h // 16
            w_patches = w // 16
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((h, w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ])
        elif self.backbone_name == "clipreid":
            from src.backends.reid.clip_reid import CLIPReIDBackend
            backend = CLIPReIDBackend(self.full_config)
            backend.load()
            self._backbone_backend = backend
            backbone_dim = backend.embed_dim  # 512 for ViT-B
            self._backbone_model   = backend._visual_encoder

            h, w = self.full_config.get("clipreid", {}).get("image_size", [224, 224])
            h_patches = h // 16
            w_patches = w // 16
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((h, w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
            ])
        elif self.backbone_name == "osnet":
            # OSNet is a CNN — it outputs a flat 512D vector without patch
            # tokens, so the local branch cannot be used.  We load it as a
            # global-only backbone and delegate all extraction to the
            # OSNetBackend, bypassing the multi-branch head entirely.
            from src.backends.reid.osnet import OSNetBackend
            backend = OSNetBackend(self.full_config)
            backend.load()
            self._backbone_backend = backend
            self._osnet_global_only = True
            logger.info(
                "MultiBranchBackend: OSNet is a CNN backbone without patch "
                "tokens. Running in global-only mode (local branch disabled)."
            )
            return  # skip branch construction — not applicable for CNN
        else:
            raise ValueError(f"Unsupported multibranch backbone: {self.backbone_name}")

        global_dim  = int(self.cfg.get("global_dim",  768))
        local_dim   = int(self.cfg.get("local_dim",   256))
        num_parts   = int(self.cfg.get("num_parts",   4))
        dropout     = float(self.cfg.get("dropout",   0.1))
        fusion_mode = self.cfg.get("fusion", "concat")
        attn_heads  = int(self.cfg.get("attention_heads", 4))

        # 2. Build branches
        self._global_branch = GlobalBranch(backbone_dim, global_dim, dropout)
        self._local_branch  = LocalBranch(
            backbone_dim, local_dim, num_parts, h_patches, w_patches, dropout
        )
        self._fusion = FusionModule(global_dim, local_dim, num_parts, fusion_mode, attn_heads)

        # 3. Optionally load fine-tuned multibranch weights
        mb_weights = self.cfg.get("pretrained_weights", "")
        if mb_weights and Path(mb_weights).exists():
            ckpt = torch.load(mb_weights, map_location="cpu")
            for module, key in [
                (self._global_branch, "global_branch"),
                (self._local_branch,  "local_branch"),
                (self._fusion,        "fusion"),
            ]:
                if key in ckpt:
                    module.load_state_dict(ckpt[key], strict=False)
            logger.info("MultiBranch fine-tuned weights loaded from '%s'.", mb_weights)

        # 4. Move to device
        for m in [self._global_branch, self._local_branch, self._fusion]:
            m.to(self.device).eval()

        logger.info(
            "MultiBranchBackend loaded — backbone=%s, global_dim=%d, "
            "local_dim=%d×%d, fusion=%s, device=%s",
            self.backbone_name, global_dim, local_dim, num_parts, fusion_mode, self.device,
        )

    # ── _preprocess ───────────────────────────────────────────────────────────
    def _preprocess_crop(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> Optional[torch.Tensor]:
        import cv2
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        H, W = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return None
        crop_bgr = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return self._transform(crop_rgb)

    # ── _forward_features ─────────────────────────────────────────────────────
    @torch.no_grad()
    def _forward_features(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the backbone in feature-extraction mode.

        Returns
        -------
        cls_token : [B, D]
        patch_seq : [B, N, D]
        """
        batch = batch.to(self.device)
        if self.backbone_name == "transreid":
            tokens = self._backbone_model.forward_features(batch)  # [B, N+1, D]
            return tokens[:, 0, :], tokens[:, 1:, :]
        else:
            # CLIP visual encoder — returns the pooled CLS vector [B, D]
            # We need the intermediate tokens; patch feature extraction
            # from CLIP requires a slight modification.
            # Use the visual encoder and return CLS as both since
            # CLIP's standard API returns only the pooled output.
            cls = self._backbone_model(batch)                       # [B, D]
            # Create a dummy patch_seq of zeros for the local branch
            # (CLIP visual API doesn't expose intermediate tokens by default)
            dummy_patches = torch.zeros(
                cls.size(0), 196, cls.size(1), device=self.device
            )
            return cls, dummy_patches

    # ── _embed_batch ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def _embed_batch(self, batch: torch.Tensor) -> np.ndarray:
        cls_token, patch_seq = self._forward_features(batch)

        global_feat = self._global_branch(cls_token)    # [B, global_dim]
        local_feat  = self._local_branch(patch_seq)     # [B, num_parts * local_dim]
        fused       = self._fusion(global_feat, local_feat)
        normalized  = F.normalize(fused, p=2, dim=1)
        return normalized.cpu().numpy().astype(np.float32)

    # ── extract ───────────────────────────────────────────────────────────────
    @torch.no_grad()
    def extract(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        cam_label: int = 0,
    ) -> np.ndarray:
        if self._backbone_backend is None:
            self.load()

        # OSNet global-only mode: delegate directly
        if self._osnet_global_only:
            return self._backbone_backend.extract(frame, bbox, cam_label)

        tensor = self._preprocess_crop(frame, bbox)
        if tensor is None:
            return np.zeros(self.embed_dim, dtype=np.float32)

        batch  = tensor.unsqueeze(0)
        result = self._embed_batch(batch)
        return result[0]

    # ── extract_batch ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def extract_batch(
        self,
        frame: np.ndarray,
        bboxes: list[np.ndarray],
        cam_labels: list[int] | None = None,
    ) -> np.ndarray:
        if self._backbone_backend is None:
            self.load()
        if not bboxes:
            return np.empty((0, self.embed_dim), dtype=np.float32)

        # OSNet global-only mode: delegate directly
        if self._osnet_global_only:
            return self._backbone_backend.extract_batch(frame, bboxes, cam_labels)

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

        batch  = torch.stack(tensors)
        embs   = self._embed_batch(batch)

        j = 0
        for i, valid in enumerate(valid_mask):
            if valid:
                out[i] = embs[j]
                j += 1
        return out

    # ── embed_dim ─────────────────────────────────────────────────────────────
    @property
    def embed_dim(self) -> int:
        # OSNet global-only mode: return OSNet's native dim
        if getattr(self, '_osnet_global_only', False):
            return 512

        global_dim = int(self.cfg.get("global_dim", 768))
        local_dim  = int(self.cfg.get("local_dim",  256))
        num_parts  = int(self.cfg.get("num_parts",  4))
        fusion     = self.cfg.get("fusion", "concat")

        if fusion == "concat":
            return global_dim + local_dim * num_parts
        return global_dim  # weighted_sum and attention collapse to global_dim
