"""
src/backends/reid/dinov2_lora.py
────────────────────────────────────
DINOv2 Foundation Model Re-ID backend with LoRA (Low-Rank Adaptation).

Uses Meta's DINOv2 ViT backbone (pretrained on 142M images via self-supervised
learning) with parameter-efficient LoRA adapters injected into the Q/V
attention projections. The backbone is FROZEN — only ~0.3% of parameters
are trainable, enabling fine-tuning on small Re-ID datasets without
catastrophic forgetting.

Feature extraction combines:
  - CLS token (global identity representation)
  - GeM pooling over patch tokens (local spatial aggregation)
  → Concatenated and L2-normalized

Reference:
  - Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision"
    TMLR 2024. https://arxiv.org/abs/2304.07193
  - Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
    ICLR 2022. https://arxiv.org/abs/2106.09685
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

_DINOV2_MEAN = (0.485, 0.456, 0.406)
_DINOV2_STD  = (0.229, 0.224, 0.225)


# ─────────────────────────────────────────────────────────────────────────────
# Generalized Mean (GeM) Pooling
# ─────────────────────────────────────────────────────────────────────────────

class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling for spatial patch tokens.

    Standard average pooling (p=1) treats all patches equally.
    GeM with p>1 upweights high-activation patches, producing more
    discriminative features for retrieval / Re-ID tasks.

    Reference: Radenović et al., "Fine-tuning CNN Image Retrieval
               with No Human Annotation", TPAMI 2019.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, D)  patch tokens

        Returns
        -------
        (B, D)  pooled features
        """
        # Clamp to avoid numerical issues with negative values
        x_clamped = x.clamp(min=self.eps)
        pooled = x_clamped.pow(self.p).mean(dim=1).pow(1.0 / self.p)
        return pooled


# ─────────────────────────────────────────────────────────────────────────────
# LoRA Injection Utilities
# ─────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a low-rank adapter.

    The original linear layer is frozen. The LoRA adapter adds a
    low-rank decomposition:  output = W·x + (B·A)·x
    where A ∈ R^{r×d_in}, B ∈ R^{d_out×r}, and r << min(d_in, d_out).

    Only A and B are trainable (~2 × r × d parameters vs d_in × d_out).
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.original = original
        self.rank     = rank
        self.scaling  = alpha / rank

        d_in  = original.in_features
        d_out = original.out_features

        # Freeze original weights
        for p in self.original.parameters():
            p.requires_grad = False

        # Low-rank matrices
        self.lora_A   = nn.Linear(d_in, rank, bias=False)
        self.lora_B   = nn.Linear(rank, d_out, bias=False)
        self.dropout  = nn.Dropout(p=dropout)

        # Initialize A with Kaiming, B with zeros (LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original(x)
        lora_out     = self.lora_B(self.lora_A(self.dropout(x)))
        return original_out + lora_out * self.scaling


def inject_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 8,
    alpha: float = 32.0,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Recursively inject LoRA adapters into target linear layers.

    Parameters
    ----------
    model          : The backbone model (DINOv2 ViT).
    target_modules : List of substrings to match layer names
                     (e.g. ["qkv", "proj"] for attention projections).
    rank           : LoRA rank (lower = fewer params, higher = more capacity).
    alpha          : LoRA scaling factor.
    dropout        : Dropout applied before the low-rank projection.

    Returns
    -------
    The model with LoRA-injected layers.
    """
    targets = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                if any(t in f"{name}.{child_name}" for t in target_modules):
                    targets.append((module, child_name, child))

    replaced = 0
    for module, child_name, child in targets:
        lora_layer = LoRALinear(child, rank, alpha, dropout)
        setattr(module, child_name, lora_layer)
        replaced += 1

    logger.info(
        "LoRA injection complete: %d linear layers adapted (rank=%d, alpha=%.1f).",
        replaced, rank, alpha,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Projection Head
# ─────────────────────────────────────────────────────────────────────────────

class ReIDProjectionHead(nn.Module):
    """
    Projects concatenated CLS + GeM features to the final embedding space.

    Architecture: Linear → BNNeck → ReLU → Dropout → Linear → L2-Norm
    """

    def __init__(
        self,
        in_dim:    int,
        out_dim:   int = 768,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 ReID Backend
# ─────────────────────────────────────────────────────────────────────────────

class DINOv2ReIDBackend(BaseReIDBackend):
    """
    DINOv2 + LoRA Re-ID backend.

    Uses Meta's DINOv2 ViT backbone with LoRA adapters for
    parameter-efficient fine-tuning on vehicle Re-ID datasets.

    Feature pipeline:
      1. Frozen DINOv2 backbone extracts CLS token + patch tokens
      2. LoRA adapters inject task-specific low-rank updates
      3. GeM pooling aggregates patch tokens
      4. CLS + GeM are concatenated and projected
      5. Output is L2-normalized

    Config keys (under 'dinov2'):
      model_name            : str   "dinov2_vits14" | "dinov2_vitb14"
      embed_dim             : int   Output embedding dimension
      image_size            : list  [H, W] input resolution
      lora_rank             : int   LoRA rank (default 8)
      lora_alpha            : float LoRA scaling factor (default 32)
      lora_dropout          : float LoRA dropout (default 0.1)
      lora_target_modules   : list  Layer name substrings to inject LoRA
      pretrained_lora_weights: str  Path to fine-tuned LoRA + head weights
      gem_pooling_p         : float GeM pooling power parameter (default 3.0)
    """

    def __init__(self, config: dict) -> None:
        self.full_config = config
        self.cfg         = config.get("dinov2", {})
        self.device      = torch.device(
            config.get("pipeline", {}).get("device", "cpu")
        )
        self._backbone:   Optional[nn.Module] = None
        self._gem_pool:   Optional[GeMPooling] = None
        self._head:       Optional[ReIDProjectionHead] = None
        self._transform   = None

    # ── load ──────────────────────────────────────────────────────────────────
    def load(self) -> None:
        model_name = self.cfg.get("model_name", "dinov2_vits14")
        h, w = self.cfg.get("image_size", [224, 224])

        # 1. Load frozen DINOv2 backbone via torch.hub
        logger.info("Loading DINOv2 backbone '%s' via torch.hub…", model_name)
        try:
            backbone = torch.hub.load(
                "facebookresearch/dinov2", model_name, pretrained=True
            )
        except Exception as e:
            logger.warning(
                "torch.hub load failed (%s). Trying local timm fallback…", e
            )
            import timm
            timm_name = model_name.replace("dinov2_", "vit_").replace("14", "_patch14_dinov2")
            backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)

        # Freeze all backbone parameters
        for param in backbone.parameters():
            param.requires_grad = False

        # 2. Inject LoRA adapters
        lora_rank    = int(self.cfg.get("lora_rank",    8))
        lora_alpha   = float(self.cfg.get("lora_alpha", 32.0))
        lora_dropout = float(self.cfg.get("lora_dropout", 0.1))
        target_modules = self.cfg.get("lora_target_modules", ["qkv"])

        backbone = inject_lora(
            backbone,
            target_modules = target_modules,
            rank           = lora_rank,
            alpha          = lora_alpha,
            dropout        = lora_dropout,
        )

        self._backbone = backbone.to(self.device)

        # 3. Determine backbone output dimension
        # DINOv2 models expose embed_dim attribute
        backbone_dim = getattr(backbone, "embed_dim", 384)
        logger.info("DINOv2 backbone dim: %d", backbone_dim)

        # 4. Build GeM pooling and projection head
        gem_p = float(self.cfg.get("gem_pooling_p", 3.0))
        self._gem_pool = GeMPooling(p=gem_p).to(self.device)

        out_dim = int(self.cfg.get("embed_dim", backbone_dim * 2))
        # Input to head = CLS (backbone_dim) + GeM (backbone_dim)
        self._head = ReIDProjectionHead(
            in_dim  = backbone_dim * 2,
            out_dim = out_dim,
            dropout = float(self.cfg.get("head_dropout", 0.1)),
        ).to(self.device)

        # 5. Load fine-tuned LoRA + head weights if available
        weights_path = Path(self.cfg.get("pretrained_lora_weights", ""))
        if weights_path.exists():
            ckpt = torch.load(weights_path, map_location="cpu")
            # Load LoRA weights (only the lora_A, lora_B parameters)
            if "backbone" in ckpt:
                missing, unexpected = self._backbone.load_state_dict(
                    ckpt["backbone"], strict=False
                )
                logger.info(
                    "DINOv2 LoRA weights loaded (missing=%d, unexpected=%d).",
                    len(missing), len(unexpected),
                )
            if "head" in ckpt:
                self._head.load_state_dict(ckpt["head"], strict=False)
                logger.info("DINOv2 projection head weights loaded.")
            if "gem_pool" in ckpt:
                self._gem_pool.load_state_dict(ckpt["gem_pool"], strict=False)
        else:
            logger.warning(
                "DINOv2 LoRA weights not found at '%s'. "
                "Using zero-shot DINOv2 features — Re-ID accuracy will be reduced.",
                weights_path,
            )

        # Set eval mode
        self._backbone.eval()
        self._head.eval()

        # 6. Build preprocessing transform
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_DINOV2_MEAN, std=_DINOV2_STD),
        ])

        # 7. TensorRT Acceleration
        if self.full_config.get("strategy", {}).get("use_tensorrt", False) and "cuda" in str(self.device):
            input_shape = (self.cfg.get("batch_size", 32), 3, h, w)

            class DINOv2Wrapper(nn.Module):
                def __init__(self, backbone, gem, head):
                    super().__init__()
                    self.backbone = backbone
                    self.gem = gem
                    self.head = head

                def forward(self, x):
                    features = self.backbone.forward_features(x)
                    if isinstance(features, dict):
                        cls_token = features.get("x_norm_clstoken", features.get("cls_token"))
                        patch_tokens = features.get("x_norm_patchtokens", features.get("patch_tokens"))
                    else:
                        cls_token = features[:, 0, :]
                        patch_tokens = features[:, 1:, :]
                    gem_out = self.gem(patch_tokens)
                    fused = torch.cat([cls_token, gem_out], dim=1)
                    return self.head(fused)

            wrapper = DINOv2Wrapper(self._backbone, self._gem_pool, self._head)
            compiled = self.compile_tensorrt(
                wrapper, input_shape=input_shape, model_name="dinov2_reid"
            )
            self._compiled_model = compiled
        else:
            self._compiled_model = None

        # Count trainable params
        trainable   = sum(p.numel() for p in self._backbone.parameters() if p.requires_grad)
        total       = sum(p.numel() for p in self._backbone.parameters())
        head_params = sum(p.numel() for p in self._head.parameters())
        logger.info(
            "DINOv2ReIDBackend loaded — model=%s, backbone_dim=%d, "
            "embed_dim=%d, trainable=%.1fK/%.1fM (%.2f%%), head=%.1fK, device=%s",
            model_name, backbone_dim, out_dim,
            trainable / 1e3, total / 1e6, 100 * trainable / total,
            head_params / 1e3, self.device,
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

    # ── _forward ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _forward(self, batch: torch.Tensor) -> np.ndarray:
        batch = batch.to(self.device)

        if self._compiled_model is not None:
            feat = self._compiled_model(batch)
            feat = F.normalize(feat, p=2, dim=1)
            return feat.cpu().numpy().astype(np.float32)

        # Extract features through DINOv2
        # DINOv2 models have different APIs depending on how they're loaded
        if hasattr(self._backbone, "forward_features"):
            features = self._backbone.forward_features(batch)
            if isinstance(features, dict):
                cls_token    = features.get("x_norm_clstoken", features.get("cls_token"))
                patch_tokens = features.get("x_norm_patchtokens", features.get("patch_tokens"))
            else:
                cls_token    = features[:, 0, :]
                patch_tokens = features[:, 1:, :]
        else:
            # Fallback: get intermediate features
            out = self._backbone(batch)
            if isinstance(out, torch.Tensor) and out.dim() == 2:
                # Model returned pooled features only
                feat = self._head(torch.cat([out, out], dim=1))
                feat = F.normalize(feat, p=2, dim=1)
                return feat.cpu().numpy().astype(np.float32)
            cls_token    = out[:, 0, :]
            patch_tokens = out[:, 1:, :]

        # GeM pool over patch tokens
        gem_feat = self._gem_pool(patch_tokens)

        # Concatenate CLS + GeM and project
        fused = torch.cat([cls_token, gem_feat], dim=1)
        emb   = self._head(fused)
        emb   = F.normalize(emb, p=2, dim=1)

        return emb.cpu().numpy().astype(np.float32)

    # ── extract ───────────────────────────────────────────────────────────────
    @torch.no_grad()
    def extract(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        cam_label: int = 0,
    ) -> np.ndarray:
        if self._backbone is None:
            self.load()
        tensor = self._preprocess_crop(frame, bbox)
        if tensor is None:
            return np.zeros(self.embed_dim, dtype=np.float32)
        batch = tensor.unsqueeze(0)
        return self._forward(batch)[0]

    # ── extract_batch ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def extract_batch(
        self,
        frame: np.ndarray,
        bboxes: list[np.ndarray],
        cam_labels: list[int] | None = None,
    ) -> np.ndarray:
        if self._backbone is None:
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

        batch = torch.stack(tensors)
        embs  = self._forward(batch)

        j = 0
        for i, valid in enumerate(valid_mask):
            if valid:
                out[i] = embs[j]
                j += 1
        return out

    # ── embed_dim ─────────────────────────────────────────────────────────────
    @property
    def embed_dim(self) -> int:
        if self._head is not None:
            return self._head.out_dim
        # Calculate from config if model not loaded yet
        model_name = self.cfg.get("model_name", "dinov2_vits14")
        backbone_dim = 384
        if "vitb" in model_name:
            backbone_dim = 768
        elif "vitl" in model_name:
            backbone_dim = 1024
        elif "vitg" in model_name:
            backbone_dim = 1536
        return int(self.cfg.get("embed_dim", backbone_dim * 2))

    @property
    def backbone_dim(self) -> int:
        """DINOv2 CLS token dimension."""
        model_name = self.cfg.get("model_name", "dinov2_vits14")
        if "vitb" in model_name:
            return 768
        elif "vitl" in model_name:
            return 1024
        elif "vitg" in model_name:
            return 1536
        return 384
