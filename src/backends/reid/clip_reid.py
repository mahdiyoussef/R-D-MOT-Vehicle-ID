"""
src/backends/reid/clipreid_backend.py
──────────────────────────────────────
CLIP-ReID backend using OpenAI's CLIP ViT-B/16 image encoder,
fine-tuned for vehicle re-identification via two-stage prompt learning.

At inference time only the image encoder branch is used.
The text / prompt-learner branch is baked into the saved weights and
is NOT needed during inference.

Reference: Shuting He et al., "CLIP-ReID: Exploiting Vision-Language Model
           for Image Re-Identification without Concrete Text Labels"
           AAAI 2023. https://arxiv.org/abs/2211.13977

Dependency:
    pip install git+https://github.com/openai/CLIP.git
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.backends.reid.base import BaseReIDBackend

logger = logging.getLogger(__name__)

# CLIP normalisation constants (different from ImageNet)
_CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


# ─────────────────────────────────────────────────────────────────────────────

def download_clipreid_weights(config: dict) -> Path:
    """
    Attempt to download CLIP-ReID fine-tuned weights in order:
      1. Check local path → return immediately if present.
      2. gdown via Google Drive ID from config.
      3. HuggingFace Hub (hf_hub_download).
      4. Print manual download instructions.
    """
    weights_path = Path(config.get("clipreid", {}).get(
        "pretrained_weights", "models/reid/clipreid_veri776.pth"
    ))
    if weights_path.exists():
        return weights_path

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    gdrive_id = config.get("clipreid", {}).get("weights_gdrive_id", "")

    # Attempt 1: gdown
    if gdrive_id and gdrive_id != "1PLACE_HOLDER_CLIPREID_ID":
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={gdrive_id}"
            gdown.download(url, str(weights_path), quiet=False)
            if weights_path.exists():
                logger.info("CLIP-ReID weights downloaded via gdown → %s", weights_path)
                return weights_path
        except Exception as e:
            logger.warning("gdown download failed: %s", e)

    # Attempt 2: HuggingFace Hub
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id="CLIP-ReID/vehicle-reid",
            filename="clipreid_veri776.pth",
            local_dir=str(weights_path.parent),
        )
        import shutil
        shutil.move(local, str(weights_path))
        logger.info("CLIP-ReID weights downloaded via HF Hub → %s", weights_path)
        return weights_path
    except Exception as e:
        logger.warning("HF Hub download failed: %s", e)

    # Attempt 3: Manual instructions
    logger.error(
        "\n" + "="*60 + "\n"
        "CLIP-ReID weights not found. Please download manually:\n"
        "  Source: https://github.com/Syliz517/CLIP-ReID\n"
        "  Target: %s\n"
        + "="*60, weights_path,
    )
    return weights_path


# ─────────────────────────────────────────────────────────────────────────────

class CLIPReIDBackend(BaseReIDBackend):
    """
    CLIP-ReID backend.

    Uses the CLIP ViT-B/16 image encoder with optional fine-tuned
    weights (Stage 2 of CLIP-ReID). At inference time only the visual
    encoder runs — the text branch is not needed.

    Outputs a 512-dimensional L2-normalized embedding per vehicle crop.
    """

    def __init__(self, config: dict) -> None:
        self.full_config = config
        self.cfg    = config.get("clipreid", {})
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._visual_encoder: Optional[torch.nn.Module] = None
        self._transform = None

    # ── load ──────────────────────────────────────────────────────────────────
    def load(self) -> None:
        try:
            import clip
        except ImportError:
            raise ImportError(
                "CLIP is not installed. Install with:\n"
                "  pip install git+https://github.com/openai/CLIP.git"
            )

        clip_model_name = self.cfg.get("clip_model", "ViT-B/16")
        logger.info("Loading CLIP model '%s'…", clip_model_name)

        # Load full CLIP model then extract only the visual encoder
        clip_model, _ = clip.load(clip_model_name, device=self.device)
        self._visual_encoder = clip_model.visual

        # Optionally load fine-tuned weights for the image encoder
        weights_path = Path(self.cfg.get("pretrained_weights", ""))
        if not weights_path.exists():
            logger.info("Fine-tuned CLIP-ReID weights not found. Attempting download…")
            weights_path = download_clipreid_weights({"clipreid": self.cfg})

        if weights_path.exists():
            ckpt = torch.load(weights_path, map_location="cpu")
            # Unwrap checkpoint wrapper keys
            for key in ("model", "state_dict", "model_state_dict", "visual"):
                if isinstance(ckpt, dict) and key in ckpt:
                    ckpt = ckpt[key]
                    break
            result = self._visual_encoder.load_state_dict(ckpt, strict=False)
            logger.info(
                "CLIP-ReID fine-tuned weights loaded from '%s'  "
                "(missing=%d, unexpected=%d)",
                weights_path,
                len(result.missing_keys),
                len(result.unexpected_keys),
            )
        else:
            logger.warning(
                "CLIP-ReID fine-tuned weights unavailable. "
                "Using zero-shot CLIP image encoder — Re-ID accuracy will be reduced."
            )

        self._visual_encoder.to(self.device).float().eval()

        # Build preprocessing transform (CLIP native 224×224)
        h, w = self.cfg.get("image_size", [224, 224])
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])

        # TensorRT Acceleration
        if self.full_config.get("strategy", {}).get("use_tensorrt", False) and "cuda" in str(self.device):
            input_shape = (self.cfg.get("batch_size", 32), 3, h, w)
            
            # Use fixed dummy class wrapper since CLIP visual encoder uses a custom module hierarchy
            class CLIPWrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    return self.m(x)
                    
            wrapper = CLIPWrapper(self._visual_encoder)
            compiled_wrapper = self.compile_tensorrt(
                wrapper,
                input_shape=input_shape,
                model_name="clipreid_vitb16"
            )
            # Rebind
            self._visual_encoder = compiled_wrapper

        logger.info(
            "CLIPReIDBackend loaded — model=%s, device=%s",
            clip_model_name, self.device,
        )

    # ── _preprocess ───────────────────────────────────────────────────────────
    def _preprocess_crop(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> Optional[torch.Tensor]:
        """Crop, validate, and transform a single bbox."""
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

    # ── extract ───────────────────────────────────────────────────────────────
    @torch.no_grad()
    def extract(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        cam_label: int = 0,
    ) -> np.ndarray:
        if self._visual_encoder is None:
            self.load()

        tensor = self._preprocess_crop(frame, bbox)
        if tensor is None:
            return np.zeros(self.embed_dim, dtype=np.float32)

        batch = tensor.unsqueeze(0).to(self.device)
        feat  = self._visual_encoder(batch)
        feat  = F.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy()[0].astype(np.float32)

    # ── extract_batch ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def extract_batch(
        self,
        frame: np.ndarray,
        bboxes: list[np.ndarray],
        cam_labels: list[int] | None = None,
    ) -> np.ndarray:
        if self._visual_encoder is None:
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

        batch = torch.stack(tensors).to(self.device)
        feat  = self._visual_encoder(batch)
        feat  = F.normalize(feat, p=2, dim=1)
        embs  = feat.cpu().numpy().astype(np.float32)

        j = 0
        for i, valid in enumerate(valid_mask):
            if valid:
                out[i] = embs[j]
                j += 1
        return out

    # ── embed_dim ─────────────────────────────────────────────────────────────
    @property
    def embed_dim(self) -> int:
        return int(self.cfg.get("embed_dim", 512))
