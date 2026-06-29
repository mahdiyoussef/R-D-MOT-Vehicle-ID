"""
src/backends/reid/base.py
─────────────────────────
Abstract base class for all Re-ID feature extractor backends.
All backends must inherit from this class and implement its interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseReIDBackend(ABC):
    """Abstract interface for all Re-ID feature extractors."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Called once at pipeline build time."""

    @abstractmethod
    def extract(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        cam_label: int = 0,
    ) -> np.ndarray:
        """
        Extract a single L2-normalized embedding.

        Parameters
        ----------
        frame     : Full BGR frame (H, W, 3) uint8
        bbox      : [x1, y1, x2, y2] float32
        cam_label : Camera index for SIE (ignored by backends that don't use it)

        Returns
        -------
        np.ndarray float32, shape (embed_dim,), L2-normalized
        """

    @abstractmethod
    def extract_batch(
        self,
        frame: np.ndarray,
        bboxes: list[np.ndarray],
        cam_labels: list[int] | None = None,
    ) -> np.ndarray:
        """
        Batch extraction — must be more efficient than looping extract().

        Returns
        -------
        np.ndarray float32, shape (N, embed_dim), each row L2-normalized
        """

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Output embedding dimensionality."""

    @property
    def backbone_dim(self) -> int:
        """Dimensionality of the backbone CLS token (defaults to embed_dim if undivided)."""
        return self.embed_dim

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def compile_tensorrt(self, model, input_shape=(32, 3, 256, 128), model_name="model"):
        """
        Compiles a PyTorch model into a TensorRT engine using torch_tensorrt.
        Caches the compiled engine to disk to speed up subsequent loads.
        """
        import torch
        import logging
        logger = logging.getLogger(__name__)

        cache_dir = torch.hub.get_dir() + "/tensorrt_cache"
        import os
        os.makedirs(cache_dir, exist_ok=True)
        trt_path = f"{cache_dir}/{model_name}_trt.ts"

        if os.path.exists(trt_path):
            logger.info("Loading cached TensorRT engine from %s", trt_path)
            try:
                # torch.jit.load works for torch_tensorrt compiled models
                trt_model = torch.jit.load(trt_path)
                return trt_model
            except Exception as e:
                logger.warning("Failed to load cached TensorRT engine: %s", e)

        logger.info("Compiling model to TensorRT... This may take a few minutes. (Shape: %s)", input_shape)
        try:
            import torch_tensorrt
            # Ensure model is in eval mode
            model.eval()
            
            # Use fixed shape for compilation since dynamic shapes can be tricky with some backbones
            # We compile for the exact max batch_size used by the embedder
            inputs = [
                torch_tensorrt.Input(
                    min_shape=[1, *input_shape[1:]],
                    opt_shape=[max(1, input_shape[0]//2), *input_shape[1:]],
                    max_shape=input_shape,
                    dtype=torch.float16 if next(model.parameters()).dtype == torch.float16 else torch.float32
                )
            ]
            
            trt_model = torch_tensorrt.compile(
                model,
                inputs=inputs,
                truncate_long_and_double=True
            )
            
            # Save the compiled engine for future use
            torch.jit.save(trt_model, trt_path)
            logger.info("TensorRT compilation successful! Engine saved to %s", trt_path)
            return trt_model
            
        except Exception as e:
            logger.error("TensorRT compilation failed: %s", e)
            logger.warning("Falling back to standard PyTorch execution.")
            return model
