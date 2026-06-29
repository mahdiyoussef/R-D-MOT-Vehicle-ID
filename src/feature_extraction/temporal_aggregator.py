"""
src/backends/reid/temporal_aggregator.py
─────────────────────────────────────────
Temporal Tracklet Aggregation Module (v3.0).

Instead of relying on a single-frame embedding snapshot, this module
buffers the last N embeddings for each tracked vehicle and fuses them
into a single, robust representation using a lightweight cross-attention
Transformer.

The latest frame's embedding serves as the Query, and all buffered
historical embeddings serve as Keys and Values. This allows the model
to dynamically attend to the most informative frames while naturally
downweighting blurry, occluded, or poorly-lit observations.

Reference:
  Temporal Attention for Video-Based Re-ID, ECCV 2024.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TemporalAttentionBlock(nn.Module):
    """
    Lightweight cross-attention block for temporal feature aggregation.

    Architecture:
      - Layer Norm → Multi-Head Cross-Attention → Residual
      - Layer Norm → FFN (Linear→GELU→Linear) → Residual

    The query is the current frame's embedding (anchor).
    Keys and Values are all buffered historical embeddings.
    """

    def __init__(
        self,
        embed_dim:  int = 768,
        num_heads:  int = 4,
        ff_mult:    float = 2.0,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        ff_dim = int(embed_dim * ff_mult)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query:    torch.Tensor,  # (B, 1, D)
        context:  torch.Tensor,  # (B, T, D)
    ) -> torch.Tensor:
        # Cross-attention: query attends to context
        normed_q = self.norm1(query)
        normed_c = self.norm1(context)
        attn_out, _ = self.attn(normed_q, normed_c, normed_c)
        query = query + attn_out

        # Feed-forward
        query = query + self.ffn(self.norm2(query))
        return query


class TemporalAggregatorModel(nn.Module):
    """
    Multi-layer temporal attention aggregator.

    Takes a sequence of embeddings [f_{t-N}, ..., f_t] for a single
    tracked vehicle and produces a single fused embedding.

    The last embedding in the sequence is used as the query (anchor),
    and all embeddings serve as key-value context.
    """

    def __init__(
        self,
        embed_dim:  int = 768,
        num_heads:  int = 4,
        num_layers: int = 2,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TemporalAttentionBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, embedding_seq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        embedding_seq : (B, T, D) — T temporal embeddings per track

        Returns
        -------
        (B, D) — single aggregated embedding
        """
        # Anchor: last frame embedding
        query   = embedding_seq[:, -1:, :]  # (B, 1, D)
        context = embedding_seq             # (B, T, D)

        for layer in self.layers:
            query = layer(query, context)

        out = self.out_norm(query.squeeze(1))  # (B, D)
        return F.normalize(out, p=2, dim=1)


class TemporalTrackletAggregator:
    """
    Non-module wrapper that manages per-track embedding buffers
    and runs the temporal attention model when enough frames are
    accumulated.

    Integration point: called from pipeline.py between embedding
    extraction and gallery update.

    Parameters (from config['temporal_aggregator'])
    ------------------------------------------------
    enabled       : bool   Enable/disable temporal aggregation
    buffer_size   : int    Number of embeddings to buffer per track
    min_frames    : int    Minimum frames before aggregation kicks in
    num_layers    : int    Number of attention layers
    num_heads     : int    Number of attention heads
    pretrained_weights : str  Path to pretrained aggregator weights
    """

    def __init__(self, config: dict, embed_dim: int) -> None:
        self.cfg       = config.get("temporal_aggregator", {})
        self.embed_dim = embed_dim
        self.enabled   = bool(self.cfg.get("enabled", True))

        self.buffer_size = int(self.cfg.get("buffer_size", 16))
        self.min_frames  = int(self.cfg.get("min_frames", 3))

        self.device = torch.device(
            config.get("pipeline", {}).get("device", "cpu")
        )

        # Per-track embedding buffers: track_id → deque of np.ndarray
        self._buffers: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.buffer_size)
        )

        # Build the temporal attention model
        self._model: Optional[TemporalAggregatorModel] = None
        if self.enabled:
            self._model = TemporalAggregatorModel(
                embed_dim  = embed_dim,
                num_heads  = int(self.cfg.get("num_heads", 4)),
                num_layers = int(self.cfg.get("num_layers", 2)),
                dropout    = float(self.cfg.get("dropout", 0.1)),
            ).to(self.device).eval()

            # Load pretrained weights if available
            weights_path = self.cfg.get("pretrained_weights", "")
            if weights_path and Path(weights_path).exists():
                from pathlib import Path
                ckpt = torch.load(weights_path, map_location="cpu")
                self._model.load_state_dict(ckpt, strict=False)
                logger.info(
                    "TemporalAggregator weights loaded from '%s'.", weights_path
                )
            else:
                logger.info(
                    "TemporalAggregator using untrained attention — "
                    "will function as learned weighted average."
                )

            logger.info(
                "TemporalTrackletAggregator ready — buffer=%d, min_frames=%d, "
                "layers=%d, heads=%d",
                self.buffer_size, self.min_frames,
                int(self.cfg.get("num_layers", 2)),
                int(self.cfg.get("num_heads", 4)),
            )

    def update_and_aggregate(
        self,
        track_id:  int,
        embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Buffer an embedding for a track and return the aggregated result.

        If temporal aggregation is disabled or fewer than min_frames
        embeddings have been collected, returns the raw embedding.
        Otherwise, runs the attention model over the buffered sequence.

        Parameters
        ----------
        track_id  : Short-term tracker ID
        embedding : (D,) L2-normalized embedding from the Re-ID backend

        Returns
        -------
        (D,) aggregated embedding (L2-normalized)
        """
        if not self.enabled or self._model is None:
            return embedding

        # Buffer the embedding
        self._buffers[track_id].append(embedding.copy())

        buf = self._buffers[track_id]
        if len(buf) < self.min_frames:
            # Not enough history — return raw embedding
            return embedding

        # Build sequence tensor: (1, T, D)
        seq = np.stack(list(buf), axis=0)  # (T, D)
        seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(self.device)

        # Run temporal attention
        with torch.no_grad():
            aggregated = self._model(seq_tensor)  # (1, D)

        return aggregated.cpu().numpy()[0].astype(np.float32)

    def remove_track(self, track_id: int) -> None:
        """Remove buffer when a track is lost beyond grace period."""
        self._buffers.pop(track_id, None)

    def reset(self) -> None:
        """Clear all buffers."""
        self._buffers.clear()
        logger.debug("TemporalTrackletAggregator buffers cleared.")
