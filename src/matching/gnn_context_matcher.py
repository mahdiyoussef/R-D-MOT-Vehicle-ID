"""
src/backends/matching/gnn_matcher.py
─────────────────────────────────────
Graph Neural Network Context-Aware Association Matcher (v3.0).

Augments the standard Hungarian assignment with a lightweight Graph
Attention Network (GATv2) that propagates spatial-temporal context
between detection nodes and gallery nodes before computing affinity.

Pipeline:
  1. Build bipartite graph: detections ↔ gallery entries
  2. Initialize node features from ReID embeddings + spatial encoding
  3. Run K rounds of GATv2 message passing
  4. Compute edge affinity scores from refined node features
  5. Solve Hungarian on GNN-refined cost matrix

When vehicles travel in groups (convoys, platoons), confidently matched
neighbors provide contextual cues that help identify occluded or
ambiguous vehicles — something pairwise cosine similarity cannot capture.

Falls back to standard Hungarian matching if:
  - torch_geometric is not installed
  - GNN weights are unavailable
  - An error occurs during message passing

Reference:
  NOWA-MOT: "Neighbor-Guided Tracking", MDPI Sensors 2025.
  Braso & Leal-Taixe, "Learning a Neural Solver for MOT", CVPR 2020.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# Try to import PyTorch and torch_geometric; fall back gracefully
_HAS_TORCH = False
_HAS_PYGEOM = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    pass

try:
    from torch_geometric.nn import GATv2Conv
    _HAS_PYGEOM = True
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Spatial-Temporal Edge Encoder
# ─────────────────────────────────────────────────────────────────────────────

if _HAS_TORCH:
    class EdgeEncoder(nn.Module):
        """
        Encodes spatial and temporal relationships between nodes
        as edge features for the GNN.

        Input per edge: [Δcx, Δcy, Δw, Δh, Δt, IoU]  (6-dim)
        Output: edge_dim-dimensional encoding
        """

        def __init__(self, edge_input_dim: int = 6, edge_dim: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(edge_input_dim, edge_dim),
                nn.ReLU(inplace=True),
                nn.Linear(edge_dim, edge_dim),
                nn.ReLU(inplace=True),
            )

        def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
            return self.net(edge_features)

    # ─────────────────────────────────────────────────────────────────────────
    # GNN Message-Passing Layers
    # ─────────────────────────────────────────────────────────────────────────

    class GNNAssociationLayer(nn.Module):
        """
        Single GNN layer using GATv2 (Graph Attention Network v2).

        Updates node features by aggregating from neighbors:
          h_i^{l+1} = MLP(h_i^l + Σ_{j∈N(i)} α_{ij} · h_j^l)

        GATv2 computes attention dynamically based on both source
        and target node features, unlike GATv1 which is limited to
        static attention.
        """

        def __init__(
            self,
            node_dim:  int = 256,
            num_heads: int = 4,
            dropout:   float = 0.1,
        ) -> None:
            super().__init__()
            assert node_dim % num_heads == 0
            self.conv = GATv2Conv(
                in_channels  = node_dim,
                out_channels = node_dim // num_heads,
                heads        = num_heads,
                dropout      = dropout,
                concat       = True,
            )
            self.norm = nn.LayerNorm(node_dim)
            self.ffn  = nn.Sequential(
                nn.Linear(node_dim, node_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(node_dim * 2, node_dim),
                nn.Dropout(dropout),
            )
            self.norm2 = nn.LayerNorm(node_dim)

        def forward(
            self,
            x:          torch.Tensor,   # (num_nodes, node_dim)
            edge_index: torch.Tensor,   # (2, num_edges)
        ) -> torch.Tensor:
            # GATv2 message passing + residual
            x = x + self.conv(self.norm(x), edge_index)
            # FFN + residual
            x = x + self.ffn(self.norm2(x))
            return x

    class GNNAssociationLayerFallback(nn.Module):
        """
        Fallback GNN layer when torch_geometric is not installed.
        Uses simple attention-based message passing on a dense adjacency.
        """

        def __init__(
            self,
            node_dim:  int = 256,
            num_heads: int = 4,
            dropout:   float = 0.1,
        ) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(
                node_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.norm1 = nn.LayerNorm(node_dim)
            self.norm2 = nn.LayerNorm(node_dim)
            self.ffn = nn.Sequential(
                nn.Linear(node_dim, node_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(node_dim * 2, node_dim),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (1, N, D) — all nodes as a single batch."""
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
            return x

    # ─────────────────────────────────────────────────────────────────────────
    # Edge Classifier
    # ─────────────────────────────────────────────────────────────────────────

    class EdgeClassifier(nn.Module):
        """
        Predicts affinity scores between detection-gallery pairs
        from their refined node features.

        Input per edge: [h_det || h_gal || edge_feat]
        Output: scalar affinity ∈ [0, 1]
        """

        def __init__(self, node_dim: int = 256, edge_dim: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(node_dim * 2 + edge_dim, node_dim),
                nn.ReLU(inplace=True),
                nn.Linear(node_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            det_feats: torch.Tensor,   # (N, node_dim)
            gal_feats: torch.Tensor,   # (M, node_dim)
            edge_feats: torch.Tensor,  # (N*M, edge_dim)
        ) -> torch.Tensor:
            """Returns (N, M) affinity matrix."""
            N = det_feats.size(0)
            M = gal_feats.size(0)

            # Expand for pairwise combination
            det_exp = det_feats.unsqueeze(1).expand(N, M, -1)  # (N, M, D)
            gal_exp = gal_feats.unsqueeze(0).expand(N, M, -1)  # (N, M, D)
            edge_r  = edge_feats.view(N, M, -1)                # (N, M, E)

            combined = torch.cat([det_exp, gal_exp, edge_r], dim=2)  # (N, M, 2D+E)
            scores   = self.net(combined).squeeze(-1)                # (N, M)
            return scores

    # ─────────────────────────────────────────────────────────────────────────
    # Full GNN Model
    # ─────────────────────────────────────────────────────────────────────────

    class GNNAssociationModel(nn.Module):
        """
        Complete GNN association model.

        Architecture:
          1. Project Re-ID embeddings to node_dim
          2. K rounds of GNN message passing
          3. Edge classifier predicts affinities
        """

        def __init__(
            self,
            embed_dim:    int = 768,
            node_dim:     int = 256,
            edge_dim:     int = 64,
            num_layers:   int = 2,
            num_heads:    int = 4,
            dropout:      float = 0.1,
        ) -> None:
            super().__init__()
            self.node_proj = nn.Sequential(
                nn.Linear(embed_dim, node_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(node_dim),
            )

            self.edge_encoder = EdgeEncoder(6, edge_dim)

            if _HAS_PYGEOM:
                self.gnn_layers = nn.ModuleList([
                    GNNAssociationLayer(node_dim, num_heads, dropout)
                    for _ in range(num_layers)
                ])
                self._use_pygeom = True
            else:
                self.gnn_layers = nn.ModuleList([
                    GNNAssociationLayerFallback(node_dim, num_heads, dropout)
                    for _ in range(num_layers)
                ])
                self._use_pygeom = False

            self.edge_classifier = EdgeClassifier(node_dim, edge_dim)

        def forward(
            self,
            det_embs:    torch.Tensor,   # (N, embed_dim)
            gal_embs:    torch.Tensor,   # (M, embed_dim)
            edge_feats:  torch.Tensor,   # (N*M, 6) raw spatial-temporal features
        ) -> torch.Tensor:
            """Returns (N, M) affinity matrix."""
            N = det_embs.size(0)
            M = gal_embs.size(0)

            # Project to node space
            det_nodes = self.node_proj(det_embs)   # (N, node_dim)
            gal_nodes = self.node_proj(gal_embs)   # (M, node_dim)

            # Encode edge features
            edge_encoded = self.edge_encoder(edge_feats)  # (N*M, edge_dim)

            # Message passing
            all_nodes = torch.cat([det_nodes, gal_nodes], dim=0)  # (N+M, node_dim)

            if self._use_pygeom:
                # Build fully-connected bipartite edge_index
                det_idx = torch.arange(N, device=det_embs.device)
                gal_idx = torch.arange(N, N + M, device=det_embs.device)
                src = det_idx.repeat_interleave(M)
                dst = gal_idx.repeat(N)
                edge_index = torch.stack([
                    torch.cat([src, dst]),
                    torch.cat([dst, src]),
                ], dim=0)

                for layer in self.gnn_layers:
                    all_nodes = layer(all_nodes, edge_index)
            else:
                # Fallback: dense attention over all nodes
                all_nodes_seq = all_nodes.unsqueeze(0)  # (1, N+M, D)
                for layer in self.gnn_layers:
                    all_nodes_seq = layer(all_nodes_seq)
                all_nodes = all_nodes_seq.squeeze(0)

            # Split back
            det_refined = all_nodes[:N]
            gal_refined = all_nodes[N:]

            # Edge classification
            affinities = self.edge_classifier(det_refined, gal_refined, edge_encoded)
            return affinities


# ─────────────────────────────────────────────────────────────────────────────
# GNN Matcher (Public API)
# ─────────────────────────────────────────────────────────────────────────────

class GNNContextMatcher:
    """
    Graph Neural Network context-aware matcher.

    Augments the standard Hungarian algorithm with learned spatial-temporal
    context propagation. Falls back gracefully to standard cosine + Hungarian
    when GNN components are unavailable.

    Config keys (under 'gnn_matcher'):
      enabled              : bool   Enable GNN augmentation
      num_layers           : int    GNN message passing rounds (default 2)
      node_dim             : int    Internal node feature dimension
      num_heads            : int    GAT attention heads
      fallback_to_hungarian: bool   Fall back on error (default True)
      pretrained_weights   : str    Path to pretrained GNN weights
      cosine_weight        : float  Blend: w*GNN + (1-w)*cosine (default 0.6)
      threshold            : float  Min affinity to accept match
    """

    def __init__(self, config: dict, embed_dim: int = 768) -> None:
        self.cfg       = config.get("gnn_matcher", {})
        self.enabled   = bool(self.cfg.get("enabled", True)) and _HAS_TORCH
        self.threshold = float(self.cfg.get("threshold", 0.45))
        self.cosine_weight = float(self.cfg.get("cosine_weight", 0.6))
        self.fallback  = bool(self.cfg.get("fallback_to_hungarian", True))

        self._model: Optional[object] = None

        if self.enabled and _HAS_TORCH:
            device = config.get("pipeline", {}).get("device", "cpu")
            self.device = torch.device(device)

            self._model = GNNAssociationModel(
                embed_dim  = embed_dim,
                node_dim   = int(self.cfg.get("node_dim", 256)),
                edge_dim   = int(self.cfg.get("edge_dim", 64)),
                num_layers = int(self.cfg.get("num_layers", 2)),
                num_heads  = int(self.cfg.get("num_heads", 4)),
                dropout    = float(self.cfg.get("dropout", 0.1)),
            ).to(self.device).eval()

            # Load pretrained weights
            from pathlib import Path
            weights_path = self.cfg.get("pretrained_weights", "")
            if weights_path and Path(weights_path).exists():
                ckpt = torch.load(weights_path, map_location="cpu")
                self._model.load_state_dict(ckpt, strict=False)
                logger.info("GNN matcher weights loaded from '%s'.", weights_path)
            else:
                logger.info(
                    "GNN matcher using untrained weights — "
                    "will blend with cosine similarity for stability."
                )

            pygeom_status = "torch_geometric" if _HAS_PYGEOM else "dense-attention fallback"
            logger.info(
                "GNNContextMatcher ready — layers=%d, heads=%d, "
                "node_dim=%d, backend=%s",
                int(self.cfg.get("num_layers", 2)),
                int(self.cfg.get("num_heads", 4)),
                int(self.cfg.get("node_dim", 256)),
                pygeom_status,
            )
        elif not _HAS_TORCH:
            logger.warning("GNN matcher disabled — PyTorch not available.")
        else:
            logger.info("GNN matcher disabled by config.")

    # ──────────────────────────────────────────────────────────────────────────
    def match(
        self,
        query_embeddings:   np.ndarray,
        gallery_embeddings: np.ndarray,
        gallery_ids:        List[int],
        query_bboxes:       np.ndarray | None = None,
        gallery_bboxes:     np.ndarray | None = None,
        temporal_gaps:      np.ndarray | None = None,
    ) -> List[Tuple[int, int]]:
        """
        Compute optimal assignment using GNN-refined affinities.

        Parameters
        ----------
        query_embeddings   : (N, D) query embeddings
        gallery_embeddings : (M, D) gallery representative embeddings
        gallery_ids        : list of M persistent IDs
        query_bboxes       : (N, 4) optional [x1,y1,x2,y2] for spatial encoding
        gallery_bboxes     : (M, 4) optional gallery bboxes
        temporal_gaps      : (M,) optional frames since last seen per gallery entry

        Returns
        -------
        list of (query_index, persistent_id) tuples
        """
        N = len(query_embeddings)
        M = len(gallery_embeddings)

        if N == 0 or M == 0:
            return []

        # Compute standard cosine similarity as baseline
        from sklearn.metrics.pairwise import cosine_similarity
        cosine_sim = cosine_similarity(
            query_embeddings.astype(np.float32),
            gallery_embeddings.astype(np.float32),
        )  # (N, M)

        # If GNN is disabled or unavailable, fall back to pure cosine + Hungarian
        if not self.enabled or self._model is None:
            return self._hungarian_solve(cosine_sim, gallery_ids)

        try:
            gnn_affinity = self._gnn_forward(
                query_embeddings, gallery_embeddings,
                query_bboxes, gallery_bboxes, temporal_gaps,
            )

            # Blend GNN affinity with cosine similarity
            w = self.cosine_weight
            blended = w * gnn_affinity + (1 - w) * cosine_sim

            return self._hungarian_solve(blended, gallery_ids)

        except Exception as e:
            logger.warning("GNN forward failed (%s). Falling back to cosine.", e)
            if self.fallback:
                return self._hungarian_solve(cosine_sim, gallery_ids)
            return []

    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _gnn_forward(
        self,
        query_embs:     np.ndarray,
        gallery_embs:   np.ndarray,
        query_bboxes:   np.ndarray | None,
        gallery_bboxes: np.ndarray | None,
        temporal_gaps:  np.ndarray | None,
    ) -> np.ndarray:
        """Run GNN and return (N, M) affinity matrix."""
        N, M = len(query_embs), len(gallery_embs)

        det_t = torch.from_numpy(query_embs.astype(np.float32)).to(self.device)
        gal_t = torch.from_numpy(gallery_embs.astype(np.float32)).to(self.device)

        # Build edge features: [Δcx, Δcy, Δw, Δh, Δt, IoU]
        edge_feats = self._build_edge_features(
            N, M, query_bboxes, gallery_bboxes, temporal_gaps
        )
        edge_t = torch.from_numpy(edge_feats.astype(np.float32)).to(self.device)

        affinities = self._model(det_t, gal_t, edge_t)
        return affinities.cpu().numpy()

    def _build_edge_features(
        self,
        N: int, M: int,
        query_bboxes:   np.ndarray | None,
        gallery_bboxes: np.ndarray | None,
        temporal_gaps:  np.ndarray | None,
    ) -> np.ndarray:
        """Build (N*M, 6) edge feature matrix."""
        feats = np.zeros((N * M, 6), dtype=np.float32)

        if query_bboxes is not None and gallery_bboxes is not None:
            for i in range(N):
                for j in range(M):
                    idx = i * M + j
                    qb = query_bboxes[i]
                    gb = gallery_bboxes[j]

                    # Center deltas
                    qcx = (qb[0] + qb[2]) / 2
                    qcy = (qb[1] + qb[3]) / 2
                    gcx = (gb[0] + gb[2]) / 2
                    gcy = (gb[1] + gb[3]) / 2
                    feats[idx, 0] = (qcx - gcx) / 1920  # normalize by frame width
                    feats[idx, 1] = (qcy - gcy) / 1080

                    # Size deltas
                    qw = qb[2] - qb[0]
                    qh = qb[3] - qb[1]
                    gw = gb[2] - gb[0]
                    gh = gb[3] - gb[1]
                    feats[idx, 2] = (qw - gw) / max(gw, 1)
                    feats[idx, 3] = (qh - gh) / max(gh, 1)

                    # Temporal gap
                    if temporal_gaps is not None:
                        feats[idx, 4] = temporal_gaps[j] / 300.0  # normalize

                    # IoU
                    ix1 = max(qb[0], gb[0])
                    iy1 = max(qb[1], gb[1])
                    ix2 = min(qb[2], gb[2])
                    iy2 = min(qb[3], gb[3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    union = qw * qh + gw * gh - inter
                    feats[idx, 5] = inter / (union + 1e-6)

        return feats

    # ──────────────────────────────────────────────────────────────────────────
    def _hungarian_solve(
        self,
        similarity_matrix: np.ndarray,
        gallery_ids: List[int],
    ) -> List[Tuple[int, int]]:
        """Standard Hungarian assignment on a similarity matrix."""
        cost_matrix = 1.0 - similarity_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        for r, c in zip(row_ind, col_ind):
            sim = similarity_matrix[r, c]
            if sim >= self.threshold:
                matches.append((int(r), gallery_ids[c]))

        return matches
