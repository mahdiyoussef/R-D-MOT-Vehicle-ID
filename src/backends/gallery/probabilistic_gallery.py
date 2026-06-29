"""
src/backends/gallery/probabilistic_gallery.py
──────────────────────────────────────────────
Probabilistic Gallery Index (v3.0).

Instead of representing each identity as a single point embedding
(EMA vector), this gallery models each identity as a Gaussian
distribution N(μ, diag(σ²)).

Matching uses Mutual Likelihood Score (MLS) instead of cosine similarity:
  MLS(q, N(μ,σ²)) = -Σ [ (q_d - μ_d)² / σ²_d  +  log(σ²_d) ]

This naturally downweights uncertain dimensions and provides
confidence-aware matching.

Mean and variance are updated incrementally using Welford's online
algorithm — no batch recomputation needed.

Reference:
  Shi & Jain, "Probabilistic Face Embeddings", ICCV 2019.
  https://arxiv.org/abs/1904.09658
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.backends.gallery.base import BaseGalleryIndex

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GaussianIdentity:
    """
    Represents a vehicle identity as a Gaussian distribution.

    Uses Welford's online algorithm for numerically stable incremental
    updates of mean and variance.

    Attributes
    ----------
    global_id      : Persistent identity ID
    mean           : Running mean μ (D,)
    m2             : Running sum of squared deviations (for variance)
    variance       : Current variance σ² = m2 / count  (D,)
    count          : Number of observations
    class_id       : Vehicle class (car, truck, etc.)
    last_seen      : Frame number of most recent observation
    confidence     : Scalar confidence score (inversely proportional to mean variance)
    """
    global_id:     int
    mean:          np.ndarray
    m2:            np.ndarray       # Welford accumulator
    variance:      np.ndarray
    count:         int = 0
    class_id:      int = 0
    last_seen:     int = 0
    confidence:    float = 0.0

    def update(self, embedding: np.ndarray) -> None:
        """
        Welford's online algorithm for incremental mean + variance.

        Numerically stable even for large count values, unlike the
        naive formula var = E[x²] - E[x]².
        """
        self.count += 1
        delta      = embedding - self.mean
        self.mean  = self.mean + delta / self.count
        delta2     = embedding - self.mean
        self.m2    = self.m2 + delta * delta2

        if self.count >= 2:
            self.variance = self.m2 / (self.count - 1)
        else:
            # Single observation: use a large prior variance
            self.variance = np.ones_like(self.mean)

        # Update confidence: inverse of mean variance (higher = more certain)
        mean_var = float(np.mean(self.variance))
        self.confidence = 1.0 / (mean_var + 1e-8)

    def mls_score(self, query: np.ndarray) -> float:
        """
        Mutual Likelihood Score — uncertainty-aware similarity metric.

        MLS(q, N(μ,σ²)) = -Σ [ (q_d - μ_d)² / σ²_d  +  log(σ²_d) ]

        Higher MLS = better match (less negative).
        The log(σ²) term penalizes high-variance (uncertain) identities,
        while the (q-μ)²/σ² term is lenient on uncertain dimensions.
        """
        var_clamped = np.maximum(self.variance, 1e-6)
        # Mahalanobis-like distance (per-dimension)
        mahal = (query - self.mean) ** 2 / var_clamped
        # Log-variance penalty
        log_var = np.log(var_clamped)
        return float(-np.sum(mahal + log_var))


# ─────────────────────────────────────────────────────────────────────────────

class ProbabilisticGalleryIndex(BaseGalleryIndex):
    """
    Gallery index using Gaussian identity representations.

    Each identity is modeled as N(μ, diag(σ²)) and matched using
    Mutual Likelihood Score instead of cosine similarity.

    Config keys (under 'probabilistic_gallery'):
      mls_threshold         : float  Minimum MLS score to accept a match
      min_absence_frames    : int    Min frames since last seen to be queryable
      variance_prior        : float  Initial variance before any observations
      min_observations      : int    Min observations before identity is queryable
    """

    def __init__(self, config: dict, embed_dim: int) -> None:
        self.cfg       = config.get("probabilistic_gallery", {})
        self.embed_dim = embed_dim

        self._identities: dict[int, GaussianIdentity] = {}
        self._mls_threshold     = float(self.cfg.get("mls_threshold", -500.0))
        self._min_absence       = int(self.cfg.get("min_absence_frames", 0))
        self._variance_prior    = float(self.cfg.get("variance_prior", 1.0))
        self._min_observations  = int(self.cfg.get("min_observations", 2))

    # ──────────────────────────────────────────────────────────────────────────
    def add(self, global_id: int, embedding: np.ndarray) -> None:
        """Add or update an embedding for a given identity."""
        emb = embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        if global_id in self._identities:
            self._identities[global_id].update(emb)
        else:
            identity = GaussianIdentity(
                global_id = global_id,
                mean      = emb.copy(),
                m2        = np.zeros(self.embed_dim, dtype=np.float32),
                variance  = np.full(self.embed_dim, self._variance_prior, dtype=np.float32),
                count     = 1,
            )
            self._identities[global_id] = identity

    # ──────────────────────────────────────────────────────────────────────────
    def update_meta(
        self,
        global_id:   int,
        class_id:    int,
        frame_n:     int,
        bbox:        np.ndarray | None = None,
    ) -> None:
        """Update metadata for an identity."""
        if global_id in self._identities:
            self._identities[global_id].class_id   = class_id
            self._identities[global_id].last_seen   = frame_n

    # ──────────────────────────────────────────────────────────────────────────
    def query(
        self,
        embedding:     np.ndarray,
        class_id:      int,
        current_frame: int,
        exclude_ids:   set[int],
        top_k:         int = 5,
    ) -> tuple[int, float] | None:
        """
        Find the best-matching identity using Mutual Likelihood Score.

        Filters:
          - Class consistency (must match class_id)
          - Temporal absence (must be absent for >= min_absence_frames)
          - Exclusion set (already assigned in current frame)
          - Minimum observations (identity must have been seen enough times)
        """
        if not self._identities:
            return None

        emb = embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        best_gid:   int | None = None
        best_score: float      = float("-inf")

        for gid, identity in self._identities.items():
            # Skip excluded
            if gid in exclude_ids:
                continue
            # Skip wrong class
            if identity.class_id != class_id:
                continue
            # Skip recently seen (still actively tracked)
            if current_frame - identity.last_seen < self._min_absence:
                continue
            # Skip under-observed identities
            if identity.count < self._min_observations:
                continue

            score = identity.mls_score(emb)
            if score > best_score:
                best_score = score
                best_gid   = gid

        if best_gid is not None and best_score >= self._mls_threshold:
            # Convert MLS to a [0, 1] confidence-like value for compatibility
            # with the rest of the pipeline's logging
            normalized_score = 1.0 / (1.0 + np.exp(-best_score / 100.0))
            return (best_gid, float(normalized_score))

        return None

    # ──────────────────────────────────────────────────────────────────────────
    def remove(self, global_id: int) -> None:
        """Remove an identity from the gallery."""
        self._identities.pop(global_id, None)

    def rebuild(self) -> None:
        """No index rebuild needed for probabilistic gallery."""
        pass

    # ──────────────────────────────────────────────────────────────────────────
    def get_identity_confidence(self, global_id: int) -> float:
        """Return the confidence score for a specific identity."""
        if global_id in self._identities:
            return self._identities[global_id].confidence
        return 0.0

    def get_all_confidences(self) -> dict[int, float]:
        """Return confidence scores for all identities."""
        return {
            gid: identity.confidence
            for gid, identity in self._identities.items()
        }

    # ──────────────────────────────────────────────────────────────────────────
    @property
    def size(self) -> int:
        return len(self._identities)

    @property
    def name(self) -> str:
        return "ProbabilisticGalleryIndex"
