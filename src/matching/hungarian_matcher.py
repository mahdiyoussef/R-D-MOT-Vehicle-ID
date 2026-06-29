"""
src/matching/hungarian_matcher.py
──────────────────────────────────
Optimal one-to-one assignment of query embeddings to gallery IDs via the
Hungarian algorithm (Kuhn-Munkres). Used internally by the cascade matcher
and as a standalone utility for batch-frame association.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class HungarianMatcher:
    """
    Optimal assignment of query embeddings to gallery IDs.

    Prevents two queries from claiming the same gallery entry —
    the key advantage over independent nearest-neighbour lookups.

    Parameters
    ----------
    similarity_threshold : float
        Minimum cosine similarity required to accept a match.
        Pairs below this threshold are treated as unmatched.
    """

    def __init__(self, similarity_threshold: float = 0.45) -> None:
        self.similarity_threshold = similarity_threshold

    def match(
        self,
        query_embeddings:   np.ndarray,
        gallery_embeddings: np.ndarray,
        gallery_ids:        List[int],
    ) -> List[Tuple[int, int]]:
        """Alias for assign_batch."""
        return self.assign_batch(query_embeddings, gallery_embeddings, gallery_ids)

    def assign_batch(
        self,
        query_embeddings:   np.ndarray,
        gallery_embeddings: np.ndarray,
        gallery_ids:        List[int],
    ) -> List[Tuple[int, int]]:
        """
        Compute optimal assignment between a batch of queries and gallery entries.

        Parameters
        ----------
        query_embeddings   : np.ndarray, shape (N, D)
        gallery_embeddings : np.ndarray, shape (M, D)
        gallery_ids        : list of M persistent IDs

        Returns
        -------
        list of (query_index, persistent_id) tuples
            Only pairs whose similarity >= threshold are returned.
        """
        n_queries = len(query_embeddings)
        n_gallery = len(gallery_embeddings)

        if n_queries == 0 or n_gallery == 0:
            return []

        # Optional: Apply k-reciprocal re-ranking if enabled
        # Default behaviour falls back to standard cosine similarity if no reranker is set.
        if getattr(self, "reranker", None) is not None:
            # Re-ranker returns distance matrix, we need similarity
            dist_matrix = self.reranker.re_rank(
                query_embeddings.astype(np.float32),
                gallery_embeddings.astype(np.float32)
            )
            similarity_matrix = 1.0 - dist_matrix
        else:
            similarity_matrix = cosine_similarity(
                query_embeddings.astype(np.float32),
                gallery_embeddings.astype(np.float32),
            )  # (N, M)

        cost_matrix = 1.0 - similarity_matrix  # scipy minimises cost

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        accepted_pairs: List[Tuple[int, int]] = []
        for row, col in zip(row_indices, col_indices):
            sim = similarity_matrix[row, col]
            if sim >= self.similarity_threshold:
                accepted_pairs.append((int(row), gallery_ids[col]))
                logger.debug(
                    "Assign  query_idx=%d → gid=%d  sim=%.3f",
                    row, gallery_ids[col], sim,
                )
            else:
                logger.debug(
                    "Reject  query_idx=%d  gid=%d  sim=%.3f < thresh=%.3f",
                    row, gallery_ids[col], sim, self.similarity_threshold,
                )

        return accepted_pairs

    def find_nearest_match(
        self,
        query_embedding:    np.ndarray,
        gallery_embeddings: np.ndarray,
        gallery_ids:        List[int],
    ) -> Tuple[int | None, float]:
        """
        Nearest-neighbour lookup for a single query (no uniqueness guarantee).
        Use for diagnostic / fallback purposes only.

        Returns
        -------
        (best_persistent_id, best_score) or (None, -1.0) if no match found.
        """
        if len(gallery_embeddings) == 0:
            return None, -1.0

        query_vec  = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_vec, gallery_embeddings)[0]
        best_idx   = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self.similarity_threshold:
            return gallery_ids[best_idx], best_score
        return None, best_score
