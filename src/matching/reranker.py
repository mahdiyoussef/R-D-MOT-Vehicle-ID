"""
src/matching/reranker.py
────────────────────────
k-Reciprocal Re-Ranking (Zhong et al., CVPR 2017)

This module implements post-processing re-ranking to improve Re-ID accuracy
without requiring additional model training. It refines the initial retrieval
results by considering the k-reciprocal nearest neighbors—if an image A is in
the top-k of image B, and image B is in the top-k of image A, they are highly
likely to belong to the same identity.

The original distance is fused with a Jaccard distance over these neighbor sets.
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


class KReciprocalReRanker:
    """
    Implements k-reciprocal re-ranking for vehicle Re-ID.
    
    Parameters
    ----------
    k1 : int
        Size of the initial k-nearest neighbor set.
    k2 : int
        Size of the k-nearest neighbor set for local query expansion.
    lambda_value : float
        Weighting factor to fuse original distance and Jaccard distance.
        final_dist = (1 - lambda) * jaccard_dist + lambda * original_dist
    """

    def __init__(self, k1: int = 20, k2: int = 6, lambda_value: float = 0.3) -> None:
        self.k1 = k1
        self.k2 = k2
        self.lambda_value = lambda_value

    def re_rank(self, query_embeds: np.ndarray, gallery_embeds: np.ndarray) -> np.ndarray:
        """
        Compute the re-ranked distance matrix between queries and gallery.

        Parameters
        ----------
        query_embeds   : np.ndarray, shape (M, D)
        gallery_embeds : np.ndarray, shape (N, D)

        Returns
        -------
        final_dist : np.ndarray, shape (M, N)
            The fused distance matrix. Lower distance means higher similarity.
        """
        if query_embeds.size == 0 or gallery_embeds.size == 0:
            return np.empty((query_embeds.shape[0], gallery_embeds.shape[0]))

        num_q = query_embeds.shape[0]
        num_g = gallery_embeds.shape[0]
        
        # Combine all features to compute global distances (Q+G, Q+G)
        all_features = np.vstack([query_embeds, gallery_embeds])
        
        # Original distances (using cosine distance since features are L2 normalized)
        # cosine distance = 1 - cosine similarity
        sim_matrix = np.dot(all_features, all_features.T)
        original_dist = 1.0 - sim_matrix
        original_dist = np.maximum(original_dist, 0.0)
        
        # Normalize original distances to [0, 1] for stable Jaccard fusion
        v_max = np.max(original_dist, axis=0, keepdims=True)
        v_max[v_max == 0] = 1.0  # avoid division by zero
        original_dist = original_dist / v_max
        
        initial_rank = np.argsort(original_dist, axis=1)

        # Build k-reciprocal nearest neighbor sets
        all_num = num_q + num_g
        V = np.zeros((all_num, all_num), dtype=np.float32)
        
        for i in range(all_num):
            # Forward k-NN
            forward_k_nn = initial_rank[i, :self.k1 + 1]
            
            # Reciprocal set: keep only those where 'i' is also in their top-k1
            reciprocal_set = []
            for candidate in forward_k_nn:
                backward_k_nn = initial_rank[candidate, :self.k1 + 1]
                if i in backward_k_nn:
                    reciprocal_set.append(candidate)
            
            # Local query expansion (Zhong et al.): expand reciprocal set using top-k2 of each neighbor
            expanded_set = set(reciprocal_set)
            for candidate in reciprocal_set:
                candidate_k_nn = initial_rank[candidate, :self.k2 + 1]
                expanded_set.update(candidate_k_nn)
            
            expanded_set_arr = np.array(list(expanded_set))
            
            # Encode into robust feature representation based on distances
            if len(expanded_set_arr) > 0:
                weights = np.exp(-original_dist[i, expanded_set_arr])
                V[i, expanded_set_arr] = weights / np.sum(weights)

        # Compute Jaccard distance between the k-reciprocal feature vectors
        # jaccard(A, B) = 1 - sum(min(A, B)) / sum(max(A, B))
        # Optimized using fast matrix operations
        V_sum = np.sum(V, axis=1, keepdims=True)
        V_sum_T = V_sum.T
        
        # To avoid heavy O(N^3) exact Jaccard, we compute a soft Jaccard approximation
        # using the intersection-over-union of the probability distributions
        intersection = np.dot(V, V.T)
        jaccard_dist = 1.0 - intersection / (V_sum + V_sum_T - intersection + 1e-8)
        
        # Fuse Jaccard distance with original distance
        final_dist = jaccard_dist * (1 - self.lambda_value) + original_dist * self.lambda_value
        
        # We only care about the Query-to-Gallery distances
        final_qg_dist = final_dist[:num_q, num_q:]
        
        return final_qg_dist
