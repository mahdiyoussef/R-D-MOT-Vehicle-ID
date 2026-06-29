"""
src/geometry/view_geometry.py
──────────────────────────────
Geometric utilities for cross-view vehicle matching.

Defines the stripe visibility map: for each pair of camera viewpoints,
only a subset of the 6 horizontal part-stripes are geometrically visible
from BOTH perspectives simultaneously. Cross-view cosine similarity is
computed ONLY on these shared-visible stripes.

Stripe indices (top → bottom of vehicle crop):
  0 = roof
  1 = upper body (windshield / upper doors)
  2 = mid-upper body
  3 = mid-lower body
  4 = lower body (door bottom / sill)
  5 = bumper / undercarriage
"""

from __future__ import annotations

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# View label constants
# ─────────────────────────────────────────────────────────────────────────────

VIEW_FRONT      = "front"
VIEW_REAR       = "rear"
VIEW_SIDE_LEFT  = "side_left"
VIEW_SIDE_RIGHT = "side_right"
VIEW_TOP_DOWN   = "top_down"
VIEW_AMBIGUOUS  = "ambiguous"

ALL_VIEWS = [VIEW_FRONT, VIEW_REAR, VIEW_SIDE_LEFT, VIEW_SIDE_RIGHT, VIEW_TOP_DOWN, VIEW_AMBIGUOUS]

# ─────────────────────────────────────────────────────────────────────────────
# Cross-view stripe visibility map
# ─────────────────────────────────────────────────────────────────────────────
# Key: frozenset({view_a, view_b})
# Value: list of stripe indices visible from BOTH views simultaneously.
# Empty list means no reliable visual stripe overlap → attribute-only matching.

_STRIPE_VISIBILITY_MAP: dict[frozenset, list[int]] = {
    frozenset({VIEW_FRONT, VIEW_SIDE_LEFT}):      [0, 1],
    frozenset({VIEW_FRONT, VIEW_SIDE_RIGHT}):     [0, 1],
    frozenset({VIEW_REAR,  VIEW_SIDE_LEFT}):      [0, 1],
    frozenset({VIEW_REAR,  VIEW_SIDE_RIGHT}):     [0, 1],
    frozenset({VIEW_FRONT, VIEW_REAR}):           [0],
    # Mirror match: side_left ↔ side_right — flip embedding before comparing
    frozenset({VIEW_SIDE_LEFT, VIEW_SIDE_RIGHT}): [0, 1, 2, 3, 4, 5],
    # Top-down: no reliable visual cosine → attribute only
    frozenset({VIEW_TOP_DOWN, VIEW_FRONT}):       [],
    frozenset({VIEW_TOP_DOWN, VIEW_REAR}):        [],
    frozenset({VIEW_TOP_DOWN, VIEW_SIDE_LEFT}):   [],
    frozenset({VIEW_TOP_DOWN, VIEW_SIDE_RIGHT}):  [],
    # Ambiguous: use all stripes with lower confidence
    frozenset({VIEW_AMBIGUOUS, VIEW_FRONT}):      [0, 1, 2, 3, 4, 5],
    frozenset({VIEW_AMBIGUOUS, VIEW_REAR}):       [0, 1, 2, 3, 4, 5],
    frozenset({VIEW_AMBIGUOUS, VIEW_SIDE_LEFT}):  [0, 1, 2, 3, 4, 5],
    frozenset({VIEW_AMBIGUOUS, VIEW_SIDE_RIGHT}): [0, 1, 2, 3, 4, 5],
}

_ALL_STRIPES = [0, 1, 2, 3, 4, 5]


def get_shared_visible_stripes(view_a: str, view_b: str) -> list[int]:
    """
    Return the stripe indices geometrically visible from BOTH viewpoints.

    Returns
    -------
    list[int]
        Stripe indices shared between the two views.
        Empty list means no visual comparison is possible (attribute-only).
        Same-view pair returns all 6 stripes.
    """
    if view_a == view_b:
        return _ALL_STRIPES
    return _STRIPE_VISIBILITY_MAP.get(frozenset({view_a, view_b}), [])


def is_mirror_view_pair(view_a: str, view_b: str) -> bool:
    """
    Returns True if the two views are a side_left ↔ side_right mirror pair.
    In this case the gallery stripe embedding should be horizontally reversed
    before computing cosine similarity.
    """
    return frozenset({view_a, view_b}) == frozenset({VIEW_SIDE_LEFT, VIEW_SIDE_RIGHT})


def compute_cross_view_cosine(
    query_part_embeds:   np.ndarray,   # (6, D)
    gallery_part_embeds: np.ndarray,   # (6, D)
    query_view:          str,
    gallery_view:        str,
) -> tuple[float, bool]:
    """
    Compute part-filtered cosine similarity using only geometrically
    shared stripes between the query and gallery viewpoints.

    Returns
    -------
    (cosine_score, is_valid)
        is_valid=False means no shared stripes exist (skip visual comparison).
    """
    shared_stripes = get_shared_visible_stripes(query_view, gallery_view)

    if not shared_stripes:
        return 0.0, False

    q_stripes = query_part_embeds[shared_stripes]    # (K, D)
    g_stripes = gallery_part_embeds[shared_stripes]  # (K, D)

    # Reverse gallery stripes for mirror-pair (horizontal flip approximation)
    if is_mirror_view_pair(query_view, gallery_view):
        g_stripes = g_stripes[::-1]

    q_mean = q_stripes.mean(axis=0)
    g_mean = g_stripes.mean(axis=0)

    q_norm = np.linalg.norm(q_mean)
    g_norm = np.linalg.norm(g_mean)

    if q_norm < 1e-6 or g_norm < 1e-6:
        return 0.0, True  # zero vector — valid attempt, zero score

    cosine_score = float(np.dot(q_mean, g_mean) / (q_norm * g_norm))
    return cosine_score, True


def are_views_compatible(view_a: str, view_b: str) -> bool:
    """
    Returns True if the two views share at least one visible stripe,
    making a visual cross-view comparison meaningful.
    """
    if view_a == view_b:
        return True
    return len(get_shared_visible_stripes(view_a, view_b)) > 0
