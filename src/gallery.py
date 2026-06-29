"""
src/gallery.py
──────────────
Stage 4 — Persistent Appearance Gallery
Maintains a long-term database of per-vehicle embeddings.
When a vehicle reappears after full occlusion (tunnel), it queries
the gallery by cosine similarity to recover the original persistent ID.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def _compute_color_signature(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Compute an HSV color histogram signature for a vehicle crop.
    Used as a lightweight secondary verification channel to prevent
    false Re-ID matches between same-model vehicles of different colors.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros(64, dtype=np.float32)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist = np.concatenate([hist_h, hist_s]).astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist


class PersistentGallery:
    """
    Long-term vehicle identity store.

    Internal structure
    ------------------
    gallery : dict
        {
          persistent_id (int): {
            "embeddings":  list[np.ndarray],   # rolling window, shape (D,)
            "last_seen":   int,                 # frame number
            "class":       int,                 # vehicle class id
            "box_size":    tuple[int,int]|None, # (width, height) in pixels
            "metadata":    dict,                # extensible payload
          }
        }

    Parameters
    ----------
    threshold    : float  Min cosine similarity to claim a gallery match.
    max_embeddings: int   Rolling window size per identity.
    timeout      : int    Frames of absence before an ID is pruned.
    """

    def __init__(
        self,
        threshold: float = 0.45,
        max_embeddings: int = 15,
        timeout: int = 5000,
        ema_alpha: float = 0.9,
    ) -> None:
        self.threshold = threshold
        self.max_embeddings = max_embeddings
        self.timeout = timeout
        self.ema_alpha = ema_alpha

        # Configurable from pipeline __init__ (tied to tracker max_age)
        self.min_absence_frames: int = 65

        # Color histogram verification threshold (Bhattacharyya distance)
        self.color_verify_threshold: float = 0.4

        # Box-size verification: max allowed ratio between query and gallery
        # box dimensions.  e.g. 0.5 means sizes must be within 50%-200% of
        # each other.  Set to 0 to disable.
        self.box_size_tolerance: float = 0.5
        self.box_size_filter_enabled: bool = False  # toggled from UI

        self.gallery: dict[int, dict] = {}
        self._next_id: int = 1

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def register_or_recover(
        self,
        embedding: np.ndarray,
        frame_n: int,
        cls_id: int,
        excluded_pids: set[int] | None = None,
        box_size: Tuple[int, int] | None = None,
    ) -> Tuple[int, str]:
        """
        Assign a persistent ID to a brand-new short-term track.

        Parameters
        ----------
        excluded_pids : set of persistent IDs that are already assigned
                        to active tracks in the current frame. These will
                        NEVER be returned as matches, preventing duplicate
                        ID assignments.
        box_size      : (width, height) in pixels of the detection bounding
                        box. Stored in the gallery and optionally used for
                        size-consistency verification during matching.

        Returns
        -------
        (persistent_id, status)
            status is 'new' or 'recovered'.
        """
        best_id, best_score = self._query(
            embedding, frame_n, cls_id,
            excluded_pids=excluded_pids,
            query_box_size=box_size,
        )

        if best_id is not None and best_score >= self.threshold:
            self._update(best_id, embedding, frame_n, box_size=box_size)
            logger.debug(
                "Gallery RECOVERED pid=%d score=%.3f frame=%d",
                best_id, best_score, frame_n,
            )
            return best_id, "recovered"

        new_id = self._next_id
        self._next_id += 1
        # Store the first embedding as both the raw list and the EMA representative
        ema_rep = embedding.copy().astype(np.float32)
        norm = np.linalg.norm(ema_rep)
        if norm > 0:
            ema_rep /= norm
        self.gallery[new_id] = {
            "embeddings": [embedding],
            "ema_representative": ema_rep,
            "color_signature": None,  # set externally via set_color_signature()
            "last_seen":  frame_n,
            "class":      cls_id,
            "box_size":   box_size,   # (w, h) in pixels at first sighting
            "metadata":   {},
        }
        logger.debug("Gallery NEW pid=%d frame=%d box=%s", new_id, frame_n, box_size)
        return new_id, "new"

    def update_known(
        self,
        persistent_id: int,
        embedding: np.ndarray,
        frame_n: int,
        box_size: Tuple[int, int] | None = None,
    ) -> None:
        """Append a new embedding for an already-identified vehicle."""
        if persistent_id not in self.gallery:
            logger.warning("update_known called for unknown pid=%d", persistent_id)
            return
        self._update(persistent_id, embedding, frame_n, box_size=box_size)

    def set_color_signature(
        self,
        persistent_id: int,
        crop_bgr: np.ndarray,
    ) -> None:
        """Store a color histogram signature for a gallery entry."""
        if persistent_id not in self.gallery:
            return
        self.gallery[persistent_id]["color_signature"] = _compute_color_signature(crop_bgr)

    def verify_color(
        self,
        persistent_id: int,
        crop_bgr: np.ndarray,
    ) -> bool:
        """
        Secondary verification: compare query color histogram
        against stored gallery color signature using Bhattacharyya distance.
        Returns True if colors are consistent (distance below threshold).
        """
        if persistent_id not in self.gallery:
            return True  # no entry to check against
        stored_sig = self.gallery[persistent_id].get("color_signature")
        if stored_sig is None:
            return True  # no color info stored yet, pass through
        query_sig = _compute_color_signature(crop_bgr)
        # Bhattacharyya distance (lower = more similar)
        distance = cv2.compareHist(
            stored_sig.astype(np.float32),
            query_sig.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        )
        return distance < self.color_verify_threshold

    def verify_box_size(
        self,
        persistent_id: int,
        query_box_size: Tuple[int, int],
    ) -> bool:
        """
        Secondary verification: check whether the query bounding box
        dimensions are within ``box_size_tolerance`` of the stored
        gallery box size.  Returns True if sizes are consistent.

        The check compares width and height ratios independently.
        A tolerance of 0.5 means each dimension must be within
        50 %–200 % of the stored value.
        """
        if not self.box_size_filter_enabled:
            return True  # filter disabled — always pass
        if persistent_id not in self.gallery:
            return True
        stored = self.gallery[persistent_id].get("box_size")
        if stored is None or query_box_size is None:
            return True  # no info to compare
        sw, sh = stored
        qw, qh = query_box_size
        if sw == 0 or sh == 0 or qw == 0 or qh == 0:
            return True
        w_ratio = min(sw, qw) / max(sw, qw)
        h_ratio = min(sh, qh) / max(sh, qh)
        tol = self.box_size_tolerance
        ok = w_ratio >= tol and h_ratio >= tol
        if not ok:
            logger.debug(
                "Box-size REJECTED pid=%d  stored=(%d,%d) query=(%d,%d) "
                "ratios=(%.2f,%.2f) tol=%.2f",
                persistent_id, sw, sh, qw, qh, w_ratio, h_ratio, tol,
            )
        return ok

    def get_representative_embeddings(
        self,
    ) -> Optional[Tuple[np.ndarray, list[int]]]:
        """
        Return a stacked embedding matrix and corresponding ID list
        for all active gallery entries.
        Uses EMA representative when available, falls back to mean.

        Returns
        -------
        (embeddings np.ndarray (M, D), ids list[int]) or None if gallery is empty.
        """
        if not self.gallery:
            return None

        emb_list: list[np.ndarray] = []
        id_list:  list[int]        = []

        for pid, data in self.gallery.items():
            # Prefer the EMA representative (Improvement #1)
            if "ema_representative" in data and data["ema_representative"] is not None:
                rep = data["ema_representative"]
            else:
                rep = np.mean(data["embeddings"], axis=0)
            emb_list.append(rep)
            id_list.append(pid)

        return np.array(emb_list, dtype=np.float32), id_list

    def prune(self, current_frame: int) -> int:
        """
        Remove IDs that haven't been seen for longer than ``timeout`` frames.

        Returns
        -------
        int  Number of IDs pruned.
        """
        stale = [
            pid for pid, d in self.gallery.items()
            if current_frame - d["last_seen"] > self.timeout
        ]
        for pid in stale:
            del self.gallery[pid]
        if stale:
            logger.info(
                "Gallery pruned %d stale IDs at frame %d. Active: %d",
                len(stale), current_frame, len(self.gallery),
            )
        return len(stale)

    def save(self, path: str | Path) -> None:
        """Serialise gallery to disk (pickle)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"gallery": self.gallery, "next_id": self._next_id},
                f,
            )
        logger.info("Gallery saved → %s  (%d IDs)", path, len(self.gallery))

    def load(self, path: str | Path) -> None:
        """Restore gallery from disk."""
        path = Path(path)
        if not path.exists():
            logger.warning("Gallery snapshot not found at '%s'. Starting fresh.", path)
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.gallery   = data["gallery"]
        self._next_id  = data["next_id"]
        logger.info("Gallery loaded ← %s  (%d IDs)", path, len(self.gallery))

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _query(
        self,
        query_emb: np.ndarray,
        current_frame: int,
        cls_id: int,
        min_absence_frames: int | None = None,
        excluded_pids: set[int] | None = None,
        query_box_size: Tuple[int, int] | None = None,
    ) -> Tuple[Optional[int], float]:
        """
        Find the best-matching gallery ID for a query embedding.
        Only searches IDs that have been absent for at least
        ``min_absence_frames`` (defaults to self.min_absence_frames,
        tied to tracker max_age) to avoid conflating active tracks.

        Parameters
        ----------
        excluded_pids  : set of persistent IDs to skip entirely.
                         Used to prevent assigning the same PID to
                         two different vehicles in the same frame.
        query_box_size : (width, height) in pixels for box-size
                         consistency filtering.
        """
        if min_absence_frames is None:
            min_absence_frames = self.min_absence_frames
        if excluded_pids is None:
            excluded_pids = set()

        best_id:    Optional[int] = None
        best_score: float         = -1.0
        q = query_emb.reshape(1, -1)

        for pid, data in self.gallery.items():
            # Skip PIDs already assigned to active tracks in this frame
            if pid in excluded_pids:
                continue
            # Skip actively tracked vehicles (seen very recently)
            if current_frame - data["last_seen"] < min_absence_frames:
                continue
            # Class consistency guard
            if data["class"] != cls_id:
                continue
            # Box-size consistency guard
            if query_box_size is not None and not self.verify_box_size(pid, query_box_size):
                continue

            # Use EMA representative for fast comparison when available
            if "ema_representative" in data and data["ema_representative"] is not None:
                rep = data["ema_representative"].reshape(1, -1)
                score = float(cosine_similarity(q, rep)[0, 0])
            else:
                gallery_embs = np.array(data["embeddings"], dtype=np.float32)
                sims  = cosine_similarity(q, gallery_embs)[0]
                score = float(np.max(sims))

            if score > best_score:
                best_score = score
                best_id    = pid

        return best_id, best_score

    def _update(
        self,
        pid: int,
        embedding: np.ndarray,
        frame_n: int,
        box_size: Tuple[int, int] | None = None,
    ) -> None:
        data = self.gallery[pid]
        data["embeddings"].append(embedding)
        # Keep rolling window
        data["embeddings"] = data["embeddings"][-self.max_embeddings :]
        data["last_seen"]  = frame_n

        # Update stored box size with latest observation
        if box_size is not None:
            data["box_size"] = box_size

        # Improvement #1: Maintain EMA representative
        if "ema_representative" in data and data["ema_representative"] is not None:
            alpha = self.ema_alpha
            ema = alpha * data["ema_representative"] + (1 - alpha) * embedding.astype(np.float32)
            norm = np.linalg.norm(ema)
            if norm > 0:
                ema /= norm
            data["ema_representative"] = ema
        else:
            # Initialize EMA from first embedding
            ema = embedding.copy().astype(np.float32)
            norm = np.linalg.norm(ema)
            if norm > 0:
                ema /= norm
            data["ema_representative"] = ema

    # ──────────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.gallery)

    def __repr__(self) -> str:
        return (
            f"PersistentGallery(size={len(self)}, "
            f"threshold={self.threshold}, next_id={self._next_id})"
        )
