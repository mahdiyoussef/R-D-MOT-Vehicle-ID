"""
database.py
-----------
Faiss-backed vector database for long-term ReID matching.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FaissReIDDatabase:
    """
    Faiss vector database with ID mapping and EMA updates.

    Parameters
    ----------
    dimension : int
        Embedding dimension.
    metric : str
        'cosine' or 'l2'.
    threshold : float
        Similarity threshold for a valid match.
    ema_alpha : float
        EMA weight for updating embeddings (higher = slower update).
    """

    def __init__(
        self,
        dimension: int,
        metric: str = "cosine",
        threshold: float = 0.85,
        ema_alpha: float = 0.9,
    ) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss is required. Install faiss-cpu or faiss-gpu.") from exc

        self.faiss = faiss
        self.metric = metric
        self.threshold = threshold
        self.ema_alpha = ema_alpha

        if metric == "cosine":
            base = faiss.IndexFlatIP(dimension)
            self._normalize = True
        elif metric == "l2":
            base = faiss.IndexFlatL2(dimension)
            self._normalize = False
        else:
            raise ValueError("metric must be 'cosine' or 'l2'")

        self.index = faiss.IndexIDMap2(base)
        self.id_to_emb: dict[int, np.ndarray] = {}
        self._next_id = 1

    def query(self, embedding: np.ndarray, top_k: int = 1) -> Tuple[Optional[int], float]:
        """
        Query the nearest neighbor.
        Returns (id, score) or (None, -1.0) if empty or below threshold.
        """
        if self.index.ntotal == 0:
            return None, -1.0

        vec = self._prepare(embedding)
        scores, ids = self.index.search(vec, top_k)

        best_id = int(ids[0][0])
        best_score = float(scores[0][0])

        if best_id < 0:
            return None, -1.0

        if self.metric == "cosine":
            if best_score >= self.threshold:
                return best_id, best_score
            return None, best_score

        # L2 distance (lower is better)
        if best_score <= self.threshold:
            return best_id, best_score
        return None, best_score

    def add_new(self, embedding: np.ndarray) -> int:
        """Add a new identity to the database and return its ID."""
        vec = self._prepare(embedding)
        pid = self._next_id
        self._next_id += 1
        self.index.add_with_ids(vec, np.array([pid], dtype=np.int64))
        self.id_to_emb[pid] = vec[0]
        return pid

    def update(self, pid: int, embedding: np.ndarray, last_seen: int | None = None) -> None:
        """Update an existing embedding using EMA (or add if missing)."""
        if pid not in self.id_to_emb:
            self.id_to_emb[pid] = self._prepare(embedding)[0]
            self.index.add_with_ids(self.id_to_emb[pid][None, :], np.array([pid], dtype=np.int64))
            return

        old = self.id_to_emb[pid]
        new = self._prepare(embedding)[0]
        updated = (self.ema_alpha * old) + ((1.0 - self.ema_alpha) * new)
        if self._normalize:
            updated = self._l2_normalize(updated)

        self._remove_id(pid)
        self.index.add_with_ids(updated[None, :], np.array([pid], dtype=np.int64))
        self.id_to_emb[pid] = updated

    def size(self) -> int:
        return self.index.ntotal

    def _prepare(self, embedding: np.ndarray) -> np.ndarray:
        vec = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if self._normalize:
            vec = self._l2_normalize(vec)
        return vec

    def _l2_normalize(self, x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.maximum(norm, 1e-12)

    def _remove_id(self, pid: int) -> None:
        selector = self.faiss.IDSelectorBatch(np.array([pid], dtype=np.int64))
        self.index.remove_ids(selector)


class MilvusReIDDatabase:
    """
    Milvus vector database with EMA updates and retention filtering.

    Parameters
    ----------
    dimension : int
        Embedding dimension.
    host : str
        Milvus host.
    port : int
        Milvus port.
    collection_name : str
        Collection name.
    metric_type : str
        'COSINE' or 'L2'.
    index_type : str
        Index type (e.g. 'HNSW').
    threshold : float
        Similarity (COSINE) or distance (L2) threshold.
    ema_alpha : float
        EMA weight for updating embeddings.
    """

    def __init__(
        self,
        dimension: int,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "vehicle_embeddings",
        metric_type: str = "COSINE",
        index_type: str = "HNSW",
        threshold: float = 0.85,
        ema_alpha: float = 0.9,
    ) -> None:
        try:
            from pymilvus import (
                connections,
                FieldSchema,
                CollectionSchema,
                DataType,
                Collection,
                utility,
            )
        except ImportError as exc:
            raise ImportError("pymilvus is required for the Milvus backend.") from exc

        self.metric_type = metric_type.upper()
        self.index_type = index_type
        self.threshold = threshold
        self.ema_alpha = ema_alpha
        self._normalize = self.metric_type == "COSINE"

        connections.connect(host=host, port=str(port))

        if not utility.has_collection(collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="last_seen", dtype=DataType.INT64),
            ]
            schema = CollectionSchema(fields, description="Vehicle ReID embeddings")
            collection = Collection(name=collection_name, schema=schema)
            index_params = {
                "index_type": self.index_type,
                "metric_type": self.metric_type,
                "params": {"M": 8, "efConstruction": 64},
            }
            collection.create_index(field_name="embedding", index_params=index_params)
        else:
            collection = Collection(name=collection_name)

        self.collection = collection
        self.collection.load()

        self._next_id = int(time.time() * 1000)

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        min_last_seen: int | None = None,
    ) -> list[tuple[int, float]]:
        """Search for nearest neighbors with optional retention filter."""
        vec = self._prepare(embedding)
        expr = None
        if min_last_seen is not None:
            expr = f"last_seen >= {int(min_last_seen)}"

        results = self.collection.search(
            data=vec.tolist(),
            anns_field="embedding",
            param={"metric_type": self.metric_type, "params": {"ef": 64}},
            limit=top_k,
            expr=expr,
            output_fields=["id", "last_seen"],
        )

        matches: list[tuple[int, float]] = []
        for hit in results[0]:
            pid = int(hit.id)
            score = float(hit.distance)
            matches.append((pid, score))
        return matches

    def add_new(self, embedding: np.ndarray, last_seen: int) -> int:
        pid = self._next_id
        self._next_id += 1
        vec = self._prepare(embedding)
        self.collection.insert([[pid], vec.tolist()[0], [int(last_seen)]])
        return pid

    def update(self, pid: int, embedding: np.ndarray, last_seen: int) -> None:
        existing = self.collection.query(
            expr=f"id == {int(pid)}",
            output_fields=["embedding"],
        )
        if existing:
            old = np.asarray(existing[0]["embedding"], dtype=np.float32)
            new = self._prepare(embedding)[0]
            updated = (self.ema_alpha * old) + ((1.0 - self.ema_alpha) * new)
            if self._normalize:
                updated = self._l2_normalize(updated)
            self.collection.delete(expr=f"id == {int(pid)}")
            self.collection.insert([[pid], updated.tolist(), [int(last_seen)]])
            return

        vec = self._prepare(embedding)
        self.collection.insert([[pid], vec.tolist()[0], [int(last_seen)]])

    def size(self) -> int:
        return int(self.collection.num_entities)

    def _prepare(self, embedding: np.ndarray) -> np.ndarray:
        vec = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if self._normalize:
            vec = self._l2_normalize(vec)
        return vec

    def _l2_normalize(self, x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.maximum(norm, 1e-12)
