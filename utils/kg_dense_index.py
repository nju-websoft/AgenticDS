from __future__ import annotations

from typing import Dict

import faiss
import numpy as np

from utils.kg_index_store import KGIndexFiles


class FieldwiseDenseIndex:
    """FAISS dense index wrapper for description/metadata field fusion."""

    def __init__(
        self,
        *,
        description_index=None,
        metadata_index=None,
        legacy_index=None,
        description_weight: float,
        metadata_weight: float,
    ) -> None:
        self.description_index = description_index
        self.metadata_index = metadata_index
        self.legacy_index = legacy_index
        self.description_weight = float(description_weight)
        self.metadata_weight = float(metadata_weight)

    @classmethod
    def load(
        cls,
        files: KGIndexFiles,
        *,
        description_weight: float,
        metadata_weight: float,
    ) -> "FieldwiseDenseIndex":
        if files.has_fieldwise_dense:
            print(f"[kg-retriever][dense] loading description index {files.description_index}", flush=True)
            description_index = faiss.read_index(str(files.description_index))
            print(f"[kg-retriever][dense] loading metadata index {files.metadata_index}", flush=True)
            metadata_index = faiss.read_index(str(files.metadata_index))
            return cls(
                description_index=description_index,
                metadata_index=metadata_index,
                description_weight=description_weight,
                metadata_weight=metadata_weight,
            )

        if files.has_legacy_dense:
            print(f"[kg-retriever][dense] loading legacy index {files.legacy_index}", flush=True)
            legacy_index = faiss.read_index(str(files.legacy_index))
            return cls(
                legacy_index=legacy_index,
                description_weight=description_weight,
                metadata_weight=metadata_weight,
            )

        raise FileNotFoundError(
            f"No dense index found in {files.cache_dir}. "
            f"Expected fieldwise indexes ({files.description_index.name}, {files.metadata_index.name}) "
            f"or legacy index ({files.legacy_index.name})."
        )

    @classmethod
    def build(cls, description_embeddings: np.ndarray, metadata_embeddings: np.ndarray) -> "FieldwiseDenseIndex":
        dimension = description_embeddings.shape[-1]
        description_index = faiss.index_factory(dimension, "Flat", faiss.METRIC_INNER_PRODUCT)
        metadata_index = faiss.index_factory(dimension, "Flat", faiss.METRIC_INNER_PRODUCT)
        description_index.add(description_embeddings)
        metadata_index.add(metadata_embeddings)
        return cls(
            description_index=description_index,
            metadata_index=metadata_index,
            description_weight=1.0,
            metadata_weight=1.0,
        )

    @property
    def is_fieldwise(self) -> bool:
        return self.description_index is not None and self.metadata_index is not None

    @property
    def ntotal(self) -> int:
        if self.is_fieldwise:
            return int(self.description_index.ntotal)
        if self.legacy_index is not None:
            return int(self.legacy_index.ntotal)
        return 0

    def write(self, files: KGIndexFiles) -> None:
        if not self.is_fieldwise:
            raise ValueError("Only fieldwise dense indexes can be written by this wrapper")
        files.cache_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.description_index, str(files.description_index))
        faiss.write_index(self.metadata_index, str(files.metadata_index))

    def search(self, query_emb: np.ndarray, top_k: int) -> Dict[int, Dict[str, float]]:
        if self.is_fieldwise:
            return self._search_fieldwise(query_emb, top_k)
        return self._search_legacy(query_emb, top_k)

    def _search_fieldwise(self, query_emb: np.ndarray, top_k: int) -> Dict[int, Dict[str, float]]:
        description_distances, description_ids = self.description_index.search(query_emb, k=top_k)
        metadata_distances, metadata_ids = self.metadata_index.search(query_emb, k=top_k)

        description_scores: Dict[int, float] = {}
        metadata_scores: Dict[int, float] = {}
        for sim, idx in zip(description_distances[0], description_ids[0]):
            if idx >= 0:
                description_scores[int(idx)] = max(float(sim), description_scores.get(int(idx), -1.0))
        for sim, idx in zip(metadata_distances[0], metadata_ids[0]):
            if idx >= 0:
                metadata_scores[int(idx)] = max(float(sim), metadata_scores.get(int(idx), -1.0))

        payload: Dict[int, Dict[str, float]] = {}
        for idx in set(description_scores) | set(metadata_scores):
            description_score = max(0.0, description_scores.get(idx, 0.0))
            metadata_score = max(0.0, metadata_scores.get(idx, 0.0))
            payload[idx] = {
                "dense": self.description_weight * description_score + self.metadata_weight * metadata_score,
                "description_dense": description_score,
                "metadata_dense": metadata_score,
            }
        return payload

    def _search_legacy(self, query_emb: np.ndarray, top_k: int) -> Dict[int, Dict[str, float]]:
        distances, ids = self.legacy_index.search(query_emb, k=top_k)
        payload: Dict[int, Dict[str, float]] = {}
        for sim, idx in zip(distances[0], ids[0]):
            if idx >= 0:
                score = max(0.0, float(sim))
                payload[int(idx)] = {"dense": score, "description_dense": score, "metadata_dense": 0.0}
        return payload
