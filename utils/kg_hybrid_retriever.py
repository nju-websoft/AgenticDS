from __future__ import annotations

import json
import resource
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence

import faiss
import numpy as np

from global_values import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_QUERY_INSTRUCTION,
    HYBRID_DENSE_TOP_K_SEARCH,
    HYBRID_DENSE_WEIGHT,
    HYBRID_DESCRIPTION_DENSE_WEIGHT,
    HYBRID_METADATA_DENSE_WEIGHT,
    HYBRID_RRF_K,
    HYBRID_SPARSE_TOP_K_SEARCH,
    HYBRID_SPARSE_WEIGHT,
    INDEX_CACHE_DIR,
    INDEX_ENTITY_TYPES,
    KG_DIR,
)
from utils.embedding_backend import LocalEmbeddingBackend, default_query_instruction
from utils.kg_dense_index import FieldwiseDenseIndex
from utils.kg_index_store import (
    entity_type_totals,
    load_index_entities,
    prepare_query_texts,
    resolve_readable_index_files,
)
from utils.kg_sparse_index import BM25SparseIndex


class KGHybridRetriever:
    """Search KG entities with field-wise dense retrieval plus BM25 sparse retrieval."""

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL_NAME,
        cache_dir: str = INDEX_CACHE_DIR,
        kg_dir: str = KG_DIR,
        entity_types: Sequence[str] | None = None,
        embedding_backend: LocalEmbeddingBackend | None = None,
        initialize_embedding_backend: bool = True,
    ) -> None:
        initialization_start = perf_counter()
        self.model_name = embedding_model
        self.requested_cache_dir = Path(cache_dir)
        self.kg_dir = kg_dir
        self.entity_types = list(entity_types or INDEX_ENTITY_TYPES)
        self.files = resolve_readable_index_files(
            model_name=self.model_name,
            cache_dir=self.requested_cache_dir,
            kg_dir=self.kg_dir,
            entity_types=self.entity_types,
        )
        self.cache_dir = self.files.cache_dir
        if self.cache_dir != self.requested_cache_dir:
            print(f"Index files were not found in {self.requested_cache_dir}; using {self.cache_dir}.")

        if not self.files.has_entities:
            raise FileNotFoundError(
                f"Index cache not found at {self.cache_dir}. "
                f"Expected file: {self.files.entities.name}. "
                "Please run 'python utils/kg_hybrid_index_builder.py' first."
            )

        if not (self.files.sparse_index / "manifest.json").is_file():
            raise FileNotFoundError(
                f"Lucene BM25 index missing: {self.files.sparse_index}. "
                "Run python -m utils.kg_lucene_index_builder before loading dense indexes."
            )

        print(
            f"[kg-retriever][entities][start] scope={self.files.entity_scope} "
            f"file={self.files.entities} size_gib={self._file_size_gib(self.files.entities):.2f} "
            f"peak_rss_gib={self._peak_rss_gib():.2f}",
            flush=True,
        )
        stage_start = perf_counter()
        self.entities = load_index_entities(self.files.entities)
        self.entity_type_totals = entity_type_totals(self.entities)
        print(
            f"[kg-retriever][entities][done] scope={self.files.entity_scope} "
            f"count={len(self.entities)} elapsed_seconds={perf_counter() - stage_start:.1f} "
            f"peak_rss_gib={self._peak_rss_gib():.2f}",
            flush=True,
        )
        print(
            f"[kg-retriever][dense][start] scope={self.files.entity_scope} "
            f"description_gib={self._file_size_gib(self.files.description_index):.2f} "
            f"metadata_gib={self._file_size_gib(self.files.metadata_index):.2f} "
            f"peak_rss_gib={self._peak_rss_gib():.2f}",
            flush=True,
        )
        stage_start = perf_counter()
        self.dense_index = FieldwiseDenseIndex.load(
            self.files,
            description_weight=HYBRID_DESCRIPTION_DENSE_WEIGHT,
            metadata_weight=HYBRID_METADATA_DENSE_WEIGHT,
        )
        print(
            f"[kg-retriever][dense][done] scope={self.files.entity_scope} "
            f"vectors={self.dense_index.ntotal} elapsed_seconds={perf_counter() - stage_start:.1f} "
            f"peak_rss_gib={self._peak_rss_gib():.2f}",
            flush=True,
        )
        self.sparse_index = self._load_sparse_index()
        self.query_instruction = EMBEDDING_QUERY_INSTRUCTION or default_query_instruction(self.model_name)
        self.embedding_backend: LocalEmbeddingBackend | None = None
        stage_start = perf_counter()
        if embedding_backend is not None:
            self.set_embedding_backend(embedding_backend)
        elif initialize_embedding_backend:
            print(
                f"[kg-retriever][embedding][start] model={self.model_name} "
                f"peak_rss_gib={self._peak_rss_gib():.2f}",
                flush=True,
            )
            self.set_embedding_backend(LocalEmbeddingBackend(self.model_name))
        print(
            f"[kg-retriever][ready] scope={self.files.entity_scope} model={self.model_name} "
            f"encoder={'attached' if self.embedding_backend is not None else 'deferred'} "
            f"embedding_seconds={perf_counter() - stage_start:.1f} "
            f"total_seconds={perf_counter() - initialization_start:.1f} "
            f"peak_rss_gib={self._peak_rss_gib():.2f}",
            flush=True,
        )

    def set_embedding_backend(self, embedding_backend: LocalEmbeddingBackend) -> None:
        if embedding_backend.model_name != self.model_name:
            raise ValueError(
                f"Embedding backend model mismatch: index={self.model_name} "
                f"backend={embedding_backend.model_name}"
            )
        self.embedding_backend = embedding_backend

    def _load_sparse_index(self) -> BM25SparseIndex:
        if self.files.sparse_index.exists():
            print(
                f"[kg-retriever][sparse][start] scope={self.files.entity_scope} "
                f"backend=lucene directory={self.files.sparse_index} "
                f"peak_rss_gib={self._peak_rss_gib():.2f}",
                flush=True,
            )
            stage_start = perf_counter()
            sparse_index = BM25SparseIndex.load(self.files.sparse_index)
            sparse_index.validate_entities(self.entities)
            print(
                f"[kg-retriever][sparse][done] scope={self.files.entity_scope} "
                f"documents={sparse_index.document_count} "
                f"elapsed_seconds={perf_counter() - stage_start:.1f} "
                f"peak_rss_gib={self._peak_rss_gib():.2f}",
                flush=True,
            )
            return sparse_index
        raise FileNotFoundError(
            f"Lucene BM25 index missing: {self.files.sparse_index}. "
            "Run python -m utils.kg_lucene_index_builder before starting retrieval."
        )

    @staticmethod
    def _peak_rss_gib() -> float:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)

    @staticmethod
    def _file_size_gib(path: Path) -> float:
        try:
            return float(path.stat().st_size) / (1024.0**3)
        except OSError:
            return 0.0

    def search(self, query: Any, top_k: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        results, _ = self.search_with_timings(query, top_k=top_k)
        return results

    def search_with_timings(
        self,
        query: Any,
        top_k: int = 10,
    ) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, float]]:
        texts = prepare_query_texts(query)
        encode_start = perf_counter()
        query_emb = self._encode_query(texts)
        encode_seconds = perf_counter() - encode_start
        dense_start = perf_counter()
        search_k = min(len(self.entities), max(top_k * 50, HYBRID_DENSE_TOP_K_SEARCH))
        dense_scores = self.dense_index.search(query_emb, top_k=search_k)
        dense_seconds = perf_counter() - dense_start
        sparse_start = perf_counter()
        sparse_scores = self._sparse_scores(texts, top_k=top_k)
        sparse_seconds = perf_counter() - sparse_start
        fusion_start = perf_counter()
        results = self._fuse_results(dense_scores, sparse_scores, top_k=top_k)
        fusion_seconds = perf_counter() - fusion_start
        return results, {
            "query_encoding_seconds": encode_seconds,
            "dense_search_seconds": dense_seconds,
            "sparse_search_seconds": sparse_seconds,
            "fusion_seconds": fusion_seconds,
        }

    def search_sparse(self, query: Any, top_k: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Run only BM25 retrieval. Used by keyword baselines and diagnostics."""
        texts = prepare_query_texts(query)
        sparse_scores = self._sparse_scores(texts, top_k=top_k)
        ranked = sorted(sparse_scores.items(), key=lambda item: item[1], reverse=True)
        return self._format_ranked_results(ranked, sparse_scores=sparse_scores, top_k=top_k)

    def search_dense(self, query: Any, top_k: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Run only dense retrieval. Useful for ablations."""
        texts = prepare_query_texts(query)
        dense_payload = self._dense_scores(texts, top_k=top_k)
        dense_scores = {idx: payload.get("dense", 0.0) for idx, payload in dense_payload.items()}
        ranked = sorted(dense_scores.items(), key=lambda item: item[1], reverse=True)
        return self._format_ranked_results(ranked, dense_payload=dense_payload, top_k=top_k)

    def _dense_scores(self, texts: Sequence[str], *, top_k: int) -> Dict[int, Dict[str, float]]:
        query_emb = self._encode_query(texts)
        search_k = min(len(self.entities), max(top_k * 50, HYBRID_DENSE_TOP_K_SEARCH))
        return self.dense_index.search(query_emb, top_k=search_k)

    def _sparse_scores(self, texts: Sequence[str], *, top_k: int) -> Dict[int, float]:
        search_k = min(len(self.entities), max(top_k * 50, HYBRID_SPARSE_TOP_K_SEARCH))
        return self.sparse_index.search(" ".join(texts), top_k=search_k)

    def _encode_query(self, texts: Sequence[str]) -> np.ndarray:
        if self.embedding_backend is None:
            raise RuntimeError("Query Encoder has not been attached to this retriever")
        texts_with_instruction = [self.query_instruction + text for text in texts]
        query_emb = self.embedding_backend.encode(list(texts_with_instruction), is_query=True).astype("float32")
        faiss.normalize_L2(query_emb)
        if query_emb.shape[0] > 1:
            query_emb = query_emb.mean(axis=0, keepdims=True)
            faiss.normalize_L2(query_emb)
        return query_emb

    def _fuse_results(
        self,
        dense_payload: Mapping[int, Mapping[str, float]],
        sparse_scores: Mapping[int, float],
        *,
        top_k: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        candidate_ids = sorted(set(dense_payload.keys()) | set(sparse_scores.keys()))
        dense_scores = {idx: dense_payload.get(idx, {}).get("dense", 0.0) for idx in candidate_ids}
        dense_ranks = self._rank_map(dense_scores)
        sparse_ranks = self._rank_map(sparse_scores)

        fused = []
        for idx in candidate_ids:
            score = 0.0
            if idx in dense_ranks:
                score += HYBRID_DENSE_WEIGHT / (HYBRID_RRF_K + dense_ranks[idx])
            if idx in sparse_ranks:
                score += HYBRID_SPARSE_WEIGHT / (HYBRID_RRF_K + sparse_ranks[idx])
            fused.append((idx, score))
        fused.sort(key=lambda item: item[1], reverse=True)

        return self._format_ranked_results(
            fused,
            dense_payload=dense_payload,
            sparse_scores=sparse_scores,
            dense_ranks=dense_ranks,
            sparse_ranks=sparse_ranks,
            top_k=top_k,
        )

    @staticmethod
    def _rank_map(score_map: Mapping[int, float]) -> Dict[int, int]:
        ranked = sorted(
            ((idx, float(score)) for idx, score in score_map.items() if float(score) > 0.0),
            key=lambda item: item[1],
            reverse=True,
        )
        return {idx: rank for rank, (idx, _) in enumerate(ranked, 1)}

    def _format_ranked_results(
        self,
        ranked: Sequence[tuple[int, float]],
        *,
        dense_payload: Mapping[int, Mapping[str, float]] | None = None,
        sparse_scores: Mapping[int, float] | None = None,
        dense_ranks: Mapping[int, int] | None = None,
        sparse_ranks: Mapping[int, int] | None = None,
        top_k: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        dense_payload = dense_payload or {}
        sparse_scores = sparse_scores or {}
        dense_ranks = dense_ranks or {}
        sparse_ranks = sparse_ranks or {}
        results: Dict[str, List[Dict[str, Any]]] = {entity_type: [] for entity_type in self.entity_type_totals}

        for idx, score in ranked:
            if idx < 0 or idx >= len(self.entities):
                continue
            entity = self.entities[idx]
            entity_type = entity.get("entity_type", "unknown")
            if len(results.setdefault(entity_type, [])) >= top_k:
                continue

            dense_item = dense_payload.get(idx, {})
            results[entity_type].append(
                {
                    "entity": entity,
                    "similarity": float(score),
                    "dense_score": float(dense_item.get("dense", 0.0)),
                    "sparse_score": float(sparse_scores.get(idx, 0.0)),
                    "desc_dense_score": float(dense_item.get("description_dense", 0.0)),
                    "metadata_dense_score": float(dense_item.get("metadata_dense", 0.0)),
                    "dense_rank": int(dense_ranks.get(idx, 0)),
                    "sparse_rank": int(sparse_ranks.get(idx, 0)),
                }
            )

        return results

    def display(self, results: Dict[str, List[Dict]], n: int = 5) -> None:
        print(f"\n{'=' * 80}")
        print("Top KG Results")
        print(f"{'=' * 80}\n")
        for entity_type, items in results.items():
            if not items:
                continue
            print(f"{entity_type.upper()} (showing {min(n, len(items))} of {len(items)})")
            for idx, item in enumerate(items[:n], 1):
                entity = item["entity"]
                header = f"{idx}. [{item['similarity']:.4f}] {entity.get('entity_type', entity_type)} | {entity.get('id', 'N/A')}"
                print(header)
                json_text = json.dumps(entity, ensure_ascii=False, sort_keys=True)
                print(f"    {json_text[:77] + '...' if len(json_text) > 80 else json_text}")
                print()

    def save(self, results: Dict[str, List[Dict]], output_file: str) -> None:
        payload = {"model": self.model_name, "results": results}
        with open(output_file, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"Saved to {output_file}")


class KGEmbeddingRetriever(KGHybridRetriever):
    """Backward-compatible alias for historical class name."""
