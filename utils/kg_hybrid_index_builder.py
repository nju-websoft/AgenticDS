from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, List, Optional, Sequence

import faiss
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from global_values import EMBEDDING_MODEL_NAME, INDEX_CACHE_DIR, INDEX_ENTITY_TYPES, KG_DIR, LOCAL_EMBEDDING_BATCH_SIZE
from global_values import (
    LOCAL_EMBEDDING_DESCRIPTION_BATCH_SIZE,
    LOCAL_EMBEDDING_DOCUMENT_MAX_LENGTH,
    LOCAL_EMBEDDING_METADATA_MAX_LENGTH,
    LOCAL_EMBEDDING_METADATA_BATCH_SIZE,
)
from utils.embedding_backend import LocalEmbeddingBackend
from utils.kg_dense_index import FieldwiseDenseIndex
from utils.kg_index_store import dense_entity_texts, index_files, load_kg_entities, normalize_entity_types, save_index_entities
from utils.kg_sparse_index import BM25SparseIndex
from utils.lucene_runtime import lucene_classes


class KGHybridIndexBuilder:
    """Build field-wise dense FAISS indexes and a BM25 sparse index for KG entities."""

    def __init__(
        self,
        *,
        datasets_file: Optional[str] = None,
        papers_file: Optional[str] = None,
        collections_file: Optional[str] = None,
        models_file: Optional[str] = None,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        cache_dir: str = INDEX_CACHE_DIR,
        kg_dir: str = KG_DIR,
        entity_types: Optional[Sequence[str]] = None,
        description_batch_size: int = LOCAL_EMBEDDING_DESCRIPTION_BATCH_SIZE,
        metadata_batch_size: int = LOCAL_EMBEDDING_METADATA_BATCH_SIZE,
        document_max_length: int = LOCAL_EMBEDDING_DOCUMENT_MAX_LENGTH,
        metadata_max_length: int = LOCAL_EMBEDDING_METADATA_MAX_LENGTH,
        progress_file: Optional[str] = None,
    ) -> None:
        self.datasets_file = datasets_file
        self.papers_file = papers_file
        self.collections_file = collections_file
        self.models_file = models_file
        self.model_name = embedding_model_name
        self.cache_dir = Path(cache_dir)
        self.kg_dir = kg_dir
        self.entity_types = normalize_entity_types(entity_types)
        self.description_batch_size = max(1, int(description_batch_size or LOCAL_EMBEDDING_DESCRIPTION_BATCH_SIZE))
        self.metadata_batch_size = max(1, int(metadata_batch_size or LOCAL_EMBEDDING_METADATA_BATCH_SIZE))
        self.document_max_length = max(1, int(document_max_length or LOCAL_EMBEDDING_DOCUMENT_MAX_LENGTH))
        self.metadata_max_length = max(1, int(metadata_max_length or LOCAL_EMBEDDING_METADATA_MAX_LENGTH))
        self.progress_file = Path(progress_file) if progress_file else None
        self.files = index_files(
            model_name=self.model_name,
            cache_dir=self.cache_dir,
            kg_dir=self.kg_dir,
            entity_types=self.entity_types,
        )
        self.files.cache_dir.mkdir(parents=True, exist_ok=True)
        lucene_classes()
        self.embedding_backend = LocalEmbeddingBackend(self.model_name)
        self.build_start_time = 0.0

    def build(self) -> None:
        self.build_start_time = perf_counter()
        self._write_progress(status="running", phase="loading_entities", processed=0, total=0)
        print(f"Loading entities for types: {', '.join(self.entity_types)}...")
        entities = load_kg_entities(
            kg_dir=self.kg_dir,
            datasets_file=self.datasets_file,
            papers_file=self.papers_file,
            collections_file=self.collections_file,
            models_file=self.models_file,
            entity_types=self.entity_types,
        )
        if not entities:
            print("No entities found.")
            self._write_progress(status="failed", phase="loading_entities", processed=0, total=0, error="No entities found")
            return

        print(
            f"Computing field-wise embeddings for {len(entities)} entities "
            f"(entity scope: {self.files.entity_scope})..."
        )
        self._write_progress(
            status="running",
            phase="preparing_texts",
            processed=0,
            total=len(entities) * 2,
            entity_count=len(entities),
        )
        description_texts: List[str] = []
        metadata_texts: List[str] = []
        for entity in entities:
            description_text, metadata_text = dense_entity_texts(entity)
            description_texts.append(description_text)
            metadata_texts.append(metadata_text)

        total_dense_texts = len(description_texts) + len(metadata_texts)
        description_embeddings = self._encode_documents(
            description_texts,
            field_name="description",
            completed_dense_texts_before=0,
            total_dense_texts=total_dense_texts,
            batch_size=self.description_batch_size,
            max_length=self.document_max_length,
        )
        metadata_embeddings = self._encode_documents(
            metadata_texts,
            field_name="metadata",
            completed_dense_texts_before=len(description_texts),
            total_dense_texts=total_dense_texts,
            batch_size=self.metadata_batch_size,
            max_length=self.metadata_max_length,
        )
        faiss.normalize_L2(description_embeddings)
        faiss.normalize_L2(metadata_embeddings)

        print(f"Embedding dimension: {description_embeddings.shape[-1]}")
        print("[phase] Building FAISS indexes...")
        self._write_progress(
            status="running",
            phase="building_faiss",
            processed=total_dense_texts,
            total=total_dense_texts,
            entity_count=len(entities),
        )
        dense_index = FieldwiseDenseIndex.build(description_embeddings, metadata_embeddings)
        dense_index.write(self.files)
        print(f"Saved description FAISS index to {self.files.description_index}")
        print(f"Saved metadata FAISS index to {self.files.metadata_index}")
        print(f"Total embeddings indexed per dense field: {dense_index.ntotal}")

        print("[phase] Building BM25 sparse index...")
        self._write_progress(
            status="running",
            phase="building_bm25",
            processed=0,
            total=len(entities),
            entity_count=len(entities),
        )
        if self.files.sparse_index.exists():
            sparse_index = BM25SparseIndex.load(self.files.sparse_index)
            sparse_index.validate_entities(entities)
        else:
            sparse_index = BM25SparseIndex.build_from_entities(
                entities,
                path=self.files.sparse_index,
                metadata={"snapshot_id": self.files.kg_name, "entity_scope": self.files.entity_scope},
            )
        sparse_index.close()
        print(f"Saved BM25 sparse index to {self.files.sparse_index}")

        save_index_entities(self.files.entities, entities)
        print(f"Saved entities to {self.files.entities}")
        total_elapsed = perf_counter() - self.build_start_time
        self._write_progress(
            status="completed",
            phase="completed",
            processed=len(entities),
            total=len(entities),
            entity_count=len(entities),
            elapsed_seconds=total_elapsed,
            eta_seconds=0.0,
        )
        print(f"[done] Total build time: {self._format_duration(total_elapsed)}")

    def mark_failed(self, error: BaseException) -> None:
        self._write_progress(
            status="failed",
            phase="failed",
            processed=0,
            total=0,
            error=f"{type(error).__name__}: {error}",
        )

    def _encode_documents(
        self,
        texts: List[str],
        *,
        field_name: str,
        completed_dense_texts_before: int,
        total_dense_texts: int,
        batch_size: int,
        max_length: int,
    ) -> np.ndarray:
        total = len(texts)
        print(
            f"[phase] Encoding {field_name} texts: {total} "
            f"(batch_size={batch_size}, max_length={max_length})"
        )
        ordered_indices = sorted(range(total), key=lambda idx: len(texts[idx]))
        output_embeddings: Optional[np.ndarray] = None
        field_start_time = perf_counter()
        progress_interval = max(batch_size * 10, max(1, total // 100))
        next_log_at = min(total, progress_interval)
        for start in range(0, total, batch_size):
            batch_indices = ordered_indices[start : start + batch_size]
            batch = [texts[idx] for idx in batch_indices]
            embeddings = self.embedding_backend.encode(
                batch,
                batch_size=batch_size,
                is_query=False,
                max_length=max_length,
            )
            if output_embeddings is None:
                output_embeddings = np.empty((total, embeddings.shape[-1]), dtype=np.float32)
            output_embeddings[np.asarray(batch_indices, dtype=np.int64)] = embeddings
            if self.embedding_backend.device.startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
            processed = min(total, start + len(batch))
            if processed >= next_log_at or processed == total:
                self._log_progress(
                    field_name=field_name,
                    processed=processed,
                    total=total,
                    completed_dense_texts_before=completed_dense_texts_before,
                    total_dense_texts=total_dense_texts,
                    field_start_time=field_start_time,
                )
                next_log_at = min(total, processed + progress_interval)
        if output_embeddings is None:
            return np.array([], dtype=np.float32)
        return output_embeddings

    def _log_progress(
        self,
        *,
        field_name: str,
        processed: int,
        total: int,
        completed_dense_texts_before: int,
        total_dense_texts: int,
        field_start_time: float,
    ) -> None:
        dense_processed = completed_dense_texts_before + processed
        elapsed_total = perf_counter() - self.build_start_time
        elapsed_field = perf_counter() - field_start_time
        field_rate = processed / max(elapsed_field, 1e-6)
        dense_rate = dense_processed / max(elapsed_total, 1e-6)
        field_eta_seconds = (total - processed) / max(field_rate, 1e-6)
        dense_eta_seconds = (total_dense_texts - dense_processed) / max(dense_rate, 1e-6)
        print(
            f"[progress:{field_name}] "
            f"field {processed}/{total} ({processed / max(total, 1):.1%}) | "
            f"dense total {dense_processed}/{total_dense_texts} ({dense_processed / max(total_dense_texts, 1):.1%}) | "
            f"field rate {field_rate:.1f}/s | dense rate {dense_rate:.1f}/s | "
            f"field ETA {self._format_duration(field_eta_seconds)} | "
            f"dense ETA {self._format_duration(dense_eta_seconds)}"
        )
        self._write_progress(
            status="running",
            phase=f"encoding_{field_name}",
            processed=dense_processed,
            total=total_dense_texts,
            field_processed=processed,
            field_total=total,
            field_rate_per_second=field_rate,
            dense_rate_per_second=dense_rate,
            eta_seconds=dense_eta_seconds,
        )

    def _write_progress(
        self,
        *,
        status: str,
        phase: str,
        processed: int,
        total: int,
        **extra: Any,
    ) -> None:
        if self.progress_file is None:
            return
        elapsed = perf_counter() - self.build_start_time if self.build_start_time else 0.0
        payload = {
            "status": status,
            "phase": phase,
            "pid": os.getpid(),
            "embedding_model": self.model_name,
            "entity_types": list(self.entity_types),
            "processed": int(processed),
            "total": int(total),
            "progress": (float(processed) / float(total)) if total else 0.0,
            "elapsed_seconds": float(extra.pop("elapsed_seconds", elapsed)),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(f"{self.progress_file}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.progress_file)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h{minutes:02d}m{secs:02d}s"
        if minutes:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AgenticDS hybrid retrieval indexes.")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--index-cache-dir", default=INDEX_CACHE_DIR)
    parser.add_argument("--kg-dir", default=KG_DIR)
    parser.add_argument("--entity-types", default=",".join(INDEX_ENTITY_TYPES))
    parser.add_argument("--datasets-file", default="")
    parser.add_argument("--papers-file", default="")
    parser.add_argument("--collections-file", default="")
    parser.add_argument("--models-file", default="")
    parser.add_argument("--description-batch-size", type=int, default=LOCAL_EMBEDDING_DESCRIPTION_BATCH_SIZE)
    parser.add_argument("--metadata-batch-size", type=int, default=LOCAL_EMBEDDING_METADATA_BATCH_SIZE)
    parser.add_argument("--document-max-length", type=int, default=LOCAL_EMBEDDING_DOCUMENT_MAX_LENGTH)
    parser.add_argument("--metadata-max-length", type=int, default=LOCAL_EMBEDDING_METADATA_MAX_LENGTH)
    parser.add_argument("--progress-file", default="")
    args = parser.parse_args()

    builder = KGHybridIndexBuilder(
        datasets_file=args.datasets_file or None,
        papers_file=args.papers_file or None,
        collections_file=args.collections_file or None,
        models_file=args.models_file or None,
        embedding_model_name=args.embedding_model,
        cache_dir=args.index_cache_dir,
        kg_dir=args.kg_dir,
        entity_types=[item.strip() for item in str(args.entity_types).split(",") if item.strip()],
        description_batch_size=args.description_batch_size,
        metadata_batch_size=args.metadata_batch_size,
        document_max_length=args.document_max_length,
        metadata_max_length=args.metadata_max_length,
        progress_file=args.progress_file or None,
    )
    try:
        builder.build()
    except BaseException as exc:
        builder.mark_failed(exc)
        raise


if __name__ == "__main__":
    main()
