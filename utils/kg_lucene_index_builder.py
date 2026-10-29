from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import ijson

from global_values import EMBEDDING_MODEL_NAME, INDEX_CACHE_DIR, KG_DIR
from utils.kg_index_store import index_files, normalize_entity_types
from utils.kg_sparse_index import BM25SparseIndex
from utils.lucene_runtime import LUCENE_VERSION


def build_sparse_indexes(
    *,
    kg_dir: str,
    cache_dir: str,
    embedding_model: str,
    entity_types: list[str],
    progress_file: str = "",
    overwrite: bool = False,
) -> None:
    """Stream the existing aligned entity stores without loading an encoder or FAISS."""
    scopes = normalize_entity_types(entity_types)
    files = [index_files(model_name=embedding_model, kg_dir=kg_dir, cache_dir=cache_dir, entity_types=[scope]) for scope in scopes]
    started = perf_counter()
    total_bytes = 0
    completed_bytes = 0
    completed_documents = 0

    def report(status: str, phase: str, **extra) -> None:
        payload = {
            "status": status, "phase": phase, "pid": os.getpid(),
            "backend": "lucene", "lucene_version": LUCENE_VERSION,
            "snapshot": str(kg_dir), "entity_types": list(scopes),
            "completed_entity_types": list(scopes[:extra.get("scope_index", 0)]),
            "elapsed_seconds": perf_counter() - started,
            "updated_at": datetime.now(timezone.utc).isoformat(), **extra,
        }
        print("[lucene-build] " + json.dumps(payload, ensure_ascii=False), flush=True)
        if progress_file:
            path = Path(progress_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)

    try:
        report("running", "checking_sources")
        for item in files:
            if not item.entities.is_file():
                raise FileNotFoundError(f"Aligned entity store missing: {item.entities}; select the embedding model used by the existing dense indexes")
            if item.sparse_index.exists() and not overwrite:
                raise FileExistsError(f"Lucene index already exists: {item.sparse_index}; use --overwrite to rebuild and retain a backup")
            total_bytes += item.entities.stat().st_size
        for scope_index, item in enumerate(files):
            report("running", "indexing", entity_type=item.entity_scope, scope_index=scope_index, source=str(item.entities), output=str(item.sparse_index))
            with item.entities.open("rb") as source:
                def progress(count: int) -> None:
                    processed_bytes = completed_bytes + source.tell()
                    fraction = min(1.0, processed_bytes / max(1, total_bytes))
                    elapsed = perf_counter() - started
                    report(
                        "running", "indexing", entity_type=item.entity_scope, scope_index=scope_index,
                        processed_documents=completed_documents + count,
                        processed_source_bytes=processed_bytes, total_source_bytes=total_bytes,
                        progress=fraction,
                        source_read_eta_seconds=elapsed * (1.0 - fraction) / fraction if fraction else None,
                    )

                index = BM25SparseIndex.build_from_entities(
                    ijson.items(source, "item"), path=item.sparse_index,
                    progress=progress, overwrite=overwrite,
                    metadata={"snapshot_id": item.kg_name, "entity_scope": item.entity_scope, "entity_store": str(item.entities)},
                )
            completed_documents += index.document_count
            index.close()
            completed_bytes += item.entities.stat().st_size
            report("running", "entity_completed", entity_type=item.entity_scope, scope_index=scope_index + 1, processed_documents=completed_documents)
        report("completed", "completed", scope_index=len(files), processed_documents=completed_documents, progress=1.0)
    except BaseException as exc:
        report("failed", "failed", error=f"{type(exc).__name__}: {exc}", processed_documents=completed_documents)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild only Lucene BM25 indexes from existing aligned entity JSON files; no GPU or dense encoding.")
    parser.add_argument("--kg-dir", default=KG_DIR)
    parser.add_argument("--index-cache-dir", default=INDEX_CACHE_DIR)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME, help="Select the existing entity stores; the resulting Lucene indexes are shared across embeddings")
    parser.add_argument("--entity-types", default="paper,collection,dataset,model")
    parser.add_argument("--progress-file", default="logs/lucene-index-build.progress.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_sparse_indexes(
        kg_dir=args.kg_dir, cache_dir=args.index_cache_dir,
        embedding_model=args.embedding_model,
        entity_types=[part.strip() for part in args.entity_types.split(",") if part.strip()],
        progress_file=args.progress_file, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
