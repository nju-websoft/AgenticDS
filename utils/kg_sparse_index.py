from __future__ import annotations

import hashlib
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping

from global_values import HYBRID_BM25_B, HYBRID_BM25_K1
from utils.kg_index_store import sparse_entity_text
from utils.lucene_runtime import LUCENE_VERSION, lucene_classes


SPARSE_INDEX_VERSION = 2
TOKENIZER = "regex:[a-z0-9]+|cjk-char"


def tokenize_bm25(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", str(text or "").lower())


def _entity_key(entity: Mapping[str, Any]) -> str:
    return f"{entity.get('entity_type') or 'dataset'}\0{entity.get('id') or ''}"


class BM25SparseIndex:
    """Apache Lucene BM25 with the project's original document tokenization."""

    def __init__(self, path: Path, metadata: dict, temporary: tempfile.TemporaryDirectory | None = None) -> None:
        classes = lucene_classes()
        self.path = path
        self.metadata = metadata
        self.document_count = int(metadata["document_count"])
        self.k1 = float(metadata["k1"])
        self.b = float(metadata["b"])
        self._temporary = temporary
        self.directory = classes["FSDirectory"].open(classes["Paths"].get(str(path.resolve())))
        self.reader = classes["DirectoryReader"].open(self.directory)
        if self.reader.numDocs() != self.document_count:
            self.close()
            raise ValueError(f"Lucene document count does not match manifest: {path}")
        self.searcher = classes["IndexSearcher"](self.reader)
        self.searcher.setSimilarity(classes["BM25Similarity"](self.k1, self.b))

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        *,
        path: str | Path | None = None,
        k1: float = HYBRID_BM25_K1,
        b: float = HYBRID_BM25_B,
        progress: Callable[[int], None] | None = None,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> "BM25SparseIndex":
        return cls._build_records(
            ((str(i), text) for i, text in enumerate(texts)),
            path=path, k1=k1, b=b, progress=progress, metadata=metadata, overwrite=overwrite,
        )

    @classmethod
    def build_from_entities(cls, entities: Iterable[Mapping[str, Any]], **kwargs: Any) -> "BM25SparseIndex":
        return cls._build_records(
            ((_entity_key(entity), sparse_entity_text(entity)) for entity in entities), **kwargs,
        )

    @classmethod
    def _build_records(
        cls,
        records: Iterable[tuple[str, str]],
        *,
        path: str | Path | None = None,
        k1: float = HYBRID_BM25_K1,
        b: float = HYBRID_BM25_B,
        progress: Callable[[int], None] | None = None,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> "BM25SparseIndex":
        classes = lucene_classes()
        temporary = tempfile.TemporaryDirectory(prefix="agenticds-lucene-") if path is None else None
        target = Path(temporary.name) / "index" if temporary else Path(path)
        if target.exists() and not overwrite:
            raise FileExistsError(f"Index already exists: {target}; pass overwrite=True to rebuild")
        target.parent.mkdir(parents=True, exist_ok=True)
        identity_digest = hashlib.sha256()
        text_digest = hashlib.sha256()
        count = 0
        oversized_terms = 0
        with tempfile.TemporaryDirectory(prefix=f".{target.name}.building-", dir=target.parent) as stage:
            directory = classes["FSDirectory"].open(classes["Paths"].get(stage))
            analyzer = classes["WhitespaceAnalyzer"](32766)
            config = classes["IndexWriterConfig"](analyzer)
            config.setRAMBufferSizeMB(256.0)
            config.setSimilarity(classes["BM25Similarity"](float(k1), float(b)))
            writer = classes["IndexWriter"](directory, config)
            try:
                for row_id, (identity, text) in enumerate(records):
                    tokens = tokenize_bm25(text)
                    valid = [term for term in tokens if len(term.encode("utf-8")) <= 32766]
                    oversized_terms += len(tokens) - len(valid)
                    contents = " ".join(valid)
                    document = classes["Document"]()
                    document.add(classes["StoredField"]("row_id", str(row_id)))
                    document.add(classes["TextField"]("contents", contents, classes["Store"].NO))
                    writer.addDocument(document)
                    identity_digest.update(identity.encode("utf-8") + b"\n")
                    text_digest.update(contents.encode("utf-8") + b"\n")
                    count += 1
                    if progress and count % 1000 == 0:
                        progress(count)
                if progress:
                    progress(count)
                writer.close()
            except BaseException:
                writer.rollback()
                raise
            finally:
                analyzer.close()
                directory.close()
            manifest = {
                "version": SPARSE_INDEX_VERSION,
                "backend": "lucene",
                "lucene_version": LUCENE_VERSION,
                "tokenizer": TOKENIZER,
                "k1": float(k1),
                "b": float(b),
                "document_count": count,
                "entity_ids_sha256": identity_digest.hexdigest(),
                "tokenized_text_sha256": text_digest.hexdigest(),
                "oversized_terms_omitted": oversized_terms,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": dict(metadata or {}),
            }
            (Path(stage) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            if target.exists():
                if not overwrite:
                    raise FileExistsError(f"Index appeared during construction: {target}; it has not been overwritten")
                backup = target.with_name(target.name + ".backup-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
                target.rename(backup)
                print(f"[lucene] previous index retained at {backup}", flush=True)
            Path(stage).rename(target)
        return cls(target, manifest, temporary=temporary)

    @classmethod
    def load(cls, path: str | Path) -> "BM25SparseIndex":
        path = Path(path)
        manifest_file = path / "manifest.json"
        if not manifest_file.is_file():
            raise FileNotFoundError(
                f"Lucene BM25 index not found: {path}. "
                "Rebuild sparse indexes with python -m utils.kg_lucene_index_builder; "
                "the old BM25 JSON files are not Lucene indexes."
            )
        metadata = json.loads(manifest_file.read_text(encoding="utf-8"))
        if (
            metadata.get("backend") != "lucene"
            or metadata.get("version") != SPARSE_INDEX_VERSION
            or metadata.get("lucene_version") != LUCENE_VERSION
            or metadata.get("tokenizer") != TOKENIZER
        ):
            raise ValueError(f"Unsupported Lucene BM25 manifest: {manifest_file}")
        return cls(path, metadata)

    def validate_entities(self, entities: Iterable[Mapping[str, Any]]) -> None:
        digest = hashlib.sha256()
        count = 0
        for entity in entities:
            digest.update(_entity_key(entity).encode("utf-8") + b"\n")
            count += 1
        if count != self.document_count or digest.hexdigest() != self.metadata["entity_ids_sha256"]:
            raise ValueError(f"Lucene row order does not match the entity/dense index: {self.path}")

    def search(self, query_text: str, top_k: int) -> Dict[int, float]:
        terms = sorted({term for term in tokenize_bm25(query_text) if len(term.encode("utf-8")) <= 32766})
        if not terms or not self.document_count:
            return {}
        classes = lucene_classes()
        if len(terms) > classes["IndexSearcher"].getMaxClauseCount():
            classes["IndexSearcher"].setMaxClauseCount(len(terms))
        builder = classes["BooleanQueryBuilder"]()
        for term in terms:
            builder.add(classes["TermQuery"](classes["Term"]("contents", term)), classes["Occur"].SHOULD)
        hits = self.searcher.search(builder.build(), max(1, int(top_k))).scoreDocs
        stored_fields = self.searcher.storedFields()
        return {int(stored_fields.document(hit.doc).get("row_id")): float(hit.score) for hit in hits}

    def close(self) -> None:
        for name in ("reader", "directory"):
            value = getattr(self, name, None)
            if value is not None:
                setattr(self, name, None)
                value.close()
        temporary = getattr(self, "_temporary", None)
        if temporary is not None:
            self._temporary = None
            temporary.cleanup()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def build_sparse_dataset_lookup(entities: Iterable[Mapping[str, object]]) -> Dict[int, str]:
    return {
        idx: str(entity.get("id"))
        for idx, entity in enumerate(entities)
        if entity.get("entity_type") == "dataset" and entity.get("id")
    }
