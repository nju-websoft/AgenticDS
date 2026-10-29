from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import ijson

from global_values import INDEX_CACHE_DIR, INDEX_ENTITY_TYPES, KG_DIR, LEGACY_EMBEDDING_CACHE_DIR
from utils.embedding_backend import sanitize_model_name


DENSE_METADATA_KEYS = [
    "tasks",
    "sub-tasks",
    "tags",
    "modalities",
    "formats",
    "languages",
    "libraries",
    "other",
    "authors",
]

SUPPORTED_ENTITY_TYPES = ("dataset", "paper", "collection", "model")
ENTITY_DESCRIPTION_MAX_CHARS = 12_000


@dataclass(frozen=True)
class KGIndexFiles:
    cache_dir: Path
    kg_name: str
    model_name: str
    entity_scope: str
    legacy_index: Path
    description_index: Path
    metadata_index: Path
    sparse_index: Path
    entities: Path

    @property
    def has_entities(self) -> bool:
        return self.entities.exists()

    @property
    def has_fieldwise_dense(self) -> bool:
        return self.description_index.exists() and self.metadata_index.exists()

    @property
    def has_legacy_dense(self) -> bool:
        return self.legacy_index.exists()

    @property
    def has_dense(self) -> bool:
        return self.has_fieldwise_dense or self.has_legacy_dense

    @property
    def is_readable(self) -> bool:
        return self.has_entities and self.has_dense


def index_files(
    *,
    model_name: str,
    cache_dir: str | Path = INDEX_CACHE_DIR,
    kg_dir: str | Path = KG_DIR,
    entity_types: Optional[Sequence[str]] = None,
) -> KGIndexFiles:
    target_dir = Path(cache_dir)
    kg_name = Path(kg_dir).name
    model_name_sanitized = sanitize_model_name(model_name)
    normalized_entity_types = normalize_entity_types(entity_types)
    entity_scope = "_".join(normalized_entity_types)
    file_stem = f"{kg_name}_{model_name_sanitized}_{entity_scope}"

    return KGIndexFiles(
        cache_dir=target_dir,
        kg_name=kg_name,
        model_name=model_name,
        entity_scope=entity_scope,
        legacy_index=target_dir / f"{file_stem}.index",
        description_index=target_dir / f"{file_stem}_description.index",
        metadata_index=target_dir / f"{file_stem}_metadata.index",
        sparse_index=target_dir / f"{kg_name}_{entity_scope}_bm25.lucene",
        entities=target_dir / f"{file_stem}.json",
    )


def resolve_readable_index_files(
    *,
    model_name: str,
    cache_dir: str | Path = INDEX_CACHE_DIR,
    kg_dir: str | Path = KG_DIR,
    legacy_cache_dir: str | Path = LEGACY_EMBEDDING_CACHE_DIR,
    entity_types: Optional[Sequence[str]] = None,
) -> KGIndexFiles:
    primary = index_files(
        model_name=model_name,
        cache_dir=cache_dir,
        kg_dir=kg_dir,
        entity_types=entity_types,
    )
    if primary.is_readable:
        return primary

    legacy_dir = Path(legacy_cache_dir)
    if primary.cache_dir != legacy_dir:
        legacy = index_files(
            model_name=model_name,
            cache_dir=legacy_dir,
            kg_dir=kg_dir,
            entity_types=entity_types,
        )
        if legacy.is_readable:
            return legacy

    return primary


def normalize_entity_types(entity_types: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    raw_values = list(entity_types or INDEX_ENTITY_TYPES)
    normalized: List[str] = []
    for value in raw_values:
        item = str(value or "").strip().lower()
        if not item:
            continue
        if item not in SUPPORTED_ENTITY_TYPES:
            raise ValueError(f"Unsupported entity type: {value}")
        if item not in normalized:
            normalized.append(item)
    if not normalized:
        raise ValueError("At least one entity type must be configured for indexing.")
    return tuple(normalized)


def load_kg_entities(
    *,
    kg_dir: str | Path = KG_DIR,
    datasets_file: Optional[str] = None,
    papers_file: Optional[str] = None,
    collections_file: Optional[str] = None,
    models_file: Optional[str] = None,
    entity_types: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    kg_path = Path(kg_dir)
    normalized_entity_types = normalize_entity_types(entity_types)
    entity_file_candidates: Dict[str, Path] = {
        "dataset": Path(datasets_file) if datasets_file else kg_path / "datasets.json",
        "paper": Path(papers_file) if papers_file else kg_path / "papers.json",
        "collection": Path(collections_file) if collections_file else kg_path / "collections.json",
        "model": Path(models_file) if models_file else kg_path / "models.json",
    }

    return list(
        iter_kg_entities(
            entity_types=normalized_entity_types,
            entity_files=entity_file_candidates,
        )
    )


def iter_kg_entities(
    *,
    entity_types: Sequence[str],
    entity_files: Mapping[str, Path],
) -> Iterator[Dict[str, Any]]:
    """Stream source JSON arrays and yield retrieval-oriented entity records."""
    for entity_type in normalize_entity_types(entity_types):
        file_path = Path(entity_files[entity_type])
        if not file_path.exists():
            print(f"Warning: File not found {file_path}")
            continue
        with file_path.open("rb") as fh:
            for item in ijson.items(fh, "item"):
                if isinstance(item, Mapping):
                    yield compact_kg_entity(item, entity_type=entity_type)


def compact_kg_entity(item: Mapping[str, Any], *, entity_type: str) -> Dict[str, Any]:
    """Normalize KG entity schemas while retaining fields required after retrieval."""
    normalized_type = normalize_entity_types([entity_type])[0]
    if normalized_type == "dataset":
        entity = dict(item)
        entity["entity_type"] = normalized_type
        return entity

    if normalized_type == "paper":
        entity = {
            "id": item.get("id"),
            "name": item.get("title") or item.get("id"),
            "description": item.get("summary") or "",
            "authors": _string_list(item.get("authors")),
            "published_at": item.get("publishedAt"),
            "upvotes": item.get("upvotes"),
        }
    elif normalized_type == "collection":
        entity = {
            "id": item.get("slug"),
            "name": item.get("title") or item.get("slug"),
            "description": item.get("description") or "",
            "upvotes": item.get("upvotes"),
        }
    else:
        entity = {
            "id": item.get("id"),
            "name": item.get("name") or item.get("id"),
            "description": item.get("description") or "",
            "tasks": _string_list(item.get("pipeline_tag")),
            "libraries": _string_list(item.get("libraries")),
            "languages": _string_list(item.get("languages")),
            "other": _string_list(item.get("other")),
            "license": item.get("license"),
            "downloads": item.get("downloads"),
            "likes": item.get("likes"),
            "created_at": item.get("createdAt"),
            "last_modified": item.get("lastModified"),
        }

    entity["entity_type"] = normalized_type
    entity["description"] = str(entity.get("description") or "")[:ENTITY_DESCRIPTION_MAX_CHARS]
    return {key: value for key, value in entity.items() if value not in (None, "", [])}


def _string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or value == "":
        return []
    return [str(value).strip()]


def load_index_entities(path: str | Path) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    with Path(path).open("rb") as fh:
        for item in ijson.items(fh, "item"):
            if isinstance(item, Mapping):
                entities.append(dict(item))
    return entities


def save_index_entities(path: str | Path, entities: Sequence[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(list(entities), fh, ensure_ascii=False)


def dense_entity_texts(entity: Mapping[str, Any]) -> Tuple[str, str]:
    description = str(entity.get("description") or "").strip()

    other_segments: List[str] = [
        str(entity.get("id") or "").strip(),
        str(entity.get("name") or "").strip(),
        f"entity_type: {entity.get('entity_type', '')}",
    ]

    for key in DENSE_METADATA_KEYS:
        values = entity.get(key)
        if isinstance(values, list) and values:
            normalized = ", ".join(str(v).strip() for v in values if str(v).strip())
            if normalized:
                other_segments.append(f"{key}: {normalized}")

    if entity.get("license"):
        other_segments.append(f"license: {entity.get('license')}")
    if entity.get("size"):
        other_segments.append(f"size: {entity.get('size')}")
    if entity.get("downloads") is not None:
        other_segments.append(f"downloads: {int(entity.get('downloads', 0))}")
    if entity.get("likes") is not None:
        other_segments.append(f"likes: {int(entity.get('likes', 0))}")
    if entity.get("upvotes") is not None:
        other_segments.append(f"upvotes: {int(entity.get('upvotes', 0))}")
    if entity.get("published_at"):
        other_segments.append(f"published_at: {entity.get('published_at')}")

    return description or str(entity.get("name") or ""), " ; ".join(part for part in other_segments if part)


def sparse_entity_text(entity: Mapping[str, Any]) -> str:
    parts: List[str] = [
        str(entity.get("id") or ""),
        str(entity.get("name") or ""),
        str(entity.get("description") or ""),
        str(entity.get("license") or ""),
        str(entity.get("size") or ""),
        str(entity.get("entity_type") or ""),
        str(entity.get("published_at") or ""),
    ]
    for key in DENSE_METADATA_KEYS:
        values = entity.get(key)
        if isinstance(values, list) and values:
            parts.append(" ".join(str(v) for v in values))
    for key in ("downloads", "likes", "upvotes"):
        if entity.get(key) is not None:
            parts.append(f"{key} {entity.get(key)}")
    return " ".join(p for p in parts if p)


def entity_type_totals(entities: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for entity in entities:
        entity_type = str(entity.get("entity_type") or "unknown")
        totals[entity_type] = totals.get(entity_type, 0) + 1
    return totals


def prepare_query_texts(query: Any) -> List[str]:
    if query is None:
        raise ValueError("query 参数不能为空")

    if isinstance(query, str):
        return [query]

    if isinstance(query, Mapping):
        query_text = str(query.get("query_text", "")).strip()
        if query_text:
            return [query_text]

        facets = query.get("facets")
        raw_query = str(query.get("query", "")).strip()
        pieces: List[str] = []
        if raw_query:
            pieces.append(raw_query)

        if isinstance(facets, Mapping):
            for key in ["tasks", "modalities", "formats", "languages", "licenses", "other", "libraries", "tags"]:
                values = facets.get(key)
                if isinstance(values, list) and values:
                    joined = ", ".join(str(item) for item in values if str(item).strip())
                    if joined:
                        pieces.append(f"{key}: {joined}")
            if facets.get("min_downloads") is not None:
                pieces.append(f"min_downloads: {int(facets['min_downloads'])}")
            if facets.get("min_likes") is not None:
                pieces.append(f"min_likes: {int(facets['min_likes'])}")

        if pieces:
            return [" ; ".join(pieces)]

        return [json.dumps(query, ensure_ascii=False, sort_keys=True)]

    if isinstance(query, Sequence) and not isinstance(query, (str, bytes, bytearray)):
        texts: List[str] = []
        for item in query:
            if isinstance(item, Mapping):
                texts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                texts.append(str(item))
        return texts

    return [str(query)]


def normalize_scores(score_map: Mapping[int, float], candidates: Sequence[int]) -> Dict[int, float]:
    values = [float(score_map.get(i, 0.0)) for i in candidates]
    if not values:
        return {}
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return {idx: (1.0 if score_map.get(idx, 0.0) > 0 else 0.0) for idx in candidates}
    return {idx: (float(score_map.get(idx, 0.0)) - lo) / (hi - lo) for idx in candidates}
