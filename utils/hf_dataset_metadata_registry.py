from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import requests

from global_values import (
    HF_DATASET_TAGS_API_URL,
    HF_DATASET_TAGS_CACHE_FILE,
    HF_DATASET_TAGS_CACHE_TTL_SECONDS,
    HF_DATASET_TAGS_TIMEOUT_SECONDS,
)


def _normalize_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", "-")
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_tag(value: Any) -> str:
    normalized = _normalize_token(value)
    if not normalized:
        return normalized
    if normalized.startswith("region") and ":" not in normalized:
        normalized = normalized.replace("region", "region:", 1)
    return normalized


def _id_to_value(raw_id: str) -> str:
    if ":" not in raw_id:
        return raw_id
    return raw_id.split(":", 1)[1]


class HFDatasetMetadataRegistry:
    """Canonical metadata registry built from HF `datasets-tags-by-type` API."""

    _default_instance: Optional["HFDatasetMetadataRegistry"] = None

    CATEGORY_TO_TYPES: Dict[str, Sequence[str]] = {
        "tasks": ("task_categories", "task_ids"),
        "modalities": ("modality",),
        "formats": ("format",),
        "licenses": ("license",),
        "languages": ("language",),
        "libraries": ("library",),
        "other": ("other",),
    }

    FILTER_LIST_FIELDS: Dict[str, str] = {
        "tasks_any": "tasks",
        "tasks_all": "tasks",
        "modalities_any": "modalities",
        "modalities_all": "modalities",
        "formats_any": "formats",
        "formats_all": "formats",
        "languages_any": "languages",
        "languages_all": "languages",
        "other_any": "other",
        "other_all": "other",
        "libraries_any": "libraries",
        "libraries_all": "libraries",
        "licenses": "licenses",
    }

    RETRIEVAL_LIST_FIELDS: Dict[str, str] = {
        "tasks": "tasks",
        "modalities": "modalities",
        "formats": "formats",
        "languages": "languages",
        "other": "other",
        "libraries": "libraries",
        "licenses": "licenses",
    }

    TASK_SYNONYMS = {
        "qa": "question-answering",
        "question answering": "question-answering",
        "q&a": "question-answering",
        "text generation": "text-generation",
        "text-generation": "text-generation",
        "text2text": "text2text-generation",
        "text to text": "text2text-generation",
        "classification": "text-classification",
        "summarization": "summarization",
        "summary": "summarization",
        "translation": "translation",
        "retrieval": "document-retrieval",
        "conversational": "conversational",
        "nsfw": "multi-class-image-classification",
        "nsfw-detection": "multi-class-image-classification",
        "adult-content-detection": "multi-class-image-classification",
        "content-moderation": "text-classification",
    }

    MODALITY_SYNONYMS = {
        "text": "text",
        "image": "image",
        "vision": "image",
        "audio": "audio",
        "speech": "audio",
        "video": "video",
        "tabular": "tabular",
        "table": "tabular",
        "timeseries": "timeseries",
        "time series": "timeseries",
        "document": "document",
        "geospatial": "geospatial",
    }

    LANGUAGE_SYNONYMS = {
        "chinese": "zh",
        "mandarin": "zh",
        "english": "en",
        "french": "fr",
        "german": "de",
        "japanese": "ja",
        "korean": "ko",
    }

    LICENSE_SYNONYMS = {
        "apache": "apache-2.0",
        "apache2": "apache-2.0",
        "apache-2": "apache-2.0",
        "mit": "mit",
        "gpl": "gpl",
        "cc by": "cc-by-4.0",
    }

    TAG_SYNONYMS = {
        "us": "region:us",
        "usa": "region:us",
        "eu": "region:eu",
        "cn": "region:cn",
        "china": "region:cn",
    }

    @classmethod
    def get_default(cls) -> "HFDatasetMetadataRegistry":
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    def __init__(
        self,
        *,
        api_url: str = HF_DATASET_TAGS_API_URL,
        cache_file: str = HF_DATASET_TAGS_CACHE_FILE,
        cache_ttl_seconds: int = HF_DATASET_TAGS_CACHE_TTL_SECONDS,
        timeout_seconds: int = HF_DATASET_TAGS_TIMEOUT_SECONDS,
    ) -> None:
        self.api_url = api_url
        self.cache_file = Path(cache_file)
        self.cache_ttl_seconds = cache_ttl_seconds
        self.timeout_seconds = timeout_seconds

        self.tags_by_type = self._load_tags_by_type()
        self.allowed_values: Dict[str, Set[str]] = {k: set() for k in self.CATEGORY_TO_TYPES}
        self.alias_to_canonical: Dict[str, Dict[str, str]] = {k: {} for k in self.CATEGORY_TO_TYPES}
        self._build_indexes()

    def _load_tags_by_type(self) -> Dict[str, List[Dict[str, Any]]]:
        payload = self._load_cache()
        if payload:
            return payload

        payload = self._fetch_remote()
        if payload:
            self._save_cache(payload)
            return payload

        return self._fallback_payload()

    def _load_cache(self) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        if not self.cache_file.exists():
            return None
        if self.cache_ttl_seconds > 0:
            age = time.time() - self.cache_file.stat().st_mtime
            if age > self.cache_ttl_seconds:
                return None

        try:
            with self.cache_file.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                return payload
        except Exception:
            return None
        return None

    def _save_cache(self, payload: Dict[str, List[Dict[str, Any]]]) -> None:
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_file.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
        except Exception:
            return

    def _fetch_remote(self) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        if not self.api_url:
            return None
        try:
            response = requests.get(self.api_url, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload
        except Exception:
            return None
        return None

    def _fallback_payload(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "task_categories": [
                {"id": "task_categories:text-generation", "label": "text-generation", "type": "task_categories"},
                {"id": "task_categories:translation", "label": "translation", "type": "task_categories"},
                {"id": "task_categories:question-answering", "label": "question-answering", "type": "task_categories"},
                {"id": "task_categories:text-classification", "label": "text-classification", "type": "task_categories"},
                {"id": "task_categories:summarization", "label": "summarization", "type": "task_categories"},
            ],
            "modality": [
                {"id": "modality:text", "label": "Text", "type": "modality"},
                {"id": "modality:image", "label": "Image", "type": "modality"},
                {"id": "modality:audio", "label": "Audio", "type": "modality"},
                {"id": "modality:tabular", "label": "Tabular", "type": "modality"},
                {"id": "modality:video", "label": "Video", "type": "modality"},
            ],
            "format": [
                {"id": "format:json", "label": "json", "type": "format"},
                {"id": "format:csv", "label": "csv", "type": "format"},
                {"id": "format:parquet", "label": "parquet", "type": "format"},
            ],
            "license": [
                {"id": "license:mit", "label": "mit", "type": "license"},
                {"id": "license:apache-2.0", "label": "apache-2.0", "type": "license"},
                {"id": "license:cc-by-4.0", "label": "cc-by-4.0", "type": "license"},
            ],
            "language": [
                {"id": "language:en", "label": "English", "type": "language"},
                {"id": "language:zh", "label": "Chinese", "type": "language"},
            ],
            "library": [
                {"id": "library:datasets", "label": "Datasets", "type": "library"},
                {"id": "library:pandas", "label": "pandas", "type": "library"},
                {"id": "library:polars", "label": "polars", "type": "library"},
            ],
            "other": [
                {"id": "medical", "label": "medical", "type": "other"},
                {"id": "finance", "label": "finance", "type": "other"},
                {"id": "legal", "label": "legal", "type": "other"},
                {"id": "code", "label": "code", "type": "other"},
            ],
        }

    def _build_indexes(self) -> None:
        for category, source_types in self.CATEGORY_TO_TYPES.items():
            alias_map = self.alias_to_canonical[category]
            allowed = self.allowed_values[category]

            for source_type in source_types:
                for item in self.tags_by_type.get(source_type, []):
                    raw_id = str(item.get("id", ""))
                    raw_label = str(item.get("label", ""))

                    canonical = _normalize_token(_id_to_value(raw_id) or raw_label)
                    if not canonical:
                        continue

                    allowed.add(canonical)

                    aliases = {
                        canonical,
                        _normalize_token(raw_id),
                        _normalize_token(raw_label),
                        _normalize_token(_id_to_value(raw_id)),
                    }
                    for alias in aliases:
                        if alias:
                            alias_map[alias] = canonical

            extra_synonyms = self._get_synonyms(category)
            for alias, canonical in extra_synonyms.items():
                if canonical in allowed:
                    alias_map[_normalize_token(alias)] = canonical

    def _get_synonyms(self, category: str) -> Dict[str, str]:
        if category == "tasks":
            return self.TASK_SYNONYMS
        if category == "modalities":
            return self.MODALITY_SYNONYMS
        if category == "languages":
            return self.LANGUAGE_SYNONYMS
        if category == "licenses":
            return self.LICENSE_SYNONYMS
        return {}

    def canonicalize(self, category: str, value: Any) -> Optional[str]:
        if value is None:
            return None

        if category == "tags":
            token = _normalize_tag(value)
            token = self.TAG_SYNONYMS.get(token, token)
            return token or None

        alias_map = self.alias_to_canonical.get(category)
        if not alias_map:
            token = _normalize_token(value)
            return token or None

        token = _normalize_token(value)
        if not token:
            return None

        if category == "languages" and re.fullmatch(r"[a-z]{2,3}", token):
            if token in self.allowed_values.get(category, set()):
                return token

        direct = alias_map.get(token)
        if direct:
            return direct

        if token in self.allowed_values.get(category, set()):
            return token

        return None

    def canonicalize_many(self, category: str, values: Optional[Iterable[Any]]) -> List[str]:
        if not values:
            return []
        ordered: List[str] = []
        seen: Set[str] = set()
        for value in values:
            normalized = self.canonicalize(category, value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    def infer_tasks_from_query_text(self, query_text: str, limit: int = 3) -> List[str]:
        text = _normalize_token(query_text)
        if not text:
            return []

        allowed = self.allowed_values.get("tasks", set())
        inferred: List[str] = []
        seen: Set[str] = set()

        def _push(task_name: str) -> None:
            task = _normalize_token(task_name)
            if not task or task in seen:
                return
            if task not in allowed:
                return
            seen.add(task)
            inferred.append(task)

        has_image = any(token in text for token in ["image", "picture", "photo", "visual", "vision"])
        has_safety = any(
            token in text
            for token in ["nsfw", "adult", "inappropriate", "moderation", "unsafe", "safety"]
        )

        if has_safety and has_image:
            _push("multi-class-image-classification")
            _push("image-classification")
            _push("multi-label-image-classification")
        elif has_safety:
            _push("text-classification")
            _push("multi-class-classification")

        keyword_task_map = {
            "object detection": "object-detection",
            "segmentation": "image-segmentation",
            "caption": "image-captioning",
            "speech recognition": "automatic-speech-recognition",
            "text to speech": "text-to-speech",
            "question answering": "question-answering",
            "qa": "question-answering",
            "translation": "translation",
            "summarization": "summarization",
            "classification": "text-classification",
        }
        for phrase, task_name in keyword_task_map.items():
            if phrase in text:
                _push(task_name)

        for alias, canonical in self.TASK_SYNONYMS.items():
            if _normalize_token(alias) in text:
                _push(canonical)

        return inferred[: max(1, int(limit))]

    def infer_modalities_from_query_text(self, query_text: str, limit: int = 2) -> List[str]:
        text = _normalize_token(query_text)
        if not text:
            return []

        allowed = self.allowed_values.get("modalities", set())
        inferred: List[str] = []
        seen: Set[str] = set()

        def _push(name: str) -> None:
            modality = _normalize_token(name)
            if not modality or modality in seen:
                return
            if modality not in allowed:
                return
            seen.add(modality)
            inferred.append(modality)

        if any(token in text for token in ["image", "picture", "photo", "visual", "vision"]):
            _push("image")
        if any(token in text for token in ["audio", "speech", "voice"]):
            _push("audio")
        if any(token in text for token in ["video", "clip"]):
            _push("video")
        if any(token in text for token in ["table", "tabular", "csv", "spreadsheet"]):
            _push("tabular")
        if any(token in text for token in ["text", "document", "corpus"]):
            _push("text")

        return inferred[: max(1, int(limit))]

    def sanitize_retrieval_facets(self, facets: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        payload = dict(facets or {})
        cleaned: Dict[str, Any] = {}

        for field, category in self.RETRIEVAL_LIST_FIELDS.items():
            normalized = self.canonicalize_many(category, payload.get(field))
            if normalized:
                cleaned[field] = normalized

        tags = self.canonicalize_many("tags", payload.get("tags"))
        if tags:
            cleaned["tags"] = tags

        for numeric_key in ("min_downloads", "min_likes"):
            if numeric_key in payload and payload[numeric_key] is not None:
                try:
                    cleaned[numeric_key] = max(0, int(payload[numeric_key]))
                except (TypeError, ValueError):
                    pass

        return cleaned

    def sanitize_metadata_filter(self, metadata_filter: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        payload = dict(metadata_filter or {})
        cleaned: Dict[str, Any] = {}

        ids = payload.get("ids")
        if ids:
            ordered_ids: List[str] = []
            seen_ids: Set[str] = set()
            for item in ids:
                normalized_id = str(item).strip()
                if not normalized_id or normalized_id in seen_ids:
                    continue
                seen_ids.add(normalized_id)
                ordered_ids.append(normalized_id)
            if ordered_ids:
                cleaned["ids"] = ordered_ids

        for field, category in self.FILTER_LIST_FIELDS.items():
            values = self.canonicalize_many(category, payload.get(field))
            if values:
                cleaned[field] = values

        tags_any = self.canonicalize_many("tags", payload.get("tags_any"))
        if tags_any:
            cleaned["tags_any"] = tags_any

        tags_all = self.canonicalize_many("tags", payload.get("tags_all"))
        if tags_all:
            cleaned["tags_all"] = tags_all

        text_contains = payload.get("text_contains")
        if text_contains:
            cleaned_text = str(text_contains).strip()
            if cleaned_text:
                cleaned["text_contains"] = cleaned_text

        for numeric_key in ("min_downloads", "min_likes"):
            if numeric_key in payload and payload[numeric_key] is not None:
                try:
                    cleaned[numeric_key] = max(0, int(payload[numeric_key]))
                except (TypeError, ValueError):
                    pass

        return cleaned

    def build_retrieval_query_text(self, raw_query: str, facets: Optional[Mapping[str, Any]] = None) -> str:
        pieces: List[str] = [str(raw_query or "").strip()]
        facet_payload = dict(facets or {})

        ordered_keys = [
            "tasks",
            "modalities",
            "formats",
            "languages",
            "licenses",
            "other",
            "libraries",
            "tags",
        ]
        for key in ordered_keys:
            values = facet_payload.get(key)
            if not values:
                continue
            if not isinstance(values, list):
                continue
            values_text = ", ".join(str(v) for v in values if str(v).strip())
            if values_text:
                pieces.append(f"{key}: {values_text}")

        if facet_payload.get("min_downloads") is not None:
            pieces.append(f"min_downloads: {int(facet_payload['min_downloads'])}")
        if facet_payload.get("min_likes") is not None:
            pieces.append(f"min_likes: {int(facet_payload['min_likes'])}")

        return " ; ".join(piece for piece in pieces if piece)
