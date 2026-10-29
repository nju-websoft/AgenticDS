"""Helper utilities for filtering dataset nodes inside the KG."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from global_values import KG_DIR
from utils.hf_dataset_metadata_registry import HFDatasetMetadataRegistry


NormalizeFunc = Callable[[Dict[str, Any]], bool]


def _normalize_values(values: Optional[Iterable[Any]]) -> List[str]:
    if not values:
        return []
    return [str(v).strip().lower() for v in values if v is not None]


def _contains_all(values: Optional[Iterable[Any]], required: Sequence[str]) -> bool:
    pool = set(_normalize_values(values))
    targets = _normalize_values(required)
    return all(item in pool for item in targets)


def _contains_any(values: Optional[Iterable[Any]], required: Sequence[str]) -> bool:
    if not required:
        return True
    pool = set(_normalize_values(values))
    targets = _normalize_values(required)
    return any(item in pool for item in targets)


@dataclass
class DatasetFilter:
    """Criteria for filtering dataset nodes.

    All list-based filters are case-insensitive. Provide strings as they appear in
    the KG (for example, ``"region:us"`` tags or ``"text-generation"`` tasks).
    """

    ids: Optional[Sequence[str]] = None
    tags_all: Optional[Sequence[str]] = None
    tags_any: Optional[Sequence[str]] = None
    tasks_all: Optional[Sequence[str]] = None
    tasks_any: Optional[Sequence[str]] = None
    modalities_all: Optional[Sequence[str]] = None
    modalities_any: Optional[Sequence[str]] = None
    licenses: Optional[Sequence[str]] = None
    languages_all: Optional[Sequence[str]] = None
    languages_any: Optional[Sequence[str]] = None
    formats_all: Optional[Sequence[str]] = None
    formats_any: Optional[Sequence[str]] = None
    other_all: Optional[Sequence[str]] = None
    other_any: Optional[Sequence[str]] = None
    libraries_all: Optional[Sequence[str]] = None
    libraries_any: Optional[Sequence[str]] = None
    min_downloads: Optional[int] = None
    min_likes: Optional[int] = None
    text_contains: Optional[str] = None
    predicate: Optional[Callable[[Dict[str, Any]], bool]] = None


class DatasetIndex:
    """In-memory view of dataset metadata for structured filtering."""

    def __init__(self, kg_dir: str = KG_DIR, dataset_file: str = "datasets.json") -> None:
        dataset_path = Path(kg_dir) / dataset_file
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        with dataset_path.open("r", encoding="utf-8") as fh:
            self._datasets: List[Dict[str, Any]] = json.load(fh)
        self._registry = HFDatasetMetadataRegistry.get_default()

    def _canon_values(self, category: str, values: Optional[Iterable[Any]]) -> List[str]:
        if not values:
            return []
        canonical = self._registry.canonicalize_many(category, values)
        if canonical:
            return canonical
        return _normalize_values(values)

    def _canon_scalar(self, category: str, value: Any) -> str:
        normalized = self._registry.canonicalize(category, value)
        if normalized:
            return normalized
        return str(value or "").strip().lower()

    def _matches(self, dataset: Dict[str, Any], params: Optional[DatasetFilter]) -> bool:
        if params is None:
            return True

        if params.ids and dataset.get("id") not in set(params.ids):
            return False

        dataset_tags = self._canon_values("tags", dataset.get("tags"))
        if params.tags_all and not _contains_all(dataset_tags, self._canon_values("tags", params.tags_all)):
            return False
        if params.tags_any and not _contains_any(dataset_tags, self._canon_values("tags", params.tags_any)):
            return False

        dataset_tasks = self._canon_values("tasks", dataset.get("tasks"))
        if params.tasks_all and not _contains_all(dataset_tasks, self._canon_values("tasks", params.tasks_all)):
            return False
        if params.tasks_any and not _contains_any(dataset_tasks, self._canon_values("tasks", params.tasks_any)):
            return False

        dataset_modalities = self._canon_values("modalities", dataset.get("modalities"))
        if params.modalities_all and not _contains_all(
            dataset_modalities,
            self._canon_values("modalities", params.modalities_all),
        ):
            return False
        if params.modalities_any and not _contains_any(
            dataset_modalities,
            self._canon_values("modalities", params.modalities_any),
        ):
            return False

        if params.licenses:
            dataset_license = self._canon_scalar("licenses", dataset.get("license"))
            normalized_licenses = set(self._canon_values("licenses", params.licenses))
            if dataset_license not in normalized_licenses:
                return False

        dataset_languages = self._canon_values("languages", dataset.get("languages"))
        if params.languages_all and not _contains_all(
            dataset_languages,
            self._canon_values("languages", params.languages_all),
        ):
            return False
        if params.languages_any and not _contains_any(
            dataset_languages,
            self._canon_values("languages", params.languages_any),
        ):
            return False

        dataset_formats = self._canon_values("formats", dataset.get("formats"))
        if params.formats_all and not _contains_all(dataset_formats, self._canon_values("formats", params.formats_all)):
            return False
        if params.formats_any and not _contains_any(dataset_formats, self._canon_values("formats", params.formats_any)):
            return False

        dataset_other = self._canon_values("other", dataset.get("other"))
        if params.other_all and not _contains_all(dataset_other, self._canon_values("other", params.other_all)):
            return False
        if params.other_any and not _contains_any(dataset_other, self._canon_values("other", params.other_any)):
            return False

        dataset_libraries = self._canon_values("libraries", dataset.get("libraries"))
        if params.libraries_all and not _contains_all(
            dataset_libraries,
            self._canon_values("libraries", params.libraries_all),
        ):
            return False
        if params.libraries_any and not _contains_any(
            dataset_libraries,
            self._canon_values("libraries", params.libraries_any),
        ):
            return False

        if params.min_downloads is not None and (dataset.get("downloads", 0) < params.min_downloads):
            return False
        if params.min_likes is not None and (dataset.get("likes", 0) < params.min_likes):
            return False

        if params.text_contains:
            haystack = " ".join(
                [
                    str(dataset.get("name", "")),
                    str(dataset.get("description", "")),
                    " ".join(dataset.get("tags", []) or []),
                ]
            ).lower()
            if params.text_contains.lower() not in haystack:
                return False

        if params.predicate and not params.predicate(dataset):
            return False

        return True

    def filter(self, params: Optional[DatasetFilter] = None, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return datasets that satisfy ``params`` (up to ``limit`` if provided)."""
        matches: List[Dict[str, Any]] = []
        for dataset in self._datasets:
            if self._matches(dataset, params):
                matches.append(dataset)
                if limit is not None and len(matches) >= limit:
                    break
        return matches


def filter_datasets(
    params: Optional[DatasetFilter] = None,
    *,
    limit: Optional[int] = None,
    kg_dir: str = KG_DIR,
    dataset_file: str = "datasets.json",
) -> List[Dict[str, Any]]:
    """Convenience wrapper that loads the index and runs a filter query."""
    index = DatasetIndex(kg_dir=kg_dir, dataset_file=dataset_file)
    return index.filter(params=params, limit=limit)
