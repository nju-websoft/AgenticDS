from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

import requests

from global_values import (
    HF_DATASET_SAMPLE_CACHE_DIR,
    HF_DATASET_SAMPLE_ENABLED,
    HF_DATASET_SAMPLE_MAX_TEXT_CHARS,
    HF_DATASET_SAMPLE_MAX_WORKERS,
    HF_DATASET_SAMPLE_TIMEOUT_SECONDS,
    HF_DATASET_VIEWER_API_BASE_URL,
)


class HFDatasetSampleStore:
    """Cache a small, traceable preview of a dataset's training configurations."""

    SUCCESS_TTL_SECONDS = 7 * 24 * 60 * 60
    ERROR_TTL_SECONDS = 15 * 60
    MAX_CONFIGS = 3

    _default_instance: Optional["HFDatasetSampleStore"] = None

    @classmethod
    def get_default(cls) -> "HFDatasetSampleStore":
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    def __init__(
        self,
        *,
        base_url: str = HF_DATASET_VIEWER_API_BASE_URL,
        cache_dir: str = HF_DATASET_SAMPLE_CACHE_DIR,
        timeout_seconds: int = HF_DATASET_SAMPLE_TIMEOUT_SECONDS,
        max_text_chars: int = HF_DATASET_SAMPLE_MAX_TEXT_CHARS,
        max_workers: int = HF_DATASET_SAMPLE_MAX_WORKERS,
        enabled: bool = HF_DATASET_SAMPLE_ENABLED,
    ) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.timeout_seconds = max(1, int(timeout_seconds))
        self.max_text_chars = max(200, int(max_text_chars))
        self.max_workers = max(1, int(max_workers))
        self.enabled = bool(enabled)

        self.session = requests.Session()

    def get_sample(self, dataset_id: str, *, sampling_hint: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        normalized_id = self._normalize_dataset_id(dataset_id)
        if not normalized_id:
            return self._empty_record(str(dataset_id), status="invalid_dataset_id", source="none")

        hint = dict(sampling_hint or {})
        cached = self._load_cached(normalized_id, hint)
        if cached is not None:
            cached["source"] = "cache"
            return cached

        if not self.enabled:
            return self._empty_record(normalized_id, status="disabled", source="disabled")

        payload = self._fetch_sample(normalized_id, hint)
        self._save_cached(normalized_id, payload)
        return payload

    def get_samples_bulk(
        self,
        dataset_ids: Iterable[str],
        max_workers: Optional[int] = None,
        sampling_hints: Optional[Mapping[str, Mapping[str, str]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        unique_ids = self._dedup_dataset_ids(dataset_ids)
        if not unique_ids:
            return {}

        results: Dict[str, Dict[str, Any]] = {}
        pending_fetch: List[str] = []

        for dataset_id in unique_ids:
            cached = self._load_cached(dataset_id, (sampling_hints or {}).get(dataset_id, {}))
            if cached is not None:
                cached["source"] = "cache"
                results[dataset_id] = cached
            else:
                pending_fetch.append(dataset_id)

        if not pending_fetch:
            return results

        if not self.enabled:
            for dataset_id in pending_fetch:
                payload = self._empty_record(dataset_id, status="disabled", source="disabled")
                results[dataset_id] = payload
            return results

        worker_count = min(len(pending_fetch), max(1, int(max_workers or self.max_workers)))
        if worker_count <= 1:
            for dataset_id in pending_fetch:
                payload = self._fetch_sample(dataset_id, (sampling_hints or {}).get(dataset_id, {}))
                self._save_cached(dataset_id, payload)
                results[dataset_id] = payload
            return results

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(self._fetch_sample, dataset_id, (sampling_hints or {}).get(dataset_id, {})): dataset_id
                for dataset_id in pending_fetch
            }
            for future in as_completed(future_map):
                dataset_id = future_map[future]
                try:
                    payload = future.result()
                except Exception as exc:  # pragma: no cover
                    payload = self._empty_record(dataset_id, status="error", source="api")
                    payload["error"] = str(exc)
                self._save_cached(dataset_id, payload)
                results[dataset_id] = payload

        return results

    def _fetch_sample(self, dataset_id: str, hint: Mapping[str, str] | None = None) -> Dict[str, Any]:
        hint = hint or {}
        record = self._empty_record(dataset_id, status="unavailable", source="api")
        record["revision"] = str(hint.get("revision") or "")

        if not self.base_url:
            record["status"] = "invalid_base_url"
            return record

        try:
            splits = self._fetch_splits(dataset_id)
        except Exception as exc:  # pragma: no cover
            record["status"] = "error"
            record["error"] = str(exc)
            return record

        if not splits:
            record["status"] = "no_split"
            return record

        available_train_configs = sorted({item["config"] for item in splits if item["split"].lower() == "train"})
        all_splits = splits
        wanted_config = str(hint.get("config") or "")
        wanted_split = str(hint.get("split") or "")
        if wanted_config or wanted_split:
            splits = [
                item for item in splits
                if (not wanted_config or item["config"] == wanted_config)
                and (not wanted_split or item["split"] == wanted_split)
            ]
            if not splits:
                record["status"] = "config_unavailable"
                return record

        picked = self._pick_split(splits)
        if not picked:
            record["status"] = "no_split"
            return record

        config = picked.get("config")
        split = picked.get("split")
        if not config or not split:
            record["status"] = "no_split"
            return record

        record["config"] = config
        record["split"] = split
        train_splits = [item for item in all_splits if item["split"].lower() == "train"]
        record["available_train_configs"] = available_train_configs
        inspected = [picked]
        inspected.extend(item for item in train_splits if (item["config"], item["split"]) != (config, split))
        inspected = inspected[: self.MAX_CONFIGS]

        for item in inspected:
            cfg, spl = item["config"], item["split"]
            try:
                first = self._fetch_rows(dataset_id=dataset_id, config=cfg, split=spl, offset=0)
                self._collect_rows(record, first, cfg, spl)
                if cfg == config and spl == split:
                    total = first.get("num_rows_total")
                    if isinstance(total, int) and total > 1:
                        for offset in sorted({(total - 1) // 2, total - 1} - {0}):
                            more = self._fetch_rows(dataset_id=dataset_id, config=cfg, split=spl, offset=offset)
                            self._collect_rows(record, more, cfg, spl)
            except Exception as exc:
                record.setdefault("fetch_errors", []).append(f"{cfg}/{spl}: {exc}")

        if record["sample_rows"]:
            first_row = record["sample_rows"][0]["row"]
            record["sample_row"] = first_row
            record["sample_text"] = self._row_to_text(first_row)
            record["status"] = "ok" if record["sample_text"] else "empty_row"
        else:
            record["status"] = "error" if record.get("fetch_errors") else "empty_rows"
        return record

    @staticmethod
    def _collect_rows(record: Dict[str, Any], payload: Mapping[str, Any], config: str, split: str) -> None:
        features = payload.get("features")
        if isinstance(features, list):
            record["sample_features"][config] = [
                str(feature.get("name")) for feature in features
                if isinstance(feature, dict) and feature.get("name")
            ]
        for item in payload.get("rows") or []:
            row = item.get("row") if isinstance(item, dict) and "row" in item else item
            if isinstance(row, dict):
                record["sample_rows"].append({
                    "config": config,
                    "split": split,
                    "offset": item.get("row_idx", 0) if isinstance(item, dict) else 0,
                    "truncated_cells": item.get("truncated_cells", []) if isinstance(item, dict) else [],
                    "row": row,
                })

    def _fetch_splits(self, dataset_id: str) -> List[Dict[str, str]]:
        url = f"{self.base_url}/splits"
        response = self.session.get(
            url,
            params={"dataset": dataset_id},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()

        rows = payload.get("splits") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return []

        cleaned: List[Dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            config = str(row.get("config") or "").strip()
            split = str(row.get("split") or "").strip()
            if not config or not split:
                continue
            cleaned.append({"config": config, "split": split})
        return cleaned

    def _pick_split(self, splits: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
        if not splits:
            return None

        split_order = {"train": 0, "validation": 1, "valid": 1, "test": 2, "dev": 3}

        def _rank(item: Dict[str, str]) -> tuple[int, str, str]:
            split = str(item.get("split") or "").strip().lower()
            return (split_order.get(split, 99), str(item.get("config") or ""), split)

        return sorted(splits, key=_rank)[0]

    def _fetch_rows(self, *, dataset_id: str, config: str, split: str, offset: int = 0) -> Dict[str, Any]:
        url = f"{self.base_url}/rows"
        response = self.session.get(
            url,
            params={
                "dataset": dataset_id,
                "config": config,
                "split": split,
                "offset": offset,
                "length": 1,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def _row_to_text(self, row: Any) -> str:
        if row is None:
            return ""

        if isinstance(row, (dict, list)):
            text = json.dumps(row, ensure_ascii=False)
        else:
            text = str(row)

        text = text.strip()
        if not text:
            return ""

        if len(text) > self.max_text_chars:
            return text[: self.max_text_chars - 3] + "..."
        return text

    def _empty_record(self, dataset_id: str, *, status: str, source: str) -> Dict[str, Any]:
        return {
            "dataset_id": dataset_id,
            "status": status,
            "source": source,
            "fetched_at": datetime.utcnow().isoformat(),
            "config": "",
            "split": "",
            "sample_text": "",
            "sample_row": None,
            "sample_rows": [],
            "sample_features": {},
            "available_train_configs": [],
            "revision": "",
        }

    def _cache_path(self, dataset_id: str) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "__", dataset_id.strip())
        return self.cache_dir / f"{safe_name}.json"

    def _load_cached(self, dataset_id: str, hint: Mapping[str, str] | None = None) -> Optional[Dict[str, Any]]:
        path = self._cache_path(dataset_id)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                hint = hint or {}
                status = str(payload.get("status") or "")
                ttl = self.SUCCESS_TTL_SECONDS if status == "ok" else self.ERROR_TTL_SECONDS
                if time.time() - path.stat().st_mtime > ttl:
                    return None
                if status == "disabled":
                    return None
                if any(
                    str(hint.get(key) or "") and str(hint[key]) != str(payload.get(key) or "")
                    for key in ("config", "split", "revision")
                ):
                    return None
                if status == "ok" and not payload.get("sample_rows"):
                    return None
                return payload
        except Exception:
            return None
        return None

    def _save_cached(self, dataset_id: str, payload: Dict[str, Any]) -> None:
        path = self._cache_path(dataset_id)
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except Exception:
            return

    def _normalize_dataset_id(self, dataset_id: str) -> str:
        return str(dataset_id or "").strip()

    def _dedup_dataset_ids(self, dataset_ids: Iterable[str]) -> List[str]:
        seen = set()
        results: List[str] = []
        for dataset_id in dataset_ids:
            normalized = self._normalize_dataset_id(dataset_id)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            results.append(normalized)
        return results
