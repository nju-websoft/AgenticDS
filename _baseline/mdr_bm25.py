from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from global_values import KG_DIR
from utils.kg_index_store import DENSE_METADATA_KEYS, load_kg_entities
from utils.kg_sparse_index import BM25SparseIndex


DEFAULT_WEIGHT_GRID_STEP = 0.05
DEFAULT_TOP_K_SEARCH = 5000


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def _name_text(entity: Mapping[str, Any]) -> str:
    name = _normalize_text(entity.get("name"))
    entity_id = _normalize_text(entity.get("id"))
    if name and entity_id and name != entity_id:
        return f"{name} {entity_id}"
    return name or entity_id


def _description_text(entity: Mapping[str, Any]) -> str:
    return _normalize_text(entity.get("description"))


def _metadata_text(entity: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for key in DENSE_METADATA_KEYS:
        values = entity.get(key)
        if isinstance(values, list) and values:
            cleaned = [str(value).strip() for value in values if str(value).strip()]
            if cleaned:
                parts.append(" ".join(cleaned))
    for key in ("license", "size"):
        value = _normalize_text(entity.get(key))
        if value:
            parts.append(value)
    if entity.get("downloads") is not None:
        parts.append(str(int(entity.get("downloads") or 0)))
    if entity.get("likes") is not None:
        parts.append(str(int(entity.get("likes") or 0)))
    return " ".join(part for part in parts if part)


def _reciprocal_rank_at_k(ranked_ids: Sequence[str], gold_id: str, *, k: int) -> float:
    for rank, candidate_id in enumerate(ranked_ids[: max(1, int(k))], 1):
        if candidate_id == gold_id:
            return 1.0 / rank
    return 0.0


@dataclass(frozen=True)
class MDRWeights:
    name: float
    description: float
    metadata: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "name": float(self.name),
            "description": float(self.description),
            "metadata": float(self.metadata),
        }


class MDRBM25Retriever:
    """Multifield BM25 baseline adapted from MDR with name/description/metadata fields."""

    def __init__(
        self,
        *,
        entities: Optional[Sequence[Mapping[str, Any]]] = None,
        kg_dir: str | Path = KG_DIR,
        top_k_search: Optional[int] = None,
    ) -> None:
        raw_entities = list(entities) if entities is not None else load_kg_entities(kg_dir=kg_dir, entity_types=["dataset"])
        self.entities: List[Dict[str, Any]] = [dict(entity) for entity in raw_entities if str(entity.get("id") or "").strip()]
        self.dataset_ids: List[str] = [str(entity.get("id") or "").strip() for entity in self.entities]
        self.id_to_index: Dict[str, int] = {dataset_id: idx for idx, dataset_id in enumerate(self.dataset_ids)}
        self.top_k_search = max(1, int(top_k_search or DEFAULT_TOP_K_SEARCH))

        self.name_index = BM25SparseIndex.build([_name_text(entity) for entity in self.entities])
        self.description_index = BM25SparseIndex.build([_description_text(entity) for entity in self.entities])
        self.metadata_index = BM25SparseIndex.build([_metadata_text(entity) for entity in self.entities])
        self.weights = MDRWeights(name=1 / 3, description=1 / 3, metadata=1 / 3)

    def search(self, query_text: str, max_results: int = 10) -> List[str]:
        ranked = self._rank_ids(query_text, self.weights)
        return ranked[: max(1, int(max_results))]

    def fit(
        self,
        query_rows: Sequence[Mapping[str, Any]],
        *,
        query_field: str = "query",
        optimize_at_k: int = 20,
        grid_step: float = DEFAULT_WEIGHT_GRID_STEP,
    ) -> MDRWeights:
        training = []
        for row in query_rows:
            query_text = _normalize_text(row.get(query_field))
            gold_id = _normalize_text(row.get("gold_dataset_id"))
            if not query_text or not gold_id:
                continue
            training.append((query_text, gold_id, self._query_score_maps(query_text, gold_id)))
        if not training:
            return self.weights

        grid = self._grid_values(grid_step)
        current = [self.weights.name, self.weights.description, self.weights.metadata]
        best_score = self._objective(training, current, optimize_at_k=optimize_at_k)

        improved = True
        while improved:
            improved = False
            for target_idx in range(3):
                local_best = current
                local_best_score = best_score
                for candidate in grid:
                    proposal = self._project_weight(current, target_idx, candidate)
                    score = self._objective(training, proposal, optimize_at_k=optimize_at_k)
                    if score > local_best_score + 1e-12:
                        local_best = proposal
                        local_best_score = score
                if local_best is not current:
                    current = local_best
                    best_score = local_best_score
                    improved = True

        self.weights = MDRWeights(name=current[0], description=current[1], metadata=current[2])
        return self.weights

    def _query_score_maps(self, query_text: str, gold_id: str) -> Dict[str, Dict[int, float]]:
        top_k = min(max(1, self.top_k_search), len(self.entities))
        name_scores = self.name_index.search(query_text, top_k=top_k)
        description_scores = self.description_index.search(query_text, top_k=top_k)
        metadata_scores = self.metadata_index.search(query_text, top_k=top_k)
        gold_idx = self.id_to_index.get(gold_id)
        if gold_idx is not None:
            name_scores.setdefault(gold_idx, 0.0)
            description_scores.setdefault(gold_idx, 0.0)
            metadata_scores.setdefault(gold_idx, 0.0)
        return {
            "name": name_scores,
            "description": description_scores,
            "metadata": metadata_scores,
        }

    def _rank_ids(self, query_text: str, weights: MDRWeights | Sequence[float]) -> List[str]:
        score_maps = self._query_score_maps(query_text, gold_id="")
        combined = self._combine_scores(score_maps, weights)
        ranked = heapq.nlargest(max(1, self.top_k_search), combined.items(), key=lambda item: item[1])
        return [self.dataset_ids[idx] for idx, score in ranked if score > 0.0]

    def _combine_scores(
        self,
        score_maps: Mapping[str, Mapping[int, float]],
        weights: MDRWeights | Sequence[float],
    ) -> Dict[int, float]:
        if isinstance(weights, MDRWeights):
            weight_values = [weights.name, weights.description, weights.metadata]
        else:
            weight_values = [float(value) for value in weights]

        combined: Dict[int, float] = {}
        for field_name, weight in zip(("name", "description", "metadata"), weight_values):
            if weight <= 0.0:
                continue
            for idx, score in score_maps.get(field_name, {}).items():
                combined[idx] = combined.get(idx, 0.0) + weight * float(score)
        return combined

    def _objective(
        self,
        training: Sequence[Tuple[str, str, Dict[str, Dict[int, float]]]],
        weights: Sequence[float],
        *,
        optimize_at_k: int,
    ) -> float:
        total = 0.0
        for _query_text, gold_id, score_maps in training:
            combined = self._combine_scores(score_maps, weights)
            ranked = heapq.nlargest(max(1, optimize_at_k), combined.items(), key=lambda item: item[1])
            ranked_ids = [self.dataset_ids[idx] for idx, _score in ranked]
            total += _reciprocal_rank_at_k(ranked_ids, gold_id, k=optimize_at_k)
        return total / max(1, len(training))

    @staticmethod
    def _grid_values(step: float) -> List[float]:
        normalized_step = max(1e-6, float(step))
        values: List[float] = []
        cursor = 0.0
        while cursor < 1.0:
            values.append(round(cursor, 6))
            cursor += normalized_step
        values.append(1.0)
        return values

    @staticmethod
    def _project_weight(current: Sequence[float], target_idx: int, target_value: float) -> List[float]:
        proposal = [float(value) for value in current]
        target_value = max(0.0, min(1.0, float(target_value)))
        proposal[target_idx] = target_value
        remainder = 1.0 - target_value
        other_indices = [idx for idx in range(len(proposal)) if idx != target_idx]
        other_sum = sum(proposal[idx] for idx in other_indices)
        if other_sum > 0.0:
            for idx in other_indices:
                proposal[idx] = proposal[idx] / other_sum * remainder
        else:
            split = remainder / max(1, len(other_indices))
            for idx in other_indices:
                proposal[idx] = split
        total = sum(proposal)
        if total > 0:
            proposal = [value / total for value in proposal]
        return proposal


def main() -> None:
    parser = argparse.ArgumentParser(description="Run snapshot-adapted MDR BM25 retrieval.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    retriever = MDRBM25Retriever()
    dataset_ids = retriever.search(args.query, max_results=args.top_k)
    print(json.dumps({"query": args.query, "top_k": int(args.top_k), "dataset_ids": dataset_ids}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
