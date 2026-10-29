"""Utilities for exploring KG neighbor relationships."""
from __future__ import annotations

import json
import resource
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from global_values import KG_DIR

@dataclass(frozen=True)
class EdgeSpec:
    source_type: str
    edge_file: str
    source_field: str
    target_field: str
    target_type: str


class KGNeighborIndex:
    def __init__(self, kg_dir: str = KG_DIR, edge_specs: Optional[Sequence[EdgeSpec]] = None):
        self.kg_dir = Path(kg_dir)
        self.edge_specs = list(edge_specs) if edge_specs else self._default_edge_specs()
        self._edges_by_source: Dict[str, List[Tuple[str, str, str]]] = {}
        self._edges_by_target: Dict[str, List[Tuple[str, str, str]]] = {}
        self._load_edges()

    def _default_edge_specs(self) -> List[EdgeSpec]:
        return [
            EdgeSpec(
                source_type="model",
                target_type="dataset",
                edge_file="model_trainedOrFineTunedOn_dataset.json",
                source_field="model_id",
                target_field="dataset_id",
            ),
            EdgeSpec(
                source_type="dataset",
                target_type="task",
                edge_file="dataset_definedFor_task.json",
                source_field="dataset_id",
                target_field="task_id",
            ),
            EdgeSpec(
                source_type="model",
                target_type="task",
                edge_file="model_definedFor_task.json",
                source_field="model_id",
                target_field="task_id",
            ),
            EdgeSpec(
                source_type="model",
                target_type="model",
                edge_file="model_adapter_model.json",
                source_field="model_id",
                target_field="base_model_id",
            ),
            EdgeSpec(
                source_type="model",
                target_type="model",
                edge_file="model_finetune_model.json",
                source_field="model_id",
                target_field="base_model_id",
            ),
            EdgeSpec(
                source_type="model",
                target_type="model",
                edge_file="model_merge_model.json",
                source_field="model_id",
                target_field="base_model_id",
            ),
            EdgeSpec(
                source_type="model",
                target_type="model",
                edge_file="model_quantized_model.json",
                source_field="model_id",
                target_field="base_model_id",
            ),
            EdgeSpec(
                source_type="dataset",
                target_type="paper",
                edge_file="dataset_cite_paper.json",
                source_field="dataset_id",
                target_field="arxiv_id",
            ),
            EdgeSpec(
                source_type="model",
                target_type="paper",
                edge_file="model_cite_paper.json",
                source_field="model_id",
                target_field="arxiv_id",
            ),
            EdgeSpec(
                source_type="org",
                target_type="model",
                edge_file="org_publish_model.json",
                source_field="org_id",
                target_field="model_id",
            ),
            EdgeSpec(
                source_type="org",
                target_type="dataset",
                edge_file="org_publish_dataset.json",
                source_field="org_id",
                target_field="dataset_id",
            ),
            EdgeSpec(
                source_type="user",
                target_type="model",
                edge_file="user_publish_model.json",
                source_field="user_id",
                target_field="model_id",
            ),
            EdgeSpec(
                source_type="user",
                target_type="dataset",
                edge_file="user_publish_dataset.json",
                source_field="user_id",
                target_field="dataset_id",
            ),
            EdgeSpec(
                source_type="space",
                target_type="model",
                edge_file="space_use_model.json",
                source_field="space_id",
                target_field="model_id",
            ),
            EdgeSpec(
                source_type="space",
                target_type="dataset",
                edge_file="space_use_dataset.json",
                source_field="space_id",
                target_field="dataset_id",
            ),
            EdgeSpec(
                source_type="collection",
                target_type="model",
                edge_file="collection_contain_model.json",
                source_field="collection_slug",
                target_field="model_id",
            ),
            EdgeSpec(
                source_type="collection",
                target_type="paper",
                edge_file="collection_contain_paper.json",
                source_field="collection_slug",
                target_field="paper_id",
            ),
            EdgeSpec(
                source_type="collection",
                target_type="dataset",
                edge_file="collection_contain_dataset.json",
                source_field="collection_slug",
                target_field="dataset_id",
            ),
        ]

    def _load_edges(self) -> None:
        total_start = perf_counter()
        loaded_files = 0
        loaded_edges = 0
        print(
            f"[kg-neighbors][start] edge_specs={len(self.edge_specs)} "
            f"peak_rss_gib={self._peak_rss_gib():.2f}",
            flush=True,
        )
        for spec in self.edge_specs:
            edge_path = self.kg_dir / spec.edge_file
            if not edge_path.exists():
                continue
            stage_start = perf_counter()
            print(
                f"[kg-neighbors][file][start] file={edge_path} "
                f"size_gib={edge_path.stat().st_size / (1024.0**3):.2f}",
                flush=True,
            )
            with edge_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            accepted = 0
            for item in payload:
                src = item.get(spec.source_field)
                dst = item.get(spec.target_field)
                if not src or not dst:
                    continue
                self._edges_by_source.setdefault(src, []).append((dst, spec.target_type, spec.edge_file))
                self._edges_by_target.setdefault(dst, []).append((src, spec.source_type, spec.edge_file))
                accepted += 1
            loaded_files += 1
            loaded_edges += accepted
            print(
                f"[kg-neighbors][file][done] file={spec.edge_file} edges={accepted} "
                f"elapsed_seconds={perf_counter() - stage_start:.1f} "
                f"peak_rss_gib={self._peak_rss_gib():.2f}",
                flush=True,
            )
        print(
            f"[kg-neighbors][done] files={loaded_files} edges={loaded_edges} "
            f"elapsed_seconds={perf_counter() - total_start:.1f} "
            f"peak_rss_gib={self._peak_rss_gib():.2f}",
            flush=True,
        )

    @staticmethod
    def _peak_rss_gib() -> float:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)

    def neighbors(
        self,
        node_id: str,
        *,
        direction: str = "out",
        target_types: Optional[Iterable[str]] = None,
        edge_files: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, str]]:
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be 'out', 'in', or 'both'")
        target_types_set: Optional[Set[str]] = set(target_types) if target_types else None
        edge_files_set: Optional[Set[str]] = set(edge_files) if edge_files else None

        results: List[Dict[str, str]] = []

        def _collect(entries: List[Tuple[str, str, str]]) -> None:
            for neighbor_id, neighbor_type, edge_name in entries:
                if target_types_set and neighbor_type not in target_types_set:
                    continue
                if edge_files_set and edge_name not in edge_files_set:
                    continue
                results.append(
                    {
                        "neighbor_id": neighbor_id,
                        "neighbor_type": neighbor_type,
                        "edge": edge_name,
                    }
                )

        if direction in {"out", "both"}:
            _collect(self._edges_by_source.get(node_id, []))
        if direction in {"in", "both"}:
            _collect(self._edges_by_target.get(node_id, []))

        return results

    def degree(
        self,
        node_id: str,
        *,
        direction: str = "out",
        target_types: Optional[Iterable[str]] = None,
        edge_files: Optional[Iterable[str]] = None,
    ) -> int:
        """Return typed path degree for degree-normalized relation propagation."""
        return len(
            self.neighbors(
                node_id,
                direction=direction,
                target_types=target_types,
                edge_files=edge_files,
            )
        )

    def traverse_paths(
        self,
        node_id: str,
        *,
        path: Sequence[Mapping[str, str]],
        terminal_ids: Optional[Iterable[str]] = None,
        exclude_terminal_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Enumerate complete typed paths without retrieval-time rank truncation."""
        frontier: List[Tuple[str, List[Dict[str, str]], List[Dict[str, str]]]] = [
            (node_id, [], [])
        ]
        for hop_index, step in enumerate(path, 1):
            next_frontier: List[Tuple[str, List[Dict[str, str]], List[Dict[str, str]]]] = []
            direction = str(step.get("direction") or "out")
            edge_file = str(step.get("edge_file") or "")
            target_type = str(step.get("target_type") or "")
            for current_id, hops, intermediates in frontier:
                for neighbor in self.neighbors(
                    current_id,
                    direction=direction,
                    target_types=[target_type] if target_type else None,
                    edge_files=[edge_file] if edge_file else None,
                ):
                    neighbor_id = str(neighbor["neighbor_id"])
                    hop = {
                        "hop": str(hop_index),
                        "edge_file": edge_file,
                        "direction": direction,
                        "source_id": current_id,
                        "target_id": neighbor_id,
                        "target_type": target_type,
                    }
                    next_intermediates = list(intermediates)
                    if hop_index < len(path):
                        next_intermediates.append(
                            {"entity_type": target_type, "entity_id": neighbor_id}
                        )
                    next_frontier.append((neighbor_id, hops + [hop], next_intermediates))
            frontier = next_frontier
            if not frontier:
                break

        allowed = {str(item) for item in terminal_ids} if terminal_ids is not None else None
        output: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, Tuple[Tuple[str, str, str], ...]]] = set()
        for target_id, hops, intermediates in frontier:
            if target_id == exclude_terminal_id or (allowed is not None and target_id not in allowed):
                continue
            signature = (
                target_id,
                tuple((item["edge_file"], item["source_id"], item["target_id"]) for item in hops),
            )
            if signature in seen:
                continue
            seen.add(signature)
            output.append(
                {
                    "target_id": target_id,
                    "intermediate_entities": intermediates,
                    "hops": hops,
                }
            )
        return output


def find_neighbors(
    node_id: str,
    *,
    target_types: Optional[Iterable[str]] = None,
    edge_files: Optional[Iterable[str]] = None,
    direction: str = "out",
    kg_dir: str = KG_DIR,
    edge_specs: Optional[Sequence[EdgeSpec]] = None,
) -> List[Dict[str, str]]:
    index = KGNeighborIndex(kg_dir=kg_dir, edge_specs=edge_specs)
    return index.neighbors(
        node_id,
        direction=direction,
        target_types=target_types,
        edge_files=edge_files,
    )
