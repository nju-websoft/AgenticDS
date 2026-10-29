from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Mapping, Sequence


@dataclass
class FusedItem:
    item_id: str
    score: float
    contributions: List[Dict[str, Any]] = field(default_factory=list)
    rrf_score: float = 0.0
    best_branch_score: float = 0.0


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    weights: Mapping[str, float] | None = None,
    rrf_k: int = 60,
    best_branch_weight: float = 0.0,
) -> List[FusedItem]:
    """Fuse consensus evidence with an optional strongest-branch rank signal."""
    rrf_k = max(1, int(rrf_k))
    best_branch_weight = min(1.0, max(0.0, float(best_branch_weight)))
    weights = weights or {}
    active = [(name, rows) for name, rows in ranked_lists.items() if rows]
    denominator = sum(max(0.0, float(weights.get(name, 1.0))) / (rrf_k + 1) for name, _ in active)
    if denominator <= 0.0:
        return []

    totals: Dict[str, float] = {}
    provenance: Dict[str, List[Dict[str, Any]]] = {}
    for list_id, rows in active:
        weight = max(0.0, float(weights.get(list_id, 1.0)))
        seen = set()
        for rank, row in enumerate(rows, 1):
            item_id = str(row.get("item_id") or row.get("dataset_id") or "").strip()
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            contribution = weight / (rrf_k + rank)
            totals[item_id] = totals.get(item_id, 0.0) + contribution
            provenance.setdefault(item_id, []).append({"list_id": list_id, "rank": rank, "raw_score": float(row.get("score", 0.0) or 0.0), "rrf_contribution": contribution / denominator, **dict(row.get("provenance") or {})})

    fused = []
    for item_id, total in totals.items():
        rrf_score = min(1.0, total / denominator)
        # A DCG-style rank discount retains strong evidence from one branch
        # without replacing the cross-branch consensus represented by RRF.
        best_branch_score = max(
            (1.0 / math.log2(float(row["rank"]) + 1.0) for row in provenance[item_id]),
            default=0.0,
        )
        score = (1.0 - best_branch_weight) * rrf_score + best_branch_weight * best_branch_score
        fused.append(
            FusedItem(
                item_id=item_id,
                score=min(1.0, score),
                contributions=provenance[item_id],
                rrf_score=rrf_score,
                best_branch_score=best_branch_score,
            )
        )
    return sorted(fused, key=lambda item: (-item.score, item.item_id))
