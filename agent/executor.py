from __future__ import annotations

import math
import re
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

from global_values import (
    AGENT_CONSTRAINT_VIOLATION_POLICY,
    AGENT_CRITIC_POOL_SIZE,
    AGENT_RERANK_SHORTLIST_MAX,
    CRITIC_MAX_CANDIDATES,
    EXPERIMENT_ANCHOR_IDENTITY_SCORING,
    EXPERIMENT_BEST_BRANCH_FUSION,
    EXPERIMENT_CONDITIONAL_SHORTLIST_ALLOCATIONS,
    EXPERIMENT_DISABLE_GRAPH_EXPANSION,
    EXPERIMENT_DISABLE_METADATA_FILTER,
    EXPERIMENT_DISABLE_PRERERANK,
    EXPERIMENT_RELIABILITY_WEIGHTED_FUSION,
    FUSION_BEST_BRANCH_WEIGHT,
    KG_DIR,
    RETRIEVAL_SERVICE_READY_TIMEOUT_SECONDS,
    RETRIEVAL_SERVICE_RPC_TIMEOUT_SECONDS,
    RETRIEVAL_SERVICE_SOCKET,
)
from services.retrieval_client import RetrievalClient
from utils.kg_find_neighbors import KGNeighborIndex
from utils.kg_hybrid_retriever import KGHybridRetriever
from utils.rank_fusion import reciprocal_rank_fusion
from utils.lucene_runtime import LUCENE_VERSION

from .schemas import Candidate, ConstraintSpec, ExecutionBundle, ExecutionTrace, RetrievalProvenance, SearchPlan, normalize_metadata_values


class ExecutorAgent:
    """Execute independent retrieval hypotheses and fuse their ranked evidence."""

    def __init__(
        self,
        debug_full_trace: bool = False,
        embedding_model: Optional[str] = None,
        retrieval_service_socket: Optional[str] = None,
    ) -> None:
        self.debug_full_trace = bool(debug_full_trace)
        self.embedding_model = embedding_model
        socket_path = str(retrieval_service_socket or RETRIEVAL_SERVICE_SOCKET or "").strip()
        self.retrieval_client: RetrievalClient | None = None
        self.retrieval_manifest: Dict[str, Any] = {}
        self._entity_retrievers: Dict[str, KGHybridRetriever] = {}
        if socket_path:
            model_name = str(embedding_model or "").strip()
            if not model_name:
                raise ValueError("embedding_model is required when using the retrieval service")
            self.retrieval_client = RetrievalClient(
                socket_path,
                expected_snapshot=KG_DIR,
                expected_embedding_model=model_name,
                timeout_seconds=RETRIEVAL_SERVICE_RPC_TIMEOUT_SECONDS,
            )
            self.retrieval_client.wait_until_ready(timeout_seconds=RETRIEVAL_SERVICE_READY_TIMEOUT_SECONDS)
            self.retrieval_manifest = self.retrieval_client.manifest()
            if (
                self.retrieval_manifest.get("sparse_backend") != "lucene"
                or self.retrieval_manifest.get("lucene_version") != LUCENE_VERSION
            ):
                raise RuntimeError("Retrieval service is not using the pinned Lucene backend; rebuild sparse indexes and restart the service")
            self.retriever = None
            self.neighbor_index = self.retrieval_client
            self.dataset_lookup: Dict[str, Dict[str, Any]] = {}
        else:
            self.retriever = KGHybridRetriever(embedding_model=embedding_model) if embedding_model else KGHybridRetriever()
            self._entity_retrievers["dataset"] = self.retriever
            self.neighbor_index = KGNeighborIndex()
            self.dataset_lookup = {
                str(row.get("id")): row
                for row in self.retriever.entities
                if row.get("entity_type") == "dataset" and row.get("id")
            }
        self._normalized_candidate_metadata: Dict[str, Dict[str, set[str]]] = {}

    def execute(self, plan: SearchPlan, steps: List[Any]) -> ExecutionBundle:
        branch_lists: Dict[str, List[Dict[str, Any]]] = {}
        anchor_lists: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        unresolved_anchors: List[Dict[str, Any]] = []
        trace: List[ExecutionTrace] = []
        relation_branch_counts: Dict[str, int] = {}

        for step in steps:
            name, params = step.name, dict(step.params or {})
            if name == "retrieve_dataset_branch":
                query_id = str(params["query_id"])
                list_id = f"dataset:{query_id}"
                hits = self._retrieve_candidates(params["query"], int(params.get("top_k", plan.retrieval_policy.per_query_top_k)))
                rows = self._dataset_hits_to_rows(hits, query_id=query_id, branch_type="dataset_query")
                branch_lists[list_id] = rows
                if params.get("capture_as_anchor"):
                    anchor_lists[(query_id, "dataset")] = hits
                self._register_dataset_metadata(hits)
                trace.append(self._trace(name, "ok", {"query_id": query_id, "list_id": list_id, "hits": len(rows), "signature": params.get("signature", "")}))

            elif name == "retrieve_relation_anchors":
                query_id = str(params.get("query_id") or "")
                entity_type = str(params.get("entity_type") or "")
                try:
                    hits = self._retrieve_entities(entity_type, params.get("query"), int(params.get("top_k", 20)))
                    anchor_lists[(query_id, entity_type)] = hits
                    trace.append(self._trace(name, "ok", {"query_id": query_id, "entity_type": entity_type, "hits": len(hits)}))
                except (FileNotFoundError, RuntimeError, ValueError) as exc:
                    unresolved_anchors.append({"query_id": query_id, "entity_type": entity_type, "reason": "index_unavailable", "detail": str(exc)})
                    trace.append(self._trace(name, "unresolved", unresolved_anchors[-1]))

            elif name == "retrieve_relation_text_control":
                relation_id = str(params.get("relation_id") or "")
                query_id = str(params.get("anchor_query_id") or "")
                hits = self._retrieve_candidates(
                    params.get("query"),
                    int(params.get("top_k", plan.retrieval_policy.per_query_top_k)),
                )
                rows = self._dataset_hits_to_rows(
                    hits,
                    query_id=query_id,
                    branch_type="relation_text_control",
                )
                for row in rows:
                    row["provenance"]["anchor_confidence"] = 1.0
                base_list_id = f"relation:{relation_id}"
                branch_occurrence = relation_branch_counts.get(base_list_id, 0) + 1
                relation_branch_counts[base_list_id] = branch_occurrence
                list_id = base_list_id if branch_occurrence == 1 else f"{base_list_id}#{branch_occurrence}"
                branch_lists[list_id] = rows
                self._register_dataset_metadata(hits)
                trace.append(
                    self._trace(
                        name,
                        "ok",
                        {
                            "relation_id": relation_id,
                            "list_id": list_id,
                            "query_id": query_id,
                            "hits": len(rows),
                            "candidate_budget": int(params.get("top_k", plan.retrieval_policy.per_query_top_k)),
                            "signature": params.get("signature", ""),
                        },
                    )
                )

            elif name == "expand_relation_anchors":
                if EXPERIMENT_DISABLE_GRAPH_EXPANSION:
                    trace.append(self._trace(name, "skipped", {"reason": "disabled_by_experiment"}))
                    continue
                relation_id = str(params.get("relation_id") or "")
                query_id = str(params.get("anchor_query_id") or "")
                entity_type = str(params.get("anchor_entity_type") or "")
                anchors = anchor_lists.get((query_id, entity_type), [])
                rows, unresolved, relation_detail = self._expand_relation_branch(anchors, params)
                base_list_id = f"relation:{relation_id}"
                branch_occurrence = relation_branch_counts.get(base_list_id, 0) + 1
                relation_branch_counts[base_list_id] = branch_occurrence
                list_id = base_list_id if branch_occurrence == 1 else f"{base_list_id}#{branch_occurrence}"
                if rows:
                    branch_lists[list_id] = rows
                if unresolved:
                    unresolved_anchors.append({"query_id": query_id, "entity_type": entity_type, "relation_id": relation_id, "reason": unresolved})
                trace.append(
                    self._trace(
                        name,
                        "ok" if rows else "unresolved",
                        {
                            "relation_id": relation_id,
                            "list_id": list_id,
                            "duplicate_relation_id": branch_occurrence > 1,
                            "anchors": len(anchors),
                            "datasets": len(rows),
                            "reason": unresolved,
                            **relation_detail,
                        },
                    )
                )

            else:
                raise ValueError(f"Unsupported retrieval operator: {name}")

        return ExecutionBundle(trace=trace, branch_lists=branch_lists, unresolved_anchors=unresolved_anchors)

    def rank_from_branch_lists(
        self,
        plan: SearchPlan,
        branch_lists: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        pool_size: int,
        trace: List[ExecutionTrace] | None = None,
    ) -> List[Candidate]:
        """Reconstruct a ranking from the cumulative cross-iteration evidence state."""
        query_weights = {item.query_id: item.weight for item in plan.retrieval_queries}
        relation_weights = {item.relation_id: item.weight for item in plan.relation_search_specs}
        weights: Dict[str, float] = {}
        groups: Dict[str, str] = {}
        for list_id in branch_lists:
            prefix, identifier = list_id.split(":", 1)
            if prefix == "relation":
                identifier = identifier.split("#", 1)[0]
                groups[list_id] = "relation"
            else:
                groups[list_id] = self._query_evidence_group(plan, identifier)
            weights[list_id] = query_weights.get(identifier, relation_weights.get(identifier, 1.0))
        weights = self._normalize_evidence_group_weights(branch_lists, weights, groups)
        candidates = self._fuse_lists(branch_lists, weights, {}, rrf_k=plan.retrieval_policy.rrf_k)
        fused_count = len(candidates)
        if trace is not None:
            trace.append(self._trace("fuse_candidate_archive", "ok", {
                "lists": len(branch_lists), "candidates": fused_count,
                "rrf_k": plan.retrieval_policy.rrf_k,
            }))
        if plan.hard_constraints:
            for candidate in candidates.values():
                violations, unknowns = self._constraint_assessment(candidate.dataset, plan.hard_constraints)
                candidate.constraint_violations = violations
                candidate.constraint_unknowns = unknowns
                candidate.filter_score = 0.0 if EXPERIMENT_DISABLE_METADATA_FILTER or violations else (0.5 if unknowns else 1.0)
        violation_count = sum(bool(item.constraint_violations) for item in candidates.values())
        unknown_count = sum(bool(item.constraint_unknowns) for item in candidates.values())
        candidates = self._apply_constraint_violation_policy(candidates)
        if trace is not None:
            trace.append(self._trace("apply_hard_constraints", "skipped" if EXPERIMENT_DISABLE_METADATA_FILTER else "ok", {
                "before": fused_count, "retained": len(candidates),
                "violation_policy": AGENT_CONSTRAINT_VIOLATION_POLICY,
                "explicit_violations": violation_count,
                "unknown_constraints": unknown_count,
            }))
        if plan.soft_preferences:
            self._score_soft_preferences(candidates, [asdict(item) for item in plan.soft_preferences])
            if trace is not None:
                trace.append(self._trace("score_soft_preferences", "ok", {"preferences": len(plan.soft_preferences)}))
        ranked = (
            self._rank_without_prerank(candidates)
            if EXPERIMENT_DISABLE_PRERERANK
            else [item[1] for item in self._rank_for_pruning(candidates, plan)]
        )
        budget = max(plan.top_k_final, int(pool_size))
        ranked = ranked[:budget]
        candidates = {item.dataset_id: item for item in ranked}
        if trace is not None:
            trace.append(self._trace("apply_fused_pool_budget", "ok", {"budget": budget, "retained": len(candidates)}))
        keep_n = min(
            max(1, int(AGENT_CRITIC_POOL_SIZE)),
            max(1, int(AGENT_RERANK_SHORTLIST_MAX)),
            max(1, int(CRITIC_MAX_CANDIDATES)),
        )
        ranked, allocations = self._apply_shortlist_allocations(
            ranked,
            candidates,
            branch_lists,
            plan.shortlist_allocations,
            keep_n,
        )
        if trace is not None:
            trace.append(self._trace("prune_and_finalize_shortlist", "ok", {
                "kept": min(len(ranked), keep_n), "pool_budget": budget,
                "shortlist_allocations": allocations,
                "mode": "retrieval_order" if EXPERIMENT_DISABLE_PRERERANK else "multi_evidence_prerank",
            }))
        return ranked[:keep_n]

    def _apply_constraint_violation_policy(self, candidates: Dict[str, Candidate]) -> Dict[str, Candidate]:
        if AGENT_CONSTRAINT_VIOLATION_POLICY == "discard" and not EXPERIMENT_DISABLE_METADATA_FILTER:
            return {dataset_id: item for dataset_id, item in candidates.items() if not item.constraint_violations}
        return candidates

    def _query_evidence_group(self, plan: SearchPlan, query_id: str) -> str:
        if query_id == "__raw__":
            return "raw"
        query = next((item for item in plan.retrieval_queries if item.query_id == query_id), None)
        return str(query.evidence_focus if query else "semantics")

    def _normalize_evidence_group_weights(
        self,
        branch_lists: Mapping[str, Sequence[Mapping[str, Any]]],
        weights: Mapping[str, float],
        groups: Mapping[str, str],
    ) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for list_id, rows in branch_lists.items():
            if not rows:
                continue
            group = str(groups.get(list_id) or "semantics")
            totals[group] = totals.get(group, 0.0) + max(0.0, float(weights.get(list_id, 1.0)))
        normalized = {
            list_id: (
                max(0.0, float(weights.get(list_id, 1.0)))
                / max(totals.get(str(groups.get(list_id) or "semantics"), 0.0), 1e-12)
            )
            for list_id, rows in branch_lists.items()
            if rows
        }
        if EXPERIMENT_RELIABILITY_WEIGHTED_FUSION:
            for list_id, rows in branch_lists.items():
                if not rows or str(groups.get(list_id) or "") != "relation" or list_id not in normalized:
                    continue
                reliability = max(
                    (
                        float((row.get("provenance") or {}).get("anchor_confidence", 0.0) or 0.0)
                        for row in rows
                    ),
                    default=0.0,
                )
                normalized[list_id] *= max(0.05, min(1.0, reliability))
        return normalized

    def _apply_shortlist_allocations(
        self,
        ranked: Sequence[Candidate],
        candidate_map: Mapping[str, Candidate],
        branch_lists: Mapping[str, Sequence[Mapping[str, Any]]],
        allocations: Any,
        keep_n: int,
    ) -> tuple[List[Candidate], Dict[str, int]]:
        requested = allocations if isinstance(allocations, Mapping) else {}
        global_ranks = {candidate.dataset_id: rank for rank, candidate in enumerate(ranked, 1)}
        reserved_ids: set[str] = set()
        realized: Dict[str, int] = {}
        budget = min(max(0, keep_n // 5), sum(max(0, int(value)) for value in requested.values()))
        for branch_id, raw_slots in requested.items():
            if budget <= 0:
                break
            slots = min(max(0, int(raw_slots)), budget)
            if slots <= 0 or branch_id not in branch_lists:
                continue
            before = len(reserved_ids)
            for row in branch_lists[branch_id]:
                dataset_id = str(row.get("item_id") or row.get("dataset_id") or "")
                candidate = candidate_map.get(dataset_id)
                if not candidate or candidate.constraint_violations or dataset_id in reserved_ids:
                    continue
                if EXPERIMENT_CONDITIONAL_SHORTLIST_ALLOCATIONS:
                    if global_ranks.get(dataset_id, 10**9) > 2 * keep_n:
                        continue
                    if str(branch_id).startswith("relation:"):
                        confidence = float((row.get("provenance") or {}).get("anchor_confidence", 0.0) or 0.0)
                        if confidence < 0.70:
                            continue
                reserved_ids.add(dataset_id)
                if len(reserved_ids) - before >= slots:
                    break
            used = len(reserved_ids) - before
            if used:
                realized[str(branch_id)] = used
                budget -= used

        selected_ids = set(reserved_ids)
        for candidate in ranked:
            if len(selected_ids) >= keep_n:
                break
            selected_ids.add(candidate.dataset_id)
        return [candidate for candidate in ranked if candidate.dataset_id in selected_ids], realized

    def _retrieve_candidates(self, query: Any, top_k: int) -> List[Dict[str, Any]]:
        if self.retrieval_client is not None:
            return self.retrieval_client.search("dataset", query, top_k=top_k).get("dataset", []) or []
        return self.retriever.search(query=query, top_k=top_k).get("dataset", []) or []

    def _retrieve_entities(self, entity_type: str, query: Any, top_k: int) -> List[Dict[str, Any]]:
        if self.retrieval_client is not None:
            return self.retrieval_client.search(entity_type, query, top_k=top_k).get(entity_type, []) or []
        if entity_type not in self._entity_retrievers:
            kwargs = {"entity_types": [entity_type]}
            if self.embedding_model:
                kwargs["embedding_model"] = self.embedding_model
            self._entity_retrievers[entity_type] = KGHybridRetriever(**kwargs)
        return self._entity_retrievers[entity_type].search(query=query, top_k=top_k).get(entity_type, []) or []

    def _dataset_hits_to_rows(self, hits: Sequence[Mapping[str, Any]], *, query_id: str, branch_type: str) -> List[Dict[str, Any]]:
        rows = []
        for hit in hits:
            dataset = hit.get("entity") or {}
            dataset_id = str(dataset.get("id") or "")
            if dataset_id:
                rows.append({"item_id": dataset_id, "score": float(hit.get("similarity", 0.0) or 0.0), "provenance": {"query_id": query_id, "branch_type": branch_type, "entity_type": "dataset"}})
        return rows

    def _register_dataset_metadata(self, hits: Sequence[Mapping[str, Any]]) -> None:
        for hit in hits:
            dataset = dict(hit.get("entity") or {})
            dataset_id = str(dataset.get("id") or "")
            if not dataset_id:
                continue
            self.dataset_lookup[dataset_id] = dataset

    def _expand_relation_branch(
        self,
        anchors: Sequence[Mapping[str, Any]],
        params: Mapping[str, Any],
    ) -> tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        if not anchors:
            return [], "no_anchor_retrieved", {"expanded_anchors": 0}
        path = [dict(item) for item in params.get("path", []) if isinstance(item, Mapping)]
        if not path:
            return [], "empty_relation_path", {"expanded_anchors": 0}
        max_anchors = min(10, max(1, int(params.get("top_k_anchors", 5))))
        max_neighbors = min(50, max(1, int(params.get("max_neighbors_per_anchor", 30))))
        max_intermediate = min(20, max(1, int(params.get("max_intermediate_per_anchor", 10))))
        max_datasets_per_intermediate = min(50, max(1, int(params.get("max_datasets_per_intermediate", 20))))
        min_anchor_score = float(params.get("min_anchor_score", 0.0) or 0.0)
        decay = min(1.0, max(0.0, float(params.get("path_decay", 0.5) or 0.5)))
        selected_anchors = list(anchors[:max_anchors])
        anchor_confidences: Dict[str, float] = {}
        if EXPERIMENT_ANCHOR_IDENTITY_SCORING:
            scored = [
                (
                    hit,
                    self._anchor_identity_confidence(
                        hit,
                        query_text=str(params.get("anchor_query_text") or ""),
                        source_evidence=str(params.get("anchor_source_evidence") or ""),
                    ),
                    rank,
                )
                for rank, hit in enumerate(anchors[:max_anchors], 1)
            ]
            scored.sort(key=lambda item: (-item[1], item[2]))
            best_confidence = scored[0][1] if scored else 0.0
            if best_confidence >= 0.70:
                scored = [item for item in scored if item[1] >= max(0.70, best_confidence - 0.08)]
            selected_anchors = [item[0] for item in scored[:max_anchors]]
            for hit, confidence, _ in scored[:max_anchors]:
                anchor_id = str((hit.get("entity") or {}).get("id") or "")
                if anchor_id:
                    anchor_confidences[anchor_id] = confidence

        expanded: List[tuple[str, str, float, List[str], float]] = []
        for hit in selected_anchors:
            entity = hit.get("entity") or {}
            anchor_id = str(entity.get("id") or "")
            anchor_score = float(hit.get("similarity", 0.0) or 0.0)
            if not anchor_id or anchor_score <= min_anchor_score:
                continue
            frontier = [(anchor_id, 1.0, [])]
            for hop, edge in enumerate(path, 1):
                next_frontier = []
                hop_limit = max_datasets_per_intermediate if hop == len(path) else max_intermediate
                if len(path) == 1:
                    hop_limit = max_neighbors
                for node_id, path_score, traversed in frontier:
                    neighbors = self.neighbor_index.neighbors(node_id, direction=str(edge.get("direction") or "out"), target_types=[str(edge.get("target_type") or "dataset")], edge_files=[str(edge.get("edge_file") or "")])
                    degree_penalty = 1.0 / math.log2(2.0 + len(neighbors))
                    for neighbor in neighbors[:hop_limit]:
                        score = path_score * degree_penalty
                        next_frontier.append((neighbor["neighbor_id"], score, traversed + [neighbor["edge"]]))
                frontier = next_frontier
            for dataset_id, path_score, traversed in frontier:
                if str(params.get("anchor_entity_type") or "") == "dataset" and dataset_id == anchor_id:
                    continue
                anchor_confidence = anchor_confidences.get(
                    anchor_id,
                    1.0 if not EXPERIMENT_ANCHOR_IDENTITY_SCORING else 0.0,
                )
                identity_factor = 1.0 if not EXPERIMENT_ANCHOR_IDENTITY_SCORING else (0.10 + 0.90 * anchor_confidence)
                expanded.append(
                    (
                        dataset_id,
                        anchor_id,
                        max(0.0, anchor_score) * identity_factor * path_score * (decay ** (len(path) - 1)),
                        traversed,
                        anchor_confidence,
                    )
                )

        if self.retrieval_client is not None:
            missing_ids = sorted({item[0] for item in expanded if item[0] not in self.dataset_lookup})
            for entity in self.retrieval_client.get_entities("dataset", missing_ids):
                dataset_id = str(entity.get("id") or "")
                if dataset_id:
                    self.dataset_lookup[dataset_id] = entity

        dataset_scores: Dict[str, float] = {}
        dataset_paths: Dict[str, Dict[str, Any]] = {}
        for dataset_id, anchor_id, score, traversed, anchor_confidence in expanded:
            if dataset_id not in self.dataset_lookup:
                continue
            dataset_scores[dataset_id] = dataset_scores.get(dataset_id, 0.0) + score
            if dataset_id not in dataset_paths or score > dataset_paths[dataset_id]["path_score"]:
                dataset_paths[dataset_id] = {
                    "anchor_id": anchor_id,
                    "relation_path": traversed,
                    "path_score": score,
                    "anchor_confidence": anchor_confidence,
                }
        rows = [{"item_id": dataset_id, "score": score, "provenance": {"query_id": str(params.get("anchor_query_id") or ""), "branch_type": "relation", "entity_type": str(params.get("anchor_entity_type") or ""), **dataset_paths[dataset_id]}} for dataset_id, score in dataset_scores.items()]
        rows.sort(key=lambda row: (-row["score"], row["item_id"]))
        return rows, "" if rows else "anchors_have_no_dataset_neighbor", {
            "expanded_anchors": len(selected_anchors),
            "selected_anchor_ids": [str((hit.get("entity") or {}).get("id") or "") for hit in selected_anchors],
            "selected_anchor_confidences": {
                key: round(value, 6) for key, value in sorted(anchor_confidences.items())
            },
        }

    @staticmethod
    def _anchor_identity_confidence(
        hit: Mapping[str, Any],
        *,
        query_text: str,
        source_evidence: str,
    ) -> float:
        entity = hit.get("entity") or {}

        def normalize(value: Any) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))

        expected = normalize(source_evidence or query_text)
        if not expected:
            return 0.0
        expected_tokens = set(expected.split())
        best = 0.0
        for value in (entity.get("title"), entity.get("name"), entity.get("id")):
            candidate = normalize(value)
            if not candidate:
                continue
            candidate_tokens = set(candidate.split())
            if candidate == expected:
                return 1.0
            if len(candidate_tokens) >= 3 and candidate in expected:
                best = max(best, 0.98)
                continue
            overlap = len(candidate_tokens & expected_tokens)
            coverage = overlap / max(1, len(candidate_tokens))
            jaccard = overlap / max(1, len(candidate_tokens | expected_tokens))
            score = 0.75 * coverage + 0.25 * jaccard
            if len(candidate_tokens) < 3:
                score = min(score, 0.65)
            best = max(best, score)
        return min(1.0, max(0.0, best))

    def _fuse_lists(self, branch_lists: Mapping[str, Sequence[Mapping[str, Any]]], weights: Mapping[str, float], candidate_map: Dict[str, Candidate], *, rrf_k: int) -> Dict[str, Candidate]:
        best_branch_weight = FUSION_BEST_BRANCH_WEIGHT if EXPERIMENT_BEST_BRANCH_FUSION else 0.0
        for fused in reciprocal_rank_fusion(
            branch_lists,
            weights=weights,
            rrf_k=rrf_k,
            best_branch_weight=best_branch_weight,
        ):
            dataset = self.dataset_lookup.get(fused.item_id)
            if not dataset:
                continue
            candidate = candidate_map.get(fused.item_id) or Candidate(fused.item_id, dataset)
            candidate.fusion_score = fused.score
            candidate.retrieval_prior_score = fused.score
            candidate.retrieval_provenance = []
            query_ids = []
            for row in fused.contributions:
                query_id = str(row.get("query_id") or "")
                if query_id and query_id not in query_ids:
                    query_ids.append(query_id)
                relation_path = [str(item) for item in row.get("relation_path", [])]
                provenance = RetrievalProvenance(
                    str(row.get("list_id") or ""),
                    query_id,
                    str(row.get("branch_type") or ""),
                    str(row.get("entity_type") or "dataset"),
                    int(row.get("rank", 0)),
                    float(row.get("raw_score", 0.0)),
                    float(row.get("rrf_contribution", 0.0)),
                    str(row.get("anchor_id") or ""),
                    relation_path,
                    float(row.get("anchor_confidence", 0.0) or 0.0),
                )
                candidate.retrieval_provenance.append(provenance)
                if provenance.branch_type == "dataset_query":
                    candidate.hybrid_score = max(candidate.hybrid_score, provenance.raw_score)
                    candidate.retrieval_rank = provenance.rank if candidate.retrieval_rank <= 0 else min(candidate.retrieval_rank, provenance.rank)
                if relation_path:
                    candidate.anchor_paths.append(
                        {
                            "anchor_id": provenance.anchor_id,
                            "anchor_confidence": provenance.anchor_confidence,
                            "relations": relation_path,
                            "score": provenance.raw_score,
                        }
                    )
                    candidate.graph_support += 1
                    candidate.graph_score += provenance.rrf_contribution
                    for edge in relation_path:
                        if edge not in candidate.graph_relation_types:
                            candidate.graph_relation_types.append(edge)
            candidate.graph_score = min(1.0, candidate.graph_score)
            candidate.matched_query_ids = query_ids
            candidate.evidence.append(f"rrf:score={fused.score:.4f},lists={len(fused.contributions)}")
            candidate_map[fused.item_id] = candidate
        return candidate_map

    def _constraint_violations(self, dataset: Mapping[str, Any], constraints: Sequence[ConstraintSpec]) -> List[str]:
        return self._constraint_assessment(dataset, constraints)[0]

    def _constraint_assessment(
        self,
        dataset: Mapping[str, Any],
        constraints: Sequence[ConstraintSpec],
    ) -> tuple[List[str], List[str]]:
        """Evaluate explicit constraints under open-world metadata semantics."""
        failed: List[str] = []
        unknown: List[str] = []
        dataset_id = str(dataset.get("id") or "")
        for item in constraints:
            value = dataset.get(item.field)
            if item.field == "text":
                value = " ".join(str(dataset.get(key) or "") for key in ("id", "name", "description"))
            if self._is_missing_constraint_value(value):
                unknown.append(item.field)
                continue
            if item.operator == "gte":
                try:
                    ok = float(value) >= float(item.value)
                except (TypeError, ValueError):
                    unknown.append(item.field)
                    continue
            elif item.operator == "lte":
                try:
                    ok = float(value) <= float(item.value)
                except (TypeError, ValueError):
                    unknown.append(item.field)
                    continue
            elif item.operator in {"contains_any", "contains_all", "in"}:
                cache_store = getattr(self, "_normalized_candidate_metadata", None)
                if cache_store is None:
                    cache_store = self._normalized_candidate_metadata = {}
                if dataset_id:
                    cache = cache_store.setdefault(dataset_id, {})
                    if item.field not in cache:
                        cache[item.field] = normalize_metadata_values(value, category=item.field)
                    actual = cache[item.field]
                else:
                    actual = normalize_metadata_values(value, category=item.field)
                required = normalize_metadata_values(item.value, category=item.field)
                ok = required.issubset(actual) if item.operator == "contains_all" else bool(actual & required)
            elif item.operator == "contains":
                ok = str(item.value).lower() in str(value or "").lower()
            else:
                ok = str(value).lower() == str(item.value).lower()
            if not ok:
                failed.append(item.field)
        return list(dict.fromkeys(failed)), list(dict.fromkeys(unknown))

    def _is_missing_constraint_value(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().casefold() in {"", "null", "none", "unknown", "<empty>"}
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False

    def _score_soft_preferences(self, candidate_map: Dict[str, Candidate], rows: Sequence[Any]) -> None:
        preferences = [row for row in rows if isinstance(row, Mapping)]
        for candidate in candidate_map.values():
            weighted, total = 0.0, 0.0
            for pref in preferences:
                field, objective = str(pref.get("field") or ""), str(pref.get("objective") or "maximize")
                weight = max(0.0, float(pref.get("weight", 1.0) or 1.0))
                value = float(candidate.dataset.get(field, 0) or 0) if field in {"downloads", "likes"} else 0.0
                reference = 100000.0 if field == "downloads" else 1000.0
                if field in {"downloads", "likes"}:
                    utility = min(1.0, math.log1p(max(0.0, value)) / math.log1p(reference))
                else:
                    actual = normalize_metadata_values(candidate.dataset.get(field), category=field)
                    desired = normalize_metadata_values(pref.get("value"), category=field)
                    utility = 1.0 if desired and actual & desired else 0.0
                if objective == "minimize":
                    utility = 1.0 - utility
                weighted += weight * utility
                total += weight
            candidate.soft_preference_score = weighted / total if total else 0.0

    def _rank_for_pruning(self, candidate_map: Dict[str, Candidate], plan: SearchPlan, limit: int | None = None) -> List[tuple[str, Candidate]]:
        for candidate in candidate_map.values():
            preference_weight = 0.15 if plan.soft_preferences else 0.0
            candidate.pre_rerank_score = (1.0 - preference_weight) * candidate.fusion_score + preference_weight * candidate.soft_preference_score
            candidate.final_score = candidate.pre_rerank_score
        rows = sorted(
            candidate_map.items(),
            key=lambda item: (
                bool(item[1].constraint_violations),
                -item[1].filter_score,
                -item[1].pre_rerank_score,
                item[0],
            ),
        )
        return rows[:limit] if limit is not None else rows

    def _rank_without_prerank(self, candidate_map: Dict[str, Candidate]) -> List[Candidate]:
        return sorted(
            candidate_map.values(),
            key=lambda item: (
                bool(item.constraint_violations),
                -item.filter_score,
                item.retrieval_rank if item.retrieval_rank > 0 else 10**9,
                -item.fusion_score,
                item.dataset_id,
            ),
        )

    def _trace(self, step: str, status: str, detail: Mapping[str, Any]) -> ExecutionTrace:
        if not self.debug_full_trace:
            detail = {key: value for key, value in detail.items() if key not in {"violations", "dataset_ids"}}
        return ExecutionTrace(step, status, dict(detail))
