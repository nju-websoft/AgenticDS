from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping

from global_values import (
    AGENT_USE_LLM_ENGINEER,
    ENGINEER_LLM_TIMEOUT_SECONDS,
    EXPERIMENT_DISABLE_LLM_ENGINEER,
    EXPERIMENT_GRAPH_TEXT_CONTROL,
    EXPERIMENT_RAW_QUERY_BRANCH,
)
from llm import LLMGateway

from .schemas import ActionStep, ExecutionProtocol, RetrievalQuerySpec, SearchPlan

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompt" / "engineer_prompt.md"


class EngineerAgent:
    """Allocate bounded retrieval effort to a planner-defined hypothesis set."""

    def __init__(self, llm_gateway: LLMGateway) -> None:
        self.llm = llm_gateway
        self.system_prompt = self._load_prompt(PROMPT_PATH)

    def compile_plan(self, plan: SearchPlan) -> ExecutionProtocol:
        policy = self._default_execution_policy(plan)
        if AGENT_USE_LLM_ENGINEER and not EXPERIMENT_DISABLE_LLM_ENGINEER:
            request = {
                "retrieval_queries": [asdict(item) for item in plan.retrieval_queries],
                "relation_search_specs": [asdict(item) for item in plan.relation_search_specs],
                "hard_constraints": [asdict(item) for item in plan.hard_constraints],
                "soft_preferences": [asdict(item) for item in plan.soft_preferences],
                "retrieval_policy_bounds": asdict(plan.retrieval_policy),
                "default_execution_policy": policy,
            }
            result = self.llm.generate_json(
                task_name="engineer",
                system_prompt=self.system_prompt,
                user_prompt=json.dumps(request, ensure_ascii=False),
                default=policy,
                timeout_seconds=ENGINEER_LLM_TIMEOUT_SECONDS,
            )
            policy = self._validate_execution_policy(result, plan, policy)
        return ExecutionProtocol(self._compile_steps(plan, policy), int(policy["fused_pool_size"]))

    def _default_execution_policy(self, plan: SearchPlan) -> Dict[str, Any]:
        return {
            "query_budgets": {item.query_id: plan.retrieval_policy.per_query_top_k for item in plan.retrieval_queries},
            "relation_anchor_budgets": {item.relation_id: item.top_k_anchors for item in plan.relation_search_specs},
            "relation_neighbor_budgets": {item.relation_id: item.max_neighbors_per_anchor for item in plan.relation_search_specs},
            "enabled_relation_ids": [item.relation_id for item in plan.relation_search_specs],
            "fused_pool_size": plan.retrieval_policy.fused_pool_size,
        }

    def _validate_execution_policy(self, payload: Any, plan: SearchPlan, fallback: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, Mapping):
            return fallback
        known_queries = {item.query_id: item for item in plan.retrieval_queries}
        known_relations = {item.relation_id for item in plan.relation_search_specs}
        raw_budgets = payload.get("query_budgets") if isinstance(payload.get("query_budgets"), Mapping) else {}
        lower = min(60, max(10, plan.top_k_final))
        upper = 60
        budgets = {query_id: min(upper, max(lower, int(raw_budgets.get(query_id, fallback["query_budgets"][query_id]) or lower))) for query_id in known_queries}
        raw_anchor_budgets = payload.get("relation_anchor_budgets") if isinstance(payload.get("relation_anchor_budgets"), Mapping) else {}
        raw_neighbor_budgets = payload.get("relation_neighbor_budgets") if isinstance(payload.get("relation_neighbor_budgets"), Mapping) else {}
        anchor_budgets = {relation_id: min(10, max(1, int(raw_anchor_budgets.get(relation_id, fallback["relation_anchor_budgets"][relation_id]) or 5))) for relation_id in known_relations}
        neighbor_budgets = {relation_id: min(50, max(1, int(raw_neighbor_budgets.get(relation_id, fallback["relation_neighbor_budgets"][relation_id]) or 30))) for relation_id in known_relations}
        remaining_anchor_budget = 24
        relation_ids = sorted(anchor_budgets)
        for index, relation_id in enumerate(relation_ids):
            reserve = len(relation_ids) - index - 1
            anchor_budgets[relation_id] = min(anchor_budgets[relation_id], max(1, remaining_anchor_budget - reserve))
            remaining_anchor_budget -= anchor_budgets[relation_id]
        enabled = payload.get("enabled_relation_ids") if isinstance(payload.get("enabled_relation_ids"), list) else fallback["enabled_relation_ids"]
        enabled = [str(item) for item in enabled if str(item) in known_relations]
        required_query_ids = {item.query_id for item in plan.retrieval_queries if item.required}
        enabled = list(dict.fromkeys(enabled + [item.relation_id for item in plan.relation_search_specs if item.anchor_query_id in required_query_ids]))
        return {
            "query_budgets": budgets,
            "relation_anchor_budgets": anchor_budgets,
            "relation_neighbor_budgets": neighbor_budgets,
            "enabled_relation_ids": enabled,
            "fused_pool_size": min(500, max(300, int(payload.get("fused_pool_size", fallback["fused_pool_size"]) or fallback["fused_pool_size"]))),
        }

    def _compile_steps(self, plan: SearchPlan, policy: Mapping[str, Any]) -> List[ActionStep]:
        steps: List[ActionStep] = []
        query_by_id = {item.query_id: item for item in plan.retrieval_queries}
        anchor_pairs = {
            (item.anchor_query_id, item.anchor_entity_type)
            for item in plan.relation_search_specs
        }
        raw_query = " ".join(plan.user_query.split())
        has_equivalent_dataset_query = any(
            "dataset" in item.target_entity_types
            and " ".join(item.query_text.lower().split()) == raw_query.lower()
            for item in plan.retrieval_queries
        )
        if EXPERIMENT_RAW_QUERY_BRANCH and raw_query and not has_equivalent_dataset_query:
            steps.append(
                ActionStep(
                    "retrieve_dataset_branch",
                    {
                        "query_id": "__raw__",
                        "role": "raw_query",
                        "entity_type": "dataset",
                        "query": {
                            "entity_type": "dataset",
                            "query": raw_query,
                            "query_text": raw_query,
                            "facets": {},
                        },
                        "top_k": plan.retrieval_policy.per_query_top_k,
                        "weight": 1.0,
                        "signature": f"{raw_query.lower()}|dataset|{{}}",
                        "capture_as_anchor": False,
                    },
                )
            )
        for query in plan.retrieval_queries:
            for entity_type in query.target_entity_types:
                if entity_type == "dataset":
                    steps.append(
                        ActionStep(
                            "retrieve_dataset_branch",
                            {
                                **self._query_step_params(query, entity_type, policy),
                                "capture_as_anchor": (query.query_id, entity_type) in anchor_pairs,
                            },
                        )
                    )
                elif not EXPERIMENT_GRAPH_TEXT_CONTROL:
                    steps.append(ActionStep("retrieve_relation_anchors", self._query_step_params(query, entity_type, policy)))

        enabled_relations = set(policy.get("enabled_relation_ids", []))
        for relation in plan.relation_search_specs:
            if relation.relation_id not in enabled_relations or relation.anchor_query_id not in query_by_id:
                continue
            anchor_query = query_by_id[relation.anchor_query_id]
            if EXPERIMENT_GRAPH_TEXT_CONTROL:
                steps.append(
                    ActionStep(
                        "retrieve_relation_text_control",
                        {
                            "relation_id": relation.relation_id,
                            "anchor_query_id": relation.anchor_query_id,
                            "role": "relation_text_control",
                            "query": {
                                "entity_type": "dataset",
                                "query": anchor_query.query_text,
                                "query_text": anchor_query.query_text,
                                "facets": dict(anchor_query.facets),
                            },
                            # Match the relation branch's declared dataset-output
                            # budget while replacing structural traversal with text retrieval.
                            "top_k": int(policy["relation_neighbor_budgets"][relation.relation_id]),
                            "weight": relation.weight,
                            "signature": f"{anchor_query.query_text.lower()}|dataset|relation_text_control",
                        },
                    )
                )
                continue
            steps.append(
                ActionStep(
                    "expand_relation_anchors",
                    {
                        **asdict(relation),
                        "top_k_anchors": int(policy["relation_anchor_budgets"][relation.relation_id]),
                        "max_neighbors_per_anchor": int(policy["relation_neighbor_budgets"][relation.relation_id]),
                        "path_decay": plan.retrieval_policy.relation_path_decay,
                        "anchor_query_text": anchor_query.query_text,
                        "anchor_source_evidence": anchor_query.source_evidence,
                        "anchor_query_role": anchor_query.role,
                    },
                )
            )

        return steps

    def _query_step_params(self, query: RetrievalQuerySpec, entity_type: str, policy: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "query_id": query.query_id,
            "role": query.role,
            "entity_type": entity_type,
            "query": {"entity_type": entity_type, "query": query.query_text, "query_text": query.query_text, "facets": dict(query.facets)},
            "top_k": int(policy["query_budgets"][query.query_id]),
            "weight": 1.0,
            "signature": query.signature(),
        }

    def _load_prompt(self, path: Path) -> str:
        return path.read_text(encoding="utf-8") if path.exists() else "Return strict JSON only."
