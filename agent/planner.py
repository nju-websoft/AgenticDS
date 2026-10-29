from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from global_values import (
    AGENT_DEFAULT_TOP_K_FINAL,
    AGENT_QUALITY_THRESHOLD,
    AGENT_RETRIEVE_TOP_K_PER_QUERY,
    AGENT_USE_LLM_PLANNER,
    EXPERIMENT_ENFORCE_RELATION_PLAN_COMPLETENESS,
    PLANNER_LLM_TIMEOUT_SECONDS,
    EXPERIMENT_DISABLE_LLM_PLANNER,
    EXPERIMENT_PLANNER_RAW_QUERY_ONLY,
)
from llm import LLMGateway

from .schemas import (
    ConstraintSpec,
    CriticFeedback,
    PreferenceSpec,
    QueryContext,
    RelationSearchSpec,
    RetrievalPolicy,
    RetrievalQuerySpec,
    SearchPlan,
    LANGUAGE_ALIASES,
)

PROMPT_INITIAL_PATH = Path(__file__).resolve().parent.parent / "prompt" / "planner_prompt.md"
PROMPT_REPLAN_PATH = Path(__file__).resolve().parent.parent / "prompt" / "planner_replan_prompt.md"

RELATION_PATH_CATALOG: Dict[str, Dict[str, Any]] = {
    "paper_direct_dataset": {
        "anchor_type": "paper",
        "path": [{"edge_file": "dataset_cite_paper.json", "direction": "in", "target_type": "dataset"}],
    },
    "paper_model_dataset": {
        "anchor_type": "paper",
        "path": [
            {"edge_file": "model_cite_paper.json", "direction": "in", "target_type": "model"},
            {"edge_file": "model_trainedOrFineTunedOn_dataset.json", "direction": "out", "target_type": "dataset"},
        ],
    },
    "paper_collection_dataset": {
        "anchor_type": "paper",
        "path": [
            {"edge_file": "collection_contain_paper.json", "direction": "in", "target_type": "collection"},
            {"edge_file": "collection_contain_dataset.json", "direction": "out", "target_type": "dataset"},
        ],
    },
    "model_direct_dataset": {
        "anchor_type": "model",
        "path": [{"edge_file": "model_trainedOrFineTunedOn_dataset.json", "direction": "out", "target_type": "dataset"}],
    },
    "model_collection_dataset": {
        "anchor_type": "model",
        "path": [
            {"edge_file": "collection_contain_model.json", "direction": "in", "target_type": "collection"},
            {"edge_file": "collection_contain_dataset.json", "direction": "out", "target_type": "dataset"},
        ],
    },
    "collection_direct_dataset": {
        "anchor_type": "collection",
        "path": [{"edge_file": "collection_contain_dataset.json", "direction": "out", "target_type": "dataset"}],
    },
    "collection_model_dataset": {
        "anchor_type": "collection",
        "path": [
            {"edge_file": "collection_contain_model.json", "direction": "out", "target_type": "model"},
            {"edge_file": "model_trainedOrFineTunedOn_dataset.json", "direction": "out", "target_type": "dataset"},
        ],
    },
    "collection_paper_dataset": {
        "anchor_type": "collection",
        "path": [
            {"edge_file": "collection_contain_paper.json", "direction": "out", "target_type": "paper"},
            {"edge_file": "dataset_cite_paper.json", "direction": "in", "target_type": "dataset"},
        ],
    },
    "dataset_collection_dataset": {
        "anchor_type": "dataset",
        "path": [
            {"edge_file": "collection_contain_dataset.json", "direction": "in", "target_type": "collection"},
            {"edge_file": "collection_contain_dataset.json", "direction": "out", "target_type": "dataset"},
        ],
    },
    "dataset_model_dataset": {
        "anchor_type": "dataset",
        "path": [
            {"edge_file": "model_trainedOrFineTunedOn_dataset.json", "direction": "in", "target_type": "model"},
            {"edge_file": "model_trainedOrFineTunedOn_dataset.json", "direction": "out", "target_type": "dataset"},
        ],
    },
}
DEFAULT_PATH_BY_ANCHOR = {
    "paper": "paper_direct_dataset",
    "model": "model_direct_dataset",
    "collection": "collection_direct_dataset",
}
ALLOWED_CONSTRAINT_FIELDS = {"downloads", "likes", "languages", "tasks", "tags", "modalities", "license", "formats", "size", "split"}
ALLOWED_CONSTRAINT_OPERATORS = {"gte", "lte", "equals", "contains", "contains_any", "contains_all", "in"}
ALLOWED_INTENT_CODES = {"name_match", "task_match", "domain_match", "format_match", "training_data", "constraint_match", "anchor_support"}


class PlannerAgent:
    """Construct a diversified, falsifiable retrieval hypothesis."""

    def __init__(self, llm_gateway: LLMGateway) -> None:
        self.llm = llm_gateway
        self.initial_prompt = self._load_prompt(PROMPT_INITIAL_PATH)
        self.replan_prompt = self._load_prompt(PROMPT_REPLAN_PATH)

    def plan(self, context: QueryContext) -> SearchPlan:
        if EXPERIMENT_PLANNER_RAW_QUERY_ONLY:
            return self._raw_query_only_plan(context)
        baseline = self._rule_initial_plan(context)
        if not AGENT_USE_LLM_PLANNER or EXPERIMENT_DISABLE_LLM_PLANNER:
            return baseline
        payload = self.llm.generate_json(
            task_name="planner_initial",
            system_prompt=self.initial_prompt,
            user_prompt=json.dumps({"query": context.raw_query, "requested_top_k": context.requested_top_k, "locale": context.locale}, ensure_ascii=False),
            default=self._llm_plan_payload(baseline),
            timeout_seconds=PLANNER_LLM_TIMEOUT_SECONDS,
            thinking=True,
        )
        return self._normalize_plan(payload, baseline=baseline, raw_query=context.raw_query, iteration=1)

    def refine_plan(
        self,
        previous: SearchPlan,
        feedback: CriticFeedback,
        *,
        executed_query_signatures: Sequence[str] | None = None,
        coverage_statistics: Mapping[str, Any] | None = None,
    ) -> SearchPlan:
        if EXPERIMENT_PLANNER_RAW_QUERY_ONLY:
            return previous
        baseline = self._rule_replan(previous, feedback)
        if not AGENT_USE_LLM_PLANNER or EXPERIMENT_DISABLE_LLM_PLANNER:
            return baseline
        request = {
            "previous_plan": self._llm_plan_payload(previous),
            "executed_query_signatures": list(executed_query_signatures or []),
            "coverage_statistics": dict(coverage_statistics or {}),
            "critic_feedback": asdict(feedback),
            "constraint": "Add at most two non-duplicate repair queries; preserve supported existing queries.",
        }
        payload = self.llm.generate_json(
            task_name="planner_replan",
            system_prompt=self.replan_prompt,
            user_prompt=json.dumps(request, ensure_ascii=False),
            default=self._llm_plan_payload(baseline),
            timeout_seconds=PLANNER_LLM_TIMEOUT_SECONDS,
            thinking=True,
        )
        raw_query = previous.user_query
        return self._normalize_plan(payload, baseline=baseline, raw_query=raw_query, iteration=previous.origin_iteration + 1)

    def _raw_query_only_plan(self, context: QueryContext) -> SearchPlan:
        """Represent the no-Planner intervention without hidden decomposition."""
        raw = context.raw_query.strip()
        top_k_final = max(1, int(context.requested_top_k or AGENT_DEFAULT_TOP_K_FINAL))
        query = RetrievalQuerySpec(
            query_id="q_raw",
            role="task_intent",
            query_text=raw,
            target_entity_types=["dataset"],
            weight=1.0,
            intent_codes=[],
            evidence_focus="raw",
        )
        return SearchPlan(
            user_query=raw,
            retrieval_queries=[query],
            relation_search_specs=[],
            hard_constraints=[],
            soft_preferences=[],
            retrieval_policy=RetrievalPolicy(per_query_top_k=AGENT_RETRIEVE_TOP_K_PER_QUERY),
            intent_codes=[],
            top_k_final=top_k_final,
            quality_target={
                "min_candidates": top_k_final,
                "min_quality": AGENT_QUALITY_THRESHOLD,
                "min_intent_coverage": 0.7,
            },
            plan_outline=[],
            reasoning="planner_ablation_raw_query_only",
            origin_iteration=1,
            shortlist_allocations={},
        )

    def _rule_initial_plan(self, context: QueryContext) -> SearchPlan:
        raw = context.raw_query.strip()
        top_k_final = max(1, int(context.requested_top_k or AGENT_DEFAULT_TOP_K_FINAL))
        constraints, preferences = self._extract_constraints_and_preferences(raw)
        anchor = self._extract_dataset_name_hint(raw)
        queries: List[RetrievalQuerySpec] = []
        if anchor:
            queries.append(RetrievalQuerySpec("q_exact", "exact_name", anchor, ["dataset"], 1.0, intent_codes=["name_match"], source_evidence=anchor, evidence_focus="identity"))
        queries.extend(
            [
                RetrievalQuerySpec("q_task", "task_intent", raw, ["dataset"], 1.0, intent_codes=["task_match", "domain_match"], evidence_focus="semantics"),
                RetrievalQuerySpec("q_format", "data_format", f"{raw} dataset schema supervision fields examples splits", ["dataset"], 1.0, intent_codes=["format_match", "training_data"], evidence_focus="structure"),
                RetrievalQuerySpec(
                    "q_constraint" if constraints or preferences else "q_evidence",
                    "constraints" if constraints or preferences else "task_intent",
                    self._constraint_query(raw, constraints, preferences),
                    ["dataset"],
                    1.0,
                    intent_codes=["constraint_match"] if constraints or preferences else ["task_match", "domain_match"],
                    evidence_focus="constraints" if constraints or preferences else "provenance",
                ),
            ]
        )
        anchor_type = self._infer_relation_anchor_type(raw)
        if anchor_type:
            queries.append(RetrievalQuerySpec(f"q_anchor_{anchor_type}", "task_intent", raw, [anchor_type], 1.0, intent_codes=["anchor_support"], evidence_focus="relation"))
        queries = self._deduplicate_queries(queries)[:5]
        relation_specs = self._relation_specs_for_queries(queries, [])
        return SearchPlan(
            user_query=raw,
            retrieval_queries=queries,
            relation_search_specs=relation_specs,
            hard_constraints=constraints,
            soft_preferences=preferences,
            retrieval_policy=RetrievalPolicy(per_query_top_k=AGENT_RETRIEVE_TOP_K_PER_QUERY),
            intent_codes=self._infer_intent_codes(raw),
            top_k_final=top_k_final,
            quality_target={
                "min_candidates": top_k_final,
                "min_quality": AGENT_QUALITY_THRESHOLD,
                "min_intent_coverage": 0.7,
            },
            plan_outline=[
                {"title": "decompose_intent", "objective": "Represent task, supervision format, constraints, and resolvable anchors as complementary hypotheses."},
                {"title": "retrieve_independent_lists", "objective": "Preserve evidence from each hypothesis as an independent ranked list."},
                {"title": "fuse_and_test", "objective": "Fuse lists, enforce explicit constraints, and test quality and intent coverage."},
            ],
            reasoning="Diversified retrieval hypotheses reduce vocabulary mismatch while preserving provenance.",
            origin_iteration=1,
            shortlist_allocations=self._default_shortlist_allocations(queries, relation_specs, top_k_final),
        )

    def _rule_replan(self, previous: SearchPlan, feedback: CriticFeedback) -> SearchPlan:
        queries = list(previous.retrieval_queries)
        next_index = len(queries) + 1
        query_text = previous.user_query
        diagnostics = list(feedback.diagnostics)
        if EXPERIMENT_ENFORCE_RELATION_PLAN_COMPLETENESS:
            anchor_codes = {"paper_anchor_unresolved", "model_anchor_unresolved", "collection_anchor_unresolved"}
            diagnostics.sort(key=lambda item: (item.code not in anchor_codes, item.severity != "critical"))
        for diagnostic in diagnostics[:2]:
            repair = self._repair_query(diagnostic.code, query_text, next_index, previous.origin_iteration + 1)
            if repair:
                queries = self._insert_repair_query(queries, repair)
                next_index += 1
        queries = self._deduplicate_queries(queries)[:5]
        return SearchPlan(
            user_query=previous.user_query,
            retrieval_queries=queries,
            relation_search_specs=self._relation_specs_for_queries(queries, previous.relation_search_specs),
            hard_constraints=list(previous.hard_constraints),
            soft_preferences=list(previous.soft_preferences),
            retrieval_policy=previous.retrieval_policy,
            intent_codes=list(previous.intent_codes),
            top_k_final=previous.top_k_final,
            quality_target=dict(previous.quality_target),
            plan_outline=list(previous.plan_outline) + [{"title": "diagnostic_repair", "objective": "Introduce only hypotheses addressing observed evidence deficits."}],
            reasoning=f"Diagnostic-driven update: {', '.join(item.code for item in feedback.diagnostics) or feedback.reason}",
            origin_iteration=previous.origin_iteration + 1,
            shortlist_allocations=dict(previous.shortlist_allocations),
        )

    def _normalize_plan(self, payload: Any, *, baseline: SearchPlan, raw_query: str, iteration: int) -> SearchPlan:
        if not isinstance(payload, Mapping):
            return baseline
        rows = payload.get("retrieval_queries")
        queries = [RetrievalQuerySpec.from_mapping(row, i + 1) for i, row in enumerate(rows or []) if isinstance(row, Mapping)]
        if iteration > 1:
            existing = list(baseline.retrieval_queries)
            additions = [item for item in queries if item.signature() not in {row.signature() for row in existing}][:2]
            for addition in additions:
                existing = self._insert_repair_query(existing, addition)
            queries = existing
        if not queries:
            queries = list(baseline.retrieval_queries)
        queries = self._validate_exact_name_queries(self._deduplicate_queries(queries), raw_query)[:5]
        if EXPERIMENT_ENFORCE_RELATION_PLAN_COMPLETENESS:
            queries = self._ensure_relation_anchor_query(queries, baseline, raw_query)
        if not any("dataset" in item.target_entity_types for item in queries):
            queries.insert(0, baseline.retrieval_queries[0])
        queries = queries[:5]
        while len(queries) < 3:
            fallback = baseline.retrieval_queries[len(queries) % len(baseline.retrieval_queries)]
            if fallback.signature() not in {item.signature() for item in queries}:
                queries.append(fallback)
            else:
                break

        constraints = self._validate_constraints(
            [ConstraintSpec.from_mapping(row) for row in payload.get("hard_constraints", []) if isinstance(row, Mapping)],
            raw_query,
        ) or list(baseline.hard_constraints)
        preferences = [PreferenceSpec.from_mapping(row) for row in payload.get("soft_preferences", []) if isinstance(row, Mapping)] or list(baseline.soft_preferences)
        relation_rows = payload.get("relation_search_specs", [])
        relation_specs = [RelationSearchSpec.from_mapping(row, i + 1) for i, row in enumerate(relation_rows) if isinstance(row, Mapping)]
        relation_specs = self._relation_specs_for_queries(queries, relation_specs or baseline.relation_search_specs)
        shortlist_allocations = self._normalize_shortlist_allocations(
            payload.get("shortlist_allocations"),
            queries,
            relation_specs,
            baseline.top_k_final,
            fallback=baseline.shortlist_allocations,
        )
        policy = RetrievalPolicy.from_mapping(asdict(baseline.retrieval_policy))
        quality = dict(baseline.quality_target)
        quality["min_quality"] = min(1.0, max(0.0, float(quality.get("min_quality", AGENT_QUALITY_THRESHOLD))))
        quality["min_intent_coverage"] = min(1.0, max(0.0, float(quality.get("min_intent_coverage", 0.7))))
        intent_codes = [str(item) for item in payload.get("intent_codes", []) if str(item) in ALLOWED_INTENT_CODES]
        if not intent_codes:
            intent_codes = list(dict.fromkeys(code for query in queries for code in query.intent_codes if code in ALLOWED_INTENT_CODES))
        if not intent_codes:
            intent_codes = list(baseline.intent_codes)
        return SearchPlan(
            user_query=raw_query,
            retrieval_queries=queries,
            relation_search_specs=relation_specs,
            hard_constraints=constraints,
            soft_preferences=preferences,
            retrieval_policy=policy,
            intent_codes=intent_codes,
            top_k_final=baseline.top_k_final,
            quality_target=quality,
            plan_outline=self._normalize_plan_outline(payload.get("plan_outline")) or list(baseline.plan_outline),
            reasoning=str(payload.get("reasoning") or baseline.reasoning),
            origin_iteration=iteration,
            shortlist_allocations=shortlist_allocations,
        )

    def _ensure_relation_anchor_query(
        self,
        queries: Sequence[RetrievalQuerySpec],
        baseline: SearchPlan,
        raw_query: str,
    ) -> List[RetrievalQuerySpec]:
        """Enforce the typed-anchor invariant implied by a relation information need."""
        anchor_type = self._infer_relation_anchor_type(raw_query)
        output = list(queries)
        if not anchor_type or any(anchor_type in item.target_entity_types for item in output):
            return output[:5]
        fallback = next(
            (item for item in baseline.retrieval_queries if anchor_type in item.target_entity_types),
            None,
        )
        if fallback is None:
            fallback = RetrievalQuerySpec(
                query_id=f"q_anchor_{anchor_type}",
                role="feedback_repair",
                query_text=raw_query,
                target_entity_types=[anchor_type],
                weight=1.0,
                intent_codes=["anchor_support"],
                source_evidence=raw_query,
                origin_iteration=baseline.origin_iteration,
                evidence_focus="relation",
            )
        if len(output) < 5:
            output.append(fallback)
        else:
            replace_index = next(
                (
                    index
                    for index in range(len(output) - 1, -1, -1)
                    if output[index].role != "exact_name" and output[index].evidence_focus == "semantics"
                ),
                len(output) - 1,
            )
            output[replace_index] = fallback
        return self._deduplicate_queries(output)[:5]

    def _relation_specs_for_queries(self, queries: Sequence[RetrievalQuerySpec], provided: Sequence[RelationSearchSpec]) -> List[RelationSearchSpec]:
        by_anchor: Dict[tuple[str, str], List[RelationSearchSpec]] = {}
        for spec in provided:
            by_anchor.setdefault((spec.anchor_query_id, spec.anchor_entity_type), []).append(spec)
        used_relation_ids: set[str] = set()

        def unique_relation_id(preferred: str, fallback: str) -> str:
            candidate = preferred.strip() or fallback
            if candidate not in used_relation_ids:
                used_relation_ids.add(candidate)
                return candidate
            candidate = fallback
            suffix = 2
            while candidate in used_relation_ids:
                candidate = f"{fallback}_{suffix}"
                suffix += 1
            used_relation_ids.add(candidate)
            return candidate

        output: List[RelationSearchSpec] = []
        for query in queries:
            for entity_type in query.target_entity_types:
                proposed_specs = by_anchor.get((query.query_id, entity_type), [])
                accepted = 0
                for proposed in proposed_specs:
                    path_id = self._resolve_path_id(proposed)
                    definition = RELATION_PATH_CATALOG.get(path_id)
                    if not definition or definition["anchor_type"] != entity_type:
                        continue
                    output.append(
                        RelationSearchSpec(
                            relation_id=unique_relation_id(
                                proposed.relation_id,
                                f"rel_{query.query_id}_{path_id}",
                            ),
                            anchor_query_id=query.query_id,
                            anchor_entity_type=entity_type,
                            path=[dict(step) for step in definition["path"]],
                            weight=1.0,
                            top_k_anchors=proposed.top_k_anchors,
                            max_neighbors_per_anchor=proposed.max_neighbors_per_anchor,
                            min_anchor_score=proposed.min_anchor_score,
                            path_id=path_id,
                            relation_intent=proposed.relation_intent,
                            max_intermediate_per_anchor=proposed.max_intermediate_per_anchor,
                            max_datasets_per_intermediate=proposed.max_datasets_per_intermediate,
                        )
                    )
                    accepted += 1
                if accepted or entity_type not in DEFAULT_PATH_BY_ANCHOR:
                    continue
                path_id = DEFAULT_PATH_BY_ANCHOR[entity_type]
                definition = RELATION_PATH_CATALOG[path_id]
                output.append(
                    RelationSearchSpec(
                        relation_id=unique_relation_id("", f"rel_{query.query_id}_{path_id}"),
                        anchor_query_id=query.query_id,
                        anchor_entity_type=entity_type,
                        path=[dict(step) for step in definition["path"]],
                        weight=1.0,
                        path_id=path_id,
                        relation_intent="direct_relation",
                    )
                )
        return output

    def _resolve_path_id(self, spec: RelationSearchSpec) -> str:
        if spec.path_id in RELATION_PATH_CATALOG:
            return spec.path_id
        signature = self._relation_path_signature(spec.path)
        return next(
            (
                path_id
                for path_id, definition in RELATION_PATH_CATALOG.items()
                if self._relation_path_signature(definition["path"]) == signature
            ),
            "",
        )

    def _normalize_shortlist_allocations(
        self,
        payload: Any,
        queries: Sequence[RetrievalQuerySpec],
        relation_specs: Sequence[RelationSearchSpec],
        top_k_final: int,
        *,
        fallback: Mapping[str, int] | None = None,
    ) -> Dict[str, int]:
        valid = {
            *(f"dataset:{item.query_id}" for item in queries if "dataset" in item.target_entity_types),
            *(f"relation:{item.relation_id}" for item in relation_specs),
        }
        rows = payload if isinstance(payload, Mapping) else fallback or {}
        budget = max(0, int(top_k_final) // 5)
        output: Dict[str, int] = {}
        remaining = budget
        for branch_id, requested in rows.items():
            branch_id = str(branch_id)
            if branch_id not in valid or remaining <= 0:
                continue
            try:
                slots = max(0, int(requested))
            except (TypeError, ValueError):
                continue
            slots = min(slots, remaining)
            if slots:
                output[branch_id] = slots
                remaining -= slots
        return output

    def _default_shortlist_allocations(
        self,
        queries: Sequence[RetrievalQuerySpec],
        relation_specs: Sequence[RelationSearchSpec],
        top_k_final: int,
    ) -> Dict[str, int]:
        budget = max(0, int(top_k_final) // 5)
        priority = [f"relation:{item.relation_id}" for item in relation_specs]
        priority.extend(
            f"dataset:{item.query_id}"
            for item in queries
            if "dataset" in item.target_entity_types and item.evidence_focus in {"identity", "structure", "provenance"}
        )
        priority.extend(f"dataset:{item.query_id}" for item in queries if "dataset" in item.target_entity_types)
        output: Dict[str, int] = {}
        for branch_id in dict.fromkeys(priority):
            if budget <= 0:
                break
            output[branch_id] = 1
            budget -= 1
        return output

    @staticmethod
    def _relation_path_signature(path: Sequence[Mapping[str, str]]) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            (
                str(step.get("edge_file") or ""),
                str(step.get("direction") or ""),
                str(step.get("target_type") or ""),
            )
            for step in path
        )

    def _validate_exact_name_queries(self, queries: Sequence[RetrievalQuerySpec], raw_query: str) -> List[RetrievalQuerySpec]:
        raw_lower = raw_query.lower()
        output = []
        for query in queries:
            if query.role == "exact_name":
                evidence = query.source_evidence.strip()
                if not evidence or evidence.lower() not in raw_lower:
                    continue
            output.append(query)
        return output

    def _validate_constraints(self, constraints: Sequence[ConstraintSpec], raw_query: str) -> List[ConstraintSpec]:
        raw_lower = raw_query.lower()
        output = []
        for item in constraints:
            if item.field not in ALLOWED_CONSTRAINT_FIELDS or item.operator not in ALLOWED_CONSTRAINT_OPERATORS:
                continue
            source = item.source_text.strip()
            if not source or source.lower() not in raw_lower:
                continue
            output.append(item)
        return output

    def _extract_constraints_and_preferences(self, query: str) -> tuple[List[ConstraintSpec], List[PreferenceSpec]]:
        constraints: List[ConstraintSpec] = []
        preferences: List[PreferenceSpec] = []
        lower = query.lower()
        for field, pattern in [("downloads", r"(?:at least|min(?:imum)?|>=)\s*([\d,]+)\s*downloads?"), ("likes", r"(?:at least|min(?:imum)?|>=)\s*([\d,]+)\s*likes?")]:
            match = re.search(pattern, lower)
            if match:
                constraints.append(ConstraintSpec(field, "gte", int(match.group(1).replace(",", "")), match.group(0)))
        for field, pattern in [("downloads", r"high downloads?|popular|widely used"), ("likes", r"high likes?|well liked")]:
            match = re.search(pattern, lower)
            if match:
                preferences.append(PreferenceSpec(field, "maximize", weight=1.0, source_text=match.group(0)))
        language_names = {name: code for name, code in LANGUAGE_ALIASES.items() if len(name) > 2}
        for name, code in language_names.items():
            if re.search(rf"\b{name}\b", lower):
                if re.search(rf"\b(?:prefer|preferably|ideally)\b[^.]*\b{name}\b", lower):
                    preferences.append(PreferenceSpec("languages", "prefer", [code], 1.0, name))
                else:
                    constraints.append(ConstraintSpec("languages", "contains_any", [code], name))
        return constraints, preferences

    def _constraint_query(self, raw: str, constraints: Sequence[ConstraintSpec], preferences: Sequence[PreferenceSpec]) -> str:
        terms = [f"{item.field} {item.value}" for item in constraints] + [f"{item.objective} {item.field}" for item in preferences]
        return f"{raw} {' '.join(terms) if terms else 'documented dataset card task labels splits examples'}".strip()

    def _repair_query(self, code: str, raw: str, index: int, iteration: int) -> RetrievalQuerySpec | None:
        if code == "missing_exact_name_match":
            anchor = self._extract_dataset_name_hint(raw)
            if not anchor:
                return None
            return RetrievalQuerySpec(f"q_repair_{index}", "exact_name", anchor, ["dataset"], 1.0, intent_codes=["name_match"], source_evidence=anchor, origin_iteration=iteration, evidence_focus="identity")
        mapping = {
            "missing_task_match": ("feedback_repair", f"{raw} task aligned training examples", ["dataset"]),
            "missing_format_match": ("feedback_repair", f"{raw} input output instruction response schema", ["dataset"]),
            "insufficient_training_data": ("feedback_repair", f"{raw} training split supervised examples", ["dataset"]),
            "paper_anchor_unresolved": ("feedback_repair", f"{raw} paper title publication arXiv", ["paper"]),
            "model_anchor_unresolved": ("feedback_repair", f"{raw} model checkpoint organization task", ["model"]),
            "collection_anchor_unresolved": ("feedback_repair", f"{raw} Hugging Face collection title topic", ["collection"]),
            "candidate_family_collapse": ("feedback_repair", f"{raw} alternative dataset sources diverse coverage", ["dataset"]),
            "low_evidence_confidence": ("feedback_repair", f"{raw} documented dataset card schema samples", ["dataset"]),
            "insufficient_candidate_count": ("feedback_repair", f"{raw} related supervised training datasets", ["dataset"]),
        }
        if code not in mapping:
            return None
        role, text, target_types = mapping[code]
        focus = "relation" if target_types != ["dataset"] else ("structure" if code in {"missing_format_match", "insufficient_training_data"} else "semantics")
        return RetrievalQuerySpec(f"q_repair_{index}", role, text, target_types, 1.0, origin_iteration=iteration, evidence_focus=focus)

    def _insert_repair_query(self, queries: Sequence[RetrievalQuerySpec], repair: RetrievalQuerySpec) -> List[RetrievalQuerySpec]:
        output = list(queries)
        if len(output) < 5:
            output.append(repair)
            return output
        repair_type = next((item for item in repair.target_entity_types if item != "dataset"), "")
        replace_index = next(
            (
                index
                for index, item in enumerate(output)
                if repair_type and repair_type in item.target_entity_types and item.role != "exact_name"
            ),
            None,
        )
        if replace_index is None:
            replace_index = next((index for index, item in enumerate(output) if item.role == "constraints"), None)
        if replace_index is None:
            replace_index = next((index for index, item in enumerate(output) if item.role == "feedback_repair"), len(output) - 1)
        output[replace_index] = repair
        return output

    def _infer_intent_codes(self, raw: str) -> List[str]:
        lower = raw.lower()
        codes = ["task_match", "domain_match"]
        if any(token in lower for token in ("instruction", "input", "output", "format", "schema", "qa", "question")):
            codes.append("format_match")
        if any(token in lower for token in ("train", "training", "sft", "finetun")):
            codes.append("training_data")
        if self._extract_dataset_name_hint(raw):
            codes.append("name_match")
        return codes

    def _infer_relation_anchor_type(self, raw: str) -> str:
        lower = raw.lower()
        paper_relation_patterns = (
            r"\bdatasets?\s+(?:used|introduced|released|evaluated)\s+(?:in|by|for)\b",
            r"\bused\s+alongside\b",
            r"\btechnical\s+report\b",
            r"\breference[sd]?\s+(?:in|by)\b",
        )
        if any(re.search(pattern, lower) for pattern in paper_relation_patterns):
            return "paper"
        if any(token in lower for token in ("paper", "publication", "arxiv", "doi")):
            return "paper"
        if any(token in lower for token in ("model", "checkpoint", "fine-tuned", "finetuned")):
            return "model"
        if "collection" in lower:
            return "collection"
        return ""

    def _extract_dataset_name_hint(self, query: str) -> str:
        match = re.search(r"\b[\w.-]+/[\w.-]+\b", query)
        return match.group(0) if match else ""

    def _deduplicate_queries(self, queries: Sequence[RetrievalQuerySpec]) -> List[RetrievalQuerySpec]:
        output, seen, seen_ids = [], set(), set()
        for query in queries:
            query.query_text = query.query_text.strip()[:512]
            if not query.query_text:
                continue
            signature = query.signature()
            if signature in seen:
                continue
            base_id = query.query_id or f"q{len(output) + 1}"
            query.query_id = base_id
            query.intent_codes = [code for code in query.intent_codes if code in ALLOWED_INTENT_CODES]
            suffix = 2
            while query.query_id in seen_ids:
                query.query_id = f"{base_id}_{suffix}"
                suffix += 1
            seen.add(signature)
            seen_ids.add(query.query_id)
            output.append(query)
        return output

    def _plan_payload(self, plan: SearchPlan) -> Dict[str, Any]:
        return {
            "retrieval_queries": [asdict(item) for item in plan.retrieval_queries],
            "relation_search_specs": [asdict(item) for item in plan.relation_search_specs],
            "hard_constraints": [asdict(item) for item in plan.hard_constraints],
            "soft_preferences": [asdict(item) for item in plan.soft_preferences],
            "retrieval_policy": asdict(plan.retrieval_policy),
            "intent_codes": list(plan.intent_codes),
            "top_k_final": plan.top_k_final,
            "quality_target": dict(plan.quality_target),
            "plan_outline": list(plan.plan_outline),
            "reasoning": plan.reasoning,
            "shortlist_allocations": dict(plan.shortlist_allocations),
        }

    def _llm_plan_payload(self, plan: SearchPlan) -> Dict[str, Any]:
        """Expose only Planner decisions; protocol constants remain deterministic."""
        return {
            "retrieval_queries": [asdict(item) for item in plan.retrieval_queries],
            "relation_search_specs": [asdict(item) for item in plan.relation_search_specs],
            "hard_constraints": [asdict(item) for item in plan.hard_constraints],
            "soft_preferences": [asdict(item) for item in plan.soft_preferences],
            "intent_codes": list(plan.intent_codes),
            "shortlist_allocations": dict(plan.shortlist_allocations),
        }

    def _normalize_plan_outline(self, payload: Any) -> List[Dict[str, Any]]:
        output = []
        for item in payload if isinstance(payload, list) else []:
            if isinstance(item, str) and item.strip():
                output.append({"title": item.strip(), "objective": ""})
            elif isinstance(item, Mapping):
                title = str(item.get("title") or item.get("step") or "").strip()
                if title:
                    output.append({"title": title, "objective": str(item.get("objective") or item.get("goal") or "").strip()})
        return output

    def _load_prompt(self, path: Path) -> str:
        return path.read_text(encoding="utf-8") if path.exists() else "Return strict JSON only."
