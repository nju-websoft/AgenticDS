from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping


QUERY_ROLES = {"exact_name", "task_intent", "data_format", "constraints", "feedback_repair"}
ENTITY_TYPES = {"dataset", "paper", "model", "collection"}
EVIDENCE_FOCI = {"identity", "semantics", "structure", "constraints", "provenance", "relation"}
LANGUAGE_ALIASES = {
    "english": "en", "en": "en", "chinese": "zh", "zh": "zh", "中文": "zh",
    "french": "fr", "fr": "fr", "german": "de", "de": "de",
    "japanese": "ja", "ja": "ja", "korean": "ko", "ko": "ko",
}


def normalize_metadata_token(value: Any, *, category: str = "") -> str:
    token = " ".join(str(value or "").strip().lower().replace("_", "-").split())
    if category == "languages":
        if token.startswith("language:"):
            token = token.split(":", 1)[1].strip()
        token = LANGUAGE_ALIASES.get(token, token)
    return token


def normalize_metadata_values(values: Any, *, category: str = "") -> set[str]:
    rows = values if isinstance(values, list) else ([] if values in (None, "") else [values])
    return {token for token in (normalize_metadata_token(item, category=category) for item in rows) if token}


@dataclass
class QueryContext:
    raw_query: str
    requested_top_k: int = 20
    locale: str = "zh-CN"
    metadata_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalQuerySpec:
    query_id: str
    role: str
    query_text: str
    target_entity_types: List[str] = field(default_factory=lambda: ["dataset"])
    weight: float = 1.0
    facets: Dict[str, Any] = field(default_factory=dict)
    intent_codes: List[str] = field(default_factory=list)
    required: bool = True
    source_evidence: str = ""
    origin_iteration: int = 1
    evidence_focus: str = "semantics"

    def signature(self) -> str:
        normalized = " ".join(self.query_text.lower().split())
        entity_types = ",".join(sorted(set(self.target_entity_types)))
        facets = json.dumps(self.facets, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"{normalized}|{entity_types}|{facets}"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], index: int = 1) -> "RetrievalQuerySpec":
        role = str(value.get("role") or "task_intent").strip().lower()
        if role not in QUERY_ROLES:
            role = "task_intent"
        target_types = value.get("target_entity_types") or [value.get("entity_type") or "dataset"]
        if not isinstance(target_types, list):
            target_types = [target_types]
        target_types = [str(item).strip().lower() for item in target_types if str(item).strip().lower() in ENTITY_TYPES]
        return cls(
            query_id=str(value.get("query_id") or f"q{index}").strip(),
            role=role,
            query_text=str(value.get("query_text") or value.get("query") or "").strip(),
            target_entity_types=target_types or ["dataset"],
            weight=1.0,
            facets=dict(value.get("facets") or {}) if isinstance(value.get("facets"), Mapping) else {},
            intent_codes=[str(item) for item in value.get("intent_codes", []) if str(item).strip()],
            required=bool(value.get("required", True)),
            source_evidence=str(value.get("source_evidence") or "").strip(),
            origin_iteration=max(1, int(value.get("origin_iteration", 1) or 1)),
            evidence_focus=(
                str(value.get("evidence_focus") or "semantics").strip().lower()
                if str(value.get("evidence_focus") or "semantics").strip().lower() in EVIDENCE_FOCI
                else "semantics"
            ),
        )


@dataclass
class RelationSearchSpec:
    relation_id: str
    anchor_query_id: str
    anchor_entity_type: str
    path: List[Dict[str, str]]
    weight: float = 1.0
    top_k_anchors: int = 5
    max_neighbors_per_anchor: int = 30
    min_anchor_score: float = 0.0
    path_id: str = ""
    relation_intent: str = ""
    max_intermediate_per_anchor: int = 10
    max_datasets_per_intermediate: int = 20

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], index: int = 1) -> "RelationSearchSpec":
        return cls(
            relation_id=str(value.get("relation_id") or f"rel{index}"),
            anchor_query_id=str(value.get("anchor_query_id") or ""),
            anchor_entity_type=str(value.get("anchor_entity_type") or "paper").lower(),
            path=[dict(item) for item in value.get("path", []) if isinstance(item, Mapping)],
            weight=1.0,
            top_k_anchors=min(10, max(1, int(value.get("top_k_anchors", 5) or 5))),
            max_neighbors_per_anchor=min(50, max(1, int(value.get("max_neighbors_per_anchor", 30) or 30))),
            min_anchor_score=max(0.0, float(value.get("min_anchor_score", 0.0) or 0.0)),
            path_id=str(value.get("path_id") or "").strip(),
            relation_intent=str(value.get("relation_intent") or "").strip(),
            max_intermediate_per_anchor=min(20, max(1, int(value.get("max_intermediate_per_anchor", 10) or 10))),
            max_datasets_per_intermediate=min(50, max(1, int(value.get("max_datasets_per_intermediate", 20) or 20))),
        )


@dataclass
class ConstraintSpec:
    field: str
    operator: str
    value: Any
    source_text: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ConstraintSpec":
        return cls(str(value.get("field") or "").strip(), str(value.get("operator") or "equals").strip(), value.get("value"), str(value.get("source_text") or "").strip())


@dataclass
class PreferenceSpec:
    field: str
    objective: str
    value: Any = None
    weight: float = 1.0
    source_text: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PreferenceSpec":
        return cls(str(value.get("field") or "").strip(), str(value.get("objective") or "maximize").strip(), value.get("value"), max(0.0, float(value.get("weight", 1.0) or 1.0)), str(value.get("source_text") or "").strip())


@dataclass
class RetrievalPolicy:
    per_query_top_k: int = 40
    relation_anchor_top_k: int = 5
    relation_neighbors_per_anchor: int = 30
    fused_pool_size: int = 300
    rrf_k: int = 60
    relation_path_decay: float = 0.7

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RetrievalPolicy":
        value = value or {}
        return cls(
            min(60, max(10, int(value.get("per_query_top_k", 40) or 40))),
            min(10, max(1, int(value.get("relation_anchor_top_k", 5) or 5))),
            min(50, max(1, int(value.get("relation_neighbors_per_anchor", 30) or 30))),
            max(20, int(value.get("fused_pool_size", 300) or 300)),
            max(1, int(value.get("rrf_k", 60) or 60)),
            min(1.0, max(0.0, float(value.get("relation_path_decay", 0.7) or 0.7))),
        )


@dataclass
class SearchPlan:
    user_query: str = ""
    retrieval_queries: List[RetrievalQuerySpec] = field(default_factory=list)
    relation_search_specs: List[RelationSearchSpec] = field(default_factory=list)
    hard_constraints: List[ConstraintSpec] = field(default_factory=list)
    soft_preferences: List[PreferenceSpec] = field(default_factory=list)
    retrieval_policy: RetrievalPolicy = field(default_factory=RetrievalPolicy)
    intent_codes: List[str] = field(default_factory=list)
    top_k_final: int = 20
    quality_target: Dict[str, Any] = field(default_factory=dict)
    plan_outline: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    origin_iteration: int = 1
    shortlist_allocations: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.retrieval_queries = [item if isinstance(item, RetrievalQuerySpec) else RetrievalQuerySpec.from_mapping(item, i + 1) for i, item in enumerate(self.retrieval_queries) if isinstance(item, (RetrievalQuerySpec, Mapping))]
        self.relation_search_specs = [item if isinstance(item, RelationSearchSpec) else RelationSearchSpec.from_mapping(item, i + 1) for i, item in enumerate(self.relation_search_specs) if isinstance(item, (RelationSearchSpec, Mapping))]
        self.hard_constraints = [item if isinstance(item, ConstraintSpec) else ConstraintSpec.from_mapping(item) for item in self.hard_constraints if isinstance(item, (ConstraintSpec, Mapping))]
        self.soft_preferences = [item if isinstance(item, PreferenceSpec) else PreferenceSpec.from_mapping(item) for item in self.soft_preferences if isinstance(item, (PreferenceSpec, Mapping))]
        if isinstance(self.retrieval_policy, Mapping):
            self.retrieval_policy = RetrievalPolicy.from_mapping(self.retrieval_policy)
        normalized_allocations: Dict[str, int] = {}
        for branch_id, slots in dict(self.shortlist_allocations or {}).items():
            try:
                normalized_slots = max(0, int(slots))
            except (TypeError, ValueError):
                continue
            if str(branch_id).strip() and normalized_slots > 0:
                normalized_allocations[str(branch_id)] = normalized_slots
        self.shortlist_allocations = normalized_allocations


@dataclass
class ActionStep:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionProtocol:
    steps: List[ActionStep]
    fused_pool_size: int


@dataclass
class ExecutionTrace:
    step: str
    status: str
    detail: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class RetrievalProvenance:
    list_id: str
    query_id: str
    branch_type: str
    entity_type: str
    rank: int
    raw_score: float
    rrf_contribution: float = 0.0
    anchor_id: str = ""
    relation_path: List[str] = field(default_factory=list)
    anchor_confidence: float = 0.0


@dataclass
class Candidate:
    dataset_id: str
    dataset: Dict[str, Any]
    retrieval_prior_score: float = 0.0
    filter_score: float = 0.0
    graph_score: float = 0.0
    hybrid_score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    desc_dense_score: float = 0.0
    metadata_dense_score: float = 0.0
    retrieval_rank: int = 0
    graph_support: int = 0
    graph_relation_types: List[str] = field(default_factory=list)
    fusion_score: float = 0.0
    soft_preference_score: float = 0.0
    pre_rerank_score: float = 0.0
    final_score: float = 0.0
    evidence: List[str] = field(default_factory=list)
    retrieval_provenance: List[RetrievalProvenance] = field(default_factory=list)
    matched_query_ids: List[str] = field(default_factory=list)
    anchor_paths: List[Dict[str, Any]] = field(default_factory=list)
    intent_matches: Dict[str, float] = field(default_factory=dict)
    constraint_violations: List[str] = field(default_factory=list)
    constraint_unknowns: List[str] = field(default_factory=list)
    confidence: str = "low"
    evidence_codes: List[str] = field(default_factory=list)
    training_applicability: str = "unknown"
    supervision_issues: List[str] = field(default_factory=list)
    family_key: str = ""


@dataclass
class Diagnostic:
    code: str
    severity: str = "warning"
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggested_query_role: str = "feedback_repair"


@dataclass
class CriticFeedback:
    accept: bool
    quality_score: float
    quantity_score: float
    overall_score: float
    reason: str
    candidate_scores: List[Dict[str, Any]] = field(default_factory=list)
    intent_coverage_score: float = 0.0
    quantity_gate: bool = False
    diagnostics: List[Diagnostic] = field(default_factory=list)
    unresolved_anchors: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.diagnostics = [item if isinstance(item, Diagnostic) else Diagnostic(**item) for item in self.diagnostics if isinstance(item, (Diagnostic, Mapping))]


@dataclass
class IterationResult:
    iteration: int
    plan: SearchPlan
    steps: List[ActionStep]
    candidates: List[Candidate]
    trace: List[ExecutionTrace]
    critic_feedback: CriticFeedback
    stage_timings: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchState:
    query_context: QueryContext | None = None
    executed_query_signatures: List[str] = field(default_factory=list)
    branch_lists: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    seen_candidate_ids: List[str] = field(default_factory=list)
    diagnostic_history: List[List[str]] = field(default_factory=list)
    iteration_metrics: List[Dict[str, Any]] = field(default_factory=list)
    best_iteration: int = 0
    best_utility: float = -1.0
    best_selection_key: List[float] = field(default_factory=list)
    no_improvement_rounds: int = 0


@dataclass
class SessionResult:
    query: QueryContext
    iterations: List[IterationResult]
    final_candidates: List[Candidate]
    selected_iteration: int = 0
    debug_artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionBundle:
    trace: List[ExecutionTrace]
    branch_lists: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    unresolved_anchors: List[Dict[str, Any]] = field(default_factory=list)
