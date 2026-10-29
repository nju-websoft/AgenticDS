from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from global_values import (
    AGENT_CONSTRAINT_VIOLATION_POLICY,
    AGENT_MAX_ITERATIONS,
    AGENT_ROUND_SELECTION_POLICY,
    AGENT_ROUND_UTILITY_QUALITY_WEIGHT,
    DEBUG_FULL_TRACE,
    EXPERIMENT_DISABLE_CRITIC,
    EXPERIMENT_FORCE_SINGLE_ITERATION,
)
from llm import LLMGateway

from .critic import CriticAgent
from .engineer import EngineerAgent
from .executor import ExecutorAgent
from .planner import PlannerAgent
from .schemas import CriticFeedback, IterationResult, QueryContext, SearchState, SessionResult


class MultiAgentDatasetRetriever:
    """Run a diagnostic-driven retrieval process with explicit cross-round evidence state."""

    def __init__(
        self,
        max_iterations: int = AGENT_MAX_ITERATIONS,
        debug_full_trace: bool = DEBUG_FULL_TRACE,
        embedding_model: str | None = None,
        agent_model: str | None = None,
        retrieval_service_socket: str | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.debug_full_trace = bool(debug_full_trace)
        self.embedding_model = embedding_model
        self.agent_model = str(agent_model or "").strip() or None
        self.retrieval_service_socket = str(retrieval_service_socket or "").strip() or None
        self.llm_gateway = LLMGateway(model_name=self.agent_model) if self.agent_model else LLMGateway()
        self.llm_gateway.set_debug(self.debug_full_trace)
        self.planner = PlannerAgent(self.llm_gateway)
        self.engineer = EngineerAgent(self.llm_gateway)
        self.executor = ExecutorAgent(
            debug_full_trace=self.debug_full_trace,
            embedding_model=self.embedding_model,
            retrieval_service_socket=self.retrieval_service_socket,
        )
        self.retrieval_manifest = dict(getattr(self.executor, "retrieval_manifest", {}) or {})
        self.critic = CriticAgent(self.llm_gateway)

    def run(self, query: str, top_k: int = 20, locale: str = "zh-CN") -> SessionResult:
        if self.debug_full_trace:
            self.llm_gateway.reset_session_logs()
        context = QueryContext(query, top_k, locale)
        state = SearchState(query_context=context)
        planner_start = perf_counter()
        current_plan = self.planner.plan(context)
        planner_initial_seconds = perf_counter() - planner_start
        iterations: list[IterationResult] = []
        effective_max = 1 if EXPERIMENT_FORCE_SINGLE_ITERATION else self.max_iterations

        for iteration in range(1, effective_max + 1):
            iteration_start = perf_counter()
            signatures_before = set(state.executed_query_signatures)
            current_signatures = {item.query_id: item.signature() for item in current_plan.retrieval_queries}
            if iteration > 1 and set(current_signatures.values()).issubset(signatures_before):
                break

            engineer_start = perf_counter()
            protocol = self.engineer.compile_plan(current_plan)
            steps = protocol.steps
            engineer_seconds = perf_counter() - engineer_start

            executor_start = perf_counter()
            bundle = self.executor.execute(current_plan, steps)
            state.branch_lists.update(bundle.branch_lists)
            state.executed_query_signatures = list(dict.fromkeys(state.executed_query_signatures + list(current_signatures.values())))
            round_candidate_ids = {
                str(row.get("item_id") or "")
                for rows in bundle.branch_lists.values()
                for row in rows
                if row.get("item_id")
            }
            previous_candidate_ids = set(state.seen_candidate_ids)
            new_candidate_count = len(round_candidate_ids - previous_candidate_ids)
            state.seen_candidate_ids = sorted(previous_candidate_ids | round_candidate_ids)
            candidates = self.executor.rank_from_branch_lists(
                current_plan, state.branch_lists,
                pool_size=protocol.fused_pool_size, trace=bundle.trace,
            )
            executor_seconds = perf_counter() - executor_start

            critic_start = perf_counter()
            feedback = self._accept_without_critic(current_plan, candidates) if EXPERIMENT_DISABLE_CRITIC else self.critic.review(current_plan, candidates, iteration, unresolved_anchors=bundle.unresolved_anchors)
            critic_seconds = perf_counter() - critic_start

            diagnostic_codes = [item.code for item in feedback.diagnostics]
            selection_key = self._iteration_selection_key(feedback, candidates, top_k, new_candidate_count)
            utility = selection_key[0] if AGENT_ROUND_SELECTION_POLICY == "utility" else feedback.overall_score
            improved = not state.best_selection_key or selection_key > tuple(state.best_selection_key)
            if improved or state.best_iteration == 0:
                state.best_iteration = iteration
                state.best_utility = utility
                state.best_selection_key = list(selection_key)
                state.no_improvement_rounds = 0
            else:
                state.no_improvement_rounds += 1
            metrics = {
                "candidate_count": len(candidates),
                "quality_score": feedback.quality_score,
                "intent_coverage_score": feedback.intent_coverage_score,
                "overall_score": feedback.overall_score,
                "quantity_gate": feedback.quantity_gate,
                "new_query_count": len(set(current_signatures.values()) - signatures_before),
                "new_candidate_count": new_candidate_count,
                "branch_list_count": len(state.branch_lists),
                "fused_pool_budget": protocol.fused_pool_size,
                "selected_as_best": state.best_iteration == iteration,
                "selection_key": list(selection_key),
                "round_selection_policy": AGENT_ROUND_SELECTION_POLICY,
                "selection_utility": utility,
            }
            state.iteration_metrics.append(metrics)
            state.diagnostic_history.append(diagnostic_codes)

            replan_seconds = 0.0
            next_plan = current_plan
            should_stop = self._should_stop(state, feedback, iteration, effective_max)
            if not should_stop:
                replan_start = perf_counter()
                next_plan = self.planner.refine_plan(
                    current_plan,
                    feedback,
                    executed_query_signatures=list(state.executed_query_signatures),
                    coverage_statistics=metrics,
                )
                replan_seconds = perf_counter() - replan_start

            iterations.append(
                IterationResult(
                    iteration,
                    current_plan,
                    steps,
                    candidates,
                    bundle.trace,
                    feedback,
                    {
                        "planner_initial_seconds": planner_initial_seconds if iteration == 1 else 0.0,
                        "engineer_seconds": engineer_seconds,
                        "executor_seconds": executor_seconds,
                        "critic_seconds": critic_seconds,
                        "replan_seconds": replan_seconds,
                        "iteration_total_seconds": perf_counter() - iteration_start,
                    },
                    metrics,
                )
            )
            if should_stop:
                break
            current_plan = next_plan

        selected = state.best_iteration if state.best_iteration else (iterations[-1].iteration if iterations else 0)
        selected_result = next((item for item in iterations if item.iteration == selected), None)
        final_candidates = selected_result.candidates[:top_k] if selected_result else []
        debug_artifacts: dict[str, Any] = {
            "search_state": asdict(state),
            "search_policy": {
                "constraint_violation_policy": AGENT_CONSTRAINT_VIOLATION_POLICY,
                "round_selection_policy": AGENT_ROUND_SELECTION_POLICY,
                "round_utility_quality_weight": AGENT_ROUND_UTILITY_QUALITY_WEIGHT,
            },
        }
        if self.debug_full_trace:
            debug_artifacts.update({"debug_full_trace_enabled": True, "llm_conversations": self.llm_gateway.get_session_logs()})
        return SessionResult(context, iterations, final_candidates, selected_iteration=selected, debug_artifacts=debug_artifacts)

    def _iteration_selection_key(
        self,
        feedback: CriticFeedback,
        candidates: list[Any],
        requested_top_k: int,
        new_candidate_count: int,
    ) -> tuple[float, ...]:
        """Use either the scalar utility or the original feasibility-first ordering."""
        if AGENT_ROUND_SELECTION_POLICY == "utility":
            return (
                AGENT_ROUND_UTILITY_QUALITY_WEIGHT * feedback.quality_score
                + (1.0 - AGENT_ROUND_UTILITY_QUALITY_WEIGHT) * feedback.intent_coverage_score,
            )
        target = max(1, int(requested_top_k))
        critical_count = sum(1 for item in feedback.diagnostics if item.severity == "critical")
        return (
            float(len(candidates) >= target),
            float(feedback.accept),
            float(-critical_count),
            float(feedback.intent_coverage_score),
            float(feedback.quality_score),
            float(feedback.overall_score),
            min(1.0, len(candidates) / target),
            min(1.0, max(0, new_candidate_count) / target),
        )

    def _should_stop(self, state: SearchState, feedback: CriticFeedback, iteration: int, max_iterations: int) -> bool:
        if feedback.accept or iteration >= max_iterations:
            return True
        if state.no_improvement_rounds >= 2:
            return True
        if AGENT_ROUND_SELECTION_POLICY == "utility":
            return False
        if len(state.diagnostic_history) >= 2:
            previous, current = state.diagnostic_history[-2:]
            critical = {item.code for item in feedback.diagnostics if item.severity == "critical"}
            if critical and critical.issubset(set(previous)) and critical.issubset(set(current)):
                return True
        return False

    def _accept_without_critic(self, plan, candidates) -> CriticFeedback:
        top_n = candidates[:5]
        quality = sum(item.final_score for item in top_n) / len(top_n) if top_n else 0.0
        quantity_gate = len(candidates) >= int(plan.quality_target.get("min_candidates", 1))
        return CriticFeedback(True, quality, 1.0 if quantity_gate else 0.0, quality, "critic_disabled", intent_coverage_score=1.0, quantity_gate=quantity_gate)

    def run_as_dict(self, query: str, top_k: int = 20, locale: str = "zh-CN") -> dict[str, Any]:
        return self.run(query, top_k, locale).to_dict()

    def save_result(self, query: str, output_file: str, top_k: int = 20, locale: str = "zh-CN") -> None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(self.run_as_dict(query, top_k, locale), fh, ensure_ascii=False, indent=2)
