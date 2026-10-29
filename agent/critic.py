from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import re
from threading import Lock
from typing import Any, Dict, List, Mapping, Optional

from global_values import (
    AGENT_ROUND_UTILITY_QUALITY_WEIGHT,
    AGENT_QUALITY_THRESHOLD,
    CRITIC_BATCH_SIZE,
    CRITIC_LLM_TIMEOUT_SECONDS,
    CRITIC_MAX_WORKERS,
    CRITIC_RERANK_WEIGHT_LLM,
    CRITIC_RERANK_WEIGHT_PRERERANK,
    CRITIC_SAMPLE_MAX_TEXT_CHARS,
    CRITIC_SERIAL_LLM_SCORING,
    EXPERIMENT_DISABLE_CRITIC_GRAPH_EVIDENCE,
    EXPERIMENT_DISABLE_CRITIC_SAMPLE_EVIDENCE,
)
from llm import LLMGateway
from utils.hf_dataset_evidence import build_candidate_evidence
from utils.hf_dataset_sample_store import HFDatasetSampleStore

from .schemas import Candidate, CriticFeedback, Diagnostic, SearchPlan, normalize_metadata_token, normalize_metadata_values

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompt" / "critic_prompt.md"

CRITIC_CONFIDENCE_LEVELS = {"low", "medium", "high"}
CRITIC_EVIDENCE_CODES = {
    "name_match",
    "task_match",
    "domain_match",
    "format_match",
    "training_data",
    "training_split_support",
    "supervision_pair_support",
    "reference_answer_support",
    "anchor_support",
    "hard_constraint_violation",
    "constraint_match",
    "sample_support",
    "metadata_support",
    "graph_support",
    "retrieval_support",
    "weak_evidence",
    "contradictory_evidence",
}


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = float(default)
    return max(0.0, min(1.0, num))


class CriticAgent:
    """Evaluate candidates and decide whether to iterate."""

    BATCH_SIZE = max(1, int(CRITIC_BATCH_SIZE))
    MAX_WORKERS = max(1, int(CRITIC_MAX_WORKERS))
    LLM_TIMEOUT_SECONDS = max(1, int(CRITIC_LLM_TIMEOUT_SECONDS))
    SINGLETON_RETRY_TIMEOUT_SECONDS = max(20, LLM_TIMEOUT_SECONDS)
    SINGLETON_FINAL_RETRY_TIMEOUT_SECONDS = max(45, LLM_TIMEOUT_SECONDS)
    SINGLETON_MINIMAL_RETRY_TIMEOUT_SECONDS = max(60, LLM_TIMEOUT_SECONDS)
    SERIAL_LLM_SCORING = bool(CRITIC_SERIAL_LLM_SCORING)
    SAMPLE_MAX_TEXT_CHARS = max(0, int(CRITIC_SAMPLE_MAX_TEXT_CHARS))
    SAMPLE_MINIMAL_MAX_TEXT_CHARS = max(80, SAMPLE_MAX_TEXT_CHARS // 2) if SAMPLE_MAX_TEXT_CHARS > 0 else 160
    RERANK_WEIGHT_LLM = _clamp01(CRITIC_RERANK_WEIGHT_LLM, default=0.65)
    RERANK_WEIGHT_PRERERANK = _clamp01(CRITIC_RERANK_WEIGHT_PRERERANK, default=0.35)

    def __init__(self, llm_gateway: Optional[LLMGateway] = None) -> None:
        self.llm = llm_gateway
        self.system_prompt = self._load_prompt(PROMPT_PATH)
        self.scoring_prompt = self._build_scoring_prompt()
        self.sample_store = HFDatasetSampleStore.get_default()
        self._stats_lock = Lock()
        self._run_stats: Dict[str, int] = {}

    def reset_run_stats(self) -> None:
        self._run_stats = {
            "initial_batch_calls": 0,
            "initial_batch_fallbacks": 0,
            "initial_singleton_calls": 0,
            "initial_singleton_fallbacks": 0,
            "split_events": 0,
            "singleton_retry_calls": 0,
            "singleton_retry_successes": 0,
            "singleton_slow_retry_calls": 0,
            "singleton_slow_retry_successes": 0,
            "singleton_minimal_retry_calls": 0,
            "singleton_minimal_retry_successes": 0,
            "invalid_semantic_payloads": 0,
            "heuristic_fallbacks": 0,
        }

    def get_run_stats(self) -> Dict[str, int]:
        return dict(self._run_stats)

    def _bump_stat(self, key: str, amount: int = 1) -> None:
        with self._stats_lock:
            self._run_stats[key] = int(self._run_stats.get(key, 0)) + int(amount)

    def review(
        self,
        plan: SearchPlan,
        candidates: List[Candidate],
        iteration: int,
        unresolved_anchors: Optional[List[Dict[str, Any]]] = None,
    ) -> CriticFeedback:
        self.reset_run_stats()
        min_candidates = int(plan.quality_target.get("min_candidates", 5))
        min_quality = float(plan.quality_target.get("min_quality", AGENT_QUALITY_THRESHOLD))
        llm_scores, llm_annotations = self._score_all_candidates_with_llm(plan, candidates, iteration)
        candidate_scores: List[Dict[str, Any]] = []
        for candidate in candidates:
            pre_rerank_score = self._pre_critic_prior(candidate)
            llm_score = _clamp01(llm_scores.get(candidate.dataset_id), default=0.0)
            blended_score = _clamp01(
                self.RERANK_WEIGHT_LLM * llm_score
                + self.RERANK_WEIGHT_PRERERANK * pre_rerank_score,
                default=llm_score,
            )
            final_score = blended_score

            candidate.final_score = final_score
            annotations = llm_annotations.get(candidate.dataset_id, {})
            candidate.confidence = str(annotations.get("confidence") or self._confidence_from_score(llm_score))
            candidate.evidence_codes = list(annotations.get("evidence_codes") or ["weak_evidence"])
            candidate.intent_matches = {code: self._intent_support(code, candidate.evidence_codes) for code in plan.intent_codes}
            joint_intent_coverage = self._candidate_intent_coverage(plan, candidate)
            candidate.evidence.append(
                f"critic_rerank:llm={llm_score:.2f},pre={pre_rerank_score:.2f},final={final_score:.2f},training={candidate.training_applicability}"
            )
            candidate_scores.append(
                {
                    "dataset_id": candidate.dataset_id,
                    "llm_score": llm_score,
                    "pre_rerank_score": pre_rerank_score,
                    "retrieval_prior_score": _clamp01(candidate.retrieval_prior_score, default=0.0),
                    "filter_score": _clamp01(candidate.filter_score, default=0.0),
                    "graph_score": 0.0 if EXPERIMENT_DISABLE_CRITIC_GRAPH_EVIDENCE else _clamp01(candidate.graph_score, default=0.0),
                    "graph_support": 0 if EXPERIMENT_DISABLE_CRITIC_GRAPH_EVIDENCE else int(candidate.graph_support),
                    "graph_relation_types": [] if EXPERIMENT_DISABLE_CRITIC_GRAPH_EVIDENCE else list(candidate.graph_relation_types),
                    "hybrid_score": float(candidate.hybrid_score),
                    "retrieval_rank": int(candidate.retrieval_rank),
                    "dense_score": float(candidate.dense_score),
                    "sparse_score": float(candidate.sparse_score),
                    "final_score": final_score,
                    "blended_score": blended_score,
                    "training_applicability": candidate.training_applicability,
                    "supervision_issues": list(candidate.supervision_issues),
                    "confidence": candidate.confidence,
                    "evidence_codes": candidate.evidence_codes,
                    "intent_matches": dict(candidate.intent_matches),
                    "joint_intent_coverage": joint_intent_coverage,
                }
            )

        candidates.sort(key=lambda c: c.final_score, reverse=True)

        top_n = candidates[:5]
        if top_n:
            quality_score = sum(item.final_score for item in top_n) / len(top_n)
        else:
            quality_score = 0.0

        quantity_score = min(1.0, len(candidates) / max(1, min_candidates))
        quantity_gate = len(candidates) >= min_candidates
        intent_coverage_score = self._intent_coverage(plan, candidates[:10])
        min_coverage = float(plan.quality_target.get("min_intent_coverage", 0.7))
        diagnostics = self._deterministic_diagnostics(
            plan,
            candidates,
            quantity_gate=quantity_gate,
            quality_score=quality_score,
            min_quality=min_quality,
            intent_coverage_score=intent_coverage_score,
            min_coverage=min_coverage,
        )
        anchor_code = {
            "paper": "paper_anchor_unresolved",
            "model": "model_anchor_unresolved",
            "collection": "collection_anchor_unresolved",
        }
        for row in unresolved_anchors or []:
            code = anchor_code.get(str(row.get("entity_type") or ""))
            if code:
                diagnostics.append(Diagnostic(code, "warning", dict(row)))
        critical = any(item.severity == "critical" for item in diagnostics)
        overall_score = (
            AGENT_ROUND_UTILITY_QUALITY_WEIGHT * quality_score
            + (1.0 - AGENT_ROUND_UTILITY_QUALITY_WEIGHT) * intent_coverage_score
        )
        accept = quantity_gate and quality_score >= min_quality and intent_coverage_score >= min_coverage and not critical

        if not accept and self.llm is not None:
            llm_meta = self._llm_diagnose_failure(
                plan,
                candidates,
                iteration,
                min_quality,
                accept,
                quality_score,
                quantity_score,
                overall_score,
                deterministic_diagnostics=diagnostics,
            )
            diagnostics = self._merge_diagnostics(diagnostics, llm_meta.get("diagnostics"))
            reason = str(llm_meta.get("reason") or "quality_or_coverage_gate_failed")
        else:
            reason = "accepted" if accept else "quality_or_coverage_gate_failed"

        return CriticFeedback(
            accept=accept,
            quality_score=quality_score,
            quantity_score=quantity_score,
            overall_score=overall_score,
            reason=reason,
            candidate_scores=candidate_scores,
            intent_coverage_score=intent_coverage_score,
            quantity_gate=quantity_gate,
            diagnostics=diagnostics,
            unresolved_anchors=list(unresolved_anchors or []),
        )

    def _intent_coverage(self, plan: SearchPlan, candidates: List[Candidate]) -> float:
        if not candidates:
            return 0.0
        return max(self._candidate_intent_coverage(plan, candidate) for candidate in candidates)

    def _candidate_intent_coverage(self, plan: SearchPlan, candidate: Candidate) -> float:
        intents = list(dict.fromkeys(plan.intent_codes or ["task_match", "domain_match"]))
        if not intents:
            return 1.0
        covered = sum(1 for code in intents if candidate.intent_matches.get(code, 0.0) >= 0.5)
        return covered / len(intents)

    def _intent_support(self, intent: str, evidence_codes: List[str]) -> float:
        evidence = set(evidence_codes)
        mappings = {
            "training_data": {"training_data", "training_split_support", "supervision_pair_support", "reference_answer_support"},
            "format_match": {"format_match", "supervision_pair_support", "reference_answer_support"},
            "name_match": {"name_match"},
            "task_match": {"task_match"},
            "domain_match": {"domain_match"},
        }
        return 1.0 if evidence & mappings.get(intent, {intent}) else 0.0

    def _deterministic_diagnostics(
        self,
        plan: SearchPlan,
        candidates: List[Candidate],
        *,
        quantity_gate: bool,
        quality_score: float,
        min_quality: float,
        intent_coverage_score: float,
        min_coverage: float,
    ) -> List[Diagnostic]:
        diagnostics: List[Diagnostic] = []
        if not quantity_gate:
            diagnostics.append(Diagnostic("insufficient_candidate_count", "critical", {"count": len(candidates)}))
        top = candidates[:10]
        violating = [item.dataset_id for item in top if item.constraint_violations]
        if violating:
            diagnostics.append(Diagnostic("hard_constraint_violation", "critical", {"dataset_ids": violating[:5]}))
        if top and all(item.confidence == "low" for item in top):
            diagnostics.append(Diagnostic("low_evidence_confidence", "critical", {"low_confidence_top10": len(top)}))
        code_to_diagnostic = {
            "name_match": "missing_exact_name_match",
            "task_match": "missing_task_match",
            "format_match": "missing_format_match",
            "training_data": "insufficient_training_data",
        }
        best_joint = max(
            top,
            key=lambda item: (self._candidate_intent_coverage(plan, item), item.final_score),
            default=None,
        )
        for intent in plan.intent_codes:
            if intent in code_to_diagnostic and (
                best_joint is None or best_joint.intent_matches.get(intent, 0.0) < 0.5
            ):
                diagnostics.append(
                    Diagnostic(
                        code_to_diagnostic[intent],
                        "critical",
                        {
                            "intent": intent,
                            "best_joint_candidate": best_joint.dataset_id if best_joint else "",
                            "best_joint_coverage": self._candidate_intent_coverage(plan, best_joint) if best_joint else 0.0,
                        },
                    )
                )
        if quality_score < min_quality and not any(item.code == "low_evidence_confidence" for item in diagnostics):
            diagnostics.append(Diagnostic("low_evidence_confidence", "critical", {"quality": quality_score, "threshold": min_quality}))
        if intent_coverage_score < min_coverage and not any(item.code.startswith("missing_") for item in diagnostics):
            diagnostics.append(Diagnostic("missing_task_match", "critical", {"coverage": intent_coverage_score, "threshold": min_coverage}))
        families = [self._candidate_family(item.dataset_id) for item in top]
        if len(families) >= 5 and len(set(families)) <= max(1, len(families) // 4):
            diagnostics.append(Diagnostic("candidate_family_collapse", "warning", {"families": len(set(families)), "candidates": len(families)}))
        return diagnostics

    def _merge_diagnostics(self, base: List[Diagnostic], payload: Any) -> List[Diagnostic]:
        allowed = {
            "missing_exact_name_match", "missing_task_match", "missing_format_match",
            "hard_constraint_violation", "insufficient_training_data", "paper_anchor_unresolved",
            "model_anchor_unresolved", "collection_anchor_unresolved", "low_evidence_confidence",
            "candidate_family_collapse", "insufficient_candidate_count",
        }
        merged = {item.code: item for item in base}
        unresolved_anchor_codes = {
            "paper_anchor_unresolved",
            "model_anchor_unresolved",
            "collection_anchor_unresolved",
        }
        for row in payload if isinstance(payload, list) else []:
            if not isinstance(row, Mapping):
                continue
            code = str(row.get("code") or "")
            if code not in allowed:
                continue
            proposed_severity = str(row.get("severity") or "warning")
            if code in unresolved_anchor_codes:
                proposed_severity = "warning"
            if code in merged and merged[code].severity == "critical":
                proposed_severity = "critical"
            merged[code] = Diagnostic(code, proposed_severity, {**(merged[code].evidence if code in merged else {}), **dict(row.get("evidence") or {})})
        return list(merged.values())

    def _candidate_family(self, dataset_id: str) -> str:
        return dataset_id.split("/", 1)[0].lower() if "/" in dataset_id else dataset_id.lower()

    def _extract_llm_scores(self, payload: Mapping[str, Any]) -> Dict[str, float]:
        rows = payload.get("candidate_scores")
        if not isinstance(rows, list):
            return {}

        scores: Dict[str, float] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            dataset_id = str(row.get("dataset_id") or "").strip()
            if not dataset_id:
                continue
            if "llm_score" not in row:
                continue
            scores[dataset_id] = _clamp01(row.get("llm_score"), default=0.0)
        return scores

    def _extract_llm_annotations(self, payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        rows = payload.get("candidate_scores")
        if not isinstance(rows, list):
            return {}

        annotations: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            dataset_id = str(row.get("dataset_id") or "").strip()
            if not dataset_id:
                continue
            score = _clamp01(row.get("llm_score"), default=0.0)
            confidence = str(row.get("confidence") or "").strip().lower()
            if confidence not in CRITIC_CONFIDENCE_LEVELS:
                confidence = self._confidence_from_score(score)
            raw_codes = row.get("evidence_codes")
            if not isinstance(raw_codes, list):
                raw_codes = []
            evidence_codes: List[str] = []
            for value in raw_codes:
                code = str(value or "").strip().lower()
                if code in CRITIC_EVIDENCE_CODES and code not in evidence_codes:
                    evidence_codes.append(code)
            annotations[dataset_id] = {
                "confidence": confidence,
                "evidence_codes": evidence_codes[:4] or ["weak_evidence"],
            }
        return annotations

    def _confidence_from_score(self, score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"

    def _score_all_candidates_with_llm(
        self,
        plan: SearchPlan,
        candidates: List[Candidate],
        iteration: int,
    ) -> tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
        if not candidates:
            return {}, {}

        query_text = self._extract_query_text(plan)
        if EXPERIMENT_DISABLE_CRITIC_SAMPLE_EVIDENCE:
            sample_payload_map = {}
        else:
            sample_payload_map = self.sample_store.get_samples_bulk(
                [candidate.dataset_id for candidate in candidates],
                max_workers=min(self.MAX_WORKERS, max(1, len(candidates))),
                sampling_hints={
                    candidate.dataset_id: self._sample_hint(candidate)
                    for candidate in candidates
                },
            )
        for candidate in candidates:
            evidence = build_candidate_evidence(
                dataset=candidate.dataset or {},
                sample=sample_payload_map.get(candidate.dataset_id, {}),
                query=query_text,
                intent_codes=plan.intent_codes,
                text_budget=self.SAMPLE_MAX_TEXT_CHARS,
            )
            candidate.training_applicability = evidence["training_applicability"]["status"]
            candidate.supervision_issues = evidence["consistency"]["issues"]

        batches = [
            candidates[start : start + self.BATCH_SIZE]
            for start in range(0, len(candidates), self.BATCH_SIZE)
        ]

        all_scores: Dict[str, float] = {}
        all_annotations: Dict[str, Dict[str, Any]] = {}

        worker_count = min(self.MAX_WORKERS, len(batches))
        if self.SERIAL_LLM_SCORING and self.llm is not None:
            worker_count = 1
        if worker_count <= 1:
            for batch in batches:
                payload = self._score_batch_resilient(
                    plan,
                    batch,
                    iteration,
                    sample_payload_map=sample_payload_map,
                )
                scores = self._extract_llm_scores(payload)
                annotations = self._extract_llm_annotations(payload)
                all_scores.update(scores)
                all_annotations.update(annotations)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(
                        self._score_batch_resilient,
                        plan,
                        batch,
                        iteration,
                        sample_payload_map,
                    ): batch
                    for batch in batches
                }
                for future in as_completed(future_map):
                    payload = future.result()
                    scores = self._extract_llm_scores(payload)
                    annotations = self._extract_llm_annotations(payload)
                    all_scores.update(scores)
                    all_annotations.update(annotations)

        # Ensure every dataset has gone through LLM scoring at least once.
        missing = [candidate for candidate in candidates if candidate.dataset_id not in all_scores]
        if missing:
            payload = self._score_batch_resilient(
                plan,
                missing,
                iteration,
                sample_payload_map=sample_payload_map,
            )
            all_scores.update(self._extract_llm_scores(payload))
            all_annotations.update(self._extract_llm_annotations(payload))

        for candidate in candidates:
            if candidate.dataset_id not in all_scores:
                fallback = self._heuristic_score_batch(
                    plan,
                    [candidate],
                    iteration,
                    sample_payload_map=sample_payload_map,
                )
                rows = fallback.get("candidate_scores") if isinstance(fallback, Mapping) else None
                if isinstance(rows, list) and rows and isinstance(rows[0], Mapping):
                    score = _clamp01(rows[0].get("llm_score"), default=0.0)
                    all_scores[candidate.dataset_id] = score
                    all_annotations.update(self._extract_llm_annotations(fallback))
                else:
                    all_scores[candidate.dataset_id] = 0.0
                    all_annotations[candidate.dataset_id] = {
                        "confidence": "low",
                        "evidence_codes": ["weak_evidence"],
                    }

        return all_scores, all_annotations

    @staticmethod
    def _sample_hint(candidate: Candidate) -> Dict[str, str]:
        dataset = candidate.dataset or {}
        return {
            "config": str(dataset.get("config_name") or dataset.get("config") or ""),
            "split": str(dataset.get("split_name") or dataset.get("split") or ""),
            "revision": str(dataset.get("revision") or dataset.get("sha") or dataset.get("last_modified") or ""),
        }

    def _pre_critic_prior(self, candidate: Candidate) -> float:
        if candidate.pre_rerank_score > 0.0:
            return _clamp01(candidate.pre_rerank_score, default=0.0)

        prior = (
            0.55 * _clamp01(candidate.retrieval_prior_score, default=0.0)
            + 0.25 * _clamp01(candidate.filter_score, default=0.0)
            + 0.20 * _clamp01(candidate.graph_score, default=0.0)
        )
        if prior <= 0.0:
            prior = _clamp01(candidate.final_score, default=0.0)
        return _clamp01(prior, default=0.0)

    def _score_batch_resilient(
        self,
        plan: SearchPlan,
        candidates: List[Candidate],
        iteration: int,
        sample_payload_map: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if not candidates:
            return {"candidate_scores": []}

        if len(candidates) == 1:
            self._bump_stat("initial_singleton_calls")
        else:
            self._bump_stat("initial_batch_calls")
        payload = self._llm_score_batch(
            plan,
            candidates,
            iteration,
            sample_payload_map=sample_payload_map,
            allow_heuristic_fallback=False,
        )
        if self._is_usable_score_payload(payload, candidates):
            return payload
        if not self._is_fallback_payload(payload):
            self._bump_stat("invalid_semantic_payloads")
        if len(candidates) == 1:
            self._bump_stat("initial_singleton_fallbacks")
        else:
            self._bump_stat("initial_batch_fallbacks")

        if len(candidates) > 1:
            self._bump_stat("split_events")
            split_at = max(1, len(candidates) // 2)
            left = self._score_batch_resilient(
                plan,
                candidates[:split_at],
                iteration,
                sample_payload_map=sample_payload_map,
            )
            right = self._score_batch_resilient(
                plan,
                candidates[split_at:],
                iteration,
                sample_payload_map=sample_payload_map,
            )
            return self._merge_candidate_score_rows(left, right)

        self._bump_stat("singleton_retry_calls")
        payload_retry = self._llm_score_batch(
            plan,
            candidates,
            iteration,
            sample_payload_map=sample_payload_map,
            allow_heuristic_fallback=False,
            timeout_seconds=self.SINGLETON_RETRY_TIMEOUT_SECONDS,
        )
        if self._is_usable_score_payload(payload_retry, candidates):
            self._bump_stat("singleton_retry_successes")
            return payload_retry
        if not self._is_fallback_payload(payload_retry):
            self._bump_stat("invalid_semantic_payloads")

        self._bump_stat("singleton_slow_retry_calls")
        payload_retry_slow = self._llm_score_batch(
            plan,
            candidates,
            iteration,
            sample_payload_map=sample_payload_map,
            allow_heuristic_fallback=False,
            timeout_seconds=self.SINGLETON_FINAL_RETRY_TIMEOUT_SECONDS,
        )
        if self._is_usable_score_payload(payload_retry_slow, candidates):
            self._bump_stat("singleton_slow_retry_successes")
            return payload_retry_slow
        if not self._is_fallback_payload(payload_retry_slow):
            self._bump_stat("invalid_semantic_payloads")

        self._bump_stat("singleton_minimal_retry_calls")
        payload_minimal = self._llm_score_batch(
            plan,
            candidates,
            iteration,
            sample_payload_map=sample_payload_map,
            allow_heuristic_fallback=False,
            timeout_seconds=self.SINGLETON_MINIMAL_RETRY_TIMEOUT_SECONDS,
            minimal_payload=True,
        )
        if self._is_usable_score_payload(payload_minimal, candidates):
            self._bump_stat("singleton_minimal_retry_successes")
            return payload_minimal
        if not self._is_fallback_payload(payload_minimal):
            self._bump_stat("invalid_semantic_payloads")

        self._bump_stat("heuristic_fallbacks")
        return self._heuristic_score_batch(
            plan,
            candidates,
            iteration,
            sample_payload_map=sample_payload_map,
        )

    def _merge_candidate_score_rows(self, *payloads: Mapping[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Dict[str, Any]] = {}
        for payload in payloads:
            rows = payload.get("candidate_scores") if isinstance(payload, Mapping) else None
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                dataset_id = str(row.get("dataset_id") or "").strip()
                if not dataset_id:
                    continue
                merged[dataset_id] = {
                    "dataset_id": dataset_id,
                    "llm_score": _clamp01(row.get("llm_score"), default=0.0),
                    **self._extract_llm_annotations({"candidate_scores": [row]}).get(
                        dataset_id,
                        {"confidence": "low", "evidence_codes": ["weak_evidence"]},
                    ),
                }
        return {"candidate_scores": list(merged.values())}

    def _llm_score_batch(
        self,
        plan: SearchPlan,
        candidates: List[Candidate],
        iteration: int,
        sample_payload_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
        allow_heuristic_fallback: bool = True,
        timeout_seconds: Optional[int] = None,
        minimal_payload: bool = False,
    ) -> Dict[str, Any]:
        if self.llm is None:
            return self._heuristic_score_batch(
                plan,
                candidates,
                iteration,
                sample_payload_map=sample_payload_map,
            )

        if sample_payload_map is None:
            sample_payload_map = self.sample_store.get_samples_bulk(
                [candidate.dataset_id for candidate in candidates],
                max_workers=min(self.MAX_WORKERS, max(1, len(candidates))),
            )
        request = self._build_candidate_scoring_request(
            plan,
            candidates,
            iteration,
            sample_payload_map=sample_payload_map,
            minimal_payload=minimal_payload,
        )
        default_payload = {
            "_critic_fallback": True,
            "candidate_scores": [
                {
                    "dataset_id": candidate.dataset_id,
                    "llm_score": 0.0,
                    "confidence": "low",
                    "evidence_codes": ["weak_evidence"],
                }
                for candidate in candidates
            ]
        }

        result = self.llm.generate_json(
            task_name="critic",
            system_prompt=self.scoring_prompt,
            user_prompt=json.dumps(request, ensure_ascii=False),
            default=default_payload,
            timeout_seconds=timeout_seconds or self.LLM_TIMEOUT_SECONDS,
            thinking=False,
        )
        payload = result if isinstance(result, dict) else default_payload
        if not allow_heuristic_fallback:
            return payload
        if self._is_fallback_payload(payload):
            return self._heuristic_score_batch(
                plan,
                candidates,
                iteration,
                sample_payload_map=sample_payload_map,
            )
        return payload

    def _build_candidate_scoring_request(
        self,
        plan: SearchPlan,
        candidates: List[Candidate],
        iteration: int,
        sample_payload_map: Mapping[str, Mapping[str, Any]],
        minimal_payload: bool,
    ) -> Dict[str, Any]:
        query_text = self._extract_query_text(plan)
        return {
            "mode": "candidate_scoring_only",
            "iteration": iteration,
            "query_text": query_text,
            "hard_constraints": self._build_hard_constraints(plan),
            "soft_preferences": self._build_soft_preferences(plan),
            "intent_codes": list(plan.intent_codes),
            "quality_target": plan.quality_target,
            "candidates": [
                self._build_candidate_payload(
                    plan,
                    candidate,
                    sample_payload=sample_payload_map.get(candidate.dataset_id, {}),
                    minimal_payload=minimal_payload,
                )
                for candidate in candidates
            ],
        }

    def _build_candidate_payload(
        self,
        plan: SearchPlan,
        candidate: Candidate,
        *,
        sample_payload: Mapping[str, Any],
        minimal_payload: bool,
    ) -> Dict[str, Any]:
        sample_evidence = build_candidate_evidence(
            dataset=candidate.dataset or {},
            sample={} if EXPERIMENT_DISABLE_CRITIC_SAMPLE_EVIDENCE else sample_payload,
            query=self._extract_query_text(plan),
            intent_codes=plan.intent_codes,
            text_budget=self.SAMPLE_MINIMAL_MAX_TEXT_CHARS if minimal_payload else self.SAMPLE_MAX_TEXT_CHARS,
        )

        if EXPERIMENT_DISABLE_CRITIC_GRAPH_EVIDENCE:
            kg_signals = {
                "graph_score": 0.0,
                "support": 0,
                "relation_types": [],
            }
        else:
            kg_signals = {
                "graph_score": float(candidate.graph_score),
                "support": int(candidate.graph_support),
                "relation_types": list(candidate.graph_relation_types)[:4 if minimal_payload else 6],
            }

        payload = {
            "dataset_id": candidate.dataset_id,
            "title": str((candidate.dataset or {}).get("name") or ""),
            "candidate_summary": self._build_candidate_summary(plan, candidate, minimal_payload=minimal_payload),
            "matched_facets": self._build_matched_facets(plan, candidate),
            "retrieval_signals": {
                "retrieval_rank": int(candidate.retrieval_rank),
                "hybrid_score": float(candidate.hybrid_score),
                "dense_score": float(candidate.dense_score),
                "bm25_score": float(candidate.sparse_score),
                "description_dense_score": float(candidate.desc_dense_score),
                "metadata_dense_score": float(candidate.metadata_dense_score),
                "filter_score": float(candidate.filter_score),
            },
            "kg_signals": kg_signals,
            "retrieval_provenance": [
                {
                    "query_id": item.query_id,
                    "branch_type": item.branch_type,
                    "rank": item.rank,
                    "rrf_contribution": item.rrf_contribution,
                    "anchor_id": item.anchor_id,
                    "anchor_confidence": item.anchor_confidence,
                    "relation_path": item.relation_path,
                }
                for item in candidate.retrieval_provenance[:6]
            ],
            "pre_rerank_score": float(candidate.pre_rerank_score),
            "hard_constraint_violations": list(candidate.constraint_violations),
            "sample_evidence": sample_evidence,
        }
        return payload

    def _build_hard_constraints(self, plan: SearchPlan) -> Dict[str, Any]:
        if plan.hard_constraints:
            return {
                item.field: {"operator": item.operator, "value": item.value, "source_text": item.source_text}
                for item in plan.hard_constraints
            }
        return {}

    def _build_soft_preferences(self, plan: SearchPlan) -> List[str]:
        if plan.soft_preferences:
            return [
                f"{item.objective}:{item.field}:{item.value if item.value is not None else ''}"
                for item in plan.soft_preferences
            ]
        query_text = self._extract_query_text(plan)
        if not query_text:
            return []

        patterns = [
            r"\bideally\b",
            r"\bpreferably\b",
            r"\bif possible\b",
            r"\bnice to have\b",
            r"\bhigh downloads?\b",
            r"\bpopular\b",
        ]
        lowered = query_text.lower()
        preferences: List[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, lowered):
                preferences.append(lowered[match.start() : match.end()])
        return preferences

    def _build_candidate_summary(
        self,
        plan: SearchPlan,
        candidate: Candidate,
        *,
        minimal_payload: bool,
    ) -> str:
        dataset = candidate.dataset or {}
        tasks = self._dedup_strings(dataset.get("tasks", []), limit=2 if minimal_payload else 3)
        modalities = self._dedup_strings(dataset.get("modalities", []), limit=1 if minimal_payload else 2)
        languages = self._dedup_strings(dataset.get("languages", []), limit=1 if minimal_payload else 2)
        tags = self._dedup_strings(dataset.get("tags", []), limit=2 if minimal_payload else 3)
        description = self._truncate_summary_text(str(dataset.get("description") or ""), limit=70 if minimal_payload else 120)

        relevant_languages = self._build_matched_facets(plan, candidate).get("languages", [])
        pieces: List[str] = []
        if tasks:
            pieces.append(f"tasks: {', '.join(tasks)}")
        if modalities:
            pieces.append(f"modality: {', '.join(modalities)}")
        if relevant_languages:
            pieces.append(f"languages: {', '.join(relevant_languages[:2])}")
        elif languages and self._has_language_constraint(plan):
            pieces.append(f"languages: {', '.join(languages)}")

        signal_tags = [
            tag
            for tag in tags
            if any(token in tag.lower() for token in ("benchmark", "instruction", "preference", "safety", "reason", "code"))
        ]
        if signal_tags:
            pieces.append(f"signals: {', '.join(signal_tags)}")

        if description and (not pieces or len("; ".join(pieces)) < (70 if minimal_payload else 110)):
            pieces.append(description)

        if not pieces:
            fallback_parts = [candidate.dataset_id]
            if dataset.get("name"):
                fallback_parts.append(str(dataset.get("name")))
            return " | ".join(fallback_parts[:2])
        return "; ".join(pieces[:3 if minimal_payload else 4])

    def _build_matched_facets(self, plan: SearchPlan, candidate: Candidate) -> Dict[str, List[str]]:
        dataset = candidate.dataset or {}
        constraint_values = self._collect_constraint_values(plan)
        matched: Dict[str, List[str]] = {}

        for field in ("tasks", "languages", "modalities", "tags"):
            expected = constraint_values.get(field, set())
            if not expected:
                continue
            raw_values = dataset.get(field, [])
            if not isinstance(raw_values, list):
                raw_values = [raw_values]
            rows: List[str] = []
            seen: set[str] = set()
            for raw in raw_values:
                text = str(raw or "").strip()
                normalized = self._normalize_simple_token(text)
                if not text or normalized not in expected or normalized in seen:
                    continue
                seen.add(normalized)
                rows.append(text)
            if rows:
                matched[field] = rows
        return matched

    def _collect_constraint_values(self, plan: SearchPlan) -> Dict[str, set[str]]:
        values: Dict[str, set[str]] = {field: set() for field in ("tasks", "languages", "modalities", "tags")}
        for constraint in plan.hard_constraints:
            if constraint.field not in values:
                continue
            raw_values = constraint.value if isinstance(constraint.value, list) else [constraint.value]
            values[constraint.field].update(self._normalize_simple_token(v) for v in raw_values if str(v or "").strip())
        for query in plan.retrieval_queries:
            for field in values:
                raw_values = query.facets.get(field, [])
                raw_values = raw_values if isinstance(raw_values, list) else [raw_values]
                values[field].update(self._normalize_simple_token(v) for v in raw_values if str(v or "").strip())
        return values

    def _dedup_strings(self, values: Any, *, limit: int) -> List[str]:
        if not isinstance(values, list):
            values = [values] if values else []
        rows: List[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            normalized = self._normalize_simple_token(text)
            if not text or normalized in seen:
                continue
            seen.add(normalized)
            rows.append(text)
            if len(rows) >= limit:
                break
        return rows

    def _truncate_summary_text(self, text: str, *, limit: int) -> str:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned:
            return ""
        sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0]
        if len(sentence) <= limit:
            return sentence
        return sentence[: max(0, limit - 3)].rstrip(" ,;:") + "..."

    def _normalize_simple_token(self, value: Any) -> str:
        return normalize_metadata_token(value)

    def _has_language_constraint(self, plan: SearchPlan) -> bool:
        constraint_values = self._collect_constraint_values(plan)
        return bool(constraint_values.get("languages"))

    def _is_fallback_payload(self, payload: Mapping[str, Any]) -> bool:
        rows = payload.get("candidate_scores")
        if not isinstance(rows, list) or not rows:
            return True

        if payload.get("_llm_errors"):
            return True
        if payload.get("_critic_fallback"):
            return True
        return False

    def _is_usable_score_payload(
        self,
        payload: Mapping[str, Any],
        candidates: List[Candidate],
    ) -> bool:
        """Reject structurally valid but semantically self-contradictory scores."""
        if self._is_fallback_payload(payload):
            return False

        rows = payload.get("candidate_scores")
        if not isinstance(rows, list):
            return False
        expected_ids = {candidate.dataset_id for candidate in candidates}
        observed_ids: set[str] = set()
        positive_codes = {
            "name_match",
            "task_match",
            "domain_match",
            "format_match",
            "training_data",
            "training_split_support",
            "supervision_pair_support",
            "reference_answer_support",
            "anchor_support",
            "constraint_match",
            "sample_support",
        }
        for row in rows:
            if not isinstance(row, Mapping):
                return False
            dataset_id = str(row.get("dataset_id") or "").strip()
            if dataset_id not in expected_ids or dataset_id in observed_ids:
                return False
            observed_ids.add(dataset_id)
            try:
                score = float(row.get("llm_score"))
            except (TypeError, ValueError):
                return False
            if not math.isfinite(score) or score < 0.0 or score > 1.0:
                return False
            codes = {
                str(value or "").strip().lower()
                for value in (row.get("evidence_codes") or [])
            }
            # A zero relevance score cannot simultaneously claim direct
            # positive evidence. This catches copied schema placeholders while
            # preserving legitimate confident negative judgments.
            if score <= 0.01 and codes & positive_codes:
                return False
        return observed_ids == expected_ids

    def _heuristic_score_batch(
        self,
        plan: SearchPlan,
        candidates: List[Candidate],
        iteration: int,
        sample_payload_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        preferred_tasks = self._extract_preferred_tasks(plan)
        preferred_languages = self._extract_preferred_languages(plan)
        dataset_name_hint = self._extract_dataset_name_hint(plan)
        query_tokens = self._extract_query_tokens(plan)

        if sample_payload_map is None:
            sample_payload_map = self.sample_store.get_samples_bulk(
                [candidate.dataset_id for candidate in candidates],
                max_workers=min(self.MAX_WORKERS, max(1, len(candidates))),
            )

        query_text = self._extract_query_text(plan).lower()

        scores: List[Dict[str, Any]] = []
        for candidate in candidates:
            dataset = candidate.dataset or {}
            sample_payload = sample_payload_map.get(candidate.dataset_id, {}) if sample_payload_map else {}
            sample_text = str(sample_payload.get("sample_text") or "")

            score = _clamp01(
                0.55 * self._pre_critic_prior(candidate)
                + 0.20 * candidate.retrieval_prior_score
                + 0.15 * candidate.filter_score
                + 0.10 * candidate.graph_score,
                default=0.05,
            )
            evidence_codes: List[str] = []

            if dataset_name_hint and self._is_name_match(candidate, dataset_name_hint):
                score = max(score, 0.98)
                evidence_codes.append("name_match")

            candidate_tokens = self._extract_candidate_tokens(candidate, sample_text=sample_text)
            if query_tokens and candidate_tokens:
                overlap = len(query_tokens & candidate_tokens) / max(1, len(query_tokens))
                if overlap > 0:
                    score = _clamp01(score + min(0.35, 0.35 * overlap), default=score)
                    evidence_codes.append("metadata_support")

            query_has_image = "image" in query_tokens or "photo" in query_tokens or "picture" in query_tokens
            query_has_safety = bool(
                {"nsfw", "adult", "inappropriate", "moderation", "unsafe", "hentai", "porn", "sexy"}
                & query_tokens
            )

            dataset_modalities = self._normalize_tokens(dataset.get("modalities", []))
            if query_has_image:
                if "image" in dataset_modalities:
                    score = _clamp01(score + 0.12, default=score)
                    evidence_codes.append("constraint_match")
                else:
                    score = _clamp01(score - 0.06, default=score)

            if query_has_safety:
                safety_tokens = {"nsfw", "adult", "inappropriate", "moderation", "unsafe", "hentai", "porn", "sexy"}
                safety_overlap = len(safety_tokens & candidate_tokens)
                if safety_overlap > 0:
                    score = _clamp01(score + min(0.45, 0.12 * safety_overlap), default=score)
                    evidence_codes.append("domain_match")
                else:
                    score = _clamp01(score - 0.08, default=score)

            if "benchmark" in query_tokens and "benchmark" in candidate_tokens:
                score = _clamp01(score + 0.08, default=score)
            if "tabular" in query_tokens and "tabular" in candidate_tokens:
                score = _clamp01(score + 0.1, default=score)

            vision_material_tokens = {"pbr", "material", "texture", "rendering", "graphics"}
            if len(vision_material_tokens & query_tokens) >= 2:
                vision_overlap = len(vision_material_tokens & candidate_tokens)
                if vision_overlap > 0:
                    score = _clamp01(score + min(0.12, 0.04 * vision_overlap), default=score)
                    evidence_codes.append("domain_match")

            if sample_text and query_text:
                if any(token in sample_text.lower() for token in query_tokens if len(token) >= 4):
                    score = _clamp01(score + 0.05, default=score)
                    evidence_codes.append("sample_support")

            dataset_tasks = self._normalize_tokens(dataset.get("tasks", []))
            if preferred_tasks:
                overlap = len(dataset_tasks & preferred_tasks) / max(1, len(preferred_tasks))
                if overlap > 0:
                    score = _clamp01(score + 0.2 * overlap, default=score)
                    evidence_codes.append("task_match")

            dataset_languages = self._normalize_tokens(dataset.get("languages", []))
            if preferred_languages:
                overlap = len(dataset_languages & preferred_languages) / max(1, len(preferred_languages))
                if overlap > 0:
                    score = _clamp01(score + 0.1 * overlap, default=score)
                    evidence_codes.append("constraint_match")

            downloads = max(0, int(dataset.get("downloads", 0) or 0))
            likes = max(0, int(dataset.get("likes", 0) or 0))
            popularity_boost = min(0.08, math.log1p(downloads) / 140.0 + math.log1p(likes) / 220.0)
            score = _clamp01(score + popularity_boost, default=score)

            if candidate.graph_support > 0 or candidate.graph_score > 0:
                evidence_codes.append("graph_support")
            if candidate.retrieval_prior_score > 0 or candidate.hybrid_score > 0:
                evidence_codes.append("retrieval_support")
            evidence_codes = list(dict.fromkeys(evidence_codes))[:4] or ["weak_evidence"]

            scores.append(
                {
                    "dataset_id": candidate.dataset_id,
                    "llm_score": score,
                    "confidence": "low",
                    "evidence_codes": evidence_codes,
                }
            )

        return {
            "mode": "candidate_scoring_only",
            "iteration": iteration,
            "candidate_scores": scores,
        }

    def _extract_query_text(self, plan: SearchPlan) -> str:
        return plan.user_query.strip()

    def _extract_query_tokens(self, plan: SearchPlan) -> set[str]:
        values: List[str] = []
        query_text = self._extract_query_text(plan)
        if query_text:
            values.append(query_text)

        for query in plan.retrieval_queries:
            for key in ("tasks", "tags", "languages"):
                value = query.facets.get(key)
                if isinstance(value, list):
                    values.extend(str(item) for item in value)

        return self._tokenize(" ".join(values))

    def _extract_candidate_tokens(self, candidate: Candidate, sample_text: str = "") -> set[str]:
        dataset = candidate.dataset or {}
        text_parts = [
            str(candidate.dataset_id or ""),
            str(dataset.get("name") or ""),
            str(dataset.get("description") or ""),
            " ".join(str(v) for v in (dataset.get("tasks") or [])),
            " ".join(str(v) for v in (dataset.get("tags") or [])),
            " ".join(str(v) for v in (dataset.get("modalities") or [])),
            " ".join(str(v) for v in (dataset.get("other") or [])),
            sample_text,
        ]
        return self._tokenize(" ".join(text_parts))

    def _tokenize(self, text: str) -> set[str]:
        if not text:
            return set()
        tokens = re.findall(r"[a-z0-9][a-z0-9._/-]{1,}|[\u4e00-\u9fff]", text.lower())
        return {tok for tok in tokens if tok}

    def _extract_dataset_name_hint(self, plan: SearchPlan) -> str:
        exact = next((item for item in plan.retrieval_queries if item.role == "exact_name" and item.source_evidence), None)
        return exact.source_evidence.lower() if exact else ""

    def _is_name_match(self, candidate: Candidate, dataset_name_hint: str) -> bool:
        if not dataset_name_hint:
            return False
        dataset = candidate.dataset or {}
        dataset_id = str(candidate.dataset_id or "").lower()
        dataset_name = str(dataset.get("name") or "").lower()
        return dataset_name_hint in dataset_id or dataset_name_hint in dataset_name

    def _extract_preferred_tasks(self, plan: SearchPlan) -> set[str]:
        values: List[Any] = []
        for constraint in plan.hard_constraints:
            if constraint.field == "tasks":
                values.extend(constraint.value if isinstance(constraint.value, list) else [constraint.value])
        for query in plan.retrieval_queries:
            values.extend(query.facets.get("tasks", []) or [])
        return self._normalize_tokens(values)

    def _extract_preferred_languages(self, plan: SearchPlan) -> set[str]:
        values: List[Any] = []
        for constraint in plan.hard_constraints:
            if constraint.field == "languages":
                values.extend(constraint.value if isinstance(constraint.value, list) else [constraint.value])
        for query in plan.retrieval_queries:
            values.extend(query.facets.get("languages", []) or [])
        return self._normalize_tokens(values, category="languages")

    def _normalize_tokens(self, values: Any, *, category: str = "") -> set[str]:
        return normalize_metadata_values(values, category=category)

    def _llm_diagnose_failure(
        self,
        plan: SearchPlan,
        candidates: List[Candidate],
        iteration: int,
        min_quality: float,
        accept: bool,
        quality_score: float,
        quantity_score: float,
        overall_score: float,
        deterministic_diagnostics: Optional[List[Diagnostic]] = None,
    ) -> Dict[str, Any]:
        if self.llm is None:
            return {
                "reason": "llm_unavailable",
                "diagnostics": [],
            }

        summary = [
            {
                "dataset_id": candidate.dataset_id,
                "llm_score": round(candidate.final_score, 4),
                "confidence": candidate.confidence,
                "evidence_codes": list(candidate.evidence_codes),
                "matched_query_ids": list(candidate.matched_query_ids),
            }
            for candidate in candidates[:20]
        ]

        request = {
            "mode": "iteration_decision_only",
            "iteration": iteration,
            "min_quality": min_quality,
            "top_k_final": plan.top_k_final,
            "quality_target": plan.quality_target,
            "query_text": plan.user_query,
            "hard_constraints": self._build_hard_constraints(plan),
            "candidate_count": len(candidates),
            "candidate_scores": summary,
            "intent_codes": list(plan.intent_codes),
            "instruction": "Diagnose failed gates. Do not override deterministic acceptance.",
            "deterministic_diagnostics": [
                {"code": item.code, "severity": item.severity, "evidence": item.evidence}
                for item in (deterministic_diagnostics or [])
            ],
            "current_feedback": {
                "accept": accept,
                "quality_score": quality_score,
                "quantity_score": quantity_score,
                "overall_score": overall_score,
            },
        }
        default_payload = {
            "reason": "derived_from_llm_scores",
            "diagnostics": [],
        }

        result = self.llm.generate_json(
            task_name="critic",
            system_prompt=self.system_prompt,
            user_prompt=json.dumps(request, ensure_ascii=False),
            default=default_payload,
            timeout_seconds=self.LLM_TIMEOUT_SECONDS,
        )
        return result if isinstance(result, dict) else default_payload

    def _load_prompt(self, path: Path) -> str:
        if not path.exists():
            return "You are Critic Agent. Return strict JSON object only."
        return path.read_text(encoding="utf-8")

    def _build_scoring_prompt(self) -> str:
        return (
            "You are Critic Agent for dataset retrieval. "
            "Each candidate includes candidate_summary, matched_facets, retrieval_signals, kg_signals, "
            "pre_rerank_score, and sample_evidence with field-linked snippets, metadata, applicability, and consistency checks. "
            "Score each candidate from 0 to 1 by relevance to query, constraints, and evidence quality. "
            "Judge the query as a conjunctive information need: high scores require the same candidate to satisfy its essential dimensions; "
            "evidence distributed across different candidates does not make one candidate fully relevant. "
            "For training-data requests, assess whether the observed rows provide the requested supervision, including reference code when requested. "
            "A missing preview, incomplete configuration coverage, or absent evidence is unknown, not proof of a violation. "
            "Do not infer that an answer is medically or mathematically correct merely because a consistency check passed. "
            "Use sample_evidence as supporting evidence when present, but do not require it for every candidate. "
            "Treat retrieval_signals and kg_signals as supporting evidence, not as automatic truth. "
            "Return one compact row per candidate with llm_score, confidence, and evidence_codes; never write reasons. "
            "confidence must be low, medium, or high. evidence_codes must contain at most six values chosen from: "
            "name_match, task_match, domain_match, format_match, training_data, training_split_support, supervision_pair_support, "
            "reference_answer_support, anchor_support, hard_constraint_violation, constraint_match, sample_support, metadata_support, graph_support, "
            "retrieval_support, weak_evidence, contradictory_evidence. "
            "The numeric score must reflect your judgment; do not copy the illustrative value from the schema. "
            "A score of exactly 0 means no relevance evidence and must not be paired with positive evidence codes. "
            "Return JSON only with schema: {\"candidate_scores\":[{\"dataset_id\":\"...\",\"llm_score\":0.73,"
            "\"confidence\":\"medium\",\"evidence_codes\":[\"task_match\",\"sample_support\"]}]}. "
            "Do not include markdown or extra keys. "
            "You must score every candidate in input candidates."
        )
