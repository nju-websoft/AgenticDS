import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, DefaultDict, Dict, List

from agent import MultiAgentDatasetRetriever
from build_query_bank import default_query_bank_path
from global_values import EVAL_QUERY_BANK_SIZE, EVAL_SAMPLE_SIZE
from utils.kg_hybrid_retriever import KGHybridRetriever


def _build_agent_process(payload: dict) -> dict:
    iterations = payload.get("iterations", []) or []
    final_candidates = payload.get("final_candidates", []) or []

    stage_totals: DefaultDict[str, float] = defaultdict(float)
    process_iterations = []
    for iteration in iterations:
        plan = iteration.get("plan", {}) or {}
        steps = iteration.get("steps", []) or []
        trace = iteration.get("trace", []) or []
        critic = iteration.get("critic_feedback", {}) or {}
        stage_timings = iteration.get("stage_timings", {}) or {}

        for stage, seconds in stage_timings.items():
            try:
                stage_totals[str(stage)] += float(seconds)
            except (TypeError, ValueError):
                continue

        process_iterations.append(
            {
                "iteration": iteration.get("iteration"),
                "plan": {
                    "plan_outline": plan.get("plan_outline", []),
                    "retrieval_queries": plan.get("retrieval_queries", []),
                    "relation_search_specs": plan.get("relation_search_specs", []),
                    "retrieval_policy": plan.get("retrieval_policy", {}),
                    "top_k_final": plan.get("top_k_final"),
                    "reasoning": plan.get("reasoning", ""),
                },
                "steps": steps,
                "trace": trace,
                "critic_feedback": {
                    "accept": critic.get("accept"),
                    "quality_score": critic.get("quality_score"),
                    "quantity_score": critic.get("quantity_score"),
                    "overall_score": critic.get("overall_score"),
                    "reason": critic.get("reason", ""),
                    "intent_coverage_score": critic.get("intent_coverage_score"),
                    "diagnostics": critic.get("diagnostics", []),
                    "candidate_scores": critic.get("candidate_scores", []),
                },
                "stage_timings": stage_timings,
            }
        )

    result = {
        "query": payload.get("query", {}),
        "summary": {
            "iterations": len(iterations),
            "final_candidate_count": len(final_candidates),
            "stage_timing_totals": dict(stage_totals),
        },
        "iterations": process_iterations,
    }
    if payload.get("debug_artifacts"):
        result["debug_artifacts"] = payload.get("debug_artifacts")
    return result


def _build_final_result(payload: dict, gold_dataset_id: str) -> dict:
    final_candidates = payload.get("final_candidates", []) or []
    rows = [
        {
            "id": item.get("dataset", {}).get("id"),
            "final_score": item.get("final_score", 0.0),
        }
        for item in final_candidates
    ]
    ids = [row["id"] for row in rows if row.get("id")]
    hit = gold_dataset_id in ids
    rank = ids.index(gold_dataset_id) + 1 if hit else None

    return {
        "query": payload.get("query", {}),
        "gold_dataset_id": gold_dataset_id,
        "hit_at_10": hit,
        "gold_rank": rank,
        "final_candidates": rows,
    }


def _rank_in_list(ids: List[str], gold_dataset_id: str) -> int | None:
    try:
        return ids.index(gold_dataset_id) + 1
    except ValueError:
        return None


def _run_baseline_comparison(
    hybrid_retriever: KGHybridRetriever,
    query: str,
    gold_dataset_id: str,
    *,
    top_k_final: int = 10,
    top_k_rank_eval: int = 220,
) -> Dict[str, Any]:
    hybrid_results = hybrid_retriever.search(query=query, top_k=top_k_rank_eval)
    hybrid_rows = hybrid_results.get("dataset", []) or []
    hybrid_ids = [str((row.get("entity") or {}).get("id") or "") for row in hybrid_rows if (row.get("entity") or {}).get("id")]
    hybrid_rank = _rank_in_list(hybrid_ids, gold_dataset_id)

    keyword_results = hybrid_retriever.search_sparse(query=query, top_k=top_k_rank_eval)
    keyword_rows = keyword_results.get("dataset", []) or []
    keyword_ranked = [
        str((row.get("entity") or {}).get("id") or "")
        for row in keyword_rows
        if (row.get("entity") or {}).get("id")
    ]
    keyword_rank = _rank_in_list(keyword_ranked, gold_dataset_id)

    return {
        "hybrid": {
            "hit_at_10": bool(hybrid_rank is not None and hybrid_rank <= top_k_final),
            "gold_rank": hybrid_rank,
            "top_ids": hybrid_ids[:top_k_final],
        },
        "keyword": {
            "hit_at_10": bool(keyword_rank is not None and keyword_rank <= top_k_final),
            "gold_rank": keyword_rank,
            "top_ids": keyword_ranked[:top_k_final],
        },
    }


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _load_query_bank(bank_path: Path) -> List[Dict[str, Any]]:
    if not bank_path.exists():
        raise FileNotFoundError(
            f"Query bank not found: {bank_path}. "
            f"Please build it first: python build_query_bank.py --bank-size {EVAL_QUERY_BANK_SIZE}"
        )

    with bank_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, list):
        raw_rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("queries"), list):
        raw_rows = payload["queries"]
    else:
        raise ValueError(f"Invalid query bank format in {bank_path}")

    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(raw_rows, 1):
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        gold_dataset_id = str(item.get("gold_dataset_id") or "").strip()
        if not query or not gold_dataset_id:
            continue

        row = dict(item)
        row.setdefault("query_id", f"q_{i:03d}")
        row["query"] = query
        row["gold_dataset_id"] = gold_dataset_id
        rows.append(row)

    if not rows:
        raise ValueError(f"No valid query rows found in {bank_path}")

    return rows


def _collect_stage_timing_totals(payload: dict) -> Dict[str, float]:
    iterations = payload.get("iterations", []) or []
    stage_totals: DefaultDict[str, float] = defaultdict(float)
    for iteration in iterations:
        stage_timings = iteration.get("stage_timings", {}) or {}
        for stage, seconds in stage_timings.items():
            try:
                stage_totals[str(stage)] += float(seconds)
            except (TypeError, ValueError):
                continue
    return dict(stage_totals)


def _pick_slowest_action_stage(stage_totals: Dict[str, float]) -> str | None:
    if not stage_totals:
        return None

    filtered = {
        stage: seconds
        for stage, seconds in stage_totals.items()
        if stage not in {"iteration_total_seconds"}
    }
    target = filtered or stage_totals
    return max(target.items(), key=lambda x: x[1])[0]


def run_benchmark(
    *,
    query_bank_file: Path,
    query_bank_size: int,
    sample_size: int,
    seed: int,
    debug_full_trace: bool,
) -> None:
    bank_path = query_bank_file
    query_bank = _load_query_bank(bank_path)
    if query_bank_size > 0:
        query_bank = query_bank[: min(query_bank_size, len(query_bank))]

    sample_n = min(max(1, sample_size), len(query_bank))
    rng = random.Random(seed)
    sampled = rng.sample(query_bank, sample_n)

    retriever = MultiAgentDatasetRetriever(debug_full_trace=debug_full_trace)
    hybrid_retriever = KGHybridRetriever()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("output") / "runs" / f"benchmark_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    method_hit_count = {
        "full": 0,
        "hybrid": 0,
        "keyword": 0,
    }
    total_query_seconds = 0.0
    global_stage_totals: DefaultDict[str, float] = defaultdict(float)

    for i, item in enumerate(sampled, 1):
        query = item["query"]
        gold = item["gold_dataset_id"]
        query_start = perf_counter()
        result = retriever.run_as_dict(query, top_k=10)
        query_seconds = perf_counter() - query_start
        total_query_seconds += query_seconds

        stage_totals = _collect_stage_timing_totals(result)
        for stage, seconds in stage_totals.items():
            global_stage_totals[stage] += seconds

        slowest_stage = _pick_slowest_action_stage(stage_totals)

        agent_payload = _build_agent_process(result)
        final_payload = _build_final_result(result, gold)
        baseline = _run_baseline_comparison(
            hybrid_retriever,
            query,
            gold,
            top_k_final=10,
            top_k_rank_eval=220,
        )

        full_hit = bool(final_payload.get("hit_at_10"))
        hybrid_hit = bool((baseline.get("hybrid") or {}).get("hit_at_10"))
        keyword_hit = bool((baseline.get("keyword") or {}).get("hit_at_10"))

        if full_hit:
            method_hit_count["full"] += 1
        if hybrid_hit:
            method_hit_count["hybrid"] += 1
        if keyword_hit:
            method_hit_count["keyword"] += 1

        final_payload["comparison"] = {
            "full": {
                "hit_at_10": full_hit,
                "gold_rank": final_payload.get("gold_rank"),
                "top_ids": [row.get("id") for row in final_payload.get("final_candidates", [])[:10] if row.get("id")],
            },
            "hybrid": baseline.get("hybrid") or {},
            "keyword": baseline.get("keyword") or {},
        }
        final_payload["timings"] = {
            "query_total_seconds": query_seconds,
            "stage_totals_seconds": stage_totals,
            "slowest_stage": slowest_stage,
        }

        query_dir = run_dir / f"query_{i:02d}"
        agent_path = query_dir / "agent.json"
        final_path = query_dir / "final.json"
        _save_json(agent_path, agent_payload)
        _save_json(final_path, final_payload)

        rows.append(
            {
                "query_index": i,
                "query_id": item["query_id"],
                "query": query,
                "gold_dataset_id": gold,
                "domain": item.get("domain"),
                "full_hit_at_10": full_hit,
                "full_gold_rank": final_payload.get("gold_rank"),
                "hybrid_hit_at_10": hybrid_hit,
                "hybrid_gold_rank": (baseline.get("hybrid") or {}).get("gold_rank"),
                "keyword_hit_at_10": keyword_hit,
                "keyword_gold_rank": (baseline.get("keyword") or {}).get("gold_rank"),
                "query_total_seconds": query_seconds,
                "stage_totals_seconds": stage_totals,
                "slowest_stage": slowest_stage,
                "agent_file": str(agent_path),
                "final_file": str(final_path),
            }
        )

        print("=" * 80)
        print(f"[{i}] Query: {query}")
        print(f"Gold: {gold}")
        print(
            "Hit@10 (full/hybrid/keyword): "
            f"{full_hit}/{hybrid_hit}/{keyword_hit} | "
            "rank(full/hybrid/keyword)="
            f"{final_payload.get('gold_rank')}/"
            f"{(baseline.get('hybrid') or {}).get('gold_rank')}/"
            f"{(baseline.get('keyword') or {}).get('gold_rank')}"
        )
        print(f"Query total time: {query_seconds:.3f}s")
        if stage_totals:
            print(f"Stage totals: {stage_totals}")
            print(f"Slowest stage: {slowest_stage}")
        if debug_full_trace:
            print("Debug full trace: enabled (LLM conversations + executor IDs saved in agent.json)")
        print(f"Agent process: {agent_path}")
        print(f"Final result:  {final_path}")

    average_stage_per_query = {
        stage: seconds / max(1, sample_n) for stage, seconds in sorted(global_stage_totals.items())
    }
    slowest_stage_by_avg = _pick_slowest_action_stage(average_stage_per_query)

    summary = {
        "run_timestamp": timestamp,
        "query_bank_file": str(bank_path),
        "query_bank_size": len(query_bank),
        "sample_size": sample_n,
        "seed": seed,
        "debug_full_trace": bool(debug_full_trace),
        "method_hit_count": method_hit_count,
        "method_hit_rate": {
            "full": method_hit_count["full"] / max(1, sample_n),
            "hybrid": method_hit_count["hybrid"] / max(1, sample_n),
            "keyword": method_hit_count["keyword"] / max(1, sample_n),
        },
        "timings": {
            "total_query_seconds": total_query_seconds,
            "average_query_seconds": total_query_seconds / max(1, sample_n),
            "stage_totals_seconds": dict(sorted(global_stage_totals.items())),
            "stage_avg_seconds_per_query": average_stage_per_query,
            "slowest_stage_by_avg": slowest_stage_by_avg,
        },
        "queries": rows,
    }
    summary_path = run_dir / "summary.json"
    _save_json(summary_path, summary)

    print("=" * 80)
    print(
        "Benchmark completed. Hit@10 "
        f"full/hybrid/keyword = "
        f"{method_hit_count['full']}/{sample_n} ({summary['method_hit_rate']['full']:.2%}) / "
        f"{method_hit_count['hybrid']}/{sample_n} ({summary['method_hit_rate']['hybrid']:.2%}) / "
        f"{method_hit_count['keyword']}/{sample_n} ({summary['method_hit_rate']['keyword']:.2%})"
    )
    print(f"Summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AgenticDS benchmark using reusable query bank and hit evaluation.")
    parser.add_argument("--bank-size", type=int, default=EVAL_QUERY_BANK_SIZE)
    parser.add_argument("--bank-file", type=str, default="")
    parser.add_argument("--sample-size", type=int, default=EVAL_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug-full-trace",
        action="store_true",
        help="Save full execution trace including LLM conversations and executor IDs.",
    )
    args = parser.parse_args()

    bank_size = max(10, int(args.bank_size))
    bank_file = Path(args.bank_file) if args.bank_file else default_query_bank_path(bank_size)

    run_benchmark(
        query_bank_file=bank_file,
        query_bank_size=bank_size,
        sample_size=max(1, int(args.sample_size)),
        seed=int(args.seed),
        debug_full_trace=bool(args.debug_full_trace),
    )


if __name__ == "__main__":
    main()
