import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from agent import MultiAgentDatasetRetriever
from global_values import AGENT_MAX_ITERATIONS, DEBUG_FULL_TRACE, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, RETRIEVAL_SERVICE_SOCKET


def _build_final_result(payload: dict) -> dict:
    final_candidates = payload.get("final_candidates", []) or []
    return {
        "query": payload.get("query", {}),
        "final_candidates": [
            {
                "id": item.get("dataset", {}).get("id"),
                "final_score": item.get("final_score", 0.0),
            }
            for item in final_candidates
        ],
    }


def _save_timestamped_outputs(
    agent_payload: dict,
    final_payload: dict,
    *,
    prefix: str,
    output_root: Path = Path("output/runs"),
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    agent_file = run_dir / "agent.json"
    final_file = run_dir / "final.json"
    with agent_file.open("w", encoding="utf-8") as fh:
        json.dump(agent_payload, fh, ensure_ascii=False, indent=2)
    with final_file.open("w", encoding="utf-8") as fh:
        json.dump(final_payload, fh, ensure_ascii=False, indent=2)

    return run_dir


def _print_top5(final_payload: dict) -> None:
    final_candidates = final_payload.get("final_candidates", []) or []
    print("=" * 80)
    print(f"Total candidates: {len(final_candidates)}")
    print("Top 5:")
    for i, item in enumerate(final_candidates[:5], 1):
        print(f"{i}. {item.get('id', 'N/A')} | score={item.get('final_score', 0):.4f}")


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one AgenticDS dataset query.")
    parser.add_argument("--query", required=True, help="Natural-language dataset search query.")
    parser.add_argument("--top-k", type=_positive_int, default=20)
    parser.add_argument("--agent-model", default=LLM_MODEL_NAME, help="LLM backbone model ID.")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--retrieval-service-socket", default=RETRIEVAL_SERVICE_SOCKET, help="Reuse the matching running retrieval service; omitted means in-process index loading.")
    parser.add_argument("--max-iterations", type=_positive_int, default=AGENT_MAX_ITERATIONS)
    parser.add_argument("--locale", default="en")
    parser.add_argument("--debug-full-trace", action="store_true", default=DEBUG_FULL_TRACE)
    parser.add_argument("--output-root", type=Path, default=Path("output/runs"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    query = args.query.strip()
    if not query:
        parser.error("--query must not be empty")

    try:
        retriever = MultiAgentDatasetRetriever(
            max_iterations=args.max_iterations,
            debug_full_trace=args.debug_full_trace,
            embedding_model=args.embedding_model,
            agent_model=args.agent_model,
            retrieval_service_socket=args.retrieval_service_socket or None,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Check the retrieval service socket and index paths. Build missing indexes with 'python -m utils.kg_hybrid_index_builder'.", file=sys.stderr)
        return 1

    print(f"Searching for: {query}")
    result = retriever.run_as_dict(query=query, top_k=args.top_k, locale=args.locale)
    final_payload = _build_final_result(result)

    _print_top5(final_payload)
    run_dir = _save_timestamped_outputs(result, final_payload, prefix="demo", output_root=args.output_root)
    print(f"Saved agent process to {run_dir / 'agent.json'}")
    print(f"Saved final result to {run_dir / 'final.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
