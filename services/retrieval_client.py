from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from time import monotonic, sleep
from typing import Any, Dict, Iterable, Mapping, Sequence


class RetrievalClient:
    """Typed local RPC client for one immutable retrieval index group."""

    def __init__(
        self,
        socket_path: str,
        *,
        expected_snapshot: str,
        expected_embedding_model: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.socket_path = str(socket_path)
        self.timeout_seconds = float(timeout_seconds)
        self.expected = {
            "snapshot_id": Path(expected_snapshot).name,
            "embedding_model": str(expected_embedding_model),
        }
        self.last_response_meta: Dict[str, Any] = {}

    def _request(self, method: str, params: Mapping[str, Any] | None = None, *, validate: bool = True) -> Any:
        payload = {
            "method": method,
            "params": dict(params or {}),
            "expected": self.expected if validate else {},
        }
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout_seconds)
            sock.connect(self.socket_path)
            stream = sock.makefile("rwb")
            stream.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
            stream.flush()
            raw = stream.readline()
        if not raw:
            raise RuntimeError("Retrieval service closed the connection without a response")
        response = json.loads(raw.decode("utf-8"))
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "Retrieval service request failed"))
        self.last_response_meta = dict(response.get("meta") or {})
        return response.get("result")

    def health(self) -> Dict[str, Any]:
        return dict(self._request("health", validate=False) or {})

    def wait_until_ready(self, timeout_seconds: float = 7200.0, poll_seconds: float = 5.0) -> Dict[str, Any]:
        deadline = monotonic() + timeout_seconds
        last_error = ""
        while monotonic() < deadline:
            try:
                health = self.health()
            except (OSError, RuntimeError) as exc:
                last_error = str(exc)
            else:
                if health.get("status") == "ready":
                    self.manifest()
                    return health
                if health.get("status") == "failed":
                    error = str(health.get("error") or "Retrieval service failed during startup")
                    raise RuntimeError(error)
            sleep(max(0.1, poll_seconds))
        raise TimeoutError(f"Retrieval service did not become ready: {last_error}")

    def manifest(self) -> Dict[str, Any]:
        return dict(self._request("manifest") or {})

    def stats(self) -> Dict[str, Any]:
        return dict(self._request("stats") or {})

    def search(self, entity_type: str, query: Any, top_k: int = 10) -> Dict[str, list[Dict[str, Any]]]:
        return dict(
            self._request(
                "search",
                {"entity_type": entity_type, "query": query, "top_k": max(1, int(top_k))},
            )
            or {}
        )

    def get_entities(self, entity_type: str, entity_ids: Sequence[str]) -> list[Dict[str, Any]]:
        return list(
            self._request(
                "get_entities",
                {"entity_type": entity_type, "entity_ids": list(entity_ids)},
            )
            or []
        )

    def neighbors(
        self,
        node_id: str,
        *,
        direction: str = "out",
        target_types: Iterable[str] | None = None,
        edge_files: Iterable[str] | None = None,
    ) -> list[Dict[str, str]]:
        return list(
            self._request(
                "neighbors",
                {
                    "node_id": node_id,
                    "direction": direction,
                    "target_types": list(target_types) if target_types else None,
                    "edge_files": list(edge_files) if edge_files else None,
                },
            )
            or []
        )


def main() -> None:
    from global_values import EMBEDDING_MODEL_NAME, KG_DIR

    parser = argparse.ArgumentParser(description="Inspect a running AgenticDS retrieval service.")
    parser.add_argument("command", choices=("health", "manifest", "stats"))
    parser.add_argument("--socket", required=True)
    parser.add_argument("--snapshot", default=KG_DIR)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME)
    args = parser.parse_args()
    client = RetrievalClient(
        args.socket,
        expected_snapshot=args.snapshot,
        expected_embedding_model=args.embedding_model,
    )
    result = getattr(client, args.command)()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
