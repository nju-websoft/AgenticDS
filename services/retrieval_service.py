from __future__ import annotations

import hashlib
import gc
import json
import os
import resource
import socket
import socketserver
import threading
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Mapping, Sequence

from utils.embedding_backend import LocalEmbeddingBackend
from utils.kg_find_neighbors import KGNeighborIndex
from utils.kg_hybrid_retriever import KGHybridRetriever
from utils.lucene_runtime import LUCENE_VERSION


ProgressCallback = Callable[[str, Mapping[str, Any]], None]


@contextmanager
def _index_loading_gc():
    """Collect the long-lived index heap once, rather than repeatedly while loading."""
    enabled = gc.isenabled()
    if enabled:
        gc.disable()
    try:
        yield
    finally:
        if enabled:
            try:
                gc.collect()
            finally:
                gc.enable()


def _rss_gib() -> float:
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024.0**3)
    except (OSError, ValueError, IndexError):
        return 0.0


def _peak_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


class RetrievalIndexGroup:
    """One immutable snapshot/model index group shared by all RPC requests."""

    def __init__(
        self,
        *,
        snapshot: str,
        embedding_model: str,
        entity_types: Sequence[str],
        cache_dir: str,
        query_encoder_device: str,
        progress: ProgressCallback | None = None,
    ) -> None:
        started = perf_counter()
        self.snapshot_path = Path(snapshot)
        self.snapshot_id = self.snapshot_path.name
        self.embedding_model = str(embedding_model)
        self.entity_types = tuple(dict.fromkeys(str(item).strip().lower() for item in entity_types if str(item).strip()))
        self.cache_dir = str(cache_dir)
        self.query_encoder_device = str(query_encoder_device)
        self.retrievers: Dict[str, KGHybridRetriever] = {}
        self.entity_lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._search_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._request_counts: Dict[str, int] = defaultdict(int)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._latency_totals: Dict[str, float] = defaultdict(float)
        self._search_stage_totals: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        with _index_loading_gc():
            load_order = [item for item in ("dataset", "paper", "collection", "model") if item in self.entity_types]
            for entity_type in load_order:
                self._notify(progress, f"loading_{entity_type}_index", {"entity_type": entity_type})
                retriever = KGHybridRetriever(
                    embedding_model=self.embedding_model,
                    cache_dir=self.cache_dir,
                    kg_dir=str(self.snapshot_path),
                    entity_types=[entity_type],
                    initialize_embedding_backend=False,
                )
                self.retrievers[entity_type] = retriever
                self.entity_lookup[entity_type] = {
                    str(entity.get("id")): entity for entity in retriever.entities if entity.get("id")
                }

            self._notify(progress, "loading_relation_adjacency", {})
            self.neighbor_index = KGNeighborIndex(kg_dir=str(self.snapshot_path))

            self._notify(progress, "loading_query_encoder", {"device": self.query_encoder_device})
            self.embedding_backend = LocalEmbeddingBackend(
                self.embedding_model,
                device=self.query_encoder_device,
            )
            for retriever in self.retrievers.values():
                retriever.set_embedding_backend(self.embedding_backend)
            self._notify(progress, "finalizing_index_memory", {})

        self.loaded_at = datetime.now(timezone.utc).isoformat()
        self.load_seconds = perf_counter() - started
        self._manifest = self._build_manifest()
        self._notify(progress, "ready", {"manifest": self._manifest})

    @staticmethod
    def _notify(progress: ProgressCallback | None, stage: str, detail: Mapping[str, Any]) -> None:
        print(f"[retrieval-service][stage] {stage} {json.dumps(dict(detail), ensure_ascii=False)}", flush=True)
        if progress is not None:
            progress(stage, detail)

    def _build_manifest(self) -> Dict[str, Any]:
        files: list[Dict[str, Any]] = []
        for entity_type, retriever in self.retrievers.items():
            for kind, path in (
                ("entities", retriever.files.entities),
                ("description_dense", retriever.files.description_index),
                ("metadata_dense", retriever.files.metadata_index),
                ("bm25", retriever.files.sparse_index),
            ):
                paths = sorted(child for child in path.iterdir() if child.is_file() and child.name != "write.lock") if path.is_dir() else [path]
                for actual_path in paths:
                    stat = actual_path.stat()
                    files.append({
                        "entity_type": entity_type, "kind": kind, "path": str(actual_path),
                        "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                    })
        fingerprint = hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "snapshot_id": self.snapshot_id,
            "snapshot_path": str(self.snapshot_path),
            "embedding_model": self.embedding_model,
            "sparse_backend": "lucene",
            "lucene_version": LUCENE_VERSION,
            "bm25": {key: retriever.sparse_index.metadata for key, retriever in self.retrievers.items()},
            "entity_types": list(self.entity_types),
            "entity_counts": {key: len(value) for key, value in self.entity_lookup.items()},
            "query_encoder": {
                "backend": self.embedding_backend.info.backend,
                "device": self.embedding_backend.info.device,
            },
            "index_fingerprint": fingerprint,
            "loaded_at": self.loaded_at,
            "load_seconds": self.load_seconds,
            "rss_gib": _rss_gib(),
            "peak_rss_gib": _peak_rss_gib(),
            "files": files,
        }

    def manifest(self) -> Dict[str, Any]:
        return dict(self._manifest)

    def validate_expected(self, expected: Mapping[str, Any]) -> None:
        snapshot_id = str(expected.get("snapshot_id") or "")
        embedding_model = str(expected.get("embedding_model") or "")
        if snapshot_id and snapshot_id != self.snapshot_id:
            raise ValueError(f"Snapshot mismatch: expected={snapshot_id} service={self.snapshot_id}")
        if embedding_model and embedding_model != self.embedding_model:
            raise ValueError(
                f"Embedding model mismatch: expected={embedding_model} service={self.embedding_model}"
            )

    def search(self, *, entity_type: str, query: Any, top_k: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if entity_type not in self.retrievers:
            raise ValueError(f"Entity type is not loaded: {entity_type}")
        started = perf_counter()
        with self._search_lock:
            result, timings = self.retrievers[entity_type].search_with_timings(query, top_k=max(1, int(top_k)))
        elapsed = perf_counter() - started
        with self._stats_lock:
            self._request_counts[f"search:{entity_type}"] += 1
            self._latency_totals[f"search:{entity_type}"] += elapsed
            for key, value in timings.items():
                self._search_stage_totals[entity_type][key] += float(value)
        return result, {"service_seconds": elapsed, **timings}

    def neighbors(self, **params: Any) -> tuple[list[Dict[str, str]], Dict[str, Any]]:
        started = perf_counter()
        result = self.neighbor_index.neighbors(
            str(params.get("node_id") or ""),
            direction=str(params.get("direction") or "out"),
            target_types=params.get("target_types"),
            edge_files=params.get("edge_files"),
        )
        elapsed = perf_counter() - started
        with self._stats_lock:
            self._request_counts["neighbors"] += 1
            self._latency_totals["neighbors"] += elapsed
        return result, {"service_seconds": elapsed}

    def get_entities(self, *, entity_type: str, entity_ids: Sequence[str]) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
        started = perf_counter()
        lookup = self.entity_lookup.get(entity_type)
        if lookup is None:
            raise ValueError(f"Entity type is not loaded: {entity_type}")
        result = [lookup[item] for item in dict.fromkeys(str(value) for value in entity_ids) if item in lookup]
        elapsed = perf_counter() - started
        with self._stats_lock:
            self._request_counts[f"get_entities:{entity_type}"] += 1
            self._latency_totals[f"get_entities:{entity_type}"] += elapsed
        return result, {"service_seconds": elapsed}

    def record_error(self, method: str) -> None:
        with self._stats_lock:
            self._error_counts[method] += 1

    def stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            counts = dict(self._request_counts)
            totals = dict(self._latency_totals)
            stages = {key: dict(value) for key, value in self._search_stage_totals.items()}
            errors = dict(self._error_counts)
        return {
            "request_counts": counts,
            "error_counts": errors,
            "latency_total_seconds": totals,
            "latency_mean_seconds": {
                key: totals[key] / counts[key] for key in totals if counts.get(key, 0)
            },
            "search_stage_total_seconds": stages,
            "rss_gib": _rss_gib(),
            "peak_rss_gib": _peak_rss_gib(),
        }


class RetrievalServiceState:
    def __init__(self) -> None:
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.status = "loading"
        self.stage = "starting"
        self.error = ""
        self.group: RetrievalIndexGroup | None = None
        self._lock = threading.Lock()

    def update_stage(self, stage: str) -> None:
        with self._lock:
            self.stage = stage

    def set_ready(self, group: RetrievalIndexGroup) -> None:
        with self._lock:
            self.group = group
            self.stage = "ready"
            self.status = "ready"

    def set_failed(self, error: BaseException) -> None:
        with self._lock:
            self.stage = "failed"
            self.status = "failed"
            self.error = f"{type(error).__name__}: {error}"

    def health(self) -> Dict[str, Any]:
        with self._lock:
            payload = {
                "status": self.status,
                "stage": self.stage,
                "error": self.error,
                "pid": os.getpid(),
                "started_at": self.started_at,
                "rss_gib": _rss_gib(),
                "peak_rss_gib": _peak_rss_gib(),
            }
            if self.group is not None:
                payload["manifest"] = self.group.manifest()
            return payload


class RetrievalRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        method = "unknown"
        try:
            raw = self.rfile.readline()
            if not raw:
                return
            request = json.loads(raw.decode("utf-8"))
            method = str(request.get("method") or "")
            params = request.get("params") or {}
            state: RetrievalServiceState = self.server.state  # type: ignore[attr-defined]
            if method == "health":
                self._write({"ok": True, "result": state.health()})
                return
            if state.status != "ready" or state.group is None:
                raise RuntimeError(f"Retrieval service is not ready: status={state.status} stage={state.stage}")
            group = state.group
            group.validate_expected(request.get("expected") or {})
            if method == "manifest":
                result, meta = group.manifest(), {}
            elif method == "stats":
                result, meta = group.stats(), {}
            elif method == "search":
                result, meta = group.search(
                    entity_type=str(params.get("entity_type") or "dataset"),
                    query=params.get("query"),
                    top_k=int(params.get("top_k", 10)),
                )
            elif method == "neighbors":
                result, meta = group.neighbors(**params)
            elif method == "get_entities":
                result, meta = group.get_entities(
                    entity_type=str(params.get("entity_type") or "dataset"),
                    entity_ids=params.get("entity_ids") or [],
                )
            else:
                raise ValueError(f"Unknown retrieval method: {method}")
            self._write({"ok": True, "result": result, "meta": meta})
        except (BrokenPipeError, ConnectionResetError):
            # Health probes may time out while index loading holds the GIL. The
            # client has already gone away, so there is no response to deliver.
            return
        except Exception as exc:
            state = getattr(self.server, "state", None)
            if state is not None and state.group is not None:
                state.group.record_error(method)
            try:
                self._write({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            except (BrokenPipeError, ConnectionResetError):
                return

    def _write(self, payload: Mapping[str, Any]) -> None:
        self.wfile.write(json.dumps(dict(payload), ensure_ascii=False).encode("utf-8") + b"\n")


class ThreadingUnixRetrievalServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, socket_path: str, state: RetrievalServiceState) -> None:
        self.state = state
        super().__init__(socket_path, RetrievalRequestHandler)


def socket_is_live(socket_path: str, timeout_seconds: float = 1.0) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_seconds)
            sock.connect(socket_path)
            sock.sendall(b'{"method":"health","params":{}}\n')
            return bool(sock.recv(1))
    except OSError:
        return False
