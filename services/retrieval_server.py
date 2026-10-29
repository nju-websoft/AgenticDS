from __future__ import annotations

import argparse
import json
import os
import signal
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from global_values import INDEX_CACHE_DIR, KG_DIR, LOCAL_EMBEDDING_DEVICE
from services.retrieval_service import (
    RetrievalIndexGroup,
    RetrievalServiceState,
    ThreadingUnixRetrievalServer,
    socket_is_live,
)


def _write_json(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve one immutable AgenticDS retrieval index group over a Unix socket.")
    parser.add_argument("--snapshot", default=KG_DIR)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--entity-types", default="dataset,paper,collection,model")
    parser.add_argument("--cache-dir", default=INDEX_CACHE_DIR)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--query-encoder-device", default=LOCAL_EMBEDDING_DEVICE)
    parser.add_argument("--progress-file", type=Path)
    args = parser.parse_args()

    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        if socket_is_live(str(socket_path)):
            raise RuntimeError(f"A retrieval service is already listening on {socket_path}")
        socket_path.unlink()

    state = RetrievalServiceState()

    def report(stage: str, detail: Mapping[str, Any]) -> None:
        state.update_stage(stage)
        _write_json(
            args.progress_file,
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "status": "ready" if stage == "ready" else "loading",
                "stage": stage,
                "pid": os.getpid(),
                "socket": str(socket_path),
                **dict(detail),
            },
        )

    server = ThreadingUnixRetrievalServer(str(socket_path), state)
    os.chmod(socket_path, 0o660)

    def load() -> None:
        try:
            group = RetrievalIndexGroup(
                snapshot=args.snapshot,
                embedding_model=args.embedding_model,
                entity_types=[item.strip() for item in args.entity_types.split(",") if item.strip()],
                cache_dir=args.cache_dir,
                query_encoder_device=args.query_encoder_device,
                progress=report,
            )
            state.set_ready(group)
        except BaseException as exc:
            state.set_failed(exc)
            _write_json(
                args.progress_file,
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                    "stage": "failed",
                    "pid": os.getpid(),
                    "socket": str(socket_path),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            print(f"[retrieval-service][failed] {type(exc).__name__}: {exc}", flush=True)

    loader = threading.Thread(target=load, name="retrieval-index-loader", daemon=True)
    loader.start()

    def stop(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"[retrieval-service] listening socket={socket_path} pid={os.getpid()}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        if socket_path.exists():
            socket_path.unlink()


if __name__ == "__main__":
    main()
