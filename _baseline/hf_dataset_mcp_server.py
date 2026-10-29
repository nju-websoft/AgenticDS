#!/usr/bin/env python3
"""Minimal stdio MCP server exposing Hugging Face dataset discovery only."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


SERVER_NAME = "agenticds-huggingface-datasets"
SERVER_VERSION = "1.0.0"
HF_API_BASE = "https://huggingface.co/api/datasets"
HF_DATASET_ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
USER_AGENT = "AgenticDS-Agent-Harness/1.0"
MAX_REQUEST_ATTEMPTS = 4
RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}
EVIDENCE_FILE_ENV = "AGENTICDS_HF_EVIDENCE_FILE"


TOOLS = [
    {
        "name": "search_hf_datasets",
        "description": (
            "Search the live Hugging Face dataset catalog. Use several focused searches when "
            "the request contains task, language, format, license, or training-data constraints."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
            },
        },
    },
    {
        "name": "get_hf_dataset",
        "description": (
            "Read live Hugging Face metadata and the dataset-card README for one canonical "
            "owner/name dataset ID. Use this to verify candidates before ranking them."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["dataset_id"],
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "pattern": r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$",
                },
                "max_readme_chars": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 30000,
                    "default": 12000,
                },
            },
        },
    },
]


def _retry_delay(attempt: int, retry_after: str = "") -> float:
    try:
        if retry_after:
            return min(30.0, max(0.0, float(retry_after)))
    except ValueError:
        pass
    return min(8.0, 0.5 * (2 ** max(0, attempt - 1))) + random.uniform(0.0, 0.25)


def _request_text(url: str, *, timeout: int = 30) -> str:
    last_error: Exception | None = None
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        request = Request(url, headers={"Accept": "application/json, text/plain", "User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            last_error = exc
            if exc.code not in RETRYABLE_HTTP_STATUS or attempt >= MAX_REQUEST_ATTEMPTS:
                raise
            time.sleep(_retry_delay(attempt, str(exc.headers.get("Retry-After") or "")))
        except (URLError, TimeoutError) as exc:
            last_error = exc
            if attempt >= MAX_REQUEST_ATTEMPTS:
                raise
            time.sleep(_retry_delay(attempt))
    if last_error is not None:
        raise last_error
    raise RuntimeError("Hugging Face request failed without an exception")


def _request_json(url: str) -> Any:
    return json.loads(_request_text(url))


def _compact_dataset(item: Mapping[str, Any]) -> Dict[str, Any]:
    card_data = item.get("cardData") if isinstance(item.get("cardData"), Mapping) else {}
    return {
        "id": item.get("id"),
        "author": item.get("author"),
        "downloads": item.get("downloads"),
        "likes": item.get("likes"),
        "last_modified": item.get("lastModified"),
        "private": item.get("private"),
        "gated": item.get("gated"),
        "disabled": item.get("disabled"),
        "tags": item.get("tags") or [],
        "description": item.get("description") or card_data.get("description"),
        "language": card_data.get("language"),
        "license": card_data.get("license"),
        "task_categories": card_data.get("task_categories"),
        "size_categories": card_data.get("size_categories"),
        "page": f"https://huggingface.co/datasets/{item.get('id')}",
    }


def _compact_search_dataset(item: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep broad discovery readable; full metadata remains available via get."""
    card_data = item.get("cardData") if isinstance(item.get("cardData"), Mapping) else {}
    raw_tags = [str(tag) for tag in (item.get("tags") or []) if str(tag).strip()]
    informative_prefixes = (
        "task_categories:",
        "language:",
        "license:",
        "size_categories:",
        "format:",
        "modality:",
    )
    informative_tags = [tag for tag in raw_tags if tag.startswith(informative_prefixes)][:10]
    description = str(item.get("description") or card_data.get("description") or "")
    description = " ".join(description.split())
    return {
        "id": item.get("id"),
        "downloads": item.get("downloads"),
        "likes": item.get("likes"),
        "gated": item.get("gated"),
        "tags": informative_tags,
        "description_excerpt": description[:240] or None,
    }


def _record_dataset_evidence(
    *,
    operation: str,
    request: str,
    dataset_ids: list[str],
    evidence: Any = None,
) -> Dict[str, int] | None:
    """Persist the catalog IDs actually observed by an agent-harness tool call."""
    evidence_file = str(os.getenv(EVIDENCE_FILE_ENV, "")).strip()
    if not evidence_file:
        return None
    normalized = [
        str(dataset_id).strip()
        for dataset_id in dataset_ids
        if HF_DATASET_ID_RE.fullmatch(str(dataset_id).strip())
    ]
    path = os.path.abspath(os.path.expanduser(evidence_file))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        observed: set[str] = set()
        for line in handle:
            try:
                existing = json.loads(line)
            except json.JSONDecodeError:
                continue
            for dataset_id in existing.get("dataset_ids", []) if isinstance(existing, Mapping) else []:
                value = str(dataset_id or "").strip()
                if HF_DATASET_ID_RE.fullmatch(value):
                    observed.add(value.casefold())
        normalized_keys = {dataset_id.casefold() for dataset_id in normalized}
        new_count = len(normalized_keys - observed)
        observed.update(normalized_keys)
        state = {
            "new_dataset_count": new_count,
            "observed_unique_dataset_count": len(observed),
        }
        record = {
            "operation": operation,
            "request": request,
            "dataset_ids": normalized,
            "evidence": evidence,
            **state,
        }
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return state


def search_hf_datasets(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    query = str(arguments.get("query") or "").strip()
    if not query:
        raise ValueError("query must not be empty")
    limit = min(50, max(1, int(arguments.get("limit", 20))))
    url = f"{HF_API_BASE}?{urlencode({'search': query, 'limit': limit, 'full': 'true'})}"
    payload = _request_json(url)
    if not isinstance(payload, list):
        raise ValueError("Hugging Face returned an unexpected search response")
    datasets = [_compact_search_dataset(item) for item in payload if isinstance(item, Mapping)]
    dataset_ids = [
        str(item.get("id") or "")
        for item in datasets
        if HF_DATASET_ID_RE.fullmatch(str(item.get("id") or ""))
    ]
    result = {
        "query": query,
        "result_count": len(payload),
        # Put the complete ID list before optional ranking metadata so terminal
        # truncation cannot hide the admissible candidate set from the agent.
        "dataset_ids": dataset_ids,
    }
    result["datasets"] = datasets
    _record_dataset_evidence(
        operation="search",
        request=query,
        dataset_ids=dataset_ids,
        evidence={"datasets": datasets},
    )
    return result


def get_hf_dataset(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    dataset_id = str(arguments.get("dataset_id") or "").strip().strip("/")
    if not HF_DATASET_ID_RE.fullmatch(dataset_id):
        raise ValueError("dataset_id must be a canonical owner/name Hugging Face dataset ID")
    max_readme_chars = min(30000, max(0, int(arguments.get("max_readme_chars", 12000))))
    encoded_id = quote(dataset_id, safe="/")
    metadata = _request_json(f"{HF_API_BASE}/{encoded_id}")
    if not isinstance(metadata, Mapping):
        raise ValueError("Hugging Face returned unexpected dataset metadata")

    readme = ""
    readme_error = ""
    if max_readme_chars:
        try:
            readme = _request_text(
                f"https://huggingface.co/datasets/{encoded_id}/raw/main/README.md"
            )[:max_readme_chars]
        except (HTTPError, URLError, TimeoutError) as exc:
            readme_error = f"{type(exc).__name__}: {exc}"

    result = _compact_dataset(metadata)
    result.update(
        {
            "siblings": [
                sibling.get("rfilename")
                for sibling in metadata.get("siblings", [])
                if isinstance(sibling, Mapping) and sibling.get("rfilename")
            ][:200],
            "dataset_card_readme": readme,
            "dataset_card_truncated": bool(readme) and len(readme) >= max_readme_chars,
            "dataset_card_error": readme_error or None,
        }
    )
    _record_dataset_evidence(
        operation="get",
        request=dataset_id,
        dataset_ids=[str(result.get("id") or "")],
        evidence=result,
    )
    return result


def _tool_result(payload: Any, *, is_error: bool = False) -> Dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _handle_request(message: Mapping[str, Any]) -> Dict[str, Any] | None:
    request_id = message.get("id")
    method = message.get("method")
    if request_id is None:
        return None
    if method == "initialize":
        params = message.get("params") if isinstance(message.get("params"), Mapping) else {}
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": params.get("protocolVersion", "2025-03-26"),
                "instructions": (
                    "For Hugging Face dataset retrieval, call search_hf_datasets directly to "
                    "discover canonical dataset IDs, then call get_hf_dataset directly when "
                    "metadata or dataset-card evidence can change inclusion or ranking. These "
                    "are tools, not MCP resources or resource templates."
                ),
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                    "prompts": {"listChanged": False},
                },
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
    if method == "resources/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"resources": []}}
    if method == "resources/templates/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"resourceTemplates": []}}
    if method == "prompts/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"prompts": []}}
    if method == "tools/call":
        params = message.get("params") if isinstance(message.get("params"), Mapping) else {}
        name = params.get("name")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), Mapping) else {}
        try:
            if name == "search_hf_datasets":
                result = search_hf_datasets(arguments)
            elif name == "get_hf_dataset":
                result = get_hf_dataset(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
            payload = _tool_result(result)
        except Exception as exc:
            payload = _tool_result(f"{type(exc).__name__}: {exc}", is_error=True)
        return {"jsonrpc": "2.0", "id": request_id, "result": payload}
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def _stdio_main() -> None:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
            response = _handle_request(message)
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            sys.stdout.flush()


def _cli_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Read-only Hugging Face dataset discovery tool.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--limit", type=int, default=20)
    get_parser = subparsers.add_parser("get")
    get_parser.add_argument("--dataset-id", required=True)
    get_parser.add_argument("--max-readme-chars", type=int, default=12000)
    args = parser.parse_args(argv)
    if args.command == "search":
        result = search_hf_datasets({"query": args.query, "limit": args.limit})
    else:
        result = get_hf_dataset(
            {
                "dataset_id": args.dataset_id,
                "max_readme_chars": args.max_readme_chars,
            }
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    if len(sys.argv) > 1:
        _cli_main(sys.argv[1:])
    else:
        _stdio_main()


if __name__ == "__main__":
    main()
