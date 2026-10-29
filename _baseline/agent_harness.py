from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import unquote, urlparse


DEFAULT_CLAUDE_CODE_COMMAND = (
    'claude --print --verbose --output-format stream-json '
    '--mcp-config {mcp_config_file} --strict-mcp-config --tools "" '
    '--allowedTools "mcp__huggingface__search_hf_datasets,mcp__huggingface__get_hf_dataset" '
    '--json-schema {schema_json} '
    '--model {model}'
)
DEFAULT_CODEX_COMMAND = (
    "codex -c 'model_provider=\"{provider_id}\"' "
    "-c 'model_providers.{provider_id}.name=\"API provider\"' "
    "-c 'model_providers.{provider_id}.base_url=\"{api_base_v1}\"' "
    "-c 'model_providers.{provider_id}.env_key=\"{api_key_env}\"' "
    "-c 'model_providers.{provider_id}.wire_api=\"responses\"' "
    "-c 'features.shell_tool=false' "
    "-c 'features.unified_exec=false' "
    "-c 'web_search=\"disabled\"' "
    "-c 'tools.view_image=false' "
    "-c 'mcp_servers.huggingface.command=\"{python_executable}\"' "
    "-c 'mcp_servers.huggingface.args=[\"{mcp_server}\"]' "
    "-c 'mcp_servers.huggingface.env_vars=[\"http_proxy\",\"https_proxy\",\"HTTP_PROXY\",\"HTTPS_PROXY\",\"NO_PROXY\",\"no_proxy\",\"AGENTICDS_HF_EVIDENCE_FILE\"]' "
    "-c 'mcp_servers.huggingface.required=true' "
    "-c 'mcp_servers.huggingface.enabled_tools=[\"search_hf_datasets\",\"get_hf_dataset\"]' "
    "-c 'mcp_servers.huggingface.omit_tools_from=[\"deferred\",\"code_mode\"]' "
    "-c 'mcp_servers.huggingface.default_tools_approval_mode=\"approve\"' "
    "--disable remote_models --disable plugins --disable remote_plugin --disable plugin_sharing "
    "--disable recommended_plugins --disable apps "
    "-a never exec --ephemeral --skip-git-repo-check --ignore-user-config --ignore-rules "
    "--sandbox danger-full-access --json --output-schema {schema_file} "
    "--output-last-message {response_file} --model {model} -"
)
DEFAULT_CLAUDE_CODE_FINALIZATION_COMMAND = (
    'claude --print --verbose --output-format stream-json --tools "" '
    '--json-schema {schema_json} --model {model}'
)
DEFAULT_CODEX_FINALIZATION_COMMAND = (
    "codex -c 'model_provider=\"{provider_id}\"' "
    "-c 'model_providers.{provider_id}.name=\"API provider\"' "
    "-c 'model_providers.{provider_id}.base_url=\"{api_base_v1}\"' "
    "-c 'model_providers.{provider_id}.env_key=\"{api_key_env}\"' "
    "-c 'model_providers.{provider_id}.wire_api=\"responses\"' "
    "-c 'features.shell_tool=false' -c 'features.unified_exec=false' "
    "-c 'web_search=\"disabled\"' -c 'tools.view_image=false' "
    "--disable remote_models --disable plugins --disable remote_plugin --disable plugin_sharing "
    "--disable recommended_plugins --disable apps "
    "-a never exec --ephemeral --skip-git-repo-check --ignore-user-config --ignore-rules "
    "--sandbox danger-full-access --json --output-schema {schema_file} "
    "--output-last-message {response_file} --model {model} -"
)
HF_DATASET_ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
HF_DATASET_ID_PLACEHOLDERS = {
    "owner/dataset-id",
    "owner/dataset-name",
    "organization/dataset-name",
}
HF_EVIDENCE_FILE_ENV = "AGENTICDS_HF_EVIDENCE_FILE"


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _resolve_command_executable(config: "HarnessConfig", executable: str) -> str:
    """Resolve harness CLIs in non-interactive services with a minimal PATH."""
    configured = str(executable or "").strip()
    if not configured:
        return ""

    if os.path.sep in configured:
        candidate = Path(configured).expanduser()
        return str(candidate.resolve()) if _is_executable_file(candidate) else ""

    if config.harness == "codex" and configured == "codex":
        explicit = str(os.getenv("AGENTICDS_CODEX_EXECUTABLE", "")).strip()
        if explicit:
            candidate = Path(explicit).expanduser()
            return str(candidate.resolve()) if _is_executable_file(candidate) else ""

    discovered = shutil.which(configured)
    if discovered:
        return discovered

    if config.harness != "codex" or configured != "codex":
        return ""

    candidates = [
        Path.home() / ".local" / "bin" / "codex",
        Path.home() / ".npm-global" / "bin" / "codex",
    ]
    for candidate in candidates:
        if _is_executable_file(candidate):
            return str(candidate.resolve())
    return ""


@dataclass(frozen=True)
class HarnessConfig:
    harness: str
    model: str
    command_template: str
    candidate_count: int = 0
    candidate_multiplier: float = 1.5
    timeout_seconds: int = 1800
    api_base_url: str = ""
    api_key_env: str = "PARATERA_API_KEY"
    provider_id: str = "paratera"

    def __post_init__(self) -> None:
        if self.harness not in {"claude_code", "codex"}:
            raise ValueError(f"Unsupported harness: {self.harness}")
        if not self.model.strip():
            raise ValueError("Harness model must not be empty.")
        if not self.command_template.strip():
            raise ValueError("Harness command template must not be empty.")
        if self.candidate_count < 0:
            raise ValueError("candidate_count must not be negative.")
        if self.candidate_multiplier < 1.0:
            raise ValueError("candidate_multiplier must be at least 1.0.")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive.")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.provider_id):
            raise ValueError("provider_id must contain only letters, digits, underscores, or hyphens.")


def _project_llm_setting(name: str, default: str = "") -> str:
    try:
        import global_values

        return str(getattr(global_values, name, default) or default).strip()
    except (ImportError, AttributeError):
        return default


def default_harness_config(harness: str, *, model: Optional[str] = None) -> HarnessConfig:
    normalized = str(harness or "").strip().lower()
    selected_model = str(model or os.getenv("AGENTICDS_HARNESS_MODEL", "")).strip()
    if not selected_model:
        raise ValueError(
            "Harness model is not configured. Pass a model explicitly or set AGENTICDS_HARNESS_MODEL."
        )
    if normalized == "claude_code":
        command = os.getenv("AGENTICDS_CLAUDE_CODE_COMMAND", DEFAULT_CLAUDE_CODE_COMMAND)
    elif normalized == "codex":
        command = os.getenv("AGENTICDS_CODEX_COMMAND", DEFAULT_CODEX_COMMAND)
    else:
        raise ValueError(f"Unsupported harness: {harness}")
    return HarnessConfig(
        harness=normalized,
        model=selected_model,
        command_template=command,
        candidate_count=int(os.getenv("AGENTICDS_HARNESS_CANDIDATE_COUNT", "0")),
        candidate_multiplier=float(os.getenv("AGENTICDS_HARNESS_CANDIDATE_MULTIPLIER", "1.5")),
        timeout_seconds=int(os.getenv("AGENTICDS_HARNESS_TIMEOUT_SECONDS", "1800")),
        api_base_url=str(
            os.getenv(
                "AGENTICDS_HARNESS_API_BASE_URL",
                _project_llm_setting("LLM_API_BASE_URL"),
            )
        ).strip(),
        api_key_env=str(os.getenv("AGENTICDS_HARNESS_API_KEY_ENV", "PARATERA_API_KEY")).strip(),
        provider_id=str(os.getenv("AGENTICDS_HARNESS_PROVIDER_ID", "paratera")).strip(),
    )


def _strip_code_fence(text: str) -> str:
    value = str(text or "").strip()
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _decode_json_objects(text: str) -> Iterable[Any]:
    value = _strip_code_fence(text)
    if not value:
        return
    try:
        yield json.loads(value)
        return
    except json.JSONDecodeError:
        pass

    decoded_line = False
    for line in value.splitlines():
        line = _strip_code_fence(line)
        if not line:
            continue
        try:
            yield json.loads(line)
            decoded_line = True
        except json.JSONDecodeError:
            continue
    if decoded_line:
        return

    decoder = json.JSONDecoder()
    for index, char in enumerate(value):
        if char not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        yield payload


def _find_dataset_values(payload: Any) -> Optional[List[Any]]:
    if isinstance(payload, dict):
        for key in ("dataset_ids", "candidates"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        preferred_keys = ("result", "output", "text", "message", "content", "item")
        values = [payload[key] for key in preferred_keys if key in payload]
        values.extend(value for key, value in payload.items() if key not in preferred_keys)
        for value in values:
            nested = _find_dataset_values(value)
            if nested is not None:
                return nested
    elif isinstance(payload, list):
        for value in reversed(payload):
            nested = _find_dataset_values(value)
            if nested is not None:
                return nested
    elif isinstance(payload, str):
        for decoded in _decode_json_objects(payload):
            if decoded == payload:
                continue
            nested = _find_dataset_values(decoded)
            if nested is not None:
                return nested
    return None


def normalize_hf_dataset_id(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("dataset_id") or value.get("id") or value.get("url") or ""
    text = unquote(str(value or "").strip()).strip("` ")
    if not text:
        return ""
    if text.startswith(("http://", "https://")):
        parsed = urlparse(text)
        if parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
            return ""
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 3 or parts[0].lower() != "datasets":
            return ""
        text = f"{parts[1]}/{parts[2]}"
    text = text.strip().strip("/")
    if not HF_DATASET_ID_RE.fullmatch(text):
        return ""
    if text.casefold() in HF_DATASET_ID_PLACEHOLDERS:
        return ""
    return text


def load_hf_evidence_dataset_ids(path: Path) -> List[str]:
    evidence_ids: List[str] = []
    seen: set[str] = set()
    if not path.exists():
        return evidence_ids
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        values = record.get("dataset_ids") if isinstance(record, Mapping) else None
        if not isinstance(values, list):
            continue
        for value in values:
            dataset_id = normalize_hf_dataset_id(value)
            key = dataset_id.casefold()
            if not dataset_id or key in seen:
                continue
            seen.add(key)
            evidence_ids.append(dataset_id)
    return evidence_ids


def load_hf_evidence_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def ground_dataset_ids(
    submitted_ids: Sequence[str],
    *,
    evidence_ids: Sequence[str],
) -> Dict[str, List[str]]:
    evidence_map = {
        normalized.casefold(): normalized
        for value in evidence_ids
        if (normalized := normalize_hf_dataset_id(value))
    }
    grounded: List[str] = []
    rejected: List[str] = []
    seen: set[str] = set()
    for value in submitted_ids:
        dataset_id = normalize_hf_dataset_id(value)
        key = dataset_id.casefold()
        if not dataset_id or key in seen:
            continue
        seen.add(key)
        canonical = evidence_map.get(key)
        if canonical is None:
            rejected.append(dataset_id)
        else:
            grounded.append(canonical)
    return {"grounded_dataset_ids": grounded, "rejected_dataset_ids": rejected}


def complete_dataset_ids_from_evidence(
    selected_ids: Sequence[str],
    *,
    evidence_ids: Sequence[str],
    candidate_count: int,
) -> Dict[str, List[str]]:
    """Fill duplicate/short structured output only with this run's MCP evidence."""
    completed: List[str] = []
    seen: set[str] = set()
    for value in [*selected_ids, *evidence_ids]:
        dataset_id = normalize_hf_dataset_id(value)
        key = dataset_id.casefold()
        if not dataset_id or key in seen:
            continue
        seen.add(key)
        completed.append(dataset_id)
        if len(completed) >= candidate_count:
            break
    selected_keys = {
        normalized.casefold()
        for value in selected_ids
        if (normalized := normalize_hf_dataset_id(value))
    }
    appended = [value for value in completed if value.casefold() not in selected_keys]
    return {"dataset_ids": completed, "appended_evidence_dataset_ids": appended}


def extract_dataset_ids(output: str, *, max_candidates: int) -> List[str]:
    requested: Optional[List[Any]] = None
    for payload in reversed(list(_decode_json_objects(output))):
        requested = _find_dataset_values(payload)
        if requested is not None:
            break

    selected: List[str] = []
    seen: set[str] = set()
    for value in requested or []:
        dataset_id = normalize_hf_dataset_id(value)
        key = dataset_id.casefold()
        if not dataset_id or key in seen:
            continue
        seen.add(key)
        selected.append(dataset_id)
        if len(selected) >= max(1, int(max_candidates)):
            break
    return selected


def extract_final_assistant_output(output: str, *, harness: str) -> str:
    """Extract only the terminal assistant answer, never an MCP/tool payload."""
    final_results: List[str] = []
    assistant_messages: List[str] = []
    for line in str(output or "").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        if harness == "claude_code":
            if event.get("type") == "result" and isinstance(event.get("result"), str):
                final_results.append(str(event["result"]))
                continue
            if event.get("type") != "assistant":
                continue
            message = event.get("message")
            content = message.get("content") if isinstance(message, Mapping) else None
            if isinstance(content, list):
                texts = [
                    str(block.get("text"))
                    for block in content
                    if isinstance(block, Mapping)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                ]
                if texts:
                    assistant_messages.append("\n".join(texts))
        elif harness == "codex":
            if event.get("type") != "item.completed":
                continue
            item = event.get("item")
            if (
                isinstance(item, Mapping)
                and item.get("type") == "agent_message"
                and isinstance(item.get("text"), str)
            ):
                assistant_messages.append(str(item["text"]))
        else:
            raise ValueError(f"Unsupported harness: {harness}")
    if final_results:
        return final_results[-1]
    return assistant_messages[-1] if assistant_messages else ""


def extract_harness_process_error(*, stdout: str, stderr: str) -> str:
    """Extract a CLI's terminal failure without confusing tool evidence for output."""
    structured_errors: List[str] = []
    for line in str(stdout or "").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        event_type = str(event.get("type") or "")
        if event_type == "error" and str(event.get("message") or "").strip():
            structured_errors.append(str(event["message"]).strip())
        elif event_type == "turn.failed":
            error = event.get("error")
            if isinstance(error, Mapping) and str(error.get("message") or "").strip():
                structured_errors.append(str(error["message"]).strip())
    if structured_errors:
        return structured_errors[-1]
    stderr_text = " ".join(str(stderr or "").strip().split())
    if stderr_text:
        return stderr_text
    return " ".join(str(stdout or "").strip().split())[-1000:]


def filter_dataset_ids_to_snapshot(
    dataset_ids: Sequence[str],
    *,
    snapshot_dataset_ids: Sequence[str],
    top_k: int,
) -> Dict[str, Any]:
    snapshot_map = {
        str(dataset_id).strip().casefold(): str(dataset_id).strip()
        for dataset_id in snapshot_dataset_ids
        if str(dataset_id).strip()
    }
    selected: List[str] = []
    in_snapshot: List[str] = []
    outside_snapshot: List[str] = []
    duplicate_ids: List[str] = []
    seen: set[str] = set()
    for raw_id in dataset_ids:
        dataset_id = normalize_hf_dataset_id(raw_id)
        key = dataset_id.casefold()
        if not dataset_id:
            continue
        if key in seen:
            duplicate_ids.append(dataset_id)
            continue
        seen.add(key)
        canonical = snapshot_map.get(key)
        if canonical is None:
            outside_snapshot.append(dataset_id)
            continue
        in_snapshot.append(canonical)
        if len(selected) < max(1, int(top_k)):
            selected.append(canonical)
    return {
        "raw_candidate_count": len(dataset_ids),
        "snapshot_candidate_count": len(in_snapshot),
        "in_snapshot_dataset_ids": in_snapshot,
        "selected_candidate_count": len(selected),
        "selected_dataset_ids": selected,
        "outside_snapshot_dataset_ids": outside_snapshot,
        "duplicate_dataset_ids": duplicate_ids,
        "requested_top_k": max(1, int(top_k)),
        "top_k_satisfied": len(selected) >= max(1, int(top_k)),
    }


def _output_schema(*, candidate_count: int, exact: bool = True) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["dataset_ids"],
        "properties": {
            "dataset_ids": {
                "type": "array",
                "minItems": max(1, int(candidate_count)) if exact else 1,
                "maxItems": max(1, int(candidate_count)),
                "items": {"type": "string"},
            }
        },
    }


def _task_prompt(
    query: str,
    *,
    candidate_count: int,
    harness: str,
    hf_tool_path: Path,
) -> str:
    del harness, hf_tool_path
    return f"""Act as a general-purpose coding-agent harness performing dataset retrieval.

Find and rank Hugging Face datasets for the natural-language query below. Use the provided Hugging Face MCP tools: call search_hf_datasets to search the live catalog, and call get_hf_dataset to inspect promising dataset metadata and cards. Choose the number of searches and inspections needed to resolve the query, then return the ranked final JSON.

Build an evidence-backed candidate pool before final ranking. First use search_hf_datasets until at least {candidate_count} distinct repository IDs have been observed. If focused searches stop adding candidates, progressively relax non-mandatory keywords while preserving explicit hard constraints; a broader but lower-ranked real candidate is preferable to repeating an exhausted search. Use get_hf_dataset selectively after the pool is large enough to distinguish and rank the strongest candidates.

Restrict discovery and evidence to huggingface.co by using only the specified Hugging Face tools, and return only repositories whose canonical page is https://huggingface.co/datasets/<owner>/<name>. Do not use this local repository, its files, a precomputed candidate list, or prior knowledge as a substitute for live Hugging Face search.

Every returned dataset ID must have appeared verbatim in an MCP tool result from this run. Do not infer repository names, complete naming patterns, or invent plausible owner/name IDs. You must call the Hugging Face tools before answering, but you remain free to choose how many searches and inspections are useful.

Rank by satisfaction of the query. Treat task, data format, language, license, size, quality, and train-versus-benchmark intent as constraints when the query mentions them. Verify repository IDs before returning them.

Query:
{query}

Return exactly {candidate_count} distinct dataset IDs in relevance order. Output only this JSON object, with no markdown or commentary:
{{"dataset_ids": ["owner/dataset-name", "..."]}}
"""


def _finalization_prompt(
    query: str,
    *,
    candidate_count: int,
    evidence_records: Sequence[Mapping[str, Any]],
) -> str:
    evidence_ids: List[str] = []
    seen: set[str] = set()
    search_provenance: List[Dict[str, Any]] = []
    metadata_by_id: Dict[str, Dict[str, Any]] = {}
    for record in evidence_records:
        ids: List[str] = []
        for value in record.get("dataset_ids", []) if isinstance(record.get("dataset_ids"), list) else []:
            dataset_id = normalize_hf_dataset_id(value)
            key = dataset_id.casefold()
            if dataset_id and key not in seen:
                seen.add(key)
                evidence_ids.append(dataset_id)
            if dataset_id:
                ids.append(dataset_id)
        operation = str(record.get("operation") or "")
        if operation == "search":
            search_provenance.append({"query": record.get("request"), "dataset_ids": ids})
        evidence = record.get("evidence")
        if isinstance(evidence, Mapping):
            if operation == "get":
                detail = {
                    key: evidence.get(key)
                    for key in (
                        "id", "downloads", "likes", "tags", "description", "language",
                        "license", "task_categories", "size_categories",
                    )
                    if evidence.get(key) not in (None, "", [], {})
                }
                dataset_id = normalize_hf_dataset_id(detail.get("id"))
                if dataset_id:
                    metadata_by_id[dataset_id.casefold()] = detail
            elif isinstance(evidence.get("datasets"), list):
                for item in evidence["datasets"]:
                    if not isinstance(item, Mapping):
                        continue
                    dataset_id = normalize_hf_dataset_id(item.get("id"))
                    if dataset_id and dataset_id.casefold() not in metadata_by_id:
                        metadata_by_id[dataset_id.casefold()] = dict(item)

    evidence_text = json.dumps(
        {
            "searches": search_provenance,
            "candidate_metadata": [
                metadata_by_id[key]
                for key in (dataset_id.casefold() for dataset_id in evidence_ids)
                if key in metadata_by_id
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"""Complete the dataset-retrieval task using only the persisted Hugging Face tool evidence below.

Original query:
{query}

Rank the strongest supported candidates by satisfaction of the query. Every returned ID must be selected from the admissible ID list. Do not invent IDs and do not describe additional searches. Return up to {candidate_count} distinct IDs; return exactly {candidate_count} whenever at least that many admissible IDs exist. Output only the JSON object.

Admissible dataset IDs:
{json.dumps(evidence_ids, ensure_ascii=False)}

Persisted tool evidence:
{evidence_text}

Required output:
{{"dataset_ids": ["owner/dataset-name", "..."]}}
"""


def _api_v1_base_url(base_url: str) -> str:
    value = str(base_url or "").strip().rstrip("/")
    if value.endswith("/v1"):
        return value
    return f"{value}/v1" if value else ""


def _build_command(config: HarnessConfig, paths: Mapping[str, Path]) -> List[str]:
    replacements = {
        "model": config.model,
        "workspace": str(paths["workspace"]),
        "schema_file": str(paths["schema_file"]),
        "response_file": str(paths["response_file"]),
        "mcp_config_file": str(paths["mcp_config_file"]),
        "mcp_server": str(paths["mcp_server"]),
        "python_executable": str(paths["python_executable"]),
        "api_base_url": config.api_base_url.rstrip("/"),
        "api_base_v1": _api_v1_base_url(config.api_base_url),
        "api_key_env": config.api_key_env,
        "provider_id": config.provider_id,
        "schema_json": (
            paths["schema_file"].read_text(encoding="utf-8")
            if "{schema_json}" in config.command_template
            else ""
        ),
    }
    command: List[str] = []
    for argument in shlex.split(config.command_template):
        try:
            command.append(argument.format_map(replacements))
        except KeyError as exc:
            raise ValueError(f"Unknown harness command placeholder: {exc.args[0]}") from exc
    if not command:
        raise ValueError("Harness command is empty.")
    return command


def _finalization_config(config: HarnessConfig) -> HarnessConfig:
    command_template = (
        DEFAULT_CODEX_FINALIZATION_COMMAND
        if config.harness == "codex"
        else DEFAULT_CLAUDE_CODE_FINALIZATION_COMMAND
    )
    return HarnessConfig(
        harness=config.harness,
        model=config.model,
        command_template=command_template,
        candidate_count=config.candidate_count,
        candidate_multiplier=config.candidate_multiplier,
        timeout_seconds=config.timeout_seconds,
        api_base_url=config.api_base_url,
        api_key_env=config.api_key_env,
        provider_id=config.provider_id,
    )


def _build_subprocess_env(config: HarnessConfig) -> Dict[str, str]:
    environment = os.environ.copy()
    if not config.api_base_url:
        return environment

    api_key = str(
        environment.get(config.api_key_env)
        or environment.get("AGENTICDS_HARNESS_API_KEY")
        or _project_llm_setting("LLM_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise ValueError(
            f"Harness API key is missing. Set {config.api_key_env} or AGENTICDS_HARNESS_API_KEY."
        )
    environment[config.api_key_env] = api_key
    if config.api_base_url.startswith(("http://127.0.0.1", "http://localhost")):
        no_proxy = [item.strip() for item in str(environment.get("NO_PROXY") or "").split(",") if item.strip()]
        for host in ("127.0.0.1", "localhost"):
            if host not in no_proxy:
                no_proxy.append(host)
        environment["NO_PROXY"] = ",".join(no_proxy)
        environment["no_proxy"] = environment["NO_PROXY"]
    if config.harness == "claude_code":
        environment["ANTHROPIC_BASE_URL"] = config.api_base_url.rstrip("/")
        environment["ANTHROPIC_AUTH_TOKEN"] = api_key
        environment.pop("ANTHROPIC_API_KEY", None)
    return environment


class AgentHarnessDatasetSearcher:
    """Use a coding-agent harness to search live Hugging Face, then align to a KG snapshot."""

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.last_summary: Dict[str, Any] = {}
        self._last_run_metadata: Dict[str, Any] = {}

    def requested_candidate_count(self, top_k: int) -> int:
        if self.config.candidate_count > 0:
            return max(int(top_k), int(self.config.candidate_count))
        return max(int(top_k), int(math.ceil(float(top_k) * self.config.candidate_multiplier)))

    def search(
        self,
        query: str,
        *,
        snapshot_dataset_ids: Sequence[str],
        top_k: int,
        artifact_root: Optional[Path] = None,
    ) -> List[str]:
        if not str(query or "").strip():
            return []
        if not snapshot_dataset_ids:
            raise ValueError("Snapshot dataset ID set is empty.")
        self.last_summary = {}
        self._last_run_metadata = {}

        session_root: Optional[Path] = None
        if artifact_root is not None:
            session_root = Path(artifact_root) / f"{self.config.harness}-{uuid.uuid4().hex[:12]}"
            session_root.mkdir(parents=True, exist_ok=True)

        requested_count = self.requested_candidate_count(top_k)
        raw_ids = self._run_once(
            query,
            candidate_count=requested_count,
            artifact_dir=session_root,
        )
        summary = filter_dataset_ids_to_snapshot(
            raw_ids,
            snapshot_dataset_ids=snapshot_dataset_ids,
            top_k=top_k,
        )
        summary.update(
            {
                "harness": self.config.harness,
                "model": self.config.model,
                "requested_candidate_count": requested_count,
                "candidate_multiplier": self.config.candidate_multiplier,
                "missing_top_k_count": max(0, int(top_k) - len(summary["selected_dataset_ids"])),
                "all_raw_dataset_ids": raw_ids,
                **self._last_run_metadata,
            }
        )
        self.last_summary = dict(summary)
        if session_root is not None:
            (session_root / "snapshot_filter.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return list(summary["selected_dataset_ids"])

    def _run_once(
        self,
        query: str,
        *,
        candidate_count: int,
        artifact_dir: Optional[Path],
    ) -> List[str]:
        with tempfile.TemporaryDirectory(prefix=f"agenticds-{self.config.harness}-") as tmpdir:
            workspace = Path(tmpdir)
            schema_file = workspace / "output_schema.json"
            evidence_file = workspace / "hf_evidence.jsonl"
            mcp_config_file = workspace / "mcp.json"
            mcp_server = Path(__file__).resolve().with_name("hf_dataset_mcp_server.py")
            schema_file.write_text(
                json.dumps(
                    _output_schema(candidate_count=candidate_count),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            finalization_schema_file = workspace / "finalization_output_schema.json"
            finalization_schema_file.write_text(
                json.dumps(
                    _output_schema(candidate_count=candidate_count, exact=False),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            mcp_config_file.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "huggingface": {
                                "type": "stdio",
                                "command": str(Path(sys.executable).resolve()),
                                "args": [str(mcp_server)],
                                "env": {HF_EVIDENCE_FILE_ENV: str(evidence_file)},
                            }
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            prompt = _task_prompt(
                query,
                candidate_count=candidate_count,
                harness=self.config.harness,
                hf_tool_path=mcp_server,
            )
            (workspace / "prompt.md").write_text(prompt, encoding="utf-8")
            run_started = perf_counter()
            attempts: List[Dict[str, Any]] = []

            environment = _build_subprocess_env(self.config)
            environment[HF_EVIDENCE_FILE_ENV] = str(evidence_file)
            if self.config.harness == "codex":
                codex_home = (
                    Path(artifact_dir).resolve() / ".codex_state"
                    if artifact_dir is not None
                    else workspace / ".codex"
                )
                codex_home.mkdir(parents=True, exist_ok=True)
                environment["CODEX_HOME"] = str(codex_home)

            requested_executable = shlex.split(self.config.command_template)[0]
            resolved_executable = _resolve_command_executable(self.config, requested_executable)
            if not resolved_executable:
                if artifact_dir is not None:
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    for source in workspace.iterdir():
                        if source.is_file():
                            shutil.copy2(source, artifact_dir / source.name)
                raise FileNotFoundError(
                    f"{self.config.harness} executable was not found: {requested_executable}. "
                    "Install the official standalone harness CLI, set AGENTICDS_CODEX_EXECUTABLE, "
                    "or override its command template."
                )

            (workspace / "invocation.json").write_text(
                json.dumps(
                    {
                        "harness": self.config.harness,
                        "requested_executable": requested_executable,
                        "resolved_executable": resolved_executable,
                        "model": self.config.model,
                        "timeout_seconds": self.config.timeout_seconds,
                        "recovery_policy": "one_retrieval_retry_then_same_harness_finalization",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            def invoke(
                *,
                phase: str,
                attempt_number: int,
                phase_prompt: str,
                timeout_seconds: int,
                config: HarnessConfig,
                phase_schema_file: Path,
            ) -> Dict[str, Any]:
                response_file = workspace / f"{phase}_response_{attempt_number}.json"
                paths = {
                    "workspace": workspace,
                    "schema_file": phase_schema_file,
                    "response_file": response_file,
                    "mcp_config_file": mcp_config_file,
                    "mcp_server": mcp_server,
                    "python_executable": Path(sys.executable).resolve(),
                }
                command = _build_command(config, paths)
                command[0] = resolved_executable
                started = perf_counter()
                timed_out = False
                return_code: Optional[int]
                try:
                    completed = subprocess.run(
                        command,
                        input=phase_prompt,
                        text=True,
                        capture_output=True,
                        cwd=str(workspace),
                        timeout=max(1, int(timeout_seconds)),
                        check=False,
                        env=environment,
                    )
                    stdout = completed.stdout or ""
                    stderr = completed.stderr or ""
                    return_code = int(completed.returncode)
                except subprocess.TimeoutExpired as exc:
                    timed_out = True
                    stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
                    stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
                    return_code = None
                elapsed = perf_counter() - started
                prefix = f"{phase}_attempt_{attempt_number}"
                (workspace / f"{prefix}.stdout.log").write_text(stdout, encoding="utf-8")
                (workspace / f"{prefix}.stderr.log").write_text(stderr, encoding="utf-8")
                if timed_out:
                    # Keep the conventional names used by existing diagnostics.
                    (workspace / "stdout.log").write_text(stdout, encoding="utf-8")
                    (workspace / "stderr.log").write_text(stderr, encoding="utf-8")
                    (workspace / "timeout.json").write_text(
                        json.dumps(
                            {
                                "phase": phase,
                                "attempt": attempt_number,
                                "timeout_seconds": max(1, int(timeout_seconds)),
                                "elapsed_seconds": elapsed,
                                "stdout_chars": len(stdout),
                                "stderr_chars": len(stderr),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                response_output = response_file.read_text(encoding="utf-8") if response_file.exists() else ""
                ids = extract_dataset_ids(response_output, max_candidates=candidate_count)
                output_source = "response_file"
                if not ids:
                    final_output = extract_final_assistant_output(stdout, harness=self.config.harness)
                    ids = extract_dataset_ids(final_output, max_candidates=candidate_count)
                    output_source = "terminal_assistant_event"
                result = {
                    "phase": phase,
                    "attempt": attempt_number,
                    "elapsed_seconds": elapsed,
                    "return_code": return_code,
                    "timed_out": timed_out,
                    "dataset_ids": ids,
                    "output_source": output_source,
                    "error": (
                        extract_harness_process_error(stdout=stdout, stderr=stderr)
                        if return_code not in (0, None) else ""
                    ),
                }
                attempts.append(result)
                return result

            try:
                retrieval_budget = max(1, int(self.config.timeout_seconds))
                finalization_budget = 300
                retrieval_deadline = time.monotonic() + retrieval_budget
                autonomous = invoke(
                    phase="retrieval",
                    attempt_number=1,
                    phase_prompt=prompt,
                    timeout_seconds=retrieval_budget,
                    config=self.config,
                    phase_schema_file=schema_file,
                )
                evidence_ids = load_hf_evidence_dataset_ids(evidence_file)
                model_dataset_ids = list(autonomous["dataset_ids"])
                grounding = ground_dataset_ids(model_dataset_ids, evidence_ids=evidence_ids)
                grounded_ids = grounding["grounded_dataset_ids"]
                unsupported_ids = grounding["rejected_dataset_ids"]
                autonomous_valid = (
                    autonomous["return_code"] == 0
                    and len(model_dataset_ids) == candidate_count
                    and not unsupported_ids
                    and len(grounded_ids) == candidate_count
                )

                recovery_used = False
                if not autonomous_valid and not evidence_ids and time.monotonic() < retrieval_deadline:
                    recovery_used = True
                    retry = invoke(
                        phase="retrieval",
                        attempt_number=2,
                        phase_prompt=prompt,
                        timeout_seconds=max(1, int(retrieval_deadline - time.monotonic())),
                        config=self.config,
                        phase_schema_file=schema_file,
                    )
                    autonomous = retry
                    evidence_ids = load_hf_evidence_dataset_ids(evidence_file)
                    model_dataset_ids = list(retry["dataset_ids"])
                    grounding = ground_dataset_ids(model_dataset_ids, evidence_ids=evidence_ids)
                    grounded_ids = grounding["grounded_dataset_ids"]
                    unsupported_ids = grounding["rejected_dataset_ids"]
                    autonomous_valid = (
                        retry["return_code"] == 0
                        and len(model_dataset_ids) == candidate_count
                        and not unsupported_ids
                        and len(grounded_ids) == candidate_count
                    )

                finalization_used = False
                selected_ids = list(grounded_ids)
                if not autonomous_valid and evidence_ids:
                    finalization_used = True
                    final_prompt = _finalization_prompt(
                        query,
                        candidate_count=candidate_count,
                        evidence_records=load_hf_evidence_records(evidence_file),
                    )
                    (workspace / "finalization_prompt.md").write_text(final_prompt, encoding="utf-8")
                    finalized = invoke(
                        phase="finalization",
                        attempt_number=1,
                        phase_prompt=final_prompt,
                        timeout_seconds=finalization_budget,
                        config=_finalization_config(self.config),
                        phase_schema_file=finalization_schema_file,
                    )
                    finalized_grounding = ground_dataset_ids(
                        finalized["dataset_ids"],
                        evidence_ids=evidence_ids,
                    )
                    if finalized_grounding["grounded_dataset_ids"]:
                        selected_ids = finalized_grounding["grounded_dataset_ids"]
                        unsupported_ids = finalized_grounding["rejected_dataset_ids"]

                if not selected_ids:
                    (workspace / "execution.json").write_text(
                        json.dumps(
                            {
                                "elapsed_seconds": perf_counter() - run_started,
                                "attempts": attempts,
                                "autonomous_completion": False,
                                "recovery_used": recovery_used,
                                "finalization_used": finalization_used,
                                "timed_out": any(bool(item.get("timed_out")) for item in attempts),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    errors = [str(item.get("error") or "") for item in attempts if item.get("error")]
                    raise ValueError(
                        f"{self.config.harness} produced no evidence-grounded dataset IDs"
                        + (f": {errors[-1][:500]}" if errors else ".")
                    )

                submitted_ids = list(selected_ids[:candidate_count])
                evidence_coverage = len(submitted_ids) / len(selected_ids) if selected_ids else 0.0
                elapsed_seconds = perf_counter() - run_started
                self._last_run_metadata = {
                    "harness_elapsed_seconds": elapsed_seconds,
                    "harness_return_code": autonomous.get("return_code"),
                    "evidence_dataset_count": len(evidence_ids),
                    "evidence_coverage": evidence_coverage,
                    "evidence_grounded": True,
                    "autonomous_completion": autonomous_valid,
                    "recovery_used": recovery_used,
                    "context_replay_used": False,
                    "finalization_used": finalization_used,
                    "raw_shortfall": max(0, candidate_count - len(submitted_ids)),
                    "attempts": attempts,
                }
                (workspace / "evidence_audit.json").write_text(
                    json.dumps(
                        {
                            "submitted_dataset_ids": submitted_ids,
                            "model_dataset_ids": model_dataset_ids,
                            "appended_evidence_dataset_ids": [],
                            "evidence_dataset_ids": evidence_ids,
                            "supported_dataset_ids": submitted_ids,
                            "unsupported_dataset_ids": unsupported_ids,
                            "evidence_coverage": evidence_coverage,
                            "affects_ranking": finalization_used,
                            "autonomous_completion": autonomous_valid,
                            "recovery_used": recovery_used,
                            "finalization_used": finalization_used,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                (workspace / "execution.json").write_text(
                    json.dumps(
                        {
                            "elapsed_seconds": elapsed_seconds,
                            "attempts": attempts,
                            "autonomous_completion": autonomous_valid,
                            "recovery_used": recovery_used,
                            "finalization_used": finalization_used,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                (workspace / "parsed_dataset_ids.json").write_text(
                    json.dumps(
                        {
                            "dataset_ids": submitted_ids,
                            "submitted_dataset_ids": submitted_ids,
                            "model_dataset_ids": model_dataset_ids,
                            "appended_evidence_dataset_ids": [],
                            "unsupported_evidence_dataset_ids": unsupported_ids,
                            "evidence_dataset_count": len(evidence_ids),
                            "output_source": (
                                "same_harness_finalization" if finalization_used else autonomous["output_source"]
                            ),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return submitted_ids
            finally:
                if artifact_dir is not None:
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    for source in workspace.iterdir():
                        if source.is_file():
                            shutil.copy2(source, artifact_dir / source.name)


# Compatibility alias for callers introduced with the initial harness implementation.
AgentHarnessReranker = AgentHarnessDatasetSearcher
