from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

from global_values import (
    LLM_API_BASE_URL,
    LLM_API_KEY,
    LLM_FALLBACK_CHANNEL,
    LLM_MAX_RETRIES,
    LLM_MODEL_NAME,
    LLM_PRIMARY_CHANNEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    DEBUG_FULL_TRACE,
)


class LLMGateway:
    """Multi-channel LLM gateway with fallback and JSON extraction."""

    def __init__(
        self,
        primary_channel: str = LLM_PRIMARY_CHANNEL,
        fallback_channel: str = LLM_FALLBACK_CHANNEL,
        api_base_url: str = LLM_API_BASE_URL,
        api_key: str = LLM_API_KEY,
        model_name: str = LLM_MODEL_NAME,
        timeout_seconds: int = LLM_TIMEOUT_SECONDS,
        temperature: float = LLM_TEMPERATURE,
    ) -> None:
        self.primary_channel = primary_channel
        self.fallback_channel = fallback_channel
        self.api_base_url = api_base_url.rstrip("/") if api_base_url else ""
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_retries = max(0, int(LLM_MAX_RETRIES))
        self.debug_enabled = bool(DEBUG_FULL_TRACE)
        self._session_logs: list[Dict[str, Any]] = []

    def set_debug(self, enabled: bool) -> None:
        self.debug_enabled = bool(enabled)

    def reset_session_logs(self) -> None:
        self._session_logs = []

    def get_session_logs(self) -> list[Dict[str, Any]]:
        return list(self._session_logs)

    def generate_json(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_prompt: str,
        default: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
        thinking: Optional[bool] = None,
    ) -> Dict[str, Any]:
        channels = [self.primary_channel]
        if self.fallback_channel and self.fallback_channel != self.primary_channel:
            channels.append(self.fallback_channel)

        request_timeout = self.timeout_seconds if timeout_seconds is None else int(max(1, timeout_seconds))

        debug_entry: Optional[Dict[str, Any]] = None
        if self.debug_enabled:
            debug_entry = {
                "task_name": task_name,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "timeout_seconds": request_timeout,
                "attempts": [],
            }

        errors = []
        for channel in channels:
            if channel in {"", "none", None}:
                continue

            attempt: Dict[str, Any] = {"channel": str(channel)}
            start = perf_counter()
            try:
                if channel == "rule":
                    payload = default or {}
                    if debug_entry is not None:
                        attempt["status"] = "ok"
                        attempt["mode"] = "rule_default"
                        attempt["parse_success"] = False
                        attempt["elapsed_seconds"] = perf_counter() - start
                        attempt["parsed_json"] = payload
                        debug_entry["attempts"].append(attempt)
                        debug_entry["result"] = payload
                        debug_entry["result_source"] = "rule_default"
                        debug_entry["parse_success"] = False
                        debug_entry["default_used"] = True
                        self._session_logs.append(debug_entry)
                    return payload
                if channel == "openai_compatible":
                    text, response_meta = self._call_openai_compatible(
                        system_prompt,
                        user_prompt,
                        timeout_seconds=request_timeout,
                        thinking=thinking,
                    )
                    if debug_entry is not None:
                        attempt["mode"] = "openai_compatible"
                        attempt["raw_response"] = text
                        if response_meta.get("model"):
                            attempt["response_model"] = response_meta["model"]
                        if response_meta.get("usage"):
                            attempt["usage"] = response_meta["usage"]
                    parsed = self._extract_json_dict(text)
                    if debug_entry is not None:
                        attempt["elapsed_seconds"] = perf_counter() - start
                    if parsed is not None:
                        if debug_entry is not None:
                            attempt["status"] = "ok"
                            attempt["parse_success"] = True
                            attempt["parsed_json"] = parsed
                            debug_entry["attempts"].append(attempt)
                            debug_entry["result"] = parsed
                            debug_entry["result_source"] = str(channel)
                            debug_entry["parse_success"] = True
                            debug_entry["default_used"] = False
                            self._session_logs.append(debug_entry)
                        return parsed
                    raise ValueError("LLM response does not contain valid JSON object")
                raise ValueError(f"Unsupported LLM channel: {channel}")
            except Exception as exc:  # pragma: no cover - best effort fallback path
                if debug_entry is not None:
                    attempt["status"] = "error"
                    attempt["parse_success"] = False
                    attempt["error_type"] = type(exc).__name__
                    attempt["elapsed_seconds"] = perf_counter() - start
                    attempt["error"] = str(exc)
                    debug_entry["attempts"].append(attempt)
                errors.append(f"{channel}: {exc}")

        if default is not None:
            payload = dict(default)
            payload.setdefault("_llm_errors", errors)
            if debug_entry is not None:
                debug_entry["result"] = payload
                debug_entry["result_source"] = "provided_default"
                debug_entry["parse_success"] = False
                debug_entry["default_used"] = True
                debug_entry["errors"] = list(errors)
                self._session_logs.append(debug_entry)
            return payload

        payload = {"_llm_errors": errors}
        if debug_entry is not None:
            debug_entry["result"] = payload
            debug_entry["result_source"] = "error_payload"
            debug_entry["parse_success"] = False
            debug_entry["default_used"] = False
            debug_entry["errors"] = list(errors)
            self._session_logs.append(debug_entry)
        return payload

    def _call_openai_compatible(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_seconds: Optional[int] = None,
        thinking: Optional[bool] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        if not self.api_base_url:
            raise ValueError("LLM_API_BASE_URL is empty")
        if not self.api_key:
            raise ValueError("LLM_API_KEY is empty")
        if not self.model_name:
            raise ValueError("LLM_MODEL_NAME is empty")

        base_url = self._normalize_base_url(self.api_base_url)
        request_timeout = self.timeout_seconds if timeout_seconds is None else int(max(1, timeout_seconds))
        client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=request_timeout,
            max_retries=self.max_retries,
        )
        request_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        if thinking is not None:
            request_kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if thinking else "disabled"},
            }
        response = client.chat.completions.create(
            **request_kwargs,
        )

        choices = response.choices or []
        if not choices:
            raise ValueError("No choices returned from LLM")

        message = choices[0].message
        content = message.content if message else ""
        if isinstance(content, list):
            content = "\n".join(item.get("text", "") for item in content if isinstance(item, dict))
        response_meta = {
            "model": getattr(response, "model", "") or "",
            "usage": self._extract_usage(response),
        }
        return str(content), response_meta

    def _extract_usage(self, response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return {}
        prompt_tokens = self._usage_value(usage, "prompt_tokens")
        completion_tokens = self._usage_value(usage, "completion_tokens")
        total_tokens = self._usage_value(usage, "total_tokens")
        payload: Dict[str, int] = {}
        if prompt_tokens is not None:
            payload["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            payload["completion_tokens"] = completion_tokens
        if total_tokens is not None:
            payload["total_tokens"] = total_tokens
        return payload

    def _usage_value(self, usage: Any, key: str) -> Optional[int]:
        value = getattr(usage, key, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _normalize_base_url(self, base_url: str) -> str:
        trimmed = base_url.strip().rstrip("/")
        if not trimmed:
            return trimmed
        if trimmed.endswith("/v1"):
            return trimmed + "/"
        return trimmed + "/v1/"

    def _extract_json_dict(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
        return None
