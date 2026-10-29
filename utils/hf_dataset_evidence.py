"""Extract bounded, source-linked evidence from dataset previews for retrieval."""

from __future__ import annotations

import ast
from collections import Counter
import re
from typing import Any, Dict, Iterator, Mapping


PROMPT_FIELDS = {
    "question", "question_content", "instruction", "input", "prompt", "problem",
    "problem_statement", "statement", "description", "query", "human",
}
ANSWER_FIELDS = {
    "answer", "answer_text", "output", "response", "solution", "solutions",
    "reference_solution", "canonical_solution", "accepted_solution", "completion", "target", "code",
    "correct_answer", "correct_option", "final_answer", "response_text", "assistant_response",
    "generated_solution", "expected_answer", "accepted_solutions", "gpt", "label", "labels", "answer_idx", "answer_index", "cop",
}
EXPLANATION_FIELDS = {"explanation", "rationale", "reasoning", "analysis", "derivation"}
CODE_FIELDS = {"code", "canonical_solution", "reference_solution", "accepted_solution"}
BLOCKED_PARENTS = {
    "test", "tests", "test_cases", "sample_test_cases", "sample_test_cases_output",
    "hidden_test_cases", "hidden_test_cases_output", "incorrect_submissions", "rejected",
    "validator", "checker", "generator", "metadata",
}
POSSIBLE_NESTED_SUPERVISION = {"messages", "conversations", "dialogue", "turns"}
POSSIBLE_CODE_OUTPUT_FIELDS = ANSWER_FIELDS | POSSIBLE_NESTED_SUPERVISION | {
    "correct_submissions", "accepted_submissions", "reference_code", "solution_code", "generated_code",
}
CODE_PROBLEM_ONLY_FIELDS = {
    "id", "question_id", "title", "name", "description", "question", "question_content",
    "problem", "statement", "instruct", "instruction", "prompt", "input", "source",
    "limit", "test_case", "test_cases", "sample_test_cases_input", "sample_test_cases_output",
    "hidden_test_cases", "hidden_test_cases_output", "analysis", "difficulty", "contest",
    "tags", "language", "languages", "url", "metadata",
}
TRAINING_TERMS = re.compile(r"\b(sft|finetun\w*|fine[ -]?tun\w*|instruction[ -]?tun\w*|training data(?:set)?|train(?:ing)? set)\b", re.I)
CODE_TERMS = re.compile(r"\b(code generation|code solution|accepted code|competitive programming|programming problems?)\b", re.I)


def training_requirement(query: str, intent_codes: list[str] | None = None) -> bool:
    return "training_data" in (intent_codes or []) or bool(TRAINING_TERMS.search(query))


def _fields(value: Any, prefix: str = "", depth: int = 0) -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping) and depth < 3:
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _fields(child, path, depth + 1)
    elif isinstance(value, list) and value and isinstance(value[0], Mapping) and depth < 3:
        yield from _fields(value[0], f"{prefix}[0]", depth + 1)
    elif prefix:
        yield prefix, value


def _field_role(path: str) -> str:
    parts = re.sub(r"\[\d+\]", "", path).lower().split(".")
    if any(part in BLOCKED_PARENTS for part in parts[:-1]):
        return ""
    leaf = parts[-1]
    if leaf in ANSWER_FIELDS:
        return "answer"
    if leaf in PROMPT_FIELDS:
        return "prompt"
    if leaf in EXPLANATION_FIELDS:
        return "explanation"
    return ""


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value[:4]):
        return ", ".join(value[:4]).strip()
    return ""


def _role_fields(row: Mapping[str, Any]) -> Dict[str, list[tuple[str, str]]]:
    roles: Dict[str, list[tuple[str, str]]] = {"prompt": [], "answer": [], "explanation": []}
    for path, value in _fields(row):
        role = _field_role(path)
        text = _text(value)
        if role and text:
            roles[role].append((path, text))
    for key in POSSIBLE_NESTED_SUPERVISION:
        messages = row.get(key)
        if not isinstance(messages, list):
            continue
        for index, message in enumerate(messages[:8]):
            if not isinstance(message, Mapping):
                continue
            speaker = str(message.get("role") or message.get("from") or "").lower()
            content = _text(message.get("content") or message.get("value"))
            role = "answer" if speaker in {"assistant", "gpt"} else "prompt" if speaker in {"user", "human"} else ""
            if role and content:
                roles[role].append((f"{key}[{index}].content", content))
    return roles


def _options(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("options") or row.get("choices")
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    if isinstance(raw, Mapping):
        values = [raw[letter] for letter in "ABCD" if letter in raw]
    elif isinstance(raw, list):
        values = raw
    else:
        values = [row.get(f"op{letter}") for letter in "abcd"]
    if len(values) < 2 or any(value is None for value in values):
        return []
    return [_strip_choice_prefix(str(value).strip(), index)[0] for index, value in enumerate(values)]


def _strip_choice_prefix(value: str, index: int) -> tuple[str, bool]:
    label = chr(ord("A") + index)
    match = re.match(r"^([A-D])[.)]\s+", value.strip(), re.I)
    if match and match.group(1).upper() == label:
        return value.strip()[match.end():], True
    return value.strip(), False


def _answer_index(row: Mapping[str, Any]) -> int | None:
    for key in ("answer_idx", "answer_index", "cop"):
        value = row.get(key)
        if isinstance(value, str) and re.fullmatch(r"[A-Z]", value.strip(), re.I):
            return ord(value.strip().upper()) - ord("A")
        numeric = int(value.strip()) if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()) else value
        if not isinstance(numeric, int) or isinstance(numeric, bool):
            continue
        if key == "cop":
            return numeric
        if key != "answer_index":
            continue
        options = _options(row)
        answer = _text(row.get("answer_text") or row.get("answer"))
        if not options or not answer:
            return None
        matches = [index for index, option in enumerate(options) if _normalize_option(answer) == _normalize_option(option)]
        for index in range(len(options)):
            stripped, had_prefix = _strip_choice_prefix(answer, index)
            if had_prefix and _normalize_option(stripped) == _normalize_option(options[index]):
                matches.append(index)
        for index in matches:
            if index in {numeric, numeric - 1}:
                return index
        return None
    return None


def _normalize_option(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def row_consistency(row: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Check only contradictions decidable from a row's explicit labels and options."""
    options = _options(row)
    index = _answer_index(row)
    if not options or index is None:
        return [], []
    checks = ["answer_index_in_options"]
    if not 0 <= index < len(options):
        return checks, ["answer_index_out_of_range"]

    issues = []
    answer = _text(row.get("answer_text") or row.get("answer"))
    if answer:
        checks.append("answer_text_matches_index")
        if _normalize_option(answer) != _normalize_option(options[index]):
            labelled = re.match(r"^([A-D])[.)]\s+", answer, re.I)
            wrong_label = bool(labelled and ord(labelled.group(1).upper()) - ord("A") != index)
            if wrong_label:
                issues.append("answer_label_index_conflict")
            answer, _ = _strip_choice_prefix(answer, index)
            actual = _normalize_option(answer)
            another_option = any(actual == _normalize_option(option) for other, option in enumerate(options) if other != index)
            another_label = len(actual) == 1 and actual.upper() in "ABCD" and actual.upper() != chr(ord("A") + index)
            if actual != _normalize_option(options[index]) and (another_option or another_label) and not wrong_label:
                issues.append("answer_text_index_conflict")

    explanation = _text(row.get("explanation") or row.get("exp"))
    match = re.match(r"^\s*Ans(?:wer)?\s*[.:]?\s*\(?([A-D])\)?(?=\s|[.:,;]|$)", explanation, re.I)
    if match:
        checks.append("explicit_explanation_option_matches_index")
        if ord(match.group(1).upper()) - ord("A") != index:
            issues.append("explanation_index_conflict")
    return checks, issues


def choice_signature(row: Mapping[str, Any]) -> tuple[str, tuple[str, ...], int] | None:
    """Identify directly comparable multiple-choice supervision across sources."""
    question = _normalize_option(str(row.get("question") or ""))
    options = _options(row)
    index = _answer_index(row)
    if not question or not options or index is None or not 0 <= index < len(options):
        return None
    return question, tuple(_normalize_option(option) for option in options), index


def _excerpt(value: tuple[str, str], limit: int) -> Dict[str, Any]:
    path, text = value
    if len(text) > limit:
        left = max(1, limit * 2 // 3)
        text = text[:left] + " ... " + text[-(limit - left):]
    return {"field": path, "text": text}


def _description_evidence(description: str, query: str, limit: int) -> list[str]:
    lines = [re.sub(r"\s+", " ", part).strip(" #*\t") for part in re.split(r"\n\s*\n|(?=^#{1,4} )", description, flags=re.M)]
    lines = [part for part in lines if len(part) > 15]
    if not lines:
        return []
    terms = set(re.findall(r"[a-z]{4,}", query.lower()))
    scored = []
    for index, part in enumerate(lines):
        lower = part.lower()
        score = len(terms & set(re.findall(r"[a-z]{4,}", lower)))
        score += sum(1 for term in ("train", "answer", "solution", "supervis", "benchmark", "split") if term in lower)
        scored.append((score, index, part))
    chosen = sorted(sorted(scored, key=lambda item: (-item[0], item[1]))[:2], key=lambda item: item[1])
    per_part = limit // len(chosen)
    return [part[:per_part] for _, _, part in chosen]


def _metadata_evidence(dataset: Mapping[str, Any], query: str) -> Dict[str, Any]:
    selected: Dict[str, Any] = {}
    query_lower = query.lower()
    for field in ("tasks", "modalities", "languages", "formats", "license", "libraries", "other", "tags"):
        value = dataset.get(field)
        if isinstance(value, list):
            values = list(dict.fromkeys(str(item) for item in value if item))
            if len(values) > 10:
                matching = [item for item in values if item.lower() in query_lower]
                selected[field] = {"count": len(values), "matching": matching[:5], "examples": values[:3]}
            elif values:
                selected[field] = values
        elif isinstance(value, str) and value.strip():
            selected[field] = value[:120]
    return selected


def build_candidate_evidence(
    *, dataset: Mapping[str, Any], sample: Mapping[str, Any], query: str,
    intent_codes: list[str] | None = None, text_budget: int = 480,
) -> Dict[str, Any]:
    """Return evidence and a conservative three-state SFT assessment."""
    rows = sample.get("sample_rows")
    if not isinstance(rows, list) or not rows:
        row = sample.get("sample_row")
        rows = [{"row": row, "config": sample.get("config", ""), "split": sample.get("split", ""), "offset": 0}] if isinstance(row, Mapping) else []
    rows = [item for item in rows if isinstance(item, Mapping) and isinstance(item.get("row"), Mapping)]
    target_is_training = training_requirement(query, intent_codes)
    target_is_code = bool(CODE_TERMS.search(query))
    observed = []
    language_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    checks: set[str] = set()
    schema = sample.get("sample_features") or {}
    for item in rows:
        row = item["row"]
        roles = _role_fields(row)
        observed.append((item, roles))
        language = _text(row.get("language") or row.get("lang"))
        if language:
            language_counts[language] += 1
        if not item.get("truncated_cells"):
            checked, issues = row_consistency(row)
            checks.update(checked)
            issue_counts.update(issues)

    train_observed = [(item, roles) for item, roles in observed if str(item.get("split") or "").lower() == "train"]
    answer_seen = any(roles["answer"] and roles["prompt"] for _, roles in train_observed)
    code_seen = any(
        roles["prompt"] and any(
            path.split(".")[-1] in CODE_FIELDS or
            (re.search(r"(?m)^\s*(?:def\s+\w+\s*\(|#include\b|class\s+\w+|import\s+\w+|```\w*)", value) is not None)
            for path, value in roles["answer"]
        )
        for _, roles in train_observed
    )
    status = "unknown"
    basis: list[str] = []
    if target_is_training and (code_seen if target_is_code else answer_seen):
        status = "satisfied"
        basis.append("observed_supervision_pair")
    elif target_is_training and observed and isinstance(schema, Mapping):
        available = set(sample.get("available_train_configs") or [])
        inspected = {str(item.get("config") or "") for item, _ in train_observed}
        names = {str(name).lower() for config in available for name in schema.get(config, [])}
        complete_schema = bool(available) and available <= inspected and all(schema.get(config) for config in available)
        blocked_only = bool(names) and names <= CODE_PROBLEM_ONLY_FIELDS
        tags = dataset.get("tags")
        benchmark_tag = isinstance(tags, list) and any(
            str(tag).lower() in {"benchmark", "llm-evaluation"} for tag in tags
        )
        possible_code_field = bool(names & POSSIBLE_CODE_OUTPUT_FIELDS) or any(
            re.search(r"solution|submission|completion|answer|response|assistant|generation", name)
            and not re.search(r"wrong|failed|incorrect|team|count|score|rate|percent|judge|checker|^num_", name)
            for name in names - {"problem_code", "special_judge_code"}
        )
        code_in_explanation = any(
            re.search(r"(?m)^\s*(?:def\s+\w+\s*\(|#include\b|```(?:python|cpp|java))", value)
            for _, roles in train_observed for _, value in roles["explanation"]
        )
        if not available and isinstance(sample.get("available_train_configs"), list) and sample.get("status") == "ok" and not sample.get("fetch_errors"):
            status = "violated"
            basis.append("no_training_split")
        elif target_is_code and complete_schema and benchmark_tag and not possible_code_field and not code_in_explanation:
            status = "violated"
            basis.append("benchmark_schema_lacks_reference_code")
        elif target_is_code and complete_schema and blocked_only and not sample.get("fetch_errors") and all(not item.get("truncated_cells") for item, _ in observed):
            status = "violated"
            basis.append("all_train_schemas_lack_reference_output")
    if issue_counts and status == "satisfied":
        status = "unknown"
        basis.append("observed_answer_conflict")

    example = next(((item, roles) for item, roles in observed if roles["prompt"] and roles["answer"]), None)
    if example is None and observed:
        example = observed[0]
    snippets: Dict[str, Any] = {}
    if example:
        _, roles = example
        budgets = {"prompt": max(40, text_budget // 4), "answer": max(100, text_budget // 2), "explanation": max(40, text_budget // 4)}
        for role in ("prompt", "answer", "explanation"):
            choices = roles[role]
            if role == "answer" and target_is_code:
                choices = sorted(choices, key=lambda pair: (pair[0].split(".")[-1] not in CODE_FIELDS, pair[0]))
            if choices:
                snippets[role] = _excerpt(choices[0], budgets[role])

    fields = list(dict.fromkeys(
        path.replace("[0]", "[]") for item, _ in observed for path, _ in _fields(item["row"])
    ))
    return {
        "status": str(sample.get("status") or "unavailable"),
        "source": str(sample.get("source") or ""),
        "config": str(sample.get("config") or ""),
        "split": str(sample.get("split") or ""),
        "revision": str(sample.get("revision") or ""),
        "observed_rows": len(observed),
        "available_train_configs": list(sample.get("available_train_configs") or []),
        "sampled_offsets": [item.get("offset") for item, _ in observed],
        "fields": fields[:35],
        "languages_observed": dict(language_counts),
        "snippets": snippets,
        "description_excerpt": _description_evidence(str(dataset.get("description") or ""), query, 240),
        "metadata": _metadata_evidence(dataset, query),
        "training_applicability": {"status": status, "basis": basis},
        "consistency": {"checked": sorted(checks), "issues": sorted(issue_counts)},
    }
