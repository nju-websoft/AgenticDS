from __future__ import annotations

import argparse
import html
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Sequence, Set

import requests

from global_values import KG_DIR


HF_FULL_TEXT_SEARCH_URL = "https://huggingface.co/search/full-text"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_PAGE_SIZE = 20
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class _FullTextSearchPropsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.props: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if self.props is not None:
            return
        attr_map = dict(attrs)
        if attr_map.get("data-target") == "FullTextSearch":
            self.props = attr_map.get("data-props")


FULL_TEXT_PROPS_RE = re.compile(
    r'data-target="FullTextSearch"[^>]*data-props="([\s\S]*?)"',
    flags=re.IGNORECASE,
)


def _build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    return session


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").strip().split())


def _build_query_variants(query: str) -> List[str]:
    normalized = _normalize_query(query)
    tokens = normalized.split()
    variants: List[str] = []
    for candidate in [
        normalized,
        " ".join(tokens[:16]),
        " ".join(tokens[:12]),
        " ".join(tokens[:8]),
        normalized[:120].rsplit(" ", 1)[0] if len(normalized) > 120 else normalized,
    ]:
        candidate = candidate.strip()
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


def _retry_after_seconds(response: requests.Response | None) -> float | None:
    if response is None:
        return None
    raw = str(response.headers.get("Retry-After") or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return max(0.0, value)


def _compute_retry_delay(exc: Exception, attempt_index: int, base_backoff_seconds: float) -> float:
    response = exc.response if isinstance(exc, requests.HTTPError) else None
    retry_after = _retry_after_seconds(response)
    if retry_after is not None:
        return retry_after
    multiplier = max(0, int(attempt_index))
    return max(0.0, float(base_backoff_seconds)) * (2**multiplier)


def _is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError):
        status = int(exc.response.status_code) if exc.response is not None else 0
        return status in {408, 425, 429, 500, 502, 503, 504}
    return isinstance(exc, requests.RequestException)


def _extract_search_payload(page_html: str) -> Dict[str, Any]:
    parser = _FullTextSearchPropsParser()
    parser.feed(page_html)
    decode_errors: List[Exception] = []
    if parser.props:
        try:
            return json.loads(html.unescape(parser.props))
        except json.JSONDecodeError as exc:
            decode_errors.append(exc)

    match = FULL_TEXT_PROPS_RE.search(page_html)
    if match:
        try:
            return json.loads(html.unescape(match.group(1)))
        except json.JSONDecodeError as exc:
            decode_errors.append(exc)

    if decode_errors:
        raise ValueError(
            "Failed to decode Hugging Face full-text search payload from the HTML response."
        ) from decode_errors[-1]
    raise ValueError("Failed to locate Hugging Face full-text search payload in the HTML response.")


def _dataset_id_from_doc(doc: Dict[str, Any]) -> str:
    dataset_id = str(doc.get("name") or "").strip()
    if dataset_id:
        return dataset_id

    owner = str(doc.get("repoOwner") or "").strip()
    repo_name = str(doc.get("repoName") or "").strip()
    if owner and repo_name:
        return f"{owner}/{repo_name}"
    return ""


def load_snapshot_dataset_ids(kg_dir: str | Path = KG_DIR) -> Set[str]:
    dataset_file = Path(kg_dir) / "datasets.json"
    allowed_ids: Set[str] = set()
    try:
        import ijson

        with dataset_file.open("rb") as handle:
            for dataset_id in ijson.items(handle, "item.id"):
                normalized = str(dataset_id or "").strip()
                if normalized:
                    allowed_ids.add(normalized)
    except ImportError:
        payload = json.loads(dataset_file.read_text(encoding="utf-8"))
        for row in payload or []:
            if not isinstance(row, dict):
                continue
            dataset_id = str(row.get("id") or "").strip()
            if dataset_id:
                allowed_ids.add(dataset_id)
    return allowed_ids


def search_hf_full_text(
    query: str,
    max_results: int = 10,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_pages: int = 10,
    max_retries: int = 3,
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    session: requests.Session | None = None,
    allowed_dataset_ids: Collection[str] | None = None,
) -> List[str]:
    query_text = str(query or "").strip()
    if not query_text:
        raise ValueError("query must not be empty")

    limit = max(1, int(max_results))
    http = session or _build_session()
    query_variants = _build_query_variants(query_text)
    allowed = {str(item).strip() for item in (allowed_dataset_ids or []) if str(item).strip()} or None

    results: List[str] = []
    seen: set[str] = set()
    for page in range(max(1, int(max_pages))):
        payload: Dict[str, Any] | None = None
        last_error: Exception | None = None
        for candidate_query in query_variants:
            for attempt_index in range(max(1, int(max_retries))):
                try:
                    response = http.get(
                        HF_FULL_TEXT_SEARCH_URL,
                        params={"q": candidate_query, "type": "dataset", "p": page},
                        timeout=timeout_seconds,
                    )
                    response.raise_for_status()
                    payload = _extract_search_payload(response.text)
                    break
                except (requests.RequestException, ValueError) as exc:
                    last_error = exc
                    payload = None
                    if isinstance(exc, requests.RequestException) and _is_retryable_exception(exc):
                        if attempt_index + 1 < max(1, int(max_retries)):
                            time.sleep(_compute_retry_delay(exc, attempt_index, retry_backoff_seconds))
                            continue
            if payload is not None:
                break
        if payload is None:
            assert last_error is not None
            if isinstance(last_error, requests.RequestException) and _is_retryable_exception(last_error):
                return results[:limit]
            raise last_error
        docs = payload.get("docs") or []
        if not isinstance(docs, list) or not docs:
            break

        new_count = 0
        for raw_doc in docs:
            if not isinstance(raw_doc, dict):
                continue
            dataset_id = _dataset_id_from_doc(raw_doc)
            if not dataset_id or dataset_id in seen:
                continue
            if allowed is not None and dataset_id not in allowed:
                continue
            seen.add(dataset_id)
            results.append(dataset_id)
            new_count += 1
            if len(results) >= limit:
                return results[:limit]

        if new_count == 0 or len(docs) < DEFAULT_PAGE_SIZE:
            break

    return results[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Hugging Face Full Text Search for dataset ids.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=DEFAULT_RETRY_BACKOFF_SECONDS)
    parser.add_argument("--kg-dir", type=str, default=KG_DIR)
    parser.add_argument(
        "--disable-snapshot-filter",
        action="store_true",
        help="Do not filter results by dataset ids present in the KG snapshot.",
    )
    args = parser.parse_args()

    allowed_dataset_ids = None if args.disable_snapshot_filter else load_snapshot_dataset_ids(args.kg_dir)

    dataset_ids = search_hf_full_text(
        args.query,
        max_results=args.top_k,
        timeout_seconds=args.timeout,
        max_pages=args.max_pages,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        allowed_dataset_ids=allowed_dataset_ids,
    )
    print(
        json.dumps(
            {
                "query": args.query,
                "top_k": int(args.top_k),
                "returned_count": len(dataset_ids),
                "dataset_ids": dataset_ids,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
