from __future__ import annotations

import argparse
import json
import re
from typing import List

import requests


GOOGLE_DATASET_SEARCH_URL = "https://datasetsearch.research.google.com/search"
GOOGLE_SITE_FILTER = "site:huggingface.co"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HF_DATASET_URL_RE = re.compile(
    r"https?://huggingface\.co/datasets/([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)",
    flags=re.IGNORECASE,
)


def _build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    return session


def _with_hf_site_filter(query: str) -> str:
    query_text = str(query or "").strip()
    if not query_text:
        raise ValueError("query must not be empty")
    if GOOGLE_SITE_FILTER.lower() in query_text.lower():
        return query_text
    return f"{query_text} {GOOGLE_SITE_FILTER}"


def _extract_result_block(page_html: str) -> str:
    start = page_html.find("AF_initDataCallback({key:'ds:0'")
    if start < 0:
        return page_html

    end = page_html.find("});", start)
    if end < 0:
        return page_html[start:]
    return page_html[start : end + 3]


def search_google_dataset_search(
    query: str,
    max_results: int = 10,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
) -> List[str]:
    resolved_query = _with_hf_site_filter(query)
    limit = max(1, int(max_results))
    http = session or _build_session()

    response = http.get(
        GOOGLE_DATASET_SEARCH_URL,
        params={"query": resolved_query},
        timeout=timeout_seconds,
    )
    response.raise_for_status()

    block = _extract_result_block(response.text).replace("\\/", "/")
    results: List[str] = []
    seen: set[str] = set()
    for dataset_id in HF_DATASET_URL_RE.findall(block):
        normalized = str(dataset_id).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        results.append(normalized)
        if len(results) >= limit:
            break
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query Google Dataset Search for Hugging Face dataset ids."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()

    resolved_query = _with_hf_site_filter(args.query)
    dataset_ids = search_google_dataset_search(
        args.query,
        max_results=args.top_k,
        timeout_seconds=args.timeout,
    )
    print(
        json.dumps(
            {
                "query": args.query,
                "resolved_query": resolved_query,
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
