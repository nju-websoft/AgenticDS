from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from global_values import EMBEDDING_MODEL_NAME, KG_DIR
from utils.kg_index_store import load_index_entities, load_kg_entities, resolve_readable_index_files
from utils.kg_sparse_index import BM25SparseIndex, tokenize_bm25


DEFAULT_CANDIDATE_TOP_K = 100
DEFAULT_FB_DOCS = 10
DEFAULT_FB_TERMS = 20
DEFAULT_ORIGINAL_QUERY_WEIGHT = 0.5
DEFAULT_QUERY_TERM_SCALE = 24


def _softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


class RM3BM25Retriever:
    def __init__(
        self,
        *,
        entities: Optional[Sequence[Mapping[str, Any]]] = None,
        kg_dir: str | Path = KG_DIR,
        candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
        fb_docs: int = DEFAULT_FB_DOCS,
        fb_terms: int = DEFAULT_FB_TERMS,
        original_query_weight: float = DEFAULT_ORIGINAL_QUERY_WEIGHT,
        query_term_scale: int = DEFAULT_QUERY_TERM_SCALE,
    ) -> None:
        sparse_index: Optional[BM25SparseIndex] = None
        if entities is None:
            files = resolve_readable_index_files(model_name=EMBEDDING_MODEL_NAME, kg_dir=kg_dir, entity_types=["dataset"])
            if files.entities.exists():
                raw_entities = load_index_entities(files.entities)
            else:
                raw_entities = load_kg_entities(kg_dir=kg_dir, entity_types=["dataset"])
            sparse_index = BM25SparseIndex.load(files.sparse_index)
            sparse_index.validate_entities(raw_entities)
        else:
            raw_entities = list(entities)

        self.entities: List[Dict[str, Any]] = [dict(entity) for entity in raw_entities if str(entity.get("id") or "").strip()]
        self.dataset_ids: List[str] = [str(entity.get("id") or "").strip() for entity in self.entities]
        self.documents: List[str] = [self._document_text(entity) for entity in self.entities]
        self.index = sparse_index if sparse_index is not None else BM25SparseIndex.build(self.documents)
        self.candidate_top_k = max(1, int(candidate_top_k))
        self.fb_docs = max(1, int(fb_docs))
        self.fb_terms = max(1, int(fb_terms))
        self.original_query_weight = max(0.0, min(1.0, float(original_query_weight)))
        self.query_term_scale = max(4, int(query_term_scale))

    @staticmethod
    def _document_text(entity: Mapping[str, Any]) -> str:
        from utils.kg_index_store import sparse_entity_text

        return sparse_entity_text(entity)

    def _initial_candidates(self, query: str, *, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        scores_by_idx = self.index.search(query, top_k=max(1, int(top_k or self.candidate_top_k)))
        ranked = sorted(scores_by_idx.items(), key=lambda item: item[1], reverse=True)
        return [(self.dataset_ids[idx], float(score)) for idx, score in ranked if 0 <= idx < len(self.dataset_ids)]

    def _expanded_query(self, query: str, candidates: Sequence[Tuple[str, float]]) -> str:
        original_terms = tokenize_bm25(query)
        original_counts = Counter(original_terms)
        if not original_counts:
            return query

        feedback_docs = list(candidates[: self.fb_docs])
        if not feedback_docs:
            return query

        doc_weights = _softmax([score for _dataset_id, score in feedback_docs])
        term_scores: Dict[str, float] = defaultdict(float)
        for doc_weight, (dataset_id, _score) in zip(doc_weights, feedback_docs):
            try:
                idx = self.dataset_ids.index(dataset_id)
            except ValueError:
                continue
            token_counts = Counter(tokenize_bm25(self.documents[idx]))
            doc_length = sum(token_counts.values()) or 1
            for term, tf in token_counts.items():
                term_scores[term] += doc_weight * (tf / doc_length)

        original_total = sum(original_counts.values()) or 1
        original_dist = {term: count / original_total for term, count in original_counts.items()}

        ranked_feedback_terms = [term for term, _score in sorted(term_scores.items(), key=lambda item: item[1], reverse=True)]
        selected_feedback_terms = ranked_feedback_terms[: self.fb_terms]

        combined_terms = set(original_dist.keys()) | set(selected_feedback_terms)
        combined_weights: Dict[str, float] = {}
        for term in combined_terms:
            orig = original_dist.get(term, 0.0)
            fb = term_scores.get(term, 0.0)
            combined_weights[term] = self.original_query_weight * orig + (1.0 - self.original_query_weight) * fb

        ranked_terms = sorted(combined_weights.items(), key=lambda item: item[1], reverse=True)
        expanded_tokens: List[str] = []
        for term, weight in ranked_terms[: max(len(original_dist), self.fb_terms)]:
            repeats = max(1, int(round(weight * self.query_term_scale)))
            expanded_tokens.extend([term] * repeats)
        return " ".join(expanded_tokens) if expanded_tokens else query

    def search(self, query: str, *, max_results: int = 20) -> List[str]:
        initial = self._initial_candidates(query, top_k=self.candidate_top_k)
        expanded_query = self._expanded_query(query, initial)
        reranked = self._initial_candidates(expanded_query, top_k=max(1, int(max_results)))
        return [dataset_id for dataset_id, _score in reranked[: max(1, int(max_results))]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BM25+RM3 retrieval on the snapshot BM25 index.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--fb-docs", type=int, default=DEFAULT_FB_DOCS)
    parser.add_argument("--fb-terms", type=int, default=DEFAULT_FB_TERMS)
    parser.add_argument("--original-query-weight", type=float, default=DEFAULT_ORIGINAL_QUERY_WEIGHT)
    args = parser.parse_args()

    retriever = RM3BM25Retriever(
        fb_docs=args.fb_docs,
        fb_terms=args.fb_terms,
        original_query_weight=args.original_query_weight,
    )
    dataset_ids = retriever.search(args.query, max_results=args.top_k)
    print(json.dumps({"query": args.query, "top_k": args.top_k, "dataset_ids": dataset_ids}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
