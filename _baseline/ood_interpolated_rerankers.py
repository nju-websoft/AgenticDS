from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from global_values import EMBEDDING_MODEL_NAME, KG_DIR
from utils.kg_index_store import load_index_entities, load_kg_entities, resolve_readable_index_files, sparse_entity_text
from utils.kg_sparse_index import BM25SparseIndex


DEFAULT_CANDIDATE_TOP_K = 100
DEFAULT_ALPHA_GRID = [round(0.1 * idx, 1) for idx in range(11)]


def _load_hf_with_local_cache_first(factory: Any, model_name: str, **kwargs: Any) -> Any:
    try:
        return factory.from_pretrained(model_name, local_files_only=True, **kwargs)
    except Exception:
        return factory.from_pretrained(model_name, **kwargs)


def minmax_normalize_scores(scores: Mapping[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = list(float(value) for value in scores.values())
    max_value = max(values)
    min_value = min(values)
    if max_value == min_value:
        return {str(key): 1.0 for key in scores.keys()}
    scale = max_value - min_value
    return {str(key): (float(value) - min_value) / scale for key, value in scores.items()}


def _dcg(relevances: Sequence[int]) -> float:
    return sum((((2**int(rel)) - 1) / math.log2(rank + 1)) for rank, rel in enumerate(relevances, 1))


def _ndcg_at_k(ranked_ids: Sequence[str], qrels: Mapping[str, int], *, k: int) -> float:
    rels = [int(qrels.get(candidate_id, 0)) for candidate_id in ranked_ids[: max(1, int(k))]]
    actual = _dcg(rels)
    ideal = _dcg(sorted((int(value) for value in qrels.values()), reverse=True)[: max(1, int(k))])
    return actual / ideal if ideal > 0 else 0.0


@dataclass(frozen=True)
class BM25CandidateBundle:
    candidate_ids: List[str]
    bm25_scores: Dict[str, float]
    documents: Dict[str, str]


class SnapshotBM25CandidateGenerator:
    def __init__(
        self,
        *,
        entities: Optional[Sequence[Mapping[str, Any]]] = None,
        kg_dir: str | Path = KG_DIR,
        candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    ) -> None:
        self.candidate_top_k = max(1, int(candidate_top_k))
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
        self.documents: Dict[str, str] = {
            dataset_id: sparse_entity_text(entity) for dataset_id, entity in zip(self.dataset_ids, self.entities)
        }
        self.index = sparse_index if sparse_index is not None else BM25SparseIndex.build_from_entities(self.entities)

    def bundle(self, query: str, *, top_k: Optional[int] = None) -> BM25CandidateBundle:
        limit = max(1, int(top_k or self.candidate_top_k))
        scores_by_idx = self.index.search(str(query or ""), top_k=limit)
        ranked = sorted(scores_by_idx.items(), key=lambda item: item[1], reverse=True)[:limit]
        candidate_ids: List[str] = []
        bm25_scores: Dict[str, float] = {}
        documents: Dict[str, str] = {}
        for idx, score in ranked:
            if idx < 0 or idx >= len(self.dataset_ids):
                continue
            dataset_id = self.dataset_ids[idx]
            candidate_ids.append(dataset_id)
            bm25_scores[dataset_id] = float(score)
            documents[dataset_id] = self.documents.get(dataset_id, "")
        return BM25CandidateBundle(candidate_ids=candidate_ids, bm25_scores=bm25_scores, documents=documents)


class MonoT5OODScorer:
    def __init__(
        self,
        *,
        model_name: str = "castorini/monot5-large-msmarco",
        batch_size: int = 8,
        max_length: int = 384,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(32, int(max_length))
        self.device = device
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._true_token_id = None
        self._false_token_id = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = _load_hf_with_local_cache_first(AutoTokenizer, self.model_name)
        model_kwargs = {"dtype": torch.float16} if self.device == "cuda" else {}
        self._model = _load_hf_with_local_cache_first(AutoModelForSeq2SeqLM, self.model_name, **model_kwargs)
        self._model.to(self.device)
        self._model.eval()

        true_ids = self._tokenizer.encode("true", add_special_tokens=False)
        false_ids = self._tokenizer.encode("false", add_special_tokens=False)
        if len(true_ids) != 1 or len(false_ids) != 1:
            raise ValueError("monoT5 true/false tokens are not single-token in current tokenizer.")
        self._true_token_id = int(true_ids[0])
        self._false_token_id = int(false_ids[0])

    def score(self, query: str, documents: Sequence[str]) -> List[float]:
        self._ensure_loaded()
        assert self._torch is not None and self._tokenizer is not None and self._model is not None
        texts = [f"Question: {query} Context: {doc}" for doc in documents]
        scores: List[float] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self._model.generate(
                    **encoded,
                    max_new_tokens=1,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                logits = outputs.scores[0]
                batch_scores = (logits[:, self._true_token_id] - logits[:, self._false_token_id]).detach().float().cpu().tolist()
                scores.extend(float(value) for value in batch_scores)
        return scores


class ANCEOODScorer:
    def __init__(
        self,
        *,
        model_name: str = "castorini/ance-msmarco-passage",
        batch_size: int = 32,
        query_max_length: int = 64,
        doc_max_length: int = 384,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.query_max_length = max(16, int(query_max_length))
        self.doc_max_length = max(32, int(doc_max_length))
        self.device = device
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = _load_hf_with_local_cache_first(AutoTokenizer, self.model_name)
        model_kwargs = {"dtype": torch.float16} if self.device == "cuda" else {}
        self._model = _load_hf_with_local_cache_first(AutoModel, self.model_name, **model_kwargs)
        self._model.to(self.device)
        self._model.eval()

    def _encode(self, texts: Sequence[str], *, max_length: int) -> "Any":
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        vectors = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self._model(**encoded)
                if getattr(outputs, "pooler_output", None) is not None:
                    pooled = outputs.pooler_output
                else:
                    pooled = outputs.last_hidden_state[:, 0]
                vectors.append(pooled.detach().float().cpu())
        return self._torch.cat(vectors, dim=0)

    def score(self, query: str, documents: Sequence[str]) -> List[float]:
        self._ensure_loaded()
        assert self._torch is not None
        if not documents:
            return []
        query_vec = self._encode([query], max_length=self.query_max_length)[0]
        doc_vecs = self._encode(documents, max_length=self.doc_max_length)
        scores = self._torch.mv(doc_vecs, query_vec)
        return [float(value) for value in scores.tolist()]


class InterpolatedBM25Reranker:
    def __init__(self, name: str, scorer: Any, *, alpha: float = 0.5) -> None:
        self.name = str(name)
        self.scorer = scorer
        self.alpha = float(alpha)
        self._rerank_scores: Dict[str, Dict[str, float]] = {}
        self._alpha_search: List[Dict[str, float]] = []

    @property
    def alpha_search(self) -> List[Dict[str, float]]:
        return list(self._alpha_search)

    def score_bundle(self, query_id: str, query: str, bundle: BM25CandidateBundle) -> Dict[str, float]:
        query_id = str(query_id)
        cached = self._rerank_scores.get(query_id)
        if cached is not None:
            return cached
        documents = [bundle.documents[candidate_id] for candidate_id in bundle.candidate_ids]
        raw_scores = self.scorer.score(query, documents)
        score_map = {candidate_id: float(score) for candidate_id, score in zip(bundle.candidate_ids, raw_scores)}
        self._rerank_scores[query_id] = score_map
        return score_map

    def fit_alpha(
        self,
        *,
        query_rows: Sequence[Mapping[str, Any]],
        candidate_bundles: Mapping[str, BM25CandidateBundle],
        qrels: Mapping[str, Mapping[str, int]],
        alpha_grid: Optional[Sequence[float]] = None,
        optimize_metric: str = "ndcg@10",
    ) -> float:
        metric_name = str(optimize_metric).strip().lower()
        alphas = [float(value) for value in (alpha_grid or DEFAULT_ALPHA_GRID)]
        scored_rows = []
        for row in query_rows:
            query_id = str(row.get("query_id") or "").strip()
            query_text = str(row.get("query") or "").strip()
            bundle = candidate_bundles.get(query_id)
            if not query_id or not query_text or bundle is None or not bundle.candidate_ids:
                continue
            rerank_scores = self.score_bundle(query_id, query_text, bundle)
            scored_rows.append((query_id, bundle, rerank_scores))
        if not scored_rows:
            return self.alpha

        best_alpha = self.alpha
        best_metric = -1.0
        search_log: List[Dict[str, float]] = []
        for alpha in alphas:
            ndcg10 = ndcg20 = map10 = map20 = 0.0
            for query_id, bundle, rerank_scores in scored_rows:
                ranked_ids = self._rank_ids(bundle, rerank_scores, alpha=alpha)
                query_qrels = {str(candidate_id): int(score) for candidate_id, score in (qrels.get(query_id) or {}).items()}
                ndcg10 += _ndcg_at_k(ranked_ids, query_qrels, k=10)
                ndcg20 += _ndcg_at_k(ranked_ids, query_qrels, k=20)
                map10 += self._ap_at_k(ranked_ids, query_qrels, k=10)
                map20 += self._ap_at_k(ranked_ids, query_qrels, k=20)
            denom = max(1, len(scored_rows))
            metrics = {
                "alpha": alpha,
                "ndcg@10": ndcg10 / denom,
                "ndcg@20": ndcg20 / denom,
                "map@10": map10 / denom,
                "map@20": map20 / denom,
            }
            search_log.append(metrics)
            score = metrics.get(metric_name, metrics["ndcg@10"])
            if score > best_metric + 1e-12:
                best_metric = score
                best_alpha = alpha
        self.alpha = best_alpha
        self._alpha_search = search_log
        return self.alpha

    def rank_candidates(self, query_id: str, query: str, bundle: BM25CandidateBundle, *, max_results: int = 20) -> List[str]:
        rerank_scores = self.score_bundle(query_id, query, bundle)
        ranked_ids = self._rank_ids(bundle, rerank_scores, alpha=self.alpha)
        return ranked_ids[: max(1, int(max_results))]

    @staticmethod
    def _ap_at_k(ranked_ids: Sequence[str], qrels: Mapping[str, int], *, k: int) -> float:
        relevant_total = sum(1 for value in qrels.values() if int(value) > 0)
        if relevant_total <= 0:
            return 0.0
        hits = 0
        precision_sum = 0.0
        for rank, candidate_id in enumerate(ranked_ids[: max(1, int(k))], 1):
            if int(qrels.get(candidate_id, 0)) > 0:
                hits += 1
                precision_sum += hits / rank
        return precision_sum / min(relevant_total, max(1, int(k)))

    @staticmethod
    def _rank_ids(bundle: BM25CandidateBundle, rerank_scores: Mapping[str, float], *, alpha: float) -> List[str]:
        normalized_bm25 = minmax_normalize_scores(bundle.bm25_scores)
        normalized_rerank = minmax_normalize_scores(rerank_scores)
        scored: List[tuple[str, float]] = []
        for candidate_id in bundle.candidate_ids:
            score = float(alpha) * normalized_rerank.get(candidate_id, 0.0) + (1.0 - float(alpha)) * normalized_bm25.get(candidate_id, 0.0)
            scored.append((candidate_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [candidate_id for candidate_id, _score in scored]
