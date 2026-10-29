from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# Reduce long-run CUDA allocator fragmentation during multi-million item embedding jobs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModel, AutoTokenizer

from global_values import (
    EMBEDDING_MODEL_NAME,
    HF_LOCAL_FILES_ONLY,
    LOCAL_EMBEDDING_BATCH_SIZE,
    LOCAL_EMBEDDING_DOCUMENT_MAX_LENGTH,
    LOCAL_EMBEDDING_DEVICE,
    LOCAL_EMBEDDING_METADATA_MAX_LENGTH,
    LOCAL_EMBEDDING_QUERY_MAX_LENGTH,
)


def sanitize_model_name(model_name: str) -> str:
    return str(model_name or "").replace("/", "_").replace("\\", "_")


def default_query_instruction(model_name: str) -> str:
    normalized = str(model_name or "").lower()
    if "qwen3-embedding" in normalized:
        return "Instruct: Retrieve Hugging Face datasets or models relevant to the user query.\nQuery: "
    return "Represent this query for searching relevant datasets: "


@dataclass(frozen=True)
class EmbeddingBackendInfo:
    model_name: str
    backend: str
    device: str


class LocalEmbeddingBackend:
    """Small adapter around supported local embedding backends.

    Qwen/Qwen3-Embedding-0.6B is served through transformers so it can reuse the
    local Hugging Face cache. BAAI/bge-m3 is served through FlagEmbedding when
    available, which matches the model's recommended dense embedding path.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        *,
        batch_size: int = LOCAL_EMBEDDING_BATCH_SIZE,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = str(model_name or EMBEDDING_MODEL_NAME)
        self.batch_size = max(1, int(batch_size))
        self.device = self._resolve_device(device or LOCAL_EMBEDDING_DEVICE)
        self.backend = self._select_backend()
        self._safe_batch_sizes: dict[tuple[bool, int], int] = {}

        print(f"Loading embedding model: {self.model_name} ({self.backend})...")
        if self.backend == "flag_bge_m3":
            self._load_flag_bge_m3()
        else:
            self._load_transformers()

    @property
    def info(self) -> EmbeddingBackendInfo:
        return EmbeddingBackendInfo(
            model_name=self.model_name,
            backend=self.backend,
            device=self.device,
        )

    def _select_backend(self) -> str:
        if self.model_name.lower() == "baai/bge-m3":
            try:
                import FlagEmbedding  # noqa: F401

                return "flag_bge_m3"
            except Exception:
                print("FlagEmbedding is unavailable; falling back to transformers for BAAI/bge-m3.")
        return "transformers"

    def _resolve_device(self, preferred_device: str) -> str:
        preferred = str(preferred_device or "").strip()
        if preferred:
            if preferred.startswith("cuda") and torch.cuda.is_available():
                print(f"Using configured embedding device: {preferred}")
                return preferred
            if preferred.startswith("cuda") and not torch.cuda.is_available():
                print(f"Configured device '{preferred}' is unavailable; falling back to CPU.")
            elif preferred == "cpu":
                print("Using configured embedding device: cpu")
                return "cpu"

        auto_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Auto-selected embedding device: {auto_device}")
        return auto_device

    def _load_flag_bge_m3(self) -> None:
        from FlagEmbedding import BGEM3FlagModel

        kwargs = {}
        if HF_LOCAL_FILES_ONLY:
            kwargs["local_files_only"] = True

        self.model = BGEM3FlagModel(
            self.model_name,
            normalize_embeddings=True,
            use_fp16=self.device.startswith("cuda"),
            devices=self.device,
            trust_remote_code=True,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            batch_size=self.batch_size,
            **kwargs,
        )
        self.tokenizer = None
        self.max_length = 512

    def _load_transformers(self) -> None:
        kwargs = {"trust_remote_code": True}
        if HF_LOCAL_FILES_ONLY:
            kwargs["local_files_only"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        self.model = AutoModel.from_pretrained(self.model_name, **kwargs).to(self.device)
        self.model.eval()
        self.max_length = self._resolve_max_length()

    def _resolve_max_length(self) -> int:
        model_max = getattr(self.tokenizer, "model_max_length", 2048)
        if not isinstance(model_max, int) or model_max <= 0 or model_max > 100000:
            return 2048
        return min(2048, model_max)

    def _pool_transformers_embeddings(self, outputs, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled

        last_hidden = outputs.last_hidden_state
        model_type = str(getattr(self.model.config, "model_type", "") or "").lower()
        decoder_family = model_type in {"qwen2", "qwen3", "llama", "mistral", "gemma", "gpt_neox"}
        is_decoder_only = bool(
            (
                getattr(self.model.config, "is_decoder", False)
                or decoder_family
            )
            and not getattr(self.model.config, "is_encoder_decoder", False)
        )

        if is_decoder_only:
            last_token_indices = attention_mask.sum(1) - 1
            row_selector = torch.arange(last_hidden.shape[0], device=last_hidden.device)
            return last_hidden[row_selector, last_token_indices]

        mask = attention_mask.unsqueeze(-1)
        summed = (last_hidden * mask).sum(1)
        counts = mask.sum(1).clamp(min=1)
        return summed / counts.to(last_hidden.dtype)

    def encode(
        self,
        texts: List[str],
        *,
        batch_size: Optional[int] = None,
        is_query: bool = False,
        max_length: Optional[int] = None,
    ) -> np.ndarray:
        requested_batch_size = max(1, int(batch_size or self.batch_size))
        clean_texts = [str(text or "") for text in texts]
        if not clean_texts:
            return np.array([])

        effective_max_length = self._resolve_runtime_max_length(is_query=is_query, override=max_length)
        cache_key = (bool(is_query), effective_max_length)
        actual_batch_size = min(
            requested_batch_size,
            self._safe_batch_sizes.get(cache_key, requested_batch_size),
        )

        while True:
            try:
                embeddings = self._encode_once(
                    clean_texts,
                    batch_size=actual_batch_size,
                    is_query=is_query,
                    max_length=effective_max_length,
                )
                self._safe_batch_sizes[cache_key] = actual_batch_size
                return embeddings
            except torch.OutOfMemoryError:
                if actual_batch_size <= 1:
                    raise
                fallback_batch_size = max(1, actual_batch_size // 2)
                gc.collect()
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                print(
                    "CUDA OOM while encoding "
                    f"{len(clean_texts)} texts at batch_size={actual_batch_size}; "
                    f"retrying with batch_size={fallback_batch_size}."
                )
                actual_batch_size = fallback_batch_size

    def _encode_once(
        self,
        texts: List[str],
        *,
        batch_size: int,
        is_query: bool,
        max_length: Optional[int],
    ) -> np.ndarray:
        clean_texts = [str(text or "") for text in texts]
        effective_max_length = self._resolve_runtime_max_length(is_query=is_query, override=max_length)

        if self.backend == "flag_bge_m3":
            payload = self.model.encode(
                clean_texts,
                batch_size=batch_size,
                max_length=effective_max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            return np.asarray(payload["dense_vecs"], dtype=np.float32)

        all_embeddings = []
        for i in range(0, len(clean_texts), batch_size):
            batch_texts = clean_texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=effective_max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._pool_transformers_embeddings(outputs, inputs["attention_mask"])
                all_embeddings.append(embeddings.float().cpu().numpy())

        if not all_embeddings:
            return np.array([])
        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    def _resolve_runtime_max_length(self, *, is_query: bool, override: Optional[int]) -> int:
        if override is not None:
            return max(1, int(override))
        if is_query:
            return max(1, int(LOCAL_EMBEDDING_QUERY_MAX_LENGTH))
        if self.backend == "flag_bge_m3":
            return max(1, int(LOCAL_EMBEDDING_DOCUMENT_MAX_LENGTH))
        return min(self.max_length, max(1, int(LOCAL_EMBEDDING_DOCUMENT_MAX_LENGTH)))
