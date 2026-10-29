import os

KG_DIR = os.getenv("AGENTICDS_KG_DIR", "KG/HuggingKG_V20260312172948")
INDEX_CACHE_DIR = os.getenv("AGENTICDS_INDEX_CACHE_DIR", "index_cache")
LEGACY_EMBEDDING_CACHE_DIR = "embedding_cache"
CACHE_DIR = os.getenv("AGENTICDS_CACHE_DIR", "cache")
# Backward-compatible alias for older imports. New code should use INDEX_CACHE_DIR.
EMBEDDING_CACHE_DIR = INDEX_CACHE_DIR
INDEX_ENTITY_TYPES = ("dataset",)

# Embedding Settings
EMBEDDING_MODEL_TYPE = "local"
EMBEDDING_MODEL_NAME = os.getenv("AGENTICDS_EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
LOCAL_EMBEDDING_BATCH_SIZE = int(os.getenv("AGENTICDS_LOCAL_EMBEDDING_BATCH_SIZE", "256"))
LOCAL_EMBEDDING_DESCRIPTION_BATCH_SIZE = int(os.getenv("AGENTICDS_LOCAL_EMBEDDING_DESCRIPTION_BATCH_SIZE", "256"))
LOCAL_EMBEDDING_METADATA_BATCH_SIZE = int(os.getenv("AGENTICDS_LOCAL_EMBEDDING_METADATA_BATCH_SIZE", "512"))
LOCAL_EMBEDDING_DEVICE = os.getenv("AGENTICDS_LOCAL_EMBEDDING_DEVICE", "cuda:0")
LOCAL_EMBEDDING_QUERY_MAX_LENGTH = int(os.getenv("AGENTICDS_LOCAL_EMBEDDING_QUERY_MAX_LENGTH", "512"))
LOCAL_EMBEDDING_DOCUMENT_MAX_LENGTH = int(os.getenv("AGENTICDS_LOCAL_EMBEDDING_DOCUMENT_MAX_LENGTH", "1024"))
LOCAL_EMBEDDING_METADATA_MAX_LENGTH = int(os.getenv("AGENTICDS_LOCAL_EMBEDDING_METADATA_MAX_LENGTH", "1024"))
EMBEDDING_QUERY_INSTRUCTION = ""
HF_LOCAL_FILES_ONLY = os.getenv("AGENTICDS_HF_LOCAL_FILES_ONLY", "true").strip().lower() in {"1", "true", "yes", "on"}

# Hybrid retrieval settings
HYBRID_DESCRIPTION_DENSE_WEIGHT = float(os.getenv("AGENTICDS_HYBRID_DESCRIPTION_DENSE_WEIGHT", "0.55"))
HYBRID_METADATA_DENSE_WEIGHT = float(os.getenv("AGENTICDS_HYBRID_METADATA_DENSE_WEIGHT", "0.45"))
HYBRID_DENSE_WEIGHT = float(os.getenv("AGENTICDS_HYBRID_DENSE_WEIGHT", "0.7"))
HYBRID_SPARSE_WEIGHT = float(os.getenv("AGENTICDS_HYBRID_SPARSE_WEIGHT", "0.3"))
HYBRID_RRF_K = float(os.getenv("AGENTICDS_HYBRID_RRF_K", "60.0"))
HYBRID_DENSE_TOP_K_SEARCH = int(os.getenv("AGENTICDS_HYBRID_DENSE_TOP_K_SEARCH", "1200"))
HYBRID_SPARSE_TOP_K_SEARCH = int(os.getenv("AGENTICDS_HYBRID_SPARSE_TOP_K_SEARCH", "1200"))
HYBRID_BM25_K1 = float(os.getenv("AGENTICDS_HYBRID_BM25_K1", "1.2"))
HYBRID_BM25_B = float(os.getenv("AGENTICDS_HYBRID_BM25_B", "0.75"))

# Hugging Face metadata registry settings
HF_DATASET_TAGS_API_URL = os.getenv("AGENTICDS_HF_DATASET_TAGS_API_URL", "https://huggingface.co/api/datasets-tags-by-type")
HF_DATASET_TAGS_TIMEOUT_SECONDS = int(os.getenv("AGENTICDS_HF_DATASET_TAGS_TIMEOUT_SECONDS", "10"))
HF_DATASET_TAGS_CACHE_TTL_SECONDS = int(os.getenv("AGENTICDS_HF_DATASET_TAGS_CACHE_TTL_SECONDS", "86400"))
HF_DATASET_TAGS_CACHE_FILE = os.getenv(
    "AGENTICDS_HF_DATASET_TAGS_CACHE_FILE",
    os.path.join(CACHE_DIR, "hf_datasets_tags_by_type.json"),
)

# Hugging Face dataset sample settings (dataset-viewer rows API)
HF_DATASET_VIEWER_API_BASE_URL = os.getenv(
    "AGENTICDS_HF_DATASET_VIEWER_API_BASE_URL",
    "https://datasets-server.huggingface.co",
)
HF_DATASET_SAMPLE_TIMEOUT_SECONDS = int(os.getenv("AGENTICDS_HF_DATASET_SAMPLE_TIMEOUT_SECONDS", "8"))
HF_DATASET_SAMPLE_CACHE_DIR = os.getenv(
    "AGENTICDS_HF_DATASET_SAMPLE_CACHE_DIR",
    os.path.join(CACHE_DIR, "hf_dataset_samples"),
)
HF_DATASET_SAMPLE_MAX_TEXT_CHARS = int(os.getenv("AGENTICDS_HF_DATASET_SAMPLE_MAX_TEXT_CHARS", "1200"))
HF_DATASET_SAMPLE_MAX_WORKERS = int(os.getenv("AGENTICDS_HF_DATASET_SAMPLE_MAX_WORKERS", "4"))
HF_DATASET_SAMPLE_ENABLED = os.getenv("AGENTICDS_HF_DATASET_SAMPLE_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# LLM Settings
LLM_TYPE = "openai_compatible"
LLM_NAME = "GLM-5"

# Configure credentials through environment variables, never source files.
DEFAULT_LLM_PRIMARY_CHANNEL = "openai_compatible"
DEFAULT_LLM_FALLBACK_CHANNEL = "rule"
DEFAULT_LLM_API_BASE_URL = "https://llmapi.paratera.com/"
DEFAULT_LLM_API_KEY = ""
DEFAULT_LLM_MODEL_NAME = "GLM-5"
DEFAULT_LLM_TIMEOUT_SECONDS = 30
DEFAULT_LLM_TEMPERATURE = 0.1

# Multi-channel LLM gateway settings
LLM_PRIMARY_CHANNEL = os.getenv("AGENTICDS_LLM_PRIMARY_CHANNEL", DEFAULT_LLM_PRIMARY_CHANNEL)
LLM_FALLBACK_CHANNEL = os.getenv("AGENTICDS_LLM_FALLBACK_CHANNEL", DEFAULT_LLM_FALLBACK_CHANNEL)
LLM_API_BASE_URL = os.getenv("AGENTICDS_LLM_API_BASE_URL", DEFAULT_LLM_API_BASE_URL)
LLM_API_KEY = os.getenv("AGENTICDS_LLM_API_KEY", DEFAULT_LLM_API_KEY)
LLM_MODEL_NAME = os.getenv("AGENTICDS_LLM_MODEL_NAME", DEFAULT_LLM_MODEL_NAME)
LLM_TIMEOUT_SECONDS = int(os.getenv("AGENTICDS_LLM_TIMEOUT_SECONDS", str(DEFAULT_LLM_TIMEOUT_SECONDS)))
LLM_TEMPERATURE = float(os.getenv("AGENTICDS_LLM_TEMPERATURE", str(DEFAULT_LLM_TEMPERATURE)))
LLM_MAX_RETRIES = int(os.getenv("AGENTICDS_LLM_MAX_RETRIES", "0"))

# Agent-specific latency budgets. Planning retains deliberative reasoning,
# while each agent can use a timeout appropriate to its decision complexity.
PLANNER_LLM_TIMEOUT_SECONDS = int(os.getenv("AGENTICDS_PLANNER_LLM_TIMEOUT_SECONDS", "90"))
ENGINEER_LLM_TIMEOUT_SECONDS = int(os.getenv("AGENTICDS_ENGINEER_LLM_TIMEOUT_SECONDS", "45"))

# Optional long-lived retrieval service. Empty keeps local in-process loading.
RETRIEVAL_SERVICE_SOCKET = os.getenv("AGENTICDS_RETRIEVAL_SERVICE_SOCKET", "").strip()
RETRIEVAL_SERVICE_RPC_TIMEOUT_SECONDS = float(os.getenv("AGENTICDS_RETRIEVAL_SERVICE_RPC_TIMEOUT_SECONDS", "120"))
RETRIEVAL_SERVICE_READY_TIMEOUT_SECONDS = float(os.getenv("AGENTICDS_RETRIEVAL_SERVICE_READY_TIMEOUT_SECONDS", "7200"))

# Qrels LLM gateway settings. Keep qrels drafting on DeepSeek-V4-Pro unless explicitly changed.
DEFAULT_QRELS_LLM_PRIMARY_CHANNEL = "openai_compatible"
DEFAULT_QRELS_LLM_FALLBACK_CHANNEL = "rule"
DEFAULT_QRELS_LLM_API_BASE_URL = DEFAULT_LLM_API_BASE_URL
DEFAULT_QRELS_LLM_API_KEY = DEFAULT_LLM_API_KEY
DEFAULT_QRELS_LLM_MODEL_NAME = "DeepSeek-V4-Pro"
DEFAULT_QRELS_LLM_TIMEOUT_SECONDS = 90
DEFAULT_QRELS_LLM_TEMPERATURE = 0.1

QRELS_LLM_PRIMARY_CHANNEL = os.getenv("AGENTICDS_QRELS_LLM_PRIMARY_CHANNEL", DEFAULT_QRELS_LLM_PRIMARY_CHANNEL)
QRELS_LLM_FALLBACK_CHANNEL = os.getenv("AGENTICDS_QRELS_LLM_FALLBACK_CHANNEL", DEFAULT_QRELS_LLM_FALLBACK_CHANNEL)
QRELS_LLM_API_BASE_URL = os.getenv("AGENTICDS_QRELS_LLM_API_BASE_URL", DEFAULT_QRELS_LLM_API_BASE_URL)
QRELS_LLM_API_KEY = os.getenv("AGENTICDS_QRELS_LLM_API_KEY", DEFAULT_QRELS_LLM_API_KEY)
QRELS_LLM_MODEL_NAME = os.getenv("AGENTICDS_QRELS_LLM_MODEL_NAME", DEFAULT_QRELS_LLM_MODEL_NAME)
QRELS_LLM_TIMEOUT_SECONDS = int(os.getenv("AGENTICDS_QRELS_LLM_TIMEOUT_SECONDS", str(DEFAULT_QRELS_LLM_TIMEOUT_SECONDS)))
QRELS_LLM_TEMPERATURE = float(os.getenv("AGENTICDS_QRELS_LLM_TEMPERATURE", str(DEFAULT_QRELS_LLM_TEMPERATURE)))

# Multi-agent retrieval settings
AGENT_MAX_ITERATIONS = int(os.getenv("AGENTICDS_AGENT_MAX_ITERATIONS", "3"))
AGENT_RETRIEVE_TOP_K_PER_QUERY = int(os.getenv("AGENTICDS_AGENT_TOP_K_PER_QUERY", "40"))
AGENT_DEFAULT_TOP_K_FINAL = 20
AGENT_QUALITY_THRESHOLD = float(os.getenv("AGENTICDS_AGENT_QUALITY_THRESHOLD", "0.55"))
# Constraint violations remain fallback candidates unless strict removal is selected.
AGENT_CONSTRAINT_VIOLATION_POLICY = os.getenv("AGENTICDS_CONSTRAINT_VIOLATION_POLICY", "fallback").strip().lower()
if AGENT_CONSTRAINT_VIOLATION_POLICY not in {"fallback", "discard"}:
    raise ValueError("AGENTICDS_CONSTRAINT_VIOLATION_POLICY must be 'fallback' or 'discard'")

# Utility selection and its convergence test share the same scalar objective.
AGENT_ROUND_SELECTION_POLICY = os.getenv("AGENTICDS_ROUND_SELECTION_POLICY", "lexicographic").strip().lower()
if AGENT_ROUND_SELECTION_POLICY not in {"lexicographic", "utility"}:
    raise ValueError("AGENTICDS_ROUND_SELECTION_POLICY must be 'lexicographic' or 'utility'")
AGENT_ROUND_UTILITY_QUALITY_WEIGHT = float(os.getenv("AGENTICDS_ROUND_UTILITY_QUALITY_WEIGHT", "0.7"))
if not 0.0 <= AGENT_ROUND_UTILITY_QUALITY_WEIGHT <= 1.0:
    raise ValueError("AGENTICDS_ROUND_UTILITY_QUALITY_WEIGHT must be between 0 and 1")

AGENT_USE_LLM_PLANNER = os.getenv("AGENTICDS_AGENT_USE_LLM_PLANNER", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
AGENT_USE_LLM_ENGINEER = os.getenv("AGENTICDS_AGENT_USE_LLM_ENGINEER", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
AGENT_RERANK_SHORTLIST_MAX = int(os.getenv("AGENTICDS_AGENT_RERANK_SHORTLIST_MAX", "40"))
AGENT_CRITIC_POOL_SIZE = int(os.getenv("AGENTICDS_AGENT_CRITIC_POOL_SIZE", str(AGENT_RERANK_SHORTLIST_MAX)))

CRITIC_MAX_CANDIDATES = int(os.getenv("AGENTICDS_CRITIC_MAX_CANDIDATES", "40"))
CRITIC_BATCH_SIZE = int(os.getenv("AGENTICDS_CRITIC_BATCH_SIZE", "1"))
CRITIC_MAX_WORKERS = int(os.getenv("AGENTICDS_CRITIC_MAX_WORKERS", "20"))
CRITIC_LLM_TIMEOUT_SECONDS = int(os.getenv("AGENTICDS_CRITIC_LLM_TIMEOUT_SECONDS", "30"))
CRITIC_SERIAL_LLM_SCORING = os.getenv("AGENTICDS_CRITIC_SERIAL_LLM_SCORING", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CRITIC_SAMPLE_MAX_TEXT_CHARS = int(os.getenv("AGENTICDS_CRITIC_SAMPLE_MAX_TEXT_CHARS", "480"))

# Eval query bank settings
EVAL_QUERY_BANK_SIZE = 200
EVAL_SAMPLE_SIZE = 5
EVAL_QUERY_REWRITE_MAX_WORKERS = 4

CRITIC_RERANK_WEIGHT_LLM = float(os.getenv("AGENTICDS_CRITIC_RERANK_WEIGHT_LLM", "0.65"))
CRITIC_RERANK_WEIGHT_PRERERANK = float(os.getenv("AGENTICDS_CRITIC_RERANK_WEIGHT_PRERERANK", "0.35"))

# Independently switchable retrieval components. Defaults select the validated B0 configuration.
EXPERIMENT_ANCHOR_IDENTITY_SCORING = os.getenv(
    "AGENTICDS_EXPERIMENT_ANCHOR_IDENTITY_SCORING", "true"
).strip().lower() in {"1", "true", "yes", "on"}
EXPERIMENT_CONDITIONAL_SHORTLIST_ALLOCATIONS = os.getenv(
    "AGENTICDS_EXPERIMENT_CONDITIONAL_SHORTLIST_ALLOCATIONS", "true"
).strip().lower() in {"1", "true", "yes", "on"}
EXPERIMENT_RELIABILITY_WEIGHTED_FUSION = os.getenv(
    "AGENTICDS_EXPERIMENT_RELIABILITY_WEIGHTED_FUSION", "false"
).strip().lower() in {"1", "true", "yes", "on"}
EXPERIMENT_ENFORCE_RELATION_PLAN_COMPLETENESS = os.getenv(
    "AGENTICDS_EXPERIMENT_ENFORCE_RELATION_PLAN_COMPLETENESS", "true"
).strip().lower() in {"1", "true", "yes", "on"}
EXPERIMENT_RAW_QUERY_BRANCH = os.getenv(
    "AGENTICDS_EXPERIMENT_RAW_QUERY_BRANCH", "true"
).strip().lower() in {"1", "true", "yes", "on"}
EXPERIMENT_BEST_BRANCH_FUSION = os.getenv(
    "AGENTICDS_EXPERIMENT_BEST_BRANCH_FUSION", "false"
).strip().lower() in {"1", "true", "yes", "on"}
FUSION_BEST_BRANCH_WEIGHT = min(
    1.0,
    max(0.0, float(os.getenv("AGENTICDS_FUSION_BEST_BRANCH_WEIGHT", "0.30"))),
)

# Debug / tracing settings
DEBUG_FULL_TRACE = os.getenv("AGENTICDS_DEBUG_FULL_TRACE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


# AgenticDS experiment switches. Subprocess-scoped environment variables keep one
# variant from changing the behavior of another variant in the same pipeline.
EXPERIMENT_FORCE_SINGLE_ITERATION = _env_bool("AGENTICDS_EXPERIMENT_FORCE_SINGLE_ITERATION")
EXPERIMENT_DISABLE_CRITIC = _env_bool("AGENTICDS_EXPERIMENT_DISABLE_CRITIC")
EXPERIMENT_DISABLE_LLM_PLANNER = _env_bool("AGENTICDS_EXPERIMENT_DISABLE_LLM_PLANNER")
EXPERIMENT_PLANNER_RAW_QUERY_ONLY = _env_bool("AGENTICDS_EXPERIMENT_PLANNER_RAW_QUERY_ONLY")
EXPERIMENT_DISABLE_LLM_ENGINEER = _env_bool("AGENTICDS_EXPERIMENT_DISABLE_LLM_ENGINEER")
EXPERIMENT_DISABLE_METADATA_FILTER = _env_bool("AGENTICDS_EXPERIMENT_DISABLE_METADATA_FILTER")
EXPERIMENT_DISABLE_GRAPH_EXPANSION = _env_bool("AGENTICDS_EXPERIMENT_DISABLE_GRAPH_EXPANSION")
EXPERIMENT_GRAPH_TEXT_CONTROL = _env_bool("AGENTICDS_EXPERIMENT_GRAPH_TEXT_CONTROL")
EXPERIMENT_DISABLE_PRERERANK = _env_bool("AGENTICDS_EXPERIMENT_DISABLE_PRERERANK")
EXPERIMENT_DISABLE_CRITIC_SAMPLE_EVIDENCE = _env_bool("AGENTICDS_EXPERIMENT_DISABLE_CRITIC_SAMPLE_EVIDENCE")
EXPERIMENT_DISABLE_CRITIC_GRAPH_EVIDENCE = _env_bool("AGENTICDS_EXPERIMENT_DISABLE_CRITIC_GRAPH_EVIDENCE")


def resolved_agent_config() -> dict:
    """Return the behaviorally relevant, secret-free configuration for an experiment."""
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "agent_model": LLM_MODEL_NAME,
        "llm_temperature": LLM_TEMPERATURE,
        "max_iterations": AGENT_MAX_ITERATIONS,
        "default_per_query_top_k": AGENT_RETRIEVE_TOP_K_PER_QUERY,
        "default_top_k_final": AGENT_DEFAULT_TOP_K_FINAL,
        "min_candidates_policy": "requested_top_k",
        "quality_threshold": AGENT_QUALITY_THRESHOLD,
        "sparse_backend": "lucene",
        "fusion_execution": "cross_round_archive_once",
        "constraint_violation_policy": AGENT_CONSTRAINT_VIOLATION_POLICY,
        "round_selection_policy": AGENT_ROUND_SELECTION_POLICY,
        "round_utility_quality_weight": AGENT_ROUND_UTILITY_QUALITY_WEIGHT,
        "planner_llm_timeout_seconds": PLANNER_LLM_TIMEOUT_SECONDS,
        "engineer_llm_timeout_seconds": ENGINEER_LLM_TIMEOUT_SECONDS,
        "retrieval_service_socket": RETRIEVAL_SERVICE_SOCKET or None,
        "rerank_shortlist_max": AGENT_RERANK_SHORTLIST_MAX,
        "critic_pool_size": AGENT_CRITIC_POOL_SIZE,
        "critic_max_candidates": CRITIC_MAX_CANDIDATES,
        "critic_batch_size": CRITIC_BATCH_SIZE,
        "critic_max_workers": CRITIC_MAX_WORKERS,
        "critic_sample_max_text_chars": CRITIC_SAMPLE_MAX_TEXT_CHARS,
        "critic_rerank_weight_llm": CRITIC_RERANK_WEIGHT_LLM,
        "critic_rerank_weight_prerank": CRITIC_RERANK_WEIGHT_PRERERANK,
        "switches": {
            "force_single_iteration": EXPERIMENT_FORCE_SINGLE_ITERATION,
            "disable_critic": EXPERIMENT_DISABLE_CRITIC,
            "disable_llm_planner": EXPERIMENT_DISABLE_LLM_PLANNER,
            "planner_raw_query_only": EXPERIMENT_PLANNER_RAW_QUERY_ONLY,
            "disable_llm_engineer": EXPERIMENT_DISABLE_LLM_ENGINEER,
            "disable_metadata_filter": EXPERIMENT_DISABLE_METADATA_FILTER,
            "disable_graph_expansion": EXPERIMENT_DISABLE_GRAPH_EXPANSION,
            "graph_text_control": EXPERIMENT_GRAPH_TEXT_CONTROL,
            "disable_prerank": EXPERIMENT_DISABLE_PRERERANK,
            "disable_critic_sample_evidence": EXPERIMENT_DISABLE_CRITIC_SAMPLE_EVIDENCE,
            "disable_critic_graph_evidence": EXPERIMENT_DISABLE_CRITIC_GRAPH_EVIDENCE,
            "anchor_identity_scoring": EXPERIMENT_ANCHOR_IDENTITY_SCORING,
            "conditional_shortlist_allocations": EXPERIMENT_CONDITIONAL_SHORTLIST_ALLOCATIONS,
            "reliability_weighted_fusion": EXPERIMENT_RELIABILITY_WEIGHTED_FUSION,
            "enforce_relation_plan_completeness": EXPERIMENT_ENFORCE_RELATION_PLAN_COMPLETENESS,
            "raw_query_branch": EXPERIMENT_RAW_QUERY_BRANCH,
            "best_branch_fusion": EXPERIMENT_BEST_BRANCH_FUSION,
            "best_branch_fusion_weight": FUSION_BEST_BRANCH_WEIGHT,
        },
    }
