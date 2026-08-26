from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/home/ai4mater/005-mat4ai_agent/0-mul-agent/stage3/workspace/without_dat_snapshot")
SOURCE_DIR = Path(os.getenv("MOOSE_SOURCE_DIR", str(DEFAULT_SOURCE))).resolve()
DB_PATH = ROOT / "data" / "simulation_registry.sqlite3"
PLAN_DB_PATH = ROOT / "data" / "simulation_plans.sqlite3"
TASK_DB_PATH = ROOT / "data" / "research_tasks.sqlite3"
INGEST_TASK_DB_PATH = ROOT / "data" / "ingest_tasks.sqlite3"
TASK_BACKEND = os.getenv("TASK_BACKEND", "auto").lower()  # auto, local, rq
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
TASK_QUEUE_NAME = os.getenv("TASK_QUEUE_NAME", "moose-research")
TASK_MAX_WORKERS = int(os.getenv("TASK_MAX_WORKERS", "4"))
RUNS_DIR = ROOT / "data" / "runs"
ARTIFACTS_DIR = ROOT / "data" / "artifacts"
KNOWLEDGE_RENDERED_DIR = ROOT / "data" / "knowledge_sources" / "rendered"
TEMPLATE_PATH = SOURCE_DIR / "case1_fiat_walltemp_nominal.i"
RUN_SCRIPT_PATH = SOURCE_DIR / "run.sh"
SURROGATE_DIR = Path("/home/ai4mater/005-mat4ai_agent/0-mul-agent/stage3/workspace/models")
EMBEDDING_MODEL_PATH = Path(os.getenv("EMBEDDING_MODEL_PATH", "/home/ai4mater/005-mat4ai_agent/models/bge-m3"))
RERANKER_MODEL_PATH = Path(os.getenv("RERANKER_MODEL_PATH", "/home/ai4mater/005-mat4ai_agent/models/bge-reranker-v2-m3"))
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "local").lower()  # local, milvus
# Do not use the SDK-reserved MILVUS_URI environment variable: PyMilvus reads
# it while importing and expects an HTTP URI, which breaks Milvus Lite paths.
MILVUS_URI = os.getenv("MOOSE_MILVUS_URI", "http://127.0.0.1:19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "scientific_chunks_v2")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "FLAT").upper()
# Selected by eval/retrieval_eval.py against human source-level labels.
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "dense")  # bm25, dense, hybrid, hybrid_rerank
RETRIEVAL_TIER = os.getenv("RETRIEVAL_TIER", "default").lower()  # fast, default, precision
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "parent_child")  # document, fixed, structure, parent_child
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto").lower()  # auto, cpu, cuda
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "auto").lower()  # auto, cpu, cuda
LLM_LOCAL_USES_GPU = os.getenv("LLM_LOCAL_USES_GPU", "true").lower() == "true"

# Disabled by default so the deterministic evaluation baseline stays reproducible.
LLM_ENABLED = os.getenv("MOOSE_COPILOT_LLM_ENABLED", "false").lower() == "true"
LLM_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
LLM_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
LLM_PRIMARY_MODEL = os.getenv("LLM_PRIMARY_MODEL", "qwen2.5-coder:7b")
LLM_FAST_MODEL = os.getenv("LLM_FAST_MODEL", "qwen2.5:3b")
LLM_NARRATIVE_MODEL = os.getenv("LLM_NARRATIVE_MODEL", "qwen2.5:7b")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "")
LLM_INPUT_COST_PER_1M = os.getenv("LLM_INPUT_COST_PER_1M", "0")
LLM_OUTPUT_COST_PER_1M = os.getenv("LLM_OUTPUT_COST_PER_1M", "0")
