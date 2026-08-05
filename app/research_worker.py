"""Serializable research job entry point used by local and RQ workers."""
from __future__ import annotations

import time

from app.config import DB_PATH, SOURCE_DIR, TASK_DB_PATH
from app.ingest import ingest
from app.multi_agent import run_multi_agent
from app.registry import SimulationRegistry
from app.task_manager import ResearchTaskStore
from app.trace import normalize_event, summary as trace_summary


def run_research_task(task_id: str, question: str) -> None:
    store = ResearchTaskStore(TASK_DB_PATH)
    if not store.claim(task_id): return
    started = time.perf_counter()
    try:
        registry = SimulationRegistry(DB_PATH)
        if not registry.cases(): ingest(SOURCE_DIR, registry)
        result = run_multi_agent(question, registry)
        task = store.get(task_id) or {}
        dispatch = normalize_event(
            {"node": "task_dispatch", "queue_backend": task.get("queue_backend", "local"), "queue_wait_ms": task.get("queue_wait_ms", 0.0)},
            trace_id=result["trace_id"], parent_span_id="", elapsed_ms=0.0,
        )
        result["trace"] = [dispatch, *result.get("trace", [])]
        result["trace_summary"] = trace_summary(result["trace"])
        result["queue"] = {"backend": task.get("queue_backend", "local"), "wait_ms": task.get("queue_wait_ms", 0.0)}
        store.complete(task_id, result, started)
    except Exception as exc:
        store.fail(task_id, exc, started)
