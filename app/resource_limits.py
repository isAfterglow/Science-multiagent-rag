"""Shared bounded resources for local inference backends."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from threading import Lock, Semaphore


_limits = {
    "embedding": Semaphore(int(os.getenv("EMBEDDING_MAX_CONCURRENCY", "1"))),
    "reranker": Semaphore(int(os.getenv("RERANKER_MAX_CONCURRENCY", "1"))),
    "llm": Semaphore(int(os.getenv("LLM_MAX_CONCURRENCY", "2"))),
    # RTX 3050-class cards cannot reliably host Ollama and BGE-M3/reranker at
    # the same time.  This is intentionally process-local: cross-process GPU
    # admission is exposed as an operational deployment concern, rather than
    # pretending a Python lock can govern an external Ollama process.
    "gpu_inference": Semaphore(int(os.getenv("GPU_INFERENCE_MAX_CONCURRENCY", "1"))),
}
_stats: dict[str, dict[str, float]] = {name: {"requests": 0, "wait_ms": 0.0} for name in _limits}
_active: dict[str, int] = {name: 0 for name in _limits}
_lock = Lock()


@contextmanager
def acquire(name: str):
    started = time.perf_counter()
    semaphore = _limits[name]
    semaphore.acquire()
    waited = (time.perf_counter() - started) * 1000
    with _lock:
        _stats[name]["requests"] += 1
        _stats[name]["wait_ms"] += waited
        _active[name] += 1
    try:
        yield round(waited, 3)
    finally:
        with _lock:
            _active[name] -= 1
        semaphore.release()


def metrics() -> dict:
    with _lock:
        return {name: {"requests": int(values["requests"]), "active": _active[name], "mean_wait_ms": round(values["wait_ms"] / values["requests"], 3) if values["requests"] else 0.0} for name, values in _stats.items()}


def saturated(name: str) -> bool:
    with _lock:
        # Semaphore values are intentionally opaque. Active workers at the
        # configured bound mean a new interactive request should downgrade.
        return _active[name] >= 1 if name in {"embedding", "reranker"} else False


@contextmanager
def acquire_inference(name: str, *, uses_gpu: bool):
    """Acquire a model-specific slot and, when relevant, the shared GPU slot.

    The nested order is stable for all callers, so a queued LLM, embedder and
    reranker cannot create an in-process lock cycle.  Callers retain their
    normal CPU fallback when CUDA initialization or allocation fails.
    """
    with acquire(name) as model_wait:
        if not uses_gpu:
            yield {"model_wait_ms": model_wait, "gpu_wait_ms": 0.0}
            return
        with acquire("gpu_inference") as gpu_wait:
            yield {"model_wait_ms": model_wait, "gpu_wait_ms": gpu_wait}
